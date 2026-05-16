"""Unified training runtime (Phase 1): single execution path for train/distill/resume."""

from __future__ import annotations

import copy
import hashlib
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Dict, List, Mapping, Optional, Tuple

import torch

from zynthe.core.config.config_manager import ConfigManager
from zynthe.core.models.model_loader import load_models
from zynthe.core.pkg.manifest import MANIFEST_FILENAME, ArtifactRecord, Manifest, compute_sha256
from zynthe.data.dataloaders import create_dataloaders

LOG = logging.getLogger(__name__)

_ALLOWED_ENGINES = {"legacy", "causal_lm_core", "causal_lm_core_stable"}


@dataclass
class RuntimeOptions:
    config_path: Optional[str] = None
    overrides: Dict[str, Any] = field(default_factory=dict)
    load_model_dir: Optional[str] = None
    load_checkpoint_path: Optional[str] = None
    checkpoint_non_strict: bool = False
    save_model: bool = True
    save_model_dir: Optional[str] = None
    save_checkpoint: bool = False
    checkpoint_path: Optional[str] = None

    @classmethod
    def from_namespace(
        cls, args: Any, overrides: Optional[Dict[str, Any]] = None
    ) -> "RuntimeOptions":
        return cls(
            config_path=getattr(args, "config", None),
            overrides=overrides or {},
            load_model_dir=getattr(args, "load_model_dir", None),
            load_checkpoint_path=getattr(args, "load_checkpoint_path", None),
            checkpoint_non_strict=bool(getattr(args, "checkpoint_non_strict", False)),
            save_model=bool(getattr(args, "save_model", True)),
            save_model_dir=getattr(args, "save_model_dir", None),
            save_checkpoint=bool(getattr(args, "save_checkpoint", False)),
            checkpoint_path=getattr(args, "checkpoint_path", None),
        )


@dataclass
class RuntimeResult:
    success: bool
    experiment_id: str
    experiment_dir: str
    engine: str
    config_hash: str
    preflight_can_proceed: bool
    resume_decision: Dict[str, Any]
    manifest_path: Optional[str]
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)


class UnifiedTrainingRuntime:
    """Single canonical training/distillation execution graph."""

    def run(self, options: RuntimeOptions) -> RuntimeResult:
        warnings: List[str] = []
        errors: List[str] = []
        manifest_path: Optional[str] = None
        quantization_summary_path: Optional[Path] = None

        cfg_manager = ConfigManager(config_path=options.config_path, overrides=options.overrides)
        normalized, norm_warnings = self._normalize_config(
            copy.deepcopy(cfg_manager.resolved_config)
        )
        warnings.extend(norm_warnings)

        config_hash = self._compute_config_hash(normalized)
        frozen_snapshot = self._deep_freeze(copy.deepcopy(normalized))
        engine = str(normalized.get("train", {}).get("engine", "legacy")).lower()
        seed = int(normalized.get("seed", normalized.get("runtime", {}).get("seed", 42)))

        # Keep manager config aligned with normalized config for downstream components.
        cfg_manager.resolved_config = normalized

        # Phase 1: config preflight
        from zynthe.core.preflight.analyser import run_preflight_check, validate_config_only

        cfg_validation = validate_config_only(normalized)
        if not cfg_validation.get("is_valid", False):
            errors.extend([str(e) for e in cfg_validation.get("errors", [])])
            return RuntimeResult(
                success=False,
                experiment_id=cfg_manager.experiment_id,
                experiment_dir=cfg_manager.experiment_dir,
                engine=engine,
                config_hash=config_hash,
                preflight_can_proceed=False,
                resume_decision={"decision": "abort", "reason": "config_validation_failed"},
                manifest_path=None,
                warnings=warnings,
                errors=errors,
            )

        # Phase 2: models + data
        teacher, student, tokenizer = load_models(
            cfg_manager,
            cfg_manager.device(),
        )

        if options.load_model_dir:
            student, tokenizer = self._load_reusable_student(
                options.load_model_dir,
                engine=engine,
                device=cfg_manager.device(),
                fallback_tokenizer=tokenizer,
            )

        train_loader, val_loader = create_dataloaders(normalized, tokenizer)

        # Phase 3: preflight gate
        preflight_dir = (
            Path(cfg_manager.paths.get("logs", cfg_manager.experiment_dir)) / "preflight"
        )
        preflight_dir.mkdir(parents=True, exist_ok=True)
        preflight_report = run_preflight_check(
            teacher_model=teacher,
            student_model=student,
            dataset=getattr(train_loader, "dataset", None),
            config=normalized,
            save_report=False,
            output_dir=str(preflight_dir),
        )

        preflight_can_proceed = bool(preflight_report.get("can_proceed", False))
        preflight_report_path = preflight_dir / "preflight_runtime_report.json"
        preflight_report_path.write_text(json.dumps(preflight_report, indent=2), encoding="utf-8")

        if not preflight_can_proceed:
            errors.extend([str(b) for b in preflight_report.get("blockers", [])])
            manifest_path = self._write_manifest(
                cfg_manager=cfg_manager,
                normalized_config=normalized,
                config_hash=config_hash,
                engine=engine,
                seed=seed,
                model_sizes=self._model_sizes(teacher, student),
                dataset_hash=self._dataset_hash(normalized),
                preflight_status={"can_proceed": False, "path": str(preflight_report_path)},
                resume_decision={"decision": "abort", "reason": "preflight_blocked"},
                artifacts_extra=[("preflight_report", preflight_report_path)],
                frozen_snapshot=frozen_snapshot,
            )
            return RuntimeResult(
                success=False,
                experiment_id=cfg_manager.experiment_id,
                experiment_dir=cfg_manager.experiment_dir,
                engine=engine,
                config_hash=config_hash,
                preflight_can_proceed=False,
                resume_decision={"decision": "abort", "reason": "preflight_blocked"},
                manifest_path=manifest_path,
                warnings=warnings,
                errors=errors,
            )

        # Phase 4: engine router (single source of truth)
        trainer = self._build_trainer(
            engine=engine,
            teacher=teacher,
            student=student,
            tokenizer=tokenizer,
            config=normalized,
            device=cfg_manager.device(),
            experiment_dir=cfg_manager.experiment_dir,
            train_loader=train_loader,
            warnings=warnings,
        )

        # Phase 5: resume automation
        resume_decision = self._apply_resume_logic(
            trainer=trainer,
            engine=engine,
            experiment_dir=cfg_manager.experiment_dir,
            explicit_checkpoint=options.load_checkpoint_path,
            strict=not options.checkpoint_non_strict,
            device=cfg_manager.device(),
        )

        # Phase 6: train
        trainer.fit(train_loader, val_loader)

        # Phase 6.5: optional quantization pipeline (PTQ / QAT)
        quantization_summary_path, quantization_warnings = self._run_optional_quantization(
            config=normalized,
            experiment_dir=cfg_manager.experiment_dir,
        )
        warnings.extend(quantization_warnings)

        # Phase 7: save artifacts
        self._save_optional_artifacts(
            trainer=trainer,
            tokenizer=tokenizer,
            cfg_manager=cfg_manager,
            options=options,
            engine=engine,
        )

        dataset_hash = getattr(trainer, "dataset_hash", "") or self._dataset_hash(normalized)
        manifest_path = self._write_manifest(
            cfg_manager=cfg_manager,
            normalized_config=normalized,
            config_hash=config_hash,
            engine=engine,
            seed=seed,
            model_sizes=self._model_sizes(teacher, student),
            dataset_hash=dataset_hash,
            preflight_status={"can_proceed": True, "path": str(preflight_report_path)},
            resume_decision=resume_decision,
            artifacts_extra=self._build_runtime_artifacts(
                preflight_report_path=preflight_report_path,
                quantization_summary_path=quantization_summary_path,
            ),
            frozen_snapshot=frozen_snapshot,
        )

        return RuntimeResult(
            success=True,
            experiment_id=cfg_manager.experiment_id,
            experiment_dir=cfg_manager.experiment_dir,
            engine=engine,
            config_hash=config_hash,
            preflight_can_proceed=True,
            resume_decision=resume_decision,
            manifest_path=manifest_path,
            warnings=warnings,
            errors=errors,
        )

    # ------------------------------------------------------------------
    # Normalization
    # ------------------------------------------------------------------
    def _normalize_config(self, cfg: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str]]:
        warnings: List[str] = []
        train = cfg.setdefault("train", {})
        distill = cfg.setdefault("distillation", {})
        cfg.setdefault("runtime", {})

        train.setdefault("engine", "legacy")
        train.setdefault("epochs", train.get("num_epochs", 1))
        train.setdefault("num_epochs", train.get("epochs", 1))
        train.setdefault("gradient_accumulation_steps", train.get("grad_accum_steps", 1))

        if "seed" not in cfg:
            cfg["seed"] = int(cfg.get("runtime", {}).get("seed", 42))
        cfg["runtime"]["seed"] = int(cfg["seed"])

        engine = str(train.get("engine", "legacy")).lower()
        if engine not in _ALLOWED_ENGINES:
            raise ValueError(
                f"Unsupported train.engine='{engine}'. Allowed={sorted(_ALLOWED_ENGINES)}"
            )

        task_type = str(distill.get("task_type", "")).lower()
        if engine in {"causal_lm_core", "causal_lm_core_stable"} and task_type not in {
            "",
            "causal_lm",
            "gpt",
            "language_modeling",
        }:
            warnings.append(
                "Engine is causal_lm_core* but distillation.task_type is non-LM; runtime will proceed with caution."
            )

        if engine == "legacy" and task_type in {"causal_lm", "gpt", "language_modeling"}:
            warnings.append(
                "Legacy engine selected with LM task_type; consider train.engine=causal_lm_core."
            )

        return cfg, warnings

    @staticmethod
    def _compute_config_hash(cfg: Mapping[str, Any]) -> str:
        payload = json.dumps(cfg, sort_keys=True, default=str, separators=(",", ":")).encode(
            "utf-8"
        )
        return hashlib.sha256(payload).hexdigest()

    def _deep_freeze(self, obj: Any) -> Any:
        if isinstance(obj, dict):
            return MappingProxyType({k: self._deep_freeze(v) for k, v in obj.items()})
        if isinstance(obj, list):
            return tuple(self._deep_freeze(v) for v in obj)
        return obj

    # ------------------------------------------------------------------
    # Router
    # ------------------------------------------------------------------
    def _build_trainer(
        self,
        *,
        engine: str,
        teacher: Any,
        student: Any,
        tokenizer: Any,
        config: Dict[str, Any],
        device: torch.device,
        experiment_dir: str,
        train_loader: Any,
        warnings: List[str],
    ) -> Any:
        if engine == "legacy":
            from zynthe.training.trainer import Trainer

            return Trainer(
                teacher=teacher,
                student=student,
                tokenizer=tokenizer,
                config=config,
                device=device,
                experiment_dir=experiment_dir,
            )

        from zynthe.core.distillers.causal_lm import RegressionGate, SafeCausalLMTrainer

        if engine == "causal_lm_core_stable":
            gate = RegressionGate.from_mapping(config.get("train", {}))
            gate_report = gate.run(
                teacher=teacher,
                student=student,
                tokenizer=tokenizer,
                config=config,
                device=device,
                experiment_dir=experiment_dir,
                train_loader=train_loader,
            )
            if not gate_report.passed:
                raise RuntimeError(
                    "RegressionGate failed for causal_lm_core_stable: "
                    f"reasons={gate_report.reasons}, "
                    f"max_token_loss_diff={gate_report.max_token_loss_diff:.6f}, "
                    f"max_grad_norm_diff={gate_report.max_grad_norm_diff:.6f}"
                )
            warnings.append("RegressionGate passed for causal_lm_core_stable")

        return SafeCausalLMTrainer(
            teacher=teacher,
            student=student,
            tokenizer=tokenizer,
            config=config,
            device=device,
            experiment_dir=experiment_dir,
        )

    # ------------------------------------------------------------------
    # Resume
    # ------------------------------------------------------------------
    def _apply_resume_logic(
        self,
        *,
        trainer: Any,
        engine: str,
        experiment_dir: str,
        explicit_checkpoint: Optional[str],
        strict: bool,
        device: torch.device,
    ) -> Dict[str, Any]:
        if explicit_checkpoint:
            return self._resume_from_path(
                trainer, engine, explicit_checkpoint, strict=strict, device=device
            )

        latest = self._find_latest_valid_checkpoint(experiment_dir)
        if latest is None:
            return {"decision": "fresh_start", "reason": "no_valid_checkpoint_found"}

        resume_report = self._resume_from_path(
            trainer, engine, str(latest), strict=strict, device=device
        )
        resume_report.setdefault("decision", "auto_resume")
        return resume_report

    def _resume_from_path(
        self,
        trainer: Any,
        engine: str,
        checkpoint_path: str,
        *,
        strict: bool,
        device: torch.device,
    ) -> Dict[str, Any]:
        if engine == "legacy":
            from zynthe.core.models.model_saver import load_checkpoint

            _, metadata = load_checkpoint(
                model=trainer.student,
                optimizer=trainer.optimizer,
                path=checkpoint_path,
                scheduler=getattr(trainer, "scheduler", None),
                scaler=getattr(trainer, "scaler", None),
                map_location=str(device),
                strict=bool(strict),
            )
            return {
                "decision": "resume",
                "source": checkpoint_path,
                "engine": "legacy",
                "strict": bool(strict),
                "epoch": int(metadata.epoch) if metadata else None,
                "step": int(metadata.global_step) if metadata else None,
            }

        report = trainer.resume_from_checkpoint(checkpoint_path)
        report.update({"decision": "resume", "source": checkpoint_path, "engine": engine})
        return report

    def _find_latest_valid_checkpoint(self, experiment_dir: str) -> Optional[Path]:
        ckpt_dir = Path(experiment_dir) / "checkpoints"
        if not ckpt_dir.exists():
            return None

        candidates = sorted(
            [p for p in ckpt_dir.glob("*.pt") if p.is_file()],
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        for path in candidates:
            if self._checkpoint_integrity_ok(path):
                return path
        return None

    @staticmethod
    def _checkpoint_integrity_ok(path: Path) -> bool:
        try:
            payload = torch.load(str(path), map_location="cpu", weights_only=False)
        except Exception:
            return False
        if not isinstance(payload, dict):
            return False
        return isinstance(payload.get("model_state_dict"), dict)

    # ------------------------------------------------------------------
    # Artifacts and manifest
    # ------------------------------------------------------------------
    @staticmethod
    def _build_runtime_artifacts(
        *,
        preflight_report_path: Path,
        quantization_summary_path: Optional[Path],
    ) -> List[Tuple[str, Path]]:
        artifacts: List[Tuple[str, Path]] = [("preflight_report", preflight_report_path)]
        if quantization_summary_path is not None:
            artifacts.append(("quantization_summary", quantization_summary_path))
        return artifacts

    def _run_optional_quantization(
        self,
        *,
        config: Dict[str, Any],
        experiment_dir: str,
    ) -> Tuple[Optional[Path], List[str]]:
        warnings: List[str] = []
        quant_cfg = config.get("quantization", {}) or {}
        if not bool(quant_cfg.get("enable", False)):
            return None, warnings

        mode = str(quant_cfg.get("mode", "ptq")).lower()
        if mode not in {"ptq", "qat"}:
            warnings.append(f"Unsupported quantization.mode='{mode}', skipping quantization stage.")
            return None, warnings

        runner_cfg = copy.deepcopy(config)
        runner_quant_cfg = runner_cfg.setdefault("quantization", {})
        default_output_dir = Path(experiment_dir) / "quantization"
        runner_quant_cfg.setdefault("output_dir", str(default_output_dir))

        summary: Dict[str, Any]
        if mode == "qat":
            from zynthe.core.quant.qat import QATRunner

            # Note: current QAT runner loads model bundle from config instead of in-memory trainer state.
            warnings.append(
                "QAT stage is linked in pipeline, but it currently reloads models from config artifacts."
            )
            summary = QATRunner(runner_cfg).run()
        else:
            from zynthe.core.quant.ptq import PTQRunner

            summary = PTQRunner(runner_cfg).run()

        logs_dir = Path(experiment_dir) / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)
        summary_path = logs_dir / "runtime_quantization_summary.json"
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        return summary_path, warnings

    def _save_optional_artifacts(
        self,
        *,
        trainer: Any,
        tokenizer: Any,
        cfg_manager: ConfigManager,
        options: RuntimeOptions,
        engine: str,
    ) -> None:
        if options.save_model:
            from zynthe.core.models.model_saver import save_model

            out_dir = options.save_model_dir or str(
                Path(cfg_manager.experiment_dir) / "student_model"
            )
            save_model(
                model=trainer.student,
                path=out_dir,
                tokenizer=tokenizer,
                metadata={
                    "experiment_id": cfg_manager.experiment_id,
                    "engine": engine,
                    "config_path": options.config_path,
                },
                use_safetensors=True,
            )

        if options.save_checkpoint:
            out_ckpt = options.checkpoint_path or str(
                Path(cfg_manager.experiment_dir) / "checkpoints" / "latest.pt"
            )
            if hasattr(trainer, "save_explicit_checkpoint"):
                trainer.save_explicit_checkpoint(out_ckpt)
            else:
                from zynthe.core.models.model_saver import CheckpointMetadata, save_checkpoint

                save_checkpoint(
                    model=trainer.student,
                    optimizer=trainer.optimizer,
                    path=out_ckpt,
                    scheduler=getattr(trainer, "scheduler", None),
                    scaler=getattr(trainer, "scaler", None),
                    metadata=CheckpointMetadata(
                        stage="training_complete",
                        epoch=len(getattr(trainer, "train_losses", [])),
                        global_step=0,
                        best_metric=float(getattr(trainer, "best_val_loss", 0.0)),
                        metrics={"best_val_loss": float(getattr(trainer, "best_val_loss", 0.0))},
                        extras={"experiment_id": cfg_manager.experiment_id},
                    ),
                )

    def _write_manifest(
        self,
        *,
        cfg_manager: ConfigManager,
        normalized_config: Dict[str, Any],
        config_hash: str,
        engine: str,
        seed: int,
        model_sizes: Dict[str, int],
        dataset_hash: str,
        preflight_status: Dict[str, Any],
        resume_decision: Dict[str, Any],
        artifacts_extra: Optional[List[Tuple[str, Path]]] = None,
        frozen_snapshot: Any = None,
    ) -> Optional[str]:
        try:
            exp_dir = Path(cfg_manager.experiment_dir)
            artifacts: List[ArtifactRecord] = []

            resolved_cfg_path = Path(
                cfg_manager.paths.get("resolved_config", exp_dir / "resolved_config.yaml")
            )
            if resolved_cfg_path.exists():
                artifacts.append(
                    ArtifactRecord.from_file(
                        "resolved_config",
                        resolved_cfg_path.relative_to(exp_dir),
                        resolved_cfg_path,
                    )
                )

            for logical_name, path in artifacts_extra or []:
                if path.exists() and path.is_file():
                    artifacts.append(
                        ArtifactRecord.from_file(
                            logical_name,
                            path.relative_to(exp_dir),
                            path,
                        )
                    )

            manifest = Manifest.create(
                package_name="zyn_runtime",
                artifacts=artifacts,
                metadata={
                    "engine": engine,
                    "seed": int(seed),
                    "config_hash": config_hash,
                    "dataset_hash": dataset_hash,
                    "model_sizes": model_sizes,
                    "preflight": preflight_status,
                    "resume": resume_decision,
                    "normalized_config": normalized_config,
                    "frozen_config_repr": repr(frozen_snapshot)[:4000],
                },
            )
            output = exp_dir / "logs" / MANIFEST_FILENAME
            manifest.save(output)
            return str(output)
        except Exception:
            LOG.debug("Failed to write runtime manifest", exc_info=True)
            return None

    @staticmethod
    def _dataset_hash(config: Mapping[str, Any]) -> str:
        data_cfg = config.get("data", {}) if isinstance(config, Mapping) else {}
        train_path = data_cfg.get("train_path")
        val_path = data_cfg.get("val_path")
        values: List[str] = []
        for p in (train_path, val_path):
            if not p:
                continue
            path = Path(str(p))
            if path.exists() and path.is_file():
                values.append(compute_sha256(path))
        if not values:
            return ""
        return hashlib.sha256("|".join(values).encode("utf-8")).hexdigest()

    @staticmethod
    def _model_sizes(teacher: Any, student: Any) -> Dict[str, int]:
        return {
            "teacher_total": int(sum(p.numel() for p in teacher.parameters())),
            "teacher_trainable": int(
                sum(p.numel() for p in teacher.parameters() if p.requires_grad)
            ),
            "student_total": int(sum(p.numel() for p in student.parameters())),
            "student_trainable": int(
                sum(p.numel() for p in student.parameters() if p.requires_grad)
            ),
        }

    @staticmethod
    def _load_reusable_student(
        model_dir: str,
        *,
        engine: str,
        device: torch.device,
        fallback_tokenizer: Any,
    ) -> Tuple[Any, Any]:
        from transformers import (
            AutoModelForCausalLM,
            AutoModelForSequenceClassification,
            AutoTokenizer,
        )

        from zynthe.core.models.model_saver import load_model

        model_class = (
            AutoModelForCausalLM
            if engine in {"causal_lm_core", "causal_lm_core_stable"}
            else AutoModelForSequenceClassification
        )
        student, tokenizer, _ = load_model(
            model_class,
            model_dir,
            AutoTokenizer,
            map_location=str(device),
        )
        return student, (tokenizer or fallback_tokenizer)
