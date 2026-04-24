#!/usr/bin/env python3
"""Lightweight CPU-safe system tester for Zynthe.

Usage:
  python tools/system_tester.py --config configs/causal_lm_core_minimal.yaml
"""

from __future__ import annotations

import argparse
import copy
import importlib
import importlib.util
import json
import os
import platform
import random
import sys
import tempfile
import traceback
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    from core.config.config_manager import ConfigManager

import numpy as np
import torch
import torch.nn as nn

# Keep optional heavy deps opt-out for text-only checks.
os.environ.setdefault("TRANSFORMERS_NO_TORCHVISION", "1")
os.environ.setdefault("TRANSFORMERS_NO_PIL", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# -----------------------------------------------------------------------------
# Report dataclasses
# -----------------------------------------------------------------------------


@dataclass
class BaseReport:
    status: str = "FAIL"
    category: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)


@dataclass
class EnvironmentReport(BaseReport):
    pass


@dataclass
class ConfigReport(BaseReport):
    pass


@dataclass
class DataReport(BaseReport):
    pass


@dataclass
class ModelReport(BaseReport):
    pass


@dataclass
class CoreWiringReport(BaseReport):
    pass


@dataclass
class CheckpointReport(BaseReport):
    pass


@dataclass
class RegressionMiniReport(BaseReport):
    pass


@dataclass
class SystemHealthReport:
    environment: EnvironmentReport
    config: ConfigReport
    data: DataReport
    model: ModelReport
    core: CoreWiringReport
    checkpoint: CheckpointReport
    regression: RegressionMiniReport
    overall_status: str
    blocking_issues: List[str]


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------


def _color(text: str, code: str) -> str:
    return f"\033[{code}m{text}\033[0m"


def _ok(text: str) -> str:
    return _color(text, "92")


def _warn(text: str) -> str:
    return _color(text, "93")


def _bad(text: str) -> str:
    return _color(text, "91")


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, (np.generic,)):
        return value.item()
    if isinstance(value, (Path,)):
        return str(value)
    if isinstance(value, torch.device):
        return str(value)
    try:
        json.dumps(value)
        return value
    except Exception:
        return str(value)


def _classify_exception(exc: Exception) -> str:
    name = type(exc).__name__.lower()
    msg = str(exc).lower()
    if "import" in name or "module" in msg or "no module named" in msg:
        return "ImportError"
    if "config" in name or "config" in msg:
        return "ConfigError"
    if "shape" in msg or "size mismatch" in msg or "dimension" in msg:
        return "ShapeMismatch"
    if "nan" in msg or "inf" in msg or "overflow" in msg:
        return "NumericalInstability"
    if "dependency" in msg or "not installed" in msg:
        return "MissingDependency"
    return type(exc).__name__


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _safe_batch_from_loader(loader: Any) -> Dict[str, torch.Tensor]:
    batch = next(iter(loader))
    if not isinstance(batch, dict):
        raise ValueError("Batch is not a dict")
    output: Dict[str, torch.Tensor] = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            output[k] = v
    return output


def _build_synthetic_batch(
    *,
    batch_size: int = 2,
    seq_len: int = 8,
    vocab_size: int = 128,
) -> Dict[str, torch.Tensor]:
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), dtype=torch.long)
    attention_mask = torch.ones((batch_size, seq_len), dtype=torch.long)
    labels = input_ids.clone()
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


class TinyTokenizer:
    """Minimal tokenizer-compatible callable for lightweight data wiring."""

    def __init__(self, vocab_size: int = 128, pad_token_id: int = 0):
        self.vocab_size = int(vocab_size)
        self.pad_token_id = int(pad_token_id)
        self.eos_token_id = 1
        self.pad_token = "<pad>"
        self.eos_token = "<eos>"

    def __call__(
        self,
        text: str,
        *,
        truncation: bool = True,
        max_length: int = 128,
        padding: Any = "max_length",
        return_tensors: str = "pt",
        add_special_tokens: bool = True,
    ) -> Dict[str, torch.Tensor]:
        del truncation, add_special_tokens
        tokens = [min((ord(ch) % (self.vocab_size - 2)) + 2, self.vocab_size - 1) for ch in (text or "")]
        max_len = int(max_length)
        tokens = tokens[:max_len]
        if padding == "max_length":
            tokens = tokens + [self.pad_token_id] * (max_len - len(tokens))
        attn = [0 if t == self.pad_token_id else 1 for t in tokens]
        ids = torch.tensor([tokens], dtype=torch.long)
        mask = torch.tensor([attn], dtype=torch.long)
        if return_tensors == "pt":
            return {"input_ids": ids, "attention_mask": mask}
        return {"input_ids": ids.tolist(), "attention_mask": mask.tolist()}  # pragma: no cover


class TinyCausalLM(nn.Module):
    """Tiny causal LM for fast CPU-only forward/backward tests."""

    def __init__(self, vocab_size: int = 128, hidden_size: int = 64):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size)
        self.config = type("Cfg", (), {"vocab_size": vocab_size, "model_type": "tiny_causal_lm"})()

    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None, labels: Optional[torch.Tensor] = None, **kwargs):
        del attention_mask, labels, kwargs
        h = self.embed(input_ids)
        logits = self.lm_head(h)
        return {"logits": logits}

    def resize_token_embeddings(self, new_size: int):
        new_size = int(new_size)
        old_size = self.embed.num_embeddings
        if new_size == old_size:
            return self.embed

        new_embed = nn.Embedding(new_size, self.embed.embedding_dim)
        new_head = nn.Linear(self.lm_head.in_features, new_size)
        with torch.no_grad():
            copy_n = min(old_size, new_size)
            new_embed.weight[:copy_n].copy_(self.embed.weight[:copy_n])
            new_head.weight[:copy_n].copy_(self.lm_head.weight[:copy_n])
            new_head.bias[:copy_n].copy_(self.lm_head.bias[:copy_n])
        self.embed = new_embed
        self.lm_head = new_head
        self.config.vocab_size = new_size
        return self.embed


# -----------------------------------------------------------------------------
# Main tester
# -----------------------------------------------------------------------------


class ZyntheSystemTester:
    def __init__(self, config_path: str):
        self.config_path = str(config_path)
        self.cfg_manager: Optional[ConfigManager] = None
        self.cfg: Dict[str, Any] = {}
        self.device = torch.device("cpu")

        self.tokenizer: Optional[TinyTokenizer] = None
        self.teacher: Optional[TinyCausalLM] = None
        self.student: Optional[TinyCausalLM] = None

        self.train_loader: Optional[Any] = None
        self.val_loader: Optional[Any] = None
        self.sample_batch: Optional[Dict[str, torch.Tensor]] = None

    # -----------------------------
    # 1) Environment
    # -----------------------------
    def test_environment(self) -> EnvironmentReport:
        report = EnvironmentReport()
        required_modules = ["torch", "yaml", "numpy", "transformers"]
        optional_modules = ["seaborn", "datasets", "evaluate", "typer"]
        module_status: Dict[str, bool] = {}

        try:
            for m in required_modules:
                module_status[m] = importlib.util.find_spec(m) is not None
                if not module_status[m]:
                    raise ImportError(f"Required package missing: {m}")
            for m in optional_modules:
                try:
                    module_status[m] = importlib.util.find_spec(m) is not None
                except Exception:
                    module_status[m] = False

            report.status = "PASS"
            report.details = {
                "python_version": platform.python_version(),
                "platform": platform.platform(),
                "torch_version": torch.__version__,
                "cuda_available": bool(torch.cuda.is_available()),
                "cuda_device_count": int(torch.cuda.device_count() if torch.cuda.is_available() else 0),
                "required_packages": module_status,
            }
            for name, ok in module_status.items():
                if name in optional_modules and not ok:
                    report.warnings.append(f"Optional package missing: {name}")
        except Exception as exc:
            report.status = "FAIL"
            report.category = _classify_exception(exc)
            report.errors.append(str(exc))
            report.details["traceback"] = traceback.format_exc(limit=3)

        return report

    # -----------------------------
    # 2) Config
    # -----------------------------
    def test_config(self) -> ConfigReport:
        report = ConfigReport()
        try:
            from core.config.config_manager import ConfigManager

            self.cfg_manager = ConfigManager(config_path=self.config_path)
            self.cfg = dict(self.cfg_manager.resolved_config)  # type: ignore[union-attr]
            self.device = torch.device("cpu")

            required_keys = ["train", "model", "data", "distillation"]
            missing = [k for k in required_keys if k not in self.cfg]
            if missing:
                raise ValueError(f"Missing required config sections: {missing}")

            train_cfg = self.cfg.get("train", {})
            model_cfg = self.cfg.get("model", {})
            data_cfg = self.cfg.get("data", {})
            normalization_checks = {
                "train.num_epochs_exists": "num_epochs" in train_cfg,
                "train.gradient_accumulation_steps_exists": "gradient_accumulation_steps" in train_cfg,
                "model.student_name_exists": bool(model_cfg.get("student_name")),
                "model.tokenizer_name_exists": bool(model_cfg.get("tokenizer_name")),
                "data.train_path_exists": bool(data_cfg.get("train_path")),
                "data.val_path_exists": bool(data_cfg.get("val_path")),
            }

            failed_norm = [k for k, ok in normalization_checks.items() if not ok]
            if failed_norm:
                raise ValueError(f"Config normalization checks failed: {failed_norm}")

            report.status = "PASS"
            report.details = {
                "config_path": self.config_path,
                "experiment_dir": self.cfg_manager.experiment_dir if self.cfg_manager else "",  # type: ignore[union-attr]
                "normalization_checks": normalization_checks,
            }
        except Exception as exc:
            report.status = "FAIL"
            report.category = _classify_exception(exc)
            report.errors.append(str(exc))
            report.details["traceback"] = traceback.format_exc(limit=3)

        return report

    # -----------------------------
    # 3) Data
    # -----------------------------
    def test_data(self) -> DataReport:
        report = DataReport()
        try:
            if not self.cfg:
                raise RuntimeError("Config not initialized")

            if self.tokenizer is None:
                self.tokenizer = TinyTokenizer(vocab_size=128)

            batch_size = int(self.cfg.get("train", {}).get("batch_size", 2))
            seq_len = int(self.cfg.get("model", {}).get("max_length", 32))
            seq_len = min(max(seq_len, 8), 64)

            class _MiniDataset(torch.utils.data.Dataset):
                def __len__(self):
                    return 4

                def __getitem__(self, idx):
                    del idx
                    return {
                        "input_ids": torch.randint(0, 127, (seq_len,), dtype=torch.long),
                        "attention_mask": torch.ones((seq_len,), dtype=torch.long),
                        "labels": torch.randint(0, 127, (seq_len,), dtype=torch.long),
                    }

            def _collate(records):
                keys = records[0].keys()
                return {k: torch.stack([r[k] for r in records], dim=0) for k in keys}

            self.train_loader = torch.utils.data.DataLoader(_MiniDataset(), batch_size=batch_size, shuffle=False, collate_fn=_collate)
            self.val_loader = torch.utils.data.DataLoader(_MiniDataset(), batch_size=batch_size, shuffle=False, collate_fn=_collate)
            self.sample_batch = _safe_batch_from_loader(self.train_loader)

            required_batch_keys = ["input_ids", "attention_mask"]
            lm_required = ["labels"]
            missing = [k for k in required_batch_keys if k not in self.sample_batch]
            missing += [k for k in lm_required if k not in self.sample_batch]
            if missing:
                raise ValueError(f"Batch missing required keys: {missing}")

            report.status = "PASS"
            report.details = {
                "train_batches": int(len(self.train_loader)),  # type: ignore[arg-type]
                "val_batches": int(len(self.val_loader)),  # type: ignore[arg-type]
                "data_loader_type": "torch.utils.data.DataLoader",
                "batch_shapes": {k: list(v.shape) for k, v in self.sample_batch.items()},
            }
        except Exception as exc:
            report.status = "FAIL"
            report.category = _classify_exception(exc)
            report.errors.append(str(exc))
            report.details["traceback"] = traceback.format_exc(limit=3)

        return report

    # -----------------------------
    # 4) Model
    # -----------------------------
    def test_model(self) -> ModelReport:
        report = ModelReport()
        try:
            if self.cfg_manager is None:
                raise RuntimeError("ConfigManager not initialized")

            # Keep model check local-only for CPU safety.
            teacher = TinyCausalLM(vocab_size=128, hidden_size=64)
            student = TinyCausalLM(vocab_size=128, hidden_size=64)
            if self.tokenizer is None:
                self.tokenizer = TinyTokenizer(vocab_size=128)
            self.teacher, self.student = teacher, student
            model_source = "synthetic_tiny_cpu"

            teacher.to(self.device)
            student.to(self.device)
            teacher.eval()
            student.eval()

            if self.sample_batch is None:
                vocab = int(getattr(student.config, "vocab_size", 128)) if hasattr(student, "config") else 128
                self.sample_batch = _build_synthetic_batch(vocab_size=max(vocab, 16))

            with torch.no_grad():
                inputs = {k: v.to(self.device) for k, v in self.sample_batch.items() if k in {"input_ids", "attention_mask", "labels"}}
                t_out = teacher(**inputs)
                s_out = student(**inputs)

            t_logits = t_out["logits"] if isinstance(t_out, dict) else t_out.logits
            s_logits = s_out["logits"] if isinstance(s_out, dict) else s_out.logits

            if t_logits.ndim != 3 or s_logits.ndim != 3:
                raise ValueError(f"Unexpected logits shape: teacher={tuple(t_logits.shape)} student={tuple(s_logits.shape)}")
            if not torch.isfinite(t_logits).all() or not torch.isfinite(s_logits).all():
                raise ValueError("NaN/Inf logits detected")

            report.status = "PASS"
            report.details = {
                "model_source": model_source,
                "teacher_params": int(sum(p.numel() for p in teacher.parameters())),
                "student_params": int(sum(p.numel() for p in student.parameters())),
                "teacher_logits_shape": list(t_logits.shape),
                "student_logits_shape": list(s_logits.shape),
            }
        except Exception as exc:
            report.status = "FAIL"
            report.category = _classify_exception(exc)
            report.errors.append(str(exc))
            report.details["traceback"] = traceback.format_exc(limit=3)

        return report

    # -----------------------------
    # 5) Causal-LM core wiring
    # -----------------------------
    def test_core_wiring(self) -> CoreWiringReport:
        report = CoreWiringReport()
        try:
            if self.teacher is None or self.student is None:
                raise RuntimeError("Models not initialized")

            from core.distillers.causal_lm import SafeCausalLMTrainer

            cfg = copy.deepcopy(self.cfg)
            cfg.setdefault("train", {})["use_amp"] = False
            cfg["train"]["epochs"] = 1
            cfg["train"]["batch_size"] = int(cfg["train"].get("batch_size", 2))

            trainer = SafeCausalLMTrainer(
                teacher=copy.deepcopy(self.teacher),
                student=copy.deepcopy(self.student),
                tokenizer=self.tokenizer,
                config=cfg,
                device=torch.device("cpu"),
                experiment_dir=self.cfg_manager.experiment_dir if self.cfg_manager else "experiments/system_tester",
            )

            batch = self.sample_batch or _build_synthetic_batch(vocab_size=128)
            inputs = trainer._prepare_batch(batch)

            trainer.teacher.eval()
            trainer.student.train()
            trainer.optimizer.zero_grad(set_to_none=True)

            with torch.no_grad():
                teacher_outputs = trainer.teacher(**inputs)
            student_outputs = trainer.student(**inputs)
            distill_out = trainer.distill_engine.compute_total_loss(
                student_outputs=student_outputs,
                teacher_outputs=teacher_outputs,
                labels=inputs["labels"],
            )
            loss = distill_out.total

            if not torch.isfinite(loss).all():
                raise ValueError("Non-finite loss detected in core wiring")

            loss.backward()
            finite_grads = True
            grad_count = 0
            for p in trainer.student.parameters():
                if p.grad is None:
                    continue
                grad_count += 1
                if not torch.isfinite(p.grad).all():
                    finite_grads = False
                    break
            if not finite_grads:
                raise ValueError("Non-finite gradients detected")

            trainer.optimizer.step()
            trainer.optimizer.zero_grad(set_to_none=True)

            report.status = "PASS"
            report.details = {
                "loss": float(loss.item()),
                "grad_tensors": int(grad_count),
                "optimizer_step": True,
                "valid_tokens": int(distill_out.valid_tokens),
            }
        except Exception as exc:
            report.status = "FAIL"
            report.category = _classify_exception(exc)
            report.errors.append(str(exc))
            report.details["traceback"] = traceback.format_exc(limit=3)

        return report

    # -----------------------------
    # 6) Checkpoint lightweight
    # -----------------------------
    def test_checkpoint(self) -> CheckpointReport:
        report = CheckpointReport()
        try:
            if self.teacher is None or self.student is None:
                raise RuntimeError("Models not initialized")

            from core.distillers.causal_lm import SafeCausalLMTrainer, smart_load_checkpoint

            cfg = copy.deepcopy(self.cfg)
            cfg.setdefault("train", {})["use_amp"] = False

            trainer = SafeCausalLMTrainer(
                teacher=copy.deepcopy(self.teacher),
                student=copy.deepcopy(self.student),
                tokenizer=self.tokenizer,
                config=cfg,
                device=torch.device("cpu"),
                experiment_dir=self.cfg_manager.experiment_dir if self.cfg_manager else "experiments/system_tester",
            )

            with tempfile.TemporaryDirectory(prefix="zyn_ckpt_") as tmp:
                ckpt_path = Path(tmp) / "smoke.pt"
                trainer.save_explicit_checkpoint(str(ckpt_path))

                strict_info = trainer.resume_from_checkpoint(str(ckpt_path))

                # Fallback simulation via embedding resize mismatch.
                mismatch_student = copy.deepcopy(trainer.student)
                if hasattr(mismatch_student, "resize_token_embeddings") and hasattr(mismatch_student, "config"):
                    old_vocab = int(getattr(mismatch_student.config, "vocab_size", 128))
                    mismatch_student.resize_token_embeddings(old_vocab + 7)

                mismatch_optimizer = torch.optim.AdamW(mismatch_student.parameters(), lr=1e-3)
                fallback_report, _, _ = smart_load_checkpoint(
                    path=str(ckpt_path),
                    model=mismatch_student,
                    optimizer=mismatch_optimizer,
                    scheduler=None,
                    scaler=None,
                    map_location="cpu",
                    strict_first=True,
                    allow_shape_mismatch_fallback=True,
                )

            report.status = "PASS"
            report.details = {
                "strict_loaded": bool(strict_info.get("strict_loaded", False)),
                "fallback_used": bool(fallback_report.fallback_used),
                "fallback_shape_mismatch_count": int(len(fallback_report.shape_mismatch)),
                "fallback_skipped_optimizer": bool(fallback_report.skipped_optimizer),
            }
        except Exception as exc:
            report.status = "FAIL"
            report.category = _classify_exception(exc)
            report.errors.append(str(exc))
            report.details["traceback"] = traceback.format_exc(limit=3)

        return report

    # -----------------------------
    # 7) Regression mini (3 steps)
    # -----------------------------
    def test_regression_mini(self) -> RegressionMiniReport:
        report = RegressionMiniReport()
        try:
            if self.teacher is None or self.student is None:
                raise RuntimeError("Models not initialized")

            from core.distillers.causal_lm import SafeCausalLMTrainer, trace_from_trainer, verify_reproducibility

            cfg = copy.deepcopy(self.cfg)
            cfg.setdefault("train", {})["use_amp"] = False
            seed = int(cfg.get("seed", cfg.get("runtime", {}).get("seed", 42)))

            if self.train_loader is None:
                # Build mini loader from synthetic samples.
                synthetic = [_build_synthetic_batch(batch_size=1, seq_len=8, vocab_size=64) for _ in range(6)]

                class _MiniDataset(torch.utils.data.Dataset):
                    def __len__(self):
                        return len(synthetic)

                    def __getitem__(self, idx):
                        item = synthetic[idx]
                        return {k: v.squeeze(0) for k, v in item.items()}

                def _collate(records):
                    keys = records[0].keys()
                    return {k: torch.stack([r[k] for r in records], dim=0) for k in keys}

                self.train_loader = torch.utils.data.DataLoader(_MiniDataset(), batch_size=2, shuffle=False, collate_fn=_collate)

            def _builder():
                _set_seed(seed)
                trainer = SafeCausalLMTrainer(
                    teacher=copy.deepcopy(self.teacher),
                    student=copy.deepcopy(self.student),
                    tokenizer=self.tokenizer,
                    config=cfg,
                    device=torch.device("cpu"),
                    experiment_dir=self.cfg_manager.experiment_dir if self.cfg_manager else "experiments/system_tester",
                )
                return trace_from_trainer(trainer, self.train_loader, max_steps=3)

            det = verify_reproducibility(run_builder=_builder, compare_steps=3, tolerance=1e-6)

            report.status = "PASS" if det.passed else "FAIL"
            report.category = None if det.passed else "NumericalInstability"
            report.details = {
                "passed": bool(det.passed),
                "compared_steps": int(det.compared_steps),
                "tolerance": float(det.tolerance),
                "max_abs_token_loss_diff": float(det.max_abs_token_loss_diff),
                "max_abs_grad_norm_diff": float(det.max_abs_grad_norm_diff),
            }
            if not det.passed:
                report.errors.append("Determinism check failed for mini regression")
        except Exception as exc:
            report.status = "FAIL"
            report.category = _classify_exception(exc)
            report.errors.append(str(exc))
            report.details["traceback"] = traceback.format_exc(limit=3)

        return report

    def run_all(self) -> SystemHealthReport:
        env = self.test_environment()
        cfg = self.test_config()
        data = self.test_data()
        model = self.test_model()
        core = self.test_core_wiring()
        checkpoint = self.test_checkpoint()
        regression = self.test_regression_mini()

        module_map = {
            "environment": env,
            "config": cfg,
            "data": data,
            "model": model,
            "core": core,
            "checkpoint": checkpoint,
            "regression": regression,
        }
        blocking_issues: List[str] = []
        for name, rep in module_map.items():
            if rep.status != "PASS":
                summary = f"{name}:{rep.category or 'Failure'}:{'; '.join(rep.errors[:1])}"
                blocking_issues.append(summary)

        overall = "PASS" if not blocking_issues else "FAIL"

        return SystemHealthReport(
            environment=env,
            config=cfg,
            data=data,
            model=model,
            core=core,
            checkpoint=checkpoint,
            regression=regression,
            overall_status=overall,
            blocking_issues=blocking_issues,
        )


def _print_summary(report: SystemHealthReport) -> None:
    rows = [
        ("environment", report.environment.status),
        ("config", report.config.status),
        ("data", report.data.status),
        ("model", report.model.status),
        ("core", report.core.status),
        ("checkpoint", report.checkpoint.status),
        ("regression", report.regression.status),
    ]

    print("\n=== Zynthe System Tester ===")
    for name, status in rows:
        color_status = _ok(status) if status == "PASS" else _bad(status)
        print(f" - {name:12s}: {color_status}")

    if report.overall_status == "PASS":
        print(_ok("Overall: PASS"))
    else:
        print(_bad("Overall: FAIL"))
        print(_warn("Blocking issues:"))
        for issue in report.blocking_issues:
            print(f"   - {issue}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Lightweight CPU-safe Zynthe system tester")
    parser.add_argument("--config", required=True, type=str, help="Path to config YAML")
    args = parser.parse_args()

    tester = ZyntheSystemTester(config_path=args.config)
    report = tester.run_all()

    _print_summary(report)

    out_dir = PROJECT_ROOT / "diagnostics"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "system_report.json"
    with out_path.open("w", encoding="utf-8") as handle:
        json.dump(_to_jsonable(asdict(report)), handle, indent=2)

    print(f"Report written: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
