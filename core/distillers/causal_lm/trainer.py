"""Production-safe trainer for Causal-LM distillation."""

from __future__ import annotations

import copy
import hashlib
import json
import logging
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import numpy as np
import torch
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from .checkpoint import CheckpointMeta, TrainingState, save_training_checkpoint, smart_load_checkpoint
from .distillation import CausalLMDistillationEngine, DistillationConfig
from .fault_injection import FaultInjector
from .determinism import trace_from_trainer, verify_reproducibility
from .metrics import DistillationHealthMetrics, MetricStabilityMonitor, TokenMetricsAccumulator, compute_distill_alignment
from .determinism import runtime_determinism_env
from .validation import TrainingHealthReport, gradient_sanity_check, validate_distillation_numerics

LOG = logging.getLogger(__name__)


@dataclass
class TrainerRuntimeState:
    epoch: int = 0
    global_step: int = 0
    best_val_token_loss: float = float("inf")


class SafeCausalLMTrainer:
    """Unified deterministic loop: forward -> distill -> backward -> step -> log -> checkpoint."""

    def __init__(
        self,
        *,
        teacher: Any,
        student: Any,
        tokenizer: Any,
        config: Mapping[str, Any],
        device: torch.device,
        experiment_dir: str,
        websocket_callback: Optional[Any] = None,
    ):
        self.teacher = teacher.to(device)
        self.student = student.to(device)
        self.tokenizer = tokenizer
        self.config = dict(config)
        self.device = device
        self.experiment_dir = str(experiment_dir)
        self.websocket_callback = websocket_callback

        train_cfg = self.config.get("train", {})
        distill_cfg = self.config.get("distillation", {})
        checkpoint_cfg = self.config.get("checkpoint", {})

        self.seed = int(train_cfg.get("seed", self.config.get("seed", self.config.get("runtime", {}).get("seed", 42))))
        self._set_seed(self.seed)

        self.epochs = int(train_cfg.get("epochs", train_cfg.get("num_epochs", 3)))
        self.grad_accum_steps = max(1, int(train_cfg.get("gradient_accumulation_steps", 1)))
        self.max_grad_norm = float(train_cfg.get("max_grad_norm", 1.0))
        self.learning_rate = float(train_cfg.get("lr", 2e-5))
        self.weight_decay = float(train_cfg.get("weight_decay", 0.01))
        self.log_interval = max(1, int(train_cfg.get("log_interval", 25)))
        self.fail_policy = str(train_cfg.get("fail_policy", "skip_step_with_backoff")).lower()
        self.emergency_grad_norm = float(train_cfg.get("emergency_grad_norm", 1000.0))

        self.checkpoint_every_epoch = bool(checkpoint_cfg.get("save_every_epoch", True))
        self.strict_first = bool(checkpoint_cfg.get("load_strict_first", True))
        self.allow_shape_mismatch_fallback = bool(checkpoint_cfg.get("allow_shape_mismatch_fallback", True))

        self.use_amp = bool(train_cfg.get("use_amp", train_cfg.get("mixed_precision", True))) and self.device.type == "cuda"
        self.scaler = GradScaler("cuda", enabled=self.use_amp)

        self.distill_engine = CausalLMDistillationEngine(
            DistillationConfig(
                temperature=float(distill_cfg.get("temperature", 2.0)),
                alpha=float(distill_cfg.get("alpha", 0.7)),
                use_ce=bool(distill_cfg.get("use_ce", True)),
                ignore_index=int(distill_cfg.get("ignore_index", -100)),
                shift_labels=bool(distill_cfg.get("shift_labels", True)),
                logit_clip=float(distill_cfg.get("logit_clip", 80.0)) if distill_cfg.get("logit_clip", 80.0) is not None else None,
                min_valid_tokens=int(distill_cfg.get("min_valid_tokens", 1)),
            )
        )

        self.teacher.eval()
        for p in self.teacher.parameters():
            p.requires_grad = False
        self.student.train()

        self.optimizer = AdamW(self.student.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        self.scheduler: Optional[LambdaLR] = None

        self.runtime = TrainerRuntimeState(epoch=0, global_step=0, best_val_token_loss=float("inf"))
        self.health = DistillationHealthMetrics()
        self.train_losses = []
        self.val_losses = []
        self.metrics_history = {"token_loss": [], "perplexity": [], "token_accuracy": []}
        self.step_loss_history = []
        self.metric_stability = MetricStabilityMonitor(
            freeze_window=int(train_cfg.get("frozen_loss_window", 10)),
            freeze_tolerance=float(train_cfg.get("frozen_loss_tolerance", 1e-7)),
            perplexity_warn_threshold=float(train_cfg.get("perplexity_warn_threshold", 1e6)),
        )
        self.gradient_freeze_steps = int(train_cfg.get("gradient_freeze_steps", 8))
        self.gradient_freeze_tolerance = float(train_cfg.get("gradient_freeze_tolerance", 1e-8))
        self.gradient_zero_tolerance = float(train_cfg.get("gradient_zero_tolerance", 1e-12))
        self._numerics_validated_once = False
        self.latest_health_report: Optional[TrainingHealthReport] = None
        self.fault_injector = FaultInjector.from_mapping(self.config.get("debug", {}).get("fault_injection", {}))

        self.dataset_hash = self._compute_dataset_hash()
        self.run_metadata = self._build_run_metadata()
        self._log_run_metadata()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def fit(self, train_loader, val_loader) -> Dict[str, float]:
        total_steps = max(1, math.ceil(len(train_loader) / self.grad_accum_steps) * max(self.epochs, 1))
        warmup_steps = int(self.config.get("train", {}).get("warmup_steps", 0))
        self.scheduler = self._build_scheduler(total_steps=total_steps, warmup_steps=warmup_steps)

        for epoch in range(self.runtime.epoch, self.epochs):
            self.runtime.epoch = epoch
            train_stats = self.train_one_epoch(train_loader)
            val_stats = self.evaluate(val_loader)

            self.train_losses.append(train_stats["token_loss"])
            self.val_losses.append(val_stats["token_loss"])
            self.metrics_history["token_loss"].append(val_stats["token_loss"])
            self.metrics_history["perplexity"].append(val_stats["perplexity"])
            self.metrics_history["token_accuracy"].append(val_stats["token_accuracy"])

            if val_stats["token_loss"] < self.runtime.best_val_token_loss:
                self.runtime.best_val_token_loss = val_stats["token_loss"]
                self._save_checkpoint("best.pt")

            if self.checkpoint_every_epoch:
                self._save_checkpoint(f"epoch_{epoch + 1}.pt")
                self._save_checkpoint("latest.pt")

            LOG.info(
                "epoch=%d train_token_loss=%.6f val_token_loss=%.6f val_ppl=%.4f val_token_acc=%.4f health=%s",
                epoch + 1,
                train_stats["token_loss"],
                val_stats["token_loss"],
                val_stats["perplexity"],
                val_stats["token_accuracy"],
                self.health.to_dict(),
            )

        return {
            "best_val_token_loss": float(self.runtime.best_val_token_loss),
            "best_val_perplexity": float(math.exp(min(self.runtime.best_val_token_loss, 20.0))),
        }

    def resume_from_checkpoint(self, checkpoint_path: str) -> Dict[str, Any]:
        report, state, metadata = smart_load_checkpoint(
            path=checkpoint_path,
            model=self.student,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            scaler=self.scaler,
            map_location=str(self.device),
            strict_first=self.strict_first,
            allow_shape_mismatch_fallback=self.allow_shape_mismatch_fallback,
        )
        self.runtime.epoch = int(state.epoch)
        self.runtime.global_step = int(state.global_step)
        if state.best_metric is not None:
            self.runtime.best_val_token_loss = float(state.best_metric)
        if report.fallback_used:
            self.health.fallback_checkpoint_loads += 1

        return {
            "strict_loaded": report.strict_loaded,
            "fallback_used": report.fallback_used,
            "loaded_tensors": report.loaded_tensors,
            "skipped_tensors": report.skipped_tensors,
            "shape_mismatch": len(report.shape_mismatch),
            "skipped_optimizer": report.skipped_optimizer,
            "optimizer_restored": report.optimizer_restored,
            "optimizer_reset": report.optimizer_reset,
            "rng_restored": report.rng_restored,
            "warning": report.warning,
            "metadata_seed": metadata.seed if metadata else None,
        }

    def discover_latest_valid_checkpoint(self) -> Optional[str]:
        """Find the newest valid checkpoint under experiment_dir/checkpoints."""

        checkpoint_dir = Path(self.experiment_dir) / "checkpoints"
        if not checkpoint_dir.exists() or not checkpoint_dir.is_dir():
            return None

        candidates = sorted(
            [p for p in checkpoint_dir.glob("*.pt") if p.is_file()],
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )

        for candidate in candidates:
            if self._checkpoint_integrity_ok(candidate):
                return str(candidate)
        return None

    def resume_from_latest(self) -> Optional[Dict[str, Any]]:
        """Resume safely from newest valid checkpoint if present."""

        latest = self.discover_latest_valid_checkpoint()
        if latest is None:
            return None
        return self.resume_from_checkpoint(latest)

    def save_explicit_checkpoint(self, checkpoint_path: str) -> str:
        out = save_training_checkpoint(
            path=checkpoint_path,
            model=self.student,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            scaler=self.scaler,
            state=TrainingState(
                epoch=self.runtime.epoch,
                global_step=self.runtime.global_step,
                best_metric=self.runtime.best_val_token_loss,
            ),
            metadata=self._checkpoint_meta(),
        )
        return str(out)

    def run_determinism_verification(
        self,
        train_loader,
        *,
        compare_steps: int = 10,
        tolerance: float = 1e-7,
    ):
        """Run two seeded traces and return DeterminismReport."""

        trainer_kwargs = {
            "tokenizer": self.tokenizer,
            "config": copy.deepcopy(self.config),
            "device": self.device,
            "experiment_dir": self.experiment_dir,
            "websocket_callback": None,
        }

        def _run_once():
            t_clone = copy.deepcopy(self.teacher)
            s_clone = copy.deepcopy(self.student)
            probe = SafeCausalLMTrainer(
                teacher=t_clone,
                student=s_clone,
                **trainer_kwargs,
            )
            return trace_from_trainer(probe, train_loader, max_steps=compare_steps)

        return verify_reproducibility(
            run_builder=_run_once,
            compare_steps=compare_steps,
            tolerance=tolerance,
        )

    # ------------------------------------------------------------------
    # Loop internals
    # ------------------------------------------------------------------
    def train_one_epoch(self, train_loader) -> Dict[str, float]:
        self.teacher.eval()
        self.student.train()
        self.optimizer.zero_grad(set_to_none=True)

        token_metrics = TokenMetricsAccumulator(
            ignore_index=self.distill_engine.config.ignore_index,
            shift_labels=self.distill_engine.config.shift_labels,
        )

        for batch_idx, batch in enumerate(train_loader):
            inputs = self._prepare_batch(batch)
            labels = inputs["labels"]
            current_step = self.runtime.global_step + 1

            teacher_grad_enabled = True
            with torch.no_grad():
                teacher_grad_enabled = torch.is_grad_enabled()
                teacher_outputs = self.teacher(**inputs)

            with autocast("cuda", enabled=self.use_amp):
                student_outputs = self.student(**inputs)
                distill_out = self.distill_engine.compute_total_loss(
                    student_outputs=student_outputs,
                    teacher_outputs=teacher_outputs,
                    labels=labels,
                )
                loss = distill_out.total / float(self.grad_accum_steps)

            if not self._numerics_validated_once:
                numerics_report = validate_distillation_numerics(
                    student_outputs=student_outputs,
                    teacher_outputs=teacher_outputs,
                    labels=labels,
                    temperature=self.distill_engine.config.temperature,
                    ignore_index=self.distill_engine.config.ignore_index,
                    shift_labels=self.distill_engine.config.shift_labels,
                    teacher_grad_enabled_during_forward=teacher_grad_enabled,
                )
                if not numerics_report.passed:
                    LOG.warning("Numerical validation warnings: %s", numerics_report.warnings)
                self._numerics_validated_once = True

            student_logits = self.distill_engine.extract_logits(student_outputs)
            teacher_logits = self.distill_engine.extract_logits(teacher_outputs)
            injected_logits = self.fault_injector.maybe_inject_logits(
                student_logits=student_logits,
                teacher_logits=teacher_logits,
                step=current_step,
            )
            student_logits = injected_logits["student_logits"]
            teacher_logits = injected_logits["teacher_logits"]

            if not self._validate_logits(student_logits, teacher_logits):
                self.optimizer.zero_grad(set_to_none=True)
                self.health.invalid_logits += 1
                self.health.skipped_steps += 1
                continue

            loss = self.fault_injector.maybe_inject_loss(loss, step=current_step)

            if not distill_out.is_finite or not torch.isfinite(loss).all():
                self._handle_nonfinite_loss(loss)
                self.optimizer.zero_grad(set_to_none=True)
                continue

            self.step_loss_history.append(float(loss.detach().item()))
            if self.use_amp:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            self.fault_injector.maybe_inject_gradients(self.student, step=current_step)

            grad_report = gradient_sanity_check(
                self.student,
                loss_window=self.step_loss_history,
                freeze_steps=self.gradient_freeze_steps,
                freeze_tolerance=self.gradient_freeze_tolerance,
                zero_grad_tolerance=self.gradient_zero_tolerance,
            )
            if grad_report.has_nan_grad or grad_report.has_inf_grad:
                self.health.bad_grad_steps += 1
                self.health.skipped_steps += 1
                self.optimizer.zero_grad(set_to_none=True)
                LOG.warning("Bad gradients detected (nan=%s inf=%s) at step=%d", grad_report.has_nan_grad, grad_report.has_inf_grad, current_step)
                continue
            if grad_report.global_grad_norm <= self.gradient_zero_tolerance:
                self.health.zero_grad_steps += 1
            if grad_report.frozen_loss_detected:
                self.health.frozen_loss_steps += 1

            should_step = ((batch_idx + 1) % self.grad_accum_steps == 0) or (batch_idx + 1 == len(train_loader))
            if should_step:
                step_success = self._safe_backward_step()
                if not step_success:
                    continue
                self.runtime.global_step += 1

            token_metrics.update_from_logits(student_logits.detach(), labels.detach())
            stability = self.metric_stability.update(float(loss.detach().item()))
            if stability["frozen_loss"]:
                self.health.frozen_loss_steps += 1
            if stability["perplexity_exploded"]:
                LOG.warning("Perplexity guard triggered at step=%d: ppl=%.4f", current_step, stability["perplexity"])

            if (batch_idx + 1) % self.log_interval == 0:
                step_metrics = token_metrics.compute()
                align = compute_distill_alignment(
                    student_logits=student_logits.detach(),
                    teacher_logits=teacher_logits.detach(),
                    labels=labels.detach(),
                    ignore_index=self.distill_engine.config.ignore_index,
                    shift_labels=self.distill_engine.config.shift_labels,
                    temperature=self.distill_engine.config.temperature,
                )
                LOG.info(
                    "epoch=%d batch=%d/%d step=%d token_loss=%.6f ppl=%.4f kd=%.6f ce=%.6f distill_kl=%.6f agree=%.4f health=%s",
                    self.runtime.epoch + 1,
                    batch_idx + 1,
                    len(train_loader),
                    self.runtime.global_step,
                    step_metrics["token_loss"],
                    step_metrics["perplexity"],
                    float(distill_out.kd.item()),
                    float(distill_out.ce.item()),
                    align["distill_kl"],
                    align["top1_agreement"],
                    self.health.to_dict(),
                )
                self.latest_health_report = self.build_training_health_report()

        return token_metrics.compute()

    @torch.no_grad()
    def evaluate(self, val_loader) -> Dict[str, float]:
        self.teacher.eval()
        self.student.eval()

        token_metrics = TokenMetricsAccumulator(
            ignore_index=self.distill_engine.config.ignore_index,
            shift_labels=self.distill_engine.config.shift_labels,
        )

        for batch in val_loader:
            inputs = self._prepare_batch(batch)
            student_outputs = self.student(**inputs)
            student_logits = self.distill_engine.extract_logits(student_outputs)
            labels = inputs["labels"]
            token_metrics.update_from_logits(student_logits, labels)

        self.student.train()
        return token_metrics.compute()

    # ------------------------------------------------------------------
    # Safety helpers
    # ------------------------------------------------------------------
    def _safe_backward_step(self) -> bool:
        if self.use_amp:
            self.scaler.unscale_(self.optimizer)

        grad_norm = torch.nn.utils.clip_grad_norm_(self.student.parameters(), self.max_grad_norm)
        grad_norm_value = float(grad_norm.item()) if isinstance(grad_norm, torch.Tensor) else float(grad_norm)
        if not math.isfinite(grad_norm_value):
            self.health.nan_events += 1
            self.health.skipped_steps += 1
            self.optimizer.zero_grad(set_to_none=True)
            return False

        if grad_norm_value > self.emergency_grad_norm:
            self.health.grad_explosion_events += 1
            self.health.skipped_steps += 1
            self._apply_lr_backoff(factor=0.5)
            self.optimizer.zero_grad(set_to_none=True)
            return False

        if self.use_amp:
            old_scale = self.scaler.get_scale()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            new_scale = self.scaler.get_scale()
            if new_scale < old_scale:
                self.health.overflow_events += 1
        else:
            self.optimizer.step()

        if self.scheduler is not None:
            self.scheduler.step()

        self.optimizer.zero_grad(set_to_none=True)
        return True

    def _handle_nonfinite_loss(self, loss: torch.Tensor) -> None:
        if torch.isnan(loss).any():
            self.health.nan_events += 1
        if torch.isinf(loss).any():
            self.health.inf_events += 1
        self.health.skipped_steps += 1

        msg = f"Non-finite loss at epoch={self.runtime.epoch + 1} step={self.runtime.global_step}."
        if self.fail_policy == "abort":
            raise RuntimeError(msg)
        LOG.warning("%s Action=skip_step", msg)

    def _validate_logits(self, student_logits: torch.Tensor, teacher_logits: torch.Tensor) -> bool:
        if not torch.isfinite(student_logits).all():
            return False
        if not torch.isfinite(teacher_logits).all():
            return False
        return True

    def build_training_health_report(self) -> TrainingHealthReport:
        return TrainingHealthReport(
            epoch=self.runtime.epoch + 1,
            step=self.runtime.global_step,
            nan_events=self.health.nan_events,
            overflow_events=self.health.overflow_events,
            grad_explosion_events=self.health.grad_explosion_events,
            frozen_loss_steps=self.health.frozen_loss_steps,
            fallback_checkpoint_loads=self.health.fallback_checkpoint_loads,
            invalid_logits=self.health.invalid_logits,
            skipped_steps=self.health.skipped_steps,
            zero_grad_steps=self.health.zero_grad_steps,
            bad_grad_steps=self.health.bad_grad_steps,
        )

    # ------------------------------------------------------------------
    # Reproducibility and metadata
    # ------------------------------------------------------------------
    def _set_seed(self, seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def _prepare_batch(self, batch: Mapping[str, Any]) -> Dict[str, Any]:
        inputs: Dict[str, Any] = {}
        for key, value in batch.items():
            if torch.is_tensor(value):
                inputs[key] = value.to(self.device)

        if "labels" not in inputs:
            if "input_ids" not in inputs:
                raise ValueError("Batch must include labels or input_ids for causal LM training")
            inputs["labels"] = inputs["input_ids"].clone()
        return inputs

    def _build_scheduler(self, *, total_steps: int, warmup_steps: int = 0) -> LambdaLR:
        warmup_steps = max(0, int(warmup_steps))

        def lr_lambda(step: int) -> float:
            if warmup_steps > 0 and step < warmup_steps:
                return max(float(step) / float(max(1, warmup_steps)), 1e-8)
            remaining = max(total_steps - warmup_steps, 1)
            progress = float(step - warmup_steps) / float(remaining)
            return max(0.0, 1.0 - min(max(progress, 0.0), 1.0))

        return LambdaLR(self.optimizer, lr_lambda=lr_lambda)

    def _apply_lr_backoff(self, factor: float = 0.5) -> None:
        for group in self.optimizer.param_groups:
            group["lr"] = max(float(group["lr"]) * factor, 1e-8)

    def _checkpoint_meta(self) -> CheckpointMeta:
        return CheckpointMeta(
            epoch=self.runtime.epoch,
            global_step=self.runtime.global_step,
            best_metric=self.runtime.best_val_token_loss,
            seed=self.seed,
            dataset_hash=self.dataset_hash,
            config_snapshot=self._safe_json(self.config),
            model_sizes=self._model_sizes(),
            distillation_params=self._safe_json(self.config.get("distillation", {})),
        )

    def _save_checkpoint(self, filename: str) -> None:
        checkpoint_dir = Path(self.experiment_dir) / "checkpoints"
        checkpoint_path = checkpoint_dir / filename
        save_training_checkpoint(
            path=str(checkpoint_path),
            model=self.student,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            scaler=self.scaler,
            state=TrainingState(
                epoch=self.runtime.epoch + 1,
                global_step=self.runtime.global_step,
                best_metric=self.runtime.best_val_token_loss,
            ),
            metadata=self._checkpoint_meta(),
        )

    def _checkpoint_integrity_ok(self, path: Path) -> bool:
        """Lightweight pre-load validation for checkpoint discover/resume."""

        try:
            payload = torch.load(str(path), map_location="cpu", weights_only=False)
        except Exception as exc:
            LOG.warning("Checkpoint integrity failure (%s): %s", path, exc)
            return False

        if not isinstance(payload, dict):
            return False
        model_state = payload.get("model_state_dict")
        state = payload.get("state")
        if not isinstance(model_state, dict):
            return False
        if state is not None and not isinstance(state, dict):
            return False
        return True

    def _model_sizes(self) -> Dict[str, int]:
        return {
            "teacher_trainable": int(sum(p.numel() for p in self.teacher.parameters() if p.requires_grad)),
            "teacher_total": int(sum(p.numel() for p in self.teacher.parameters())),
            "student_trainable": int(sum(p.numel() for p in self.student.parameters() if p.requires_grad)),
            "student_total": int(sum(p.numel() for p in self.student.parameters())),
        }

    def _compute_dataset_hash(self) -> str:
        data_cfg = self.config.get("data", {}) if isinstance(self.config, dict) else {}
        paths = [data_cfg.get("train_path"), data_cfg.get("val_path")]
        hasher = hashlib.sha256()
        any_file = False
        for path in paths:
            if not path:
                continue
            p = Path(path)
            if not p.exists() or not p.is_file():
                continue
            any_file = True
            hasher.update(str(p.resolve()).encode("utf-8"))
            with p.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    hasher.update(chunk)
        return hasher.hexdigest() if any_file else ""

    def _build_run_metadata(self) -> Dict[str, Any]:
        self._assert_token_only_metrics()
        return {
            "seed": self.seed,
            "device": str(self.device),
            "dataset_hash": self.dataset_hash,
            "model_sizes": self._model_sizes(),
            "distillation": self._safe_json(self.config.get("distillation", {})),
            "train": self._safe_json(self.config.get("train", {})),
            "determinism_env": runtime_determinism_env(),
        }

    def _assert_token_only_metrics(self) -> None:
        forbidden = {"accuracy", "f1", "precision", "recall", "auc", "mcc"}
        present_forbidden = [k for k in self.metrics_history.keys() if k.lower() in forbidden]
        if present_forbidden:
            raise RuntimeError(f"Token-only metrics violation detected: {present_forbidden}")

    def _log_run_metadata(self) -> None:
        logs_dir = Path(self.experiment_dir) / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)
        run_manifest = logs_dir / "causal_lm_run_manifest.json"
        with run_manifest.open("w", encoding="utf-8") as handle:
            json.dump(self.run_metadata, handle, indent=2)
        LOG.info("Run manifest: %s", run_manifest)

    @staticmethod
    def _safe_json(value: Any) -> Any:
        try:
            json.dumps(value)
            return value
        except TypeError:
            if isinstance(value, dict):
                return {str(k): SafeCausalLMTrainer._safe_json(v) for k, v in value.items()}
            if isinstance(value, (list, tuple)):
                return [SafeCausalLMTrainer._safe_json(v) for v in value]
            return str(value)
