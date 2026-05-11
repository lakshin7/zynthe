"""Regression gate for promoting causal_lm_core to controlled stable candidate."""

from __future__ import annotations

import copy
import json
import logging
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

import numpy as np
import torch

from .determinism import DeterminismTrace, trace_from_trainer
from .trainer import SafeCausalLMTrainer

LOG = logging.getLogger(__name__)


@dataclass
class RegressionReport:
    passed: bool
    compared_steps: int
    token_loss_threshold: float
    grad_norm_threshold: float
    max_token_loss_diff: float
    max_grad_norm_diff: float
    has_nan_or_inf: bool
    reasons: List[str] = field(default_factory=list)
    legacy_trace: Dict[str, List[float]] = field(default_factory=dict)
    causal_trace: Dict[str, List[float]] = field(default_factory=dict)


@dataclass
class RegressionGateConfig:
    enabled: bool = True
    compare_steps: int = 8
    token_loss_threshold: float = 1e-3
    grad_norm_threshold: float = 1e-2
    force_disable_amp: bool = True


def _set_seed(seed: int) -> None:
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _trace_from_legacy_trainer(
    trainer: Any,
    train_loader,
    *,
    max_steps: int,
) -> DeterminismTrace:
    losses: List[float] = []
    grad_norms: List[float] = []
    has_nan_or_inf = False

    trainer.teacher.eval()
    trainer.student.train()
    trainer.optimizer.zero_grad(set_to_none=True)

    for batch_idx, batch in enumerate(train_loader):
        if batch_idx >= int(max_steps):
            break
        if not batch or not isinstance(batch, dict):
            continue

        batch = {k: v.to(trainer.device) for k, v in batch.items() if hasattr(v, "to")}
        teacher_batch = trainer._filter_batch_for_model(batch, trainer._teacher_forward_params)
        student_batch = trainer._filter_batch_for_model(batch, trainer._student_forward_params)
        teacher_runtime_kwargs = trainer._build_forward_runtime_kwargs(
            trainer._teacher_forward_params
        )
        student_runtime_kwargs = trainer._build_forward_runtime_kwargs(
            trainer._student_forward_params
        )
        trainer._prepare_distiller_batch()

        with torch.no_grad():
            teacher_outputs = trainer.teacher(**teacher_batch, **teacher_runtime_kwargs)

        student_outputs = trainer.student(**student_batch, **student_runtime_kwargs)
        labels = batch.get("labels", None)
        feature_payload = trainer._collect_distiller_features()
        result = trainer.distiller.compute_loss(
            student_outputs=student_outputs,
            teacher_outputs=teacher_outputs,
            targets=labels,
            student_features=feature_payload.get("student_features"),
            teacher_features=feature_payload.get("teacher_features"),
        )
        loss = result[0] if isinstance(result, tuple) else result

        losses.append(float(loss.detach().item()))
        if not torch.isfinite(loss).all():
            has_nan_or_inf = True
            break

        trainer.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            trainer.student.parameters(),
            float(trainer.config.get("train", {}).get("max_grad_norm", 1.0)),
        )
        grad_norm_val = (
            float(grad_norm.item()) if isinstance(grad_norm, torch.Tensor) else float(grad_norm)
        )
        grad_norms.append(grad_norm_val)

        if not np.isfinite(grad_norm_val):
            has_nan_or_inf = True
            break

        trainer.optimizer.step()
        trainer.optimizer.zero_grad(set_to_none=True)

    return DeterminismTrace(losses=losses, grad_norms=grad_norms, has_nan_or_inf=has_nan_or_inf)


class RegressionGate:
    """Compares first N seeded steps between legacy trainer and causal_lm_core."""

    def __init__(self, config: Optional[RegressionGateConfig] = None):
        self.config = config or RegressionGateConfig()

    @classmethod
    def from_mapping(cls, cfg: Mapping[str, Any]) -> "RegressionGate":
        gate_cfg = dict(cfg.get("regression_gate", {})) if isinstance(cfg, Mapping) else {}
        return cls(
            RegressionGateConfig(
                enabled=bool(gate_cfg.get("enabled", True)),
                compare_steps=int(gate_cfg.get("compare_steps", 8)),
                token_loss_threshold=float(gate_cfg.get("token_loss_threshold", 1e-3)),
                grad_norm_threshold=float(gate_cfg.get("grad_norm_threshold", 1e-2)),
                force_disable_amp=bool(gate_cfg.get("force_disable_amp", True)),
            )
        )

    def run(
        self,
        *,
        teacher: Any,
        student: Any,
        tokenizer: Any,
        config: Mapping[str, Any],
        device: torch.device,
        experiment_dir: str,
        train_loader,
    ) -> RegressionReport:
        if not self.config.enabled:
            return RegressionReport(
                passed=True,
                compared_steps=0,
                token_loss_threshold=self.config.token_loss_threshold,
                grad_norm_threshold=self.config.grad_norm_threshold,
                max_token_loss_diff=0.0,
                max_grad_norm_diff=0.0,
                has_nan_or_inf=False,
                reasons=["regression_gate_disabled"],
            )

        cfg_core = copy.deepcopy(dict(config))
        cfg_legacy = copy.deepcopy(dict(config))
        seed = int(
            cfg_core.get("train", {}).get(
                "seed", cfg_core.get("seed", cfg_core.get("runtime", {}).get("seed", 42))
            )
        )

        try:
            from zynthe.training.trainer import Trainer as LegacyTrainer
        except Exception as exc:
            return RegressionReport(
                passed=False,
                compared_steps=0,
                token_loss_threshold=self.config.token_loss_threshold,
                grad_norm_threshold=self.config.grad_norm_threshold,
                max_token_loss_diff=float("inf"),
                max_grad_norm_diff=float("inf"),
                has_nan_or_inf=True,
                reasons=[f"legacy_trainer_unavailable:{exc}"],
            )

        if self.config.force_disable_amp:
            cfg_core.setdefault("train", {})["use_amp"] = False
            cfg_legacy.setdefault("train", {})["use_amp"] = False

        _set_seed(seed)
        core_trainer = SafeCausalLMTrainer(
            teacher=copy.deepcopy(teacher),
            student=copy.deepcopy(student),
            tokenizer=tokenizer,
            config=cfg_core,
            device=device,
            experiment_dir=experiment_dir,
        )

        _set_seed(seed)
        legacy_trainer = LegacyTrainer(
            teacher=copy.deepcopy(teacher),
            student=copy.deepcopy(student),
            tokenizer=tokenizer,
            config=cfg_legacy,
            device=device,
            experiment_dir=experiment_dir,
        )

        _set_seed(seed)
        core_trace = trace_from_trainer(
            core_trainer, train_loader, max_steps=self.config.compare_steps
        )
        _set_seed(seed)
        legacy_trace = _trace_from_legacy_trainer(
            legacy_trainer, train_loader, max_steps=self.config.compare_steps
        )

        steps = min(
            self.config.compare_steps,
            len(core_trace.losses),
            len(legacy_trace.losses),
            len(core_trace.grad_norms),
            len(legacy_trace.grad_norms),
        )

        reasons: List[str] = []
        if steps <= 0:
            reasons.append("insufficient_trace_steps")
            return RegressionReport(
                passed=False,
                compared_steps=0,
                token_loss_threshold=self.config.token_loss_threshold,
                grad_norm_threshold=self.config.grad_norm_threshold,
                max_token_loss_diff=float("inf"),
                max_grad_norm_diff=float("inf"),
                has_nan_or_inf=True,
                reasons=reasons,
                legacy_trace={"losses": legacy_trace.losses, "grad_norms": legacy_trace.grad_norms},
                causal_trace={"losses": core_trace.losses, "grad_norms": core_trace.grad_norms},
            )

        loss_diffs = [
            abs(float(core_trace.losses[i]) - float(legacy_trace.losses[i])) for i in range(steps)
        ]
        grad_diffs = [
            abs(float(core_trace.grad_norms[i]) - float(legacy_trace.grad_norms[i]))
            for i in range(steps)
        ]

        max_loss_diff = max(loss_diffs) if loss_diffs else 0.0
        max_grad_diff = max(grad_diffs) if grad_diffs else 0.0
        has_nan_or_inf = bool(core_trace.has_nan_or_inf or legacy_trace.has_nan_or_inf)

        if has_nan_or_inf:
            reasons.append("nan_or_inf_detected")
        if max_loss_diff >= self.config.token_loss_threshold:
            reasons.append("token_loss_diff_threshold_exceeded")
        if max_grad_diff >= self.config.grad_norm_threshold:
            reasons.append("grad_norm_diff_threshold_exceeded")

        passed = len(reasons) == 0

        report = RegressionReport(
            passed=passed,
            compared_steps=int(steps),
            token_loss_threshold=float(self.config.token_loss_threshold),
            grad_norm_threshold=float(self.config.grad_norm_threshold),
            max_token_loss_diff=float(max_loss_diff),
            max_grad_norm_diff=float(max_grad_diff),
            has_nan_or_inf=has_nan_or_inf,
            reasons=reasons,
            legacy_trace={
                "losses": legacy_trace.losses[:steps],
                "grad_norms": legacy_trace.grad_norms[:steps],
            },
            causal_trace={
                "losses": core_trace.losses[:steps],
                "grad_norms": core_trace.grad_norms[:steps],
            },
        )

        try:
            logs_dir = Path(experiment_dir) / "logs"
            logs_dir.mkdir(parents=True, exist_ok=True)
            out_path = logs_dir / "regression_gate_report.json"
            with out_path.open("w", encoding="utf-8") as handle:
                json.dump(report.__dict__, handle, indent=2)
        except Exception:
            LOG.debug("Failed to persist regression gate report", exc_info=True)

        return report
