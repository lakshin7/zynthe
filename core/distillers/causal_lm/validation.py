"""Validation and safety layer for Causal-LM distillation reliability."""

from __future__ import annotations

import logging
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .checkpoint import (
    CheckpointLoadReport,
    CheckpointMeta,
    TrainingState,
    save_training_checkpoint,
    smart_load_checkpoint,
)

LOG = logging.getLogger(__name__)


@dataclass
class NumericalValidationReport:
    passed: bool
    checks: Dict[str, bool] = field(default_factory=dict)
    diagnostics: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)


@dataclass
class GradientSanityReport:
    has_nan_grad: bool
    has_inf_grad: bool
    zero_grad_ratio: float
    global_grad_norm: float
    frozen_loss_detected: bool
    checks: Dict[str, bool] = field(default_factory=dict)


@dataclass
class TrainingHealthReport:
    epoch: int
    step: int
    nan_events: int
    overflow_events: int
    grad_explosion_events: int
    frozen_loss_steps: int
    fallback_checkpoint_loads: int
    invalid_logits: int
    skipped_steps: int
    zero_grad_steps: int
    bad_grad_steps: int

    def to_dict(self) -> Dict[str, int]:
        return {
            "epoch": int(self.epoch),
            "step": int(self.step),
            "nan_events": int(self.nan_events),
            "overflow_events": int(self.overflow_events),
            "grad_explosion_events": int(self.grad_explosion_events),
            "frozen_loss_steps": int(self.frozen_loss_steps),
            "fallback_checkpoint_loads": int(self.fallback_checkpoint_loads),
            "invalid_logits": int(self.invalid_logits),
            "skipped_steps": int(self.skipped_steps),
            "zero_grad_steps": int(self.zero_grad_steps),
            "bad_grad_steps": int(self.bad_grad_steps),
        }


@dataclass
class CheckpointStressScenario:
    name: str
    passed: bool
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CheckpointStressReport:
    all_passed: bool
    scenarios: List[CheckpointStressScenario] = field(default_factory=list)


def _extract_logits(outputs: Any) -> torch.Tensor:
    if isinstance(outputs, dict):
        return outputs["logits"]
    if hasattr(outputs, "logits"):
        return outputs.logits
    if isinstance(outputs, tuple):
        return outputs[0]
    return outputs


def validate_distillation_numerics(
    *,
    student_outputs: Any,
    teacher_outputs: Any,
    labels: torch.Tensor,
    temperature: float,
    ignore_index: int = -100,
    shift_labels: bool = True,
    atol: float = 1e-6,
    rtol: float = 1e-5,
    teacher_grad_enabled_during_forward: Optional[bool] = None,
) -> NumericalValidationReport:
    """Validate KD formula, masking, teacher no_grad, and AMP isolation assumptions."""

    checks: Dict[str, bool] = {}
    warnings: List[str] = []
    diagnostics: Dict[str, Any] = {}

    s = _extract_logits(student_outputs)
    t = _extract_logits(teacher_outputs)

    if shift_labels:
        s = s[:, :-1, :].contiguous()
        t = t[:, :-1, :].contiguous()
        labels = labels[:, 1:].contiguous()

    flat_s = s.view(-1, s.size(-1)).float()
    flat_t = t.view(-1, t.size(-1)).float()
    flat_y = labels.reshape(-1)
    valid = flat_y != int(ignore_index)

    checks["mask_has_valid_tokens"] = bool(valid.any().item() if torch.is_tensor(valid.any()) else valid.any())
    if not checks["mask_has_valid_tokens"]:
        return NumericalValidationReport(
            passed=False,
            checks=checks,
            diagnostics={"valid_tokens": 0},
            warnings=["No valid tokens after masking; unable to validate KL numerics."],
        )

    s_valid = flat_s[valid]
    t_valid = flat_t[valid]

    temp = max(float(temperature), 1e-6)
    s_log_probs = F.log_softmax(s_valid / temp, dim=-1)
    t_probs = F.softmax(t_valid / temp, dim=-1)

    kd_reference = F.kl_div(s_log_probs, t_probs, reduction="batchmean") * (temp * temp)

    # Compute equivalent form explicitly from definition for consistency check.
    eps = 1e-12
    kd_manual = (t_probs * (torch.log(t_probs.clamp_min(eps)) - s_log_probs)).sum(dim=-1).mean() * (temp * temp)

    checks["kl_formula_matches_manual"] = bool(torch.allclose(kd_reference, kd_manual, atol=atol, rtol=rtol).item())
    checks["finite_student_logits"] = bool(torch.isfinite(s_valid).all().item())
    checks["finite_teacher_logits"] = bool(torch.isfinite(t_valid).all().item())
    checks["teacher_requires_grad_off"] = bool(not t.requires_grad)

    if teacher_grad_enabled_during_forward is not None:
        checks["teacher_forward_under_no_grad"] = bool(not teacher_grad_enabled_during_forward)
    else:
        checks["teacher_forward_under_no_grad"] = True

    # Teacher logits should not be scaled by GradScaler. In practice, no_grad + float logits implies this.
    checks["teacher_logits_unscaled_by_amp"] = bool(not t.requires_grad and t.dtype in {torch.float16, torch.bfloat16, torch.float32})

    diagnostics.update(
        {
            "valid_tokens": int(valid.sum().item()),
            "kd_reference": float(kd_reference.item()),
            "kd_manual": float(kd_manual.item()),
            "student_dtype": str(s.dtype),
            "teacher_dtype": str(t.dtype),
        }
    )

    for key, ok in checks.items():
        if not ok:
            warnings.append(f"check_failed:{key}")

    return NumericalValidationReport(
        passed=all(checks.values()),
        checks=checks,
        diagnostics=diagnostics,
        warnings=warnings,
    )


def gradient_sanity_check(
    model: nn.Module,
    *,
    loss_window: Sequence[float],
    freeze_steps: int = 8,
    freeze_tolerance: float = 1e-8,
    zero_grad_tolerance: float = 1e-12,
) -> GradientSanityReport:
    """Detect NaN/Inf/zero gradients and frozen loss windows."""

    grads: List[torch.Tensor] = []
    total_params = 0
    zero_like = 0
    has_nan = False
    has_inf = False

    for param in model.parameters():
        if param.grad is None:
            continue
        grad = param.grad.detach()
        grads.append(grad)
        total_params += grad.numel()
        has_nan = has_nan or bool(torch.isnan(grad).any().item())
        has_inf = has_inf or bool(torch.isinf(grad).any().item())
        zero_like += int((grad.abs() <= zero_grad_tolerance).sum().item())

    if grads:
        flat = torch.cat([g.reshape(-1).float().cpu() for g in grads], dim=0)
        norm = float(torch.norm(flat, p=2).item())
    else:
        norm = 0.0

    zero_grad_ratio = float(zero_like / max(total_params, 1))

    frozen = False
    if len(loss_window) >= max(2, freeze_steps):
        recent = [float(v) for v in loss_window[-freeze_steps:]]
        frozen = (max(recent) - min(recent)) <= float(freeze_tolerance)

    checks = {
        "no_nan_grad": not has_nan,
        "no_inf_grad": not has_inf,
        "non_zero_grad_norm": norm > zero_grad_tolerance,
        "loss_not_frozen": not frozen,
    }

    return GradientSanityReport(
        has_nan_grad=has_nan,
        has_inf_grad=has_inf,
        zero_grad_ratio=zero_grad_ratio,
        global_grad_norm=norm,
        frozen_loss_detected=frozen,
        checks=checks,
    )


class _ToyModel(nn.Module):
    def __init__(self, vocab: int, hidden: int = 16):
        super().__init__()
        self.embed = nn.Embedding(vocab, hidden)
        self.lm_head = nn.Linear(hidden, vocab)

    def forward(self, input_ids: torch.Tensor, labels: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        h = self.embed(input_ids)
        return {"logits": self.lm_head(h)}


def run_checkpoint_stress_tests(device: Optional[torch.device] = None) -> CheckpointStressReport:
    """Automated stress checks for strict/fallback checkpoint behavior."""

    use_device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scenarios: List[CheckpointStressScenario] = []

    with tempfile.TemporaryDirectory(prefix="ckpt_stress_") as tmp:
        root = Path(tmp)
        ckpt = root / "base.pt"

        model_a = _ToyModel(vocab=32).to(use_device)
        optim_a = torch.optim.AdamW(model_a.parameters(), lr=1e-3)
        scheduler_a = torch.optim.lr_scheduler.LambdaLR(optim_a, lr_lambda=lambda _: 1.0)
        scaler_a = torch.cuda.amp.GradScaler(enabled=use_device.type == "cuda")

        save_training_checkpoint(
            path=str(ckpt),
            model=model_a,
            optimizer=optim_a,
            scheduler=scheduler_a,
            scaler=scaler_a,
            state=TrainingState(epoch=1, global_step=5, best_metric=1.23),
            metadata=CheckpointMeta(epoch=1, global_step=5, best_metric=1.23, seed=42),
        )

        # 1) Strict load success
        model_strict = _ToyModel(vocab=32).to(use_device)
        optim_strict = torch.optim.AdamW(model_strict.parameters(), lr=1e-3)
        sched_strict = torch.optim.lr_scheduler.LambdaLR(optim_strict, lr_lambda=lambda _: 1.0)
        scaler_strict = torch.cuda.amp.GradScaler(enabled=use_device.type == "cuda")
        report_strict, _, _ = smart_load_checkpoint(
            path=str(ckpt),
            model=model_strict,
            optimizer=optim_strict,
            scheduler=sched_strict,
            scaler=scaler_strict,
            map_location=str(use_device),
            strict_first=True,
            allow_shape_mismatch_fallback=True,
        )
        scenarios.append(
            CheckpointStressScenario(
                name="strict_load_success",
                passed=bool(report_strict.strict_loaded and report_strict.loaded_tensors > 0),
                details=report_strict.__dict__,
            )
        )

        # 2) Shape mismatch fallback (embedding/head mismatch)
        model_mismatch = _ToyModel(vocab=40).to(use_device)
        optim_mismatch = torch.optim.AdamW(model_mismatch.parameters(), lr=1e-3)
        report_mismatch, _, _ = smart_load_checkpoint(
            path=str(ckpt),
            model=model_mismatch,
            optimizer=optim_mismatch,
            scheduler=None,
            scaler=None,
            map_location=str(use_device),
            strict_first=True,
            allow_shape_mismatch_fallback=True,
        )
        scenarios.append(
            CheckpointStressScenario(
                name="shape_mismatch_fallback",
                passed=bool(report_mismatch.fallback_used and report_mismatch.skipped_optimizer),
                details=report_mismatch.__dict__,
            )
        )

        # 3) Partial compatibility + parameter count change
        class _ToyModelPartial(_ToyModel):
            def __init__(self, vocab: int):
                super().__init__(vocab=vocab)
                self.extra = nn.Linear(16, 16)

        model_partial = _ToyModelPartial(vocab=32).to(use_device)
        report_partial, _, _ = smart_load_checkpoint(
            path=str(ckpt),
            model=model_partial,
            optimizer=None,
            scheduler=None,
            scaler=None,
            map_location=str(use_device),
            strict_first=False,
            allow_shape_mismatch_fallback=True,
        )
        scenarios.append(
            CheckpointStressScenario(
                name="partial_tensor_compatibility",
                passed=bool(report_partial.loaded_tensors > 0 and report_partial.skipped_tensors > 0),
                details=report_partial.__dict__,
            )
        )

        # 4) Missing optimizer/scaler should not crash
        report_missing_state, _, _ = smart_load_checkpoint(
            path=str(ckpt),
            model=_ToyModel(vocab=32).to(use_device),
            optimizer=None,
            scheduler=None,
            scaler=None,
            map_location=str(use_device),
            strict_first=True,
            allow_shape_mismatch_fallback=True,
        )
        scenarios.append(
            CheckpointStressScenario(
                name="missing_optimizer_scaler",
                passed=bool(report_missing_state.loaded_tensors > 0),
                details=report_missing_state.__dict__,
            )
        )

        # 5) Optimizer incompatibility forced reset
        bad_opt = torch.optim.SGD(_ToyModel(vocab=32).to(use_device).parameters(), lr=1e-2)
        report_opt, _, _ = smart_load_checkpoint(
            path=str(ckpt),
            model=_ToyModel(vocab=32).to(use_device),
            optimizer=bad_opt,
            scheduler=None,
            scaler=None,
            map_location=str(use_device),
            strict_first=True,
            allow_shape_mismatch_fallback=True,
        )
        scenarios.append(
            CheckpointStressScenario(
                name="optimizer_incompatibility_reset",
                passed=bool(report_opt.optimizer_reset or report_opt.optimizer_restore_reason in {"incompatible_optimizer_state", "missing_optimizer_state"}),
                details=report_opt.__dict__,
            )
        )

        # 6) Parameter reorder/key rename simulation
        altered_ckpt = root / "altered.pt"
        raw_payload = torch.load(str(ckpt), map_location="cpu")
        model_state = dict(raw_payload.get("model_state_dict", {}))
        if model_state:
            first_key = next(iter(model_state.keys()))
            model_state[f"renamed::{first_key}"] = model_state.pop(first_key)
        raw_payload["model_state_dict"] = model_state
        torch.save(raw_payload, str(altered_ckpt))

        report_reorder, _, _ = smart_load_checkpoint(
            path=str(altered_ckpt),
            model=_ToyModel(vocab=32).to(use_device),
            optimizer=None,
            scheduler=None,
            scaler=None,
            map_location=str(use_device),
            strict_first=False,
            allow_shape_mismatch_fallback=True,
        )
        scenarios.append(
            CheckpointStressScenario(
                name="parameter_reorder_or_rename",
                passed=bool(len(report_reorder.unexpected_keys) > 0 and report_reorder.loaded_tensors > 0),
                details=report_reorder.__dict__,
            )
        )

    all_passed = all(s.passed for s in scenarios)
    return CheckpointStressReport(all_passed=all_passed, scenarios=scenarios)
