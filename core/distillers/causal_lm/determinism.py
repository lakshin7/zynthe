"""Determinism verification harness for Causal-LM distillation."""

from __future__ import annotations

import os
import platform
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional

import numpy as np
import torch


@dataclass
class DeterminismReport:
    passed: bool
    compared_steps: int
    tolerance: float
    max_abs_token_loss_diff: float
    max_abs_grad_norm_diff: float
    run_a_losses: List[float] = field(default_factory=list)
    run_b_losses: List[float] = field(default_factory=list)
    run_a_grad_norms: List[float] = field(default_factory=list)
    run_b_grad_norms: List[float] = field(default_factory=list)
    env: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DeterminismTrace:
    losses: List[float]
    grad_norms: List[float]
    has_nan_or_inf: bool


def runtime_determinism_env() -> Dict[str, Any]:
    cuda_available = torch.cuda.is_available()
    env: Dict[str, Any] = {
        "PYTHONHASHSEED": os.environ.get("PYTHONHASHSEED", ""),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "cuda_available": cuda_available,
        "cuda_version": torch.version.cuda,
        "cudnn_enabled": bool(torch.backends.cudnn.enabled),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "numpy_version": np.__version__,
    }
    if cuda_available:
        env["cuda_device_name"] = torch.cuda.get_device_name(0)
        env["cuda_device_count"] = int(torch.cuda.device_count())
    return env


def verify_reproducibility(
    *,
    run_builder: Callable[[], DeterminismTrace],
    compare_steps: int = 10,
    tolerance: float = 1e-8,
) -> DeterminismReport:
    """Run two seeded traces and compare token loss + grad norm step-wise."""

    trace_a = run_builder()
    trace_b = run_builder()

    steps = min(int(compare_steps), len(trace_a.losses), len(trace_b.losses), len(trace_a.grad_norms), len(trace_b.grad_norms))
    if steps <= 0:
        return DeterminismReport(
            passed=False,
            compared_steps=0,
            tolerance=float(tolerance),
            max_abs_token_loss_diff=float("inf"),
            max_abs_grad_norm_diff=float("inf"),
            run_a_losses=trace_a.losses,
            run_b_losses=trace_b.losses,
            run_a_grad_norms=trace_a.grad_norms,
            run_b_grad_norms=trace_b.grad_norms,
            env=runtime_determinism_env(),
        )

    a_loss = trace_a.losses[:steps]
    b_loss = trace_b.losses[:steps]
    a_grad = trace_a.grad_norms[:steps]
    b_grad = trace_b.grad_norms[:steps]

    loss_diffs = [abs(float(x) - float(y)) for x, y in zip(a_loss, b_loss)]
    grad_diffs = [abs(float(x) - float(y)) for x, y in zip(a_grad, b_grad)]

    max_loss_diff = max(loss_diffs) if loss_diffs else 0.0
    max_grad_diff = max(grad_diffs) if grad_diffs else 0.0

    passed = (
        not trace_a.has_nan_or_inf
        and not trace_b.has_nan_or_inf
        and max_loss_diff <= float(tolerance)
        and max_grad_diff <= float(tolerance)
    )

    return DeterminismReport(
        passed=passed,
        compared_steps=steps,
        tolerance=float(tolerance),
        max_abs_token_loss_diff=float(max_loss_diff),
        max_abs_grad_norm_diff=float(max_grad_diff),
        run_a_losses=[float(v) for v in a_loss],
        run_b_losses=[float(v) for v in b_loss],
        run_a_grad_norms=[float(v) for v in a_grad],
        run_b_grad_norms=[float(v) for v in b_grad],
        env=runtime_determinism_env(),
    )


def trace_from_trainer(
    trainer: Any,
    train_loader,
    *,
    max_steps: int = 10,
) -> DeterminismTrace:
    """Collect first-N step token losses and grad norms from SafeCausalLMTrainer."""

    losses: List[float] = []
    grad_norms: List[float] = []
    has_nan_or_inf = False

    trainer.teacher.eval()
    trainer.student.train()
    trainer.optimizer.zero_grad(set_to_none=True)

    for batch_idx, batch in enumerate(train_loader):
        if batch_idx >= int(max_steps):
            break
        inputs = trainer._prepare_batch(batch)
        labels = inputs["labels"]

        with torch.no_grad():
            t_out = trainer.teacher(**inputs)

        with torch.amp.autocast("cuda", enabled=trainer.use_amp):
            s_out = trainer.student(**inputs)
            d_out = trainer.distill_engine.compute_total_loss(
                student_outputs=s_out,
                teacher_outputs=t_out,
                labels=labels,
            )
            loss = d_out.total

        losses.append(float(loss.detach().item()))
        if not torch.isfinite(loss).all():
            has_nan_or_inf = True
            break

        trainer.optimizer.zero_grad(set_to_none=True)
        if trainer.use_amp:
            trainer.scaler.scale(loss).backward()
            trainer.scaler.unscale_(trainer.optimizer)
        else:
            loss.backward()

        grad_norm = torch.nn.utils.clip_grad_norm_(trainer.student.parameters(), trainer.max_grad_norm)
        grad_norm_val = float(grad_norm.item()) if isinstance(grad_norm, torch.Tensor) else float(grad_norm)
        grad_norms.append(grad_norm_val)
        if not np.isfinite(grad_norm_val):
            has_nan_or_inf = True
            break

        trainer.optimizer.step()
        trainer.optimizer.zero_grad(set_to_none=True)

    return DeterminismTrace(losses=losses, grad_norms=grad_norms, has_nan_or_inf=has_nan_or_inf)
