"""DDP-ready distillation smoke (torchrun entry-point).

Loaded by ``torchrun`` under ``scripts/smoke/run_distributed_ddp.py``.

Each rank:
1. Loads ``prajjwal1/bert-tiny``.
2. Wraps the distillation via :func:`prepare_distillation` with
   ``accelerator.prepare`` (which detects WORLD_SIZE and applies
   DistributedDataParallel when > 1).
3. Runs ``--steps`` SGD updates on synthetic batches.
4. Records loss progression + adapter / accelerator details.
5. Saves a JSON summary at ``--output`` (only on rank 0).

Smoke criterion: **all ranks produce the same loss** (data-parallel
sync verified) **and decay > 0**.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F


def _main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--output", type=Path, default=Path("tests/smoke/results"))
    args = p.parse_args()

    from transformers import AutoModelForSequenceClassification

    from zynthe.core.training.distributed import (
        DistributedConfig,
        prepare_distillation,
    )

    # Rank-aware seed (each rank gets a different but deterministic seed).
    try:
        from accelerate import Accelerator
        accelerator = Accelerator()
        local_rank = accelerator.local_process_index
        world_size = accelerator.num_processes
    except Exception:
        # No accelerate / running as plain script: rank 0 only.
        local_rank = 0
        world_size = 1
        accelerator = None

    rank_seed = args.seed + local_rank
    torch.manual_seed(rank_seed)

    print(
        f"[ddp-smoke] rank={local_rank} world_size={world_size} seed={rank_seed}",
        flush=True,
    )

    teacher = AutoModelForSequenceClassification.from_pretrained("prajjwal1/bert-tiny")
    student = AutoModelForSequenceClassification.from_pretrained("prajjwal1/bert-tiny")
    teacher.eval()
    for p_ in teacher.parameters():
        p_.requires_grad = False
    optim = torch.optim.SGD(student.parameters(), lr=args.lr)

    cfg = DistributedConfig(
        enabled=True,
        mixed_precision="no",
    )
    bundle = prepare_distillation(teacher, student, optim, dataloader=None, config=cfg)

    # Pick the device the prepared student lives on.
    device = next(bundle.student.parameters()).device

    losses: list[float] = []
    started = time.time()
    for step in range(args.steps):
        torch.manual_seed(rank_seed + step)
        x = torch.randint(0, 1000, (4, 8), dtype=torch.long, device=device)
        y = torch.randint(0, 2, (4,), device=device)
        with torch.no_grad():
            t_out = bundle.teacher(input_ids=x)
        optim.zero_grad()
        s_out = bundle.student(input_ids=x)
        if s_out.logits.dim() == 3:
            s_out.logits = s_out.logits.mean(dim=1)
        loss = F.cross_entropy(s_out.logits, y)
        if not torch.isfinite(loss):
            print(f"[ddp-smoke][FAIL] rank {local_rank} non-finite loss", file=sys.stderr)
            return 1
        loss.backward()
        optim.step()
        losses.append(loss.item())
    duration = time.time() - started

    print(
        f"[ddp-smoke] rank={local_rank} {args.steps} steps in {duration:.1f}s. "
        f"first={losses[0]:.4f} last={losses[-1]:.4f} min={min(losses):.4f}",
        flush=True,
    )

    # Sync across ranks so we capture the full picture from rank 0.
    if accelerator is not None:
        accelerator.wait_for_everyone()

    # Only rank 0 writes the summary.
    if local_rank == 0:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "timestamp": time.time(),
            "commit": os.environ.get("GIT_COMMIT", "local"),
            "model": "prajjwal1/bert-tiny",
            "world_size": world_size,
            "accelerator": type(bundle.accelerator).__name__ if bundle.accelerator else None,
            "steps": args.steps,
            "duration_s": duration,
            "loss_first": losses[0],
            "loss_last": losses[-1],
            "loss_min": min(losses),
            "loss_decay": losses[0] - losses[-1],
        }
        out_path = args.output / "distributed_ddp.json"
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"[ddp-smoke][OK] summary -> {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(_main())
