"""Multi-task T5 trainer for Distill step-by-step (paper §3.2).

Wraps a T5-style encoder-decoder and exposes the paper's
two-forward-pass recipe under task prefixes:

    label_forward:     f([label]     · input) → label_logits
    rationale_forward: f([rationale] · input) → rationale_logits

Packs both views into a ``{"label_logits", "rationale_logits"}``
dict consumable by :class:`RationaleDistiller.compute_loss`.

The class is a thin wrapper that does NOT own the loss — that's
delegated to the distiller so users can swap the loss formulation
(entropy, dynamic τ, projection) without touching the trainer.

Usage::

    from zynthe.core.training.rationale_trainer import (
        MultiTaskT5Trainer,
    )

    trainer = MultiTaskT5Trainer.from_pretrained(
        "patrickvonplaten/t5-tiny-random",
        label_prefix="label: ",
        rationale_prefix="rationale: ",
    )
    optim = torch.optim.SGD(trainer.student.parameters(), lr=1e-3)
    for batch in dataloader:
        loss, breakdown = trainer.train_step(batch, distiller, optim)
        optim.step()

The trainer's forward pass uses the same model twice — once with
each task prefix — and lets the distiller compute the multi-task
loss.  At test-time, only the label forward is needed; the
rationale head is unused.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class MultiTaskT5Trainer:
    """T5 wrapper for the paper's two-forward-pass multi-task recipe.

    The trainer owns:
    - The T5 model + tokenizer.
    - The label / rationale task prefixes.
    - The two forward passes.

    The trainer does NOT own:
    - The loss function (delegated to :class:`RationaleDistiller`).
    - The optimizer (the caller's responsibility).

    Attributes:
        model: The T5 (or any seq2seq) model.
        tokenizer: The associated tokenizer.
        label_prefix: Prepended to the input for the label view.
        rationale_prefix: Prepended to the input for the rationale
            view.
        device: The device the model lives on.
        label_decoder_start_token_id: Used to build the decoder
            input ids in teacher-forcing mode.  Falls back to the
            T5 default if the model has no explicit attribute.
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer: Any,
        *,
        label_prefix: str = "label: ",
        rationale_prefix: str = "rationale: ",
        device: Optional[torch.device] = None,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.label_prefix = label_prefix
        self.rationale_prefix = rationale_prefix
        self.device = device or next(model.parameters()).device
        self.label_decoder_start_token_id = getattr(
            model.config, "decoder_start_token_id", 0
        )

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def from_pretrained(
        cls,
        model_name: str,
        *,
        label_prefix: str = "label: ",
        rationale_prefix: str = "rationale: ",
        device: Optional[str] = None,
    ) -> "MultiTaskT5Trainer":
        """Load a T5-style model + tokenizer and wrap them."""
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)
        return cls(
            model,
            tokenizer,
            label_prefix=label_prefix,
            rationale_prefix=rationale_prefix,
            device=torch.device(device),
        )

    # ------------------------------------------------------------------
    # Tokenisation
    # ------------------------------------------------------------------

    def _encode(self, input_text: str, max_length: int = 128) -> Dict[str, torch.Tensor]:
        """Tokenise the input with the model's tokenizer, on device."""
        enc = self.tokenizer(
            input_text,
            return_tensors="pt",
            padding="max_length",
            max_length=max_length,
            truncation=True,
        )
        return {k: v.to(self.device) for k, v in enc.items()}

    def _teacher_forcing_targets(
        self, target_ids: torch.Tensor
    ) -> torch.Tensor:
        """Shift the target ids right for T5-style teacher forcing.

        Drops the last column and prepends ``decoder_start_token_id``.
        """
        shifted = target_ids.new_zeros(target_ids.shape)
        shifted[:, 0] = self.label_decoder_start_token_id
        shifted[:, 1:] = target_ids[:, :-1]
        return shifted

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward_label(
        self, input_text: str, *, max_length: int = 128
    ) -> torch.Tensor:
        """Inference path: run the model once with the [label] prefix
        and return the decoder logits.

        Wrapped in ``torch.no_grad()`` — this is the inference /
        inspection path, not the training path.  For training, use
        :meth:`train_step`, which teacher-forces the decoder and
        propagates gradients.
        """
        return self._forward_no_grad(self.label_prefix + input_text, max_length=max_length)

    def forward_rationale(
        self, input_text: str, *, max_length: int = 128
    ) -> torch.Tensor:
        """Inference path: run the model with the [rationale] prefix.

        See :meth:`forward_label` for the inference/training contract.
        """
        return self._forward_no_grad(self.rationale_prefix + input_text, max_length=max_length)

    def _forward_no_grad(self, full_text: str, *, max_length: int) -> torch.Tensor:
        """Shared inference path: build the decoder input from
        ``decoder_start_token_id`` and run the model under no_grad.

        The tiny / randomly-initialised T5 we use in the smoke has no
        ``decoder_start_token_id`` baked into the config, so we fall
        back to building a single-token decoder input of zeros.
        """
        encoded = self._encode(full_text, max_length=max_length)
        bsz = encoded["input_ids"].shape[0]
        decoder_input_ids = torch.full(
            (bsz, 1),
            int(self.label_decoder_start_token_id),
            dtype=torch.long,
            device=self.device,
        )
        with torch.no_grad():
            return self.model(
                input_ids=encoded["input_ids"],
                attention_mask=encoded["attention_mask"],
                decoder_input_ids=decoder_input_ids,
            ).logits

    def forward_both(
        self, input_text: str, *, max_length: int = 128
    ) -> Dict[str, torch.Tensor]:
        """Convenience: run both views, return a RationaleDistiller-shaped dict."""
        return {
            "label_logits": self.forward_label(input_text, max_length=max_length),
            "rationale_logits": self.forward_rationale(
                input_text, max_length=max_length
            ),
        }

    # ------------------------------------------------------------------
    # Multi-task training step
    # ------------------------------------------------------------------

    def train_step(
        self,
        batch: Dict[str, Any],
        distiller: Any,
        optimizer: torch.optim.Optimizer,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Run a single multi-task step.

        Args:
            batch: A dict with at least ``"input"`` (string) and
                ``"label_ids"`` / ``"rationale_ids"`` (already-tokenised
                target ids).  The trainer also accepts an optional
                ``"max_length"`` override.
            distiller: A :class:`RationaleDistiller` (or any object
                with a ``compute_loss`` method that takes the
                ``(student_outputs, targets)`` dict).
            optimizer: The optimiser to call ``zero_grad()`` and
                ``step()`` on.

        Returns:
            ``(loss, breakdown)`` matching
            :meth:`RationaleDistiller.compute_loss`.
        """
        input_text: str = batch["input"]
        max_length: int = batch.get("max_length", 128)
        label_target_ids = batch["label_ids"].to(self.device)
        rationale_target_ids = batch["rationale_ids"].to(self.device)

        # Encoder side: tokenise once, reuse the encoder for both heads.
        encoded = self._encode(input_text, max_length=max_length)

        optimizer.zero_grad()

        # Label head — teacher-forced.
        label_decoder_input = self._teacher_forcing_targets(label_target_ids)
        label_out = self.model(
            input_ids=encoded["input_ids"],
            attention_mask=encoded["attention_mask"],
            decoder_input_ids=label_decoder_input,
        )
        label_logits = label_out.logits  # (B, T_target, V)

        # Rationale head — teacher-forced.
        rationale_decoder_input = self._teacher_forcing_targets(rationale_target_ids)
        rationale_out = self.model(
            input_ids=encoded["input_ids"],
            attention_mask=encoded["attention_mask"],
            decoder_input_ids=rationale_decoder_input,
        )
        rationale_logits = rationale_out.logits  # (B, T_rationale, V)

        student_outputs = {
            "label_logits": label_logits,
            "rationale_logits": rationale_logits,
        }
        targets = {
            "label_ids": label_target_ids,
            "rationale_ids": rationale_target_ids,
        }
        loss, breakdown = distiller.compute_loss(
            student_outputs=student_outputs, targets=targets
        )
        if not loss.requires_grad:
            raise RuntimeError(
                f"train_step loss has requires_grad=False (loss={loss.item()}, "
                f"grad_fn={loss.grad_fn}); offline model may have produced "
                f"detached logits. check label_ids / rationale_ids dtypes."
            )
        loss.backward()
        optimizer.step()
        return loss, breakdown
