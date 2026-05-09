"""
Tests validating the bug-fixes applied to the Zynthé codebase.

Covers:
1. torch.load weights_only fix (checkpoint round-trip)
2. torch.amp.GradScaler / autocast deprecation fix (no FutureWarning)
3. kd_hinton.py comment indentation fix (super().__init__ is called)
4. CausalLMDistillationEngine basic forward pass
"""
from __future__ import annotations

import os
import tempfile
import warnings
from types import SimpleNamespace

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _TinyLM(nn.Module):
    """Minimal GPT-style model that returns a dict with 'logits'."""

    def __init__(self, vocab: int = 32, hidden: int = 16):
        super().__init__()
        self.embed = nn.Embedding(vocab, hidden)
        self.lm_head = nn.Linear(hidden, vocab)

    def forward(self, input_ids: torch.Tensor, **kwargs):
        h = self.embed(input_ids)
        return {"logits": self.lm_head(h)}


# ---------------------------------------------------------------------------
# 1. torch.load weights_only fix
# ---------------------------------------------------------------------------

class TestCheckpointRoundTrip:
    """Verify that checkpoints containing NumPy RNG state load correctly."""

    def test_save_and_load_with_numpy_rng_state(self):
        """save_training_checkpoint + smart_load_checkpoint must survive PyTorch 2.6+."""
        from core.distillers.causal_lm.checkpoint import (
            CheckpointMeta,
            TrainingState,
            save_training_checkpoint,
            smart_load_checkpoint,
        )

        model = _TinyLM(vocab=32)
        optim = torch.optim.AdamW(model.parameters(), lr=1e-3)

        with tempfile.TemporaryDirectory() as tmp:
            ckpt_path = os.path.join(tmp, "checkpoint.pt")

            save_training_checkpoint(
                path=ckpt_path,
                model=model,
                optimizer=optim,
                scheduler=None,
                scaler=None,
                state=TrainingState(epoch=2, global_step=10, best_metric=0.5),
                metadata=CheckpointMeta(epoch=2, global_step=10, seed=42),
            )

            assert os.path.exists(ckpt_path)

            model2 = _TinyLM(vocab=32)
            optim2 = torch.optim.AdamW(model2.parameters(), lr=1e-3)
            report, state, meta = smart_load_checkpoint(
                path=ckpt_path,
                model=model2,
                optimizer=optim2,
                scheduler=None,
                scaler=None,
                map_location="cpu",
                strict_first=True,
            )

            assert report.strict_loaded, "Expected strict load to succeed"
            assert state.epoch == 2
            assert state.global_step == 10
            assert meta is not None and meta.seed == 42

    def test_checkpoint_stress_tests_all_pass(self):
        """The full checkpoint stress-test suite must pass with the weights_only fix."""
        from core.distillers.causal_lm.validation import run_checkpoint_stress_tests

        report = run_checkpoint_stress_tests()
        for scenario in report.scenarios:
            assert scenario.passed, (
                f"Checkpoint stress scenario '{scenario.name}' failed: "
                f"{scenario.details}"
            )
        assert report.all_passed


# ---------------------------------------------------------------------------
# 2. torch.amp.GradScaler deprecation fix — no FutureWarning
# ---------------------------------------------------------------------------

class TestAMPNoDeprecationWarning:
    """torch.amp.GradScaler / autocast usage must not emit FutureWarning."""

    def test_grad_scaler_no_future_warning_validation(self):
        """validation.py scalers must not produce FutureWarning."""
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            # Import triggers module-level code; also instantiate the path directly.
            import torch
            _ = torch.amp.GradScaler("cuda", enabled=False)

        future_warnings = [
            w for w in caught
            if issubclass(w.category, FutureWarning)
            and "cuda.amp" in str(w.message).lower()
        ]
        assert not future_warnings, (
            f"Unexpected FutureWarning(s): {[str(w.message) for w in future_warnings]}"
        )

    def test_autocast_no_future_warning(self):
        """torch.amp.autocast must not produce FutureWarning for 'cuda' device type."""
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            with torch.amp.autocast("cuda", enabled=False):
                _ = torch.tensor(1.0) + torch.tensor(2.0)

        future_warnings = [
            w for w in caught
            if issubclass(w.category, FutureWarning)
            and "cuda.amp" in str(w.message).lower()
        ]
        assert not future_warnings, (
            f"Unexpected FutureWarning(s): {[str(w.message) for w in future_warnings]}"
        )


# ---------------------------------------------------------------------------
# 3. kd_hinton.py indentation fix — super().__init__ is reached
# ---------------------------------------------------------------------------

class TestKDHintonInit:
    """Verify that KDHintonDistiller.__init__ correctly calls super().__init__."""

    def test_super_init_called(self, dummy_models):
        """KDHintonDistiller must initialise its BaseDistiller parent properly."""
        from core.distillers.kd_hinton import KDHintonDistiller

        teacher, student = dummy_models
        distiller = KDHintonDistiller(
            teacher,
            student,
            config={"kd_hinton": {"temperature": 3.0, "alpha": 0.8}},
        )
        # These attributes come from BaseDistiller.__init__ — if super() was
        # never called they would be missing.
        assert hasattr(distiller, "teacher")
        assert hasattr(distiller, "student")
        assert hasattr(distiller, "device")

    def test_hint_enabled_config_respected(self, dummy_models):
        """hint_enabled flag from top-level config must override sub-config."""
        from core.distillers.kd_hinton import KDHintonDistiller

        teacher, student = dummy_models
        distiller = KDHintonDistiller(
            teacher,
            student,
            config={"hint_enabled": False, "kd_hinton": {}},
        )
        assert not distiller.hint_enabled

    def test_compute_loss_returns_finite(self, dummy_models):
        """Basic forward + loss computation must return a finite scalar."""
        from core.distillers.kd_hinton import KDHintonDistiller

        teacher, student = dummy_models
        distiller = KDHintonDistiller(
            teacher,
            student,
            config={"kd_hinton": {"temperature": 4.0, "alpha": 0.7}},
        )

        input_ids = torch.randint(0, 50, (4, 8))
        attention_mask = torch.ones(4, 8, dtype=torch.long)
        labels = torch.randint(0, 2, (4,))
        batch = {"input_ids": input_ids, "attention_mask": attention_mask}

        with torch.no_grad():
            t_out = teacher(**batch, labels=labels)
            s_out = student(**batch, labels=labels)

        loss, _ = distiller.compute_loss(
            student_outputs=s_out,
            teacher_outputs=t_out,
            targets=labels,
        )
        assert torch.isfinite(loss), f"Loss is not finite: {loss}"


# ---------------------------------------------------------------------------
# 4. CausalLMDistillationEngine basic forward pass
# ---------------------------------------------------------------------------

class TestCausalLMDistillationEngine:
    """Basic sanity checks for the fixed distillation engine."""

    def _make_outputs(self, logits: torch.Tensor):
        return SimpleNamespace(logits=logits)

    def test_normal_batch(self):
        from core.distillers.causal_lm.distillation import (
            CausalLMDistillationEngine,
            DistillationConfig,
        )

        engine = CausalLMDistillationEngine(DistillationConfig(temperature=2.0, alpha=0.7))
        B, T, V = 2, 8, 50
        s_logits = torch.randn(B, T, V)
        t_logits = torch.randn(B, T, V)
        labels = torch.randint(0, V, (B, T))
        labels[:, 0] = -100  # mask the first token

        result = engine.compute_total_loss(
            student_outputs=self._make_outputs(s_logits),
            teacher_outputs=self._make_outputs(t_logits),
            labels=labels,
        )
        assert result.is_finite, f"Loss is not finite. Warning: {result.warning}"
        assert result.valid_tokens > 0
        assert torch.isfinite(result.total)
        assert torch.isfinite(result.kd)
        assert torch.isfinite(result.ce)

    def test_all_masked_tokens_returns_zero(self):
        from core.distillers.causal_lm.distillation import (
            CausalLMDistillationEngine,
            DistillationConfig,
        )

        engine = CausalLMDistillationEngine(DistillationConfig(temperature=2.0, alpha=0.5))
        B, T, V = 2, 4, 30
        s_logits = torch.randn(B, T, V)
        t_logits = torch.randn(B, T, V)
        labels = torch.full((B, T), -100, dtype=torch.long)  # everything masked

        result = engine.compute_total_loss(
            student_outputs=self._make_outputs(s_logits),
            teacher_outputs=self._make_outputs(t_logits),
            labels=labels,
        )
        assert result.valid_tokens == 0
        assert result.is_finite
        assert result.warning is not None  # should emit a warning about no valid tokens
