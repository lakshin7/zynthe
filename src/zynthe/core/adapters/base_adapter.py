"""
Base Model Adapter — Abstract Interface
=========================================

Defines the :class:`ModelAdapter` ABC that every concrete adapter must
implement.  Adapters normalise the following across architectures:

1. **Batch preparation** — filter / reshape batch keys for a given model.
2. **Output extraction** — turn arbitrary ``ModelOutput`` into a
   standard dict with ``logits``, ``hidden_states``, ``attentions``.
3. **Hook discovery** — return a list of module names suitable for
   registering forward hooks (used by feature/attention distillers).
4. **Dimension alignment** — project teacher features into student
   dimensionality (or vice versa).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn


class ModelAdapter(ABC):
    """Abstract base class for model I/O normalisation.

    Every adapter is *stateless* with respect to the forward pass — it
    may hold lightweight projection modules but those are always created
    lazily.  This lets the same adapter instance be reused across
    training steps without accumulating hidden state.
    """

    #: Human-readable modality tag (used for logging and registry keys).
    modality: str = "unknown"

    # ------------------------------------------------------------------
    # Required
    # ------------------------------------------------------------------

    @abstractmethod
    def prepare_batch(
        self,
        batch: Dict[str, Any],
        model: nn.Module,
    ) -> Dict[str, Any]:
        """Filter and transform *batch* so it can be passed to *model*.

        Only keys accepted by ``model.forward()`` should survive.
        Additional transformations (e.g. pixel-value normalisation for
        vision models) happen here.

        Args:
            batch: Raw training batch (dict of tensors).
            model: The model that will consume the output.

        Returns:
            Cleaned batch dict ready for ``model(**result)``.
        """

    @abstractmethod
    def extract_outputs(self, raw_output: Any) -> Dict[str, Any]:
        """Normalise raw model output into a standard dict.

        Required keys (may be ``None`` when unavailable):

        - ``logits``          — final prediction tensor
        - ``hidden_states``   — tuple of layer-wise hidden states
        - ``attentions``      — tuple of layer-wise attention weights
        - ``loss``            — pre-computed loss (if the model returns one)

        Additional modality-specific keys are allowed (e.g.
        ``image_embeds`` for CLIP).

        Args:
            raw_output: Whatever ``model.forward()`` returned.

        Returns:
            Normalised output dict.
        """

    @abstractmethod
    def get_hookable_layers(self, model: nn.Module) -> List[str]:
        """Return a list of ``named_modules()`` keys suitable for hooks.

        The returned names should point to the primary representation
        layers (e.g. transformer blocks, conv stages).  Distillers use
        these to register forward hooks for feature extraction.

        Args:
            model: The model to inspect.

        Returns:
            List of fully-qualified module names.
        """

    @abstractmethod
    def align_dimensions(
        self,
        teacher_features: Dict[str, torch.Tensor],
        student_features: Dict[str, torch.Tensor],
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """Project features so teacher and student dims match.

        May insert learned linear projections or simple interpolation
        depending on the architecture.

        Args:
            teacher_features: ``{layer_name: tensor}`` from teacher.
            student_features: ``{layer_name: tensor}`` from student.

        Returns:
            ``(aligned_teacher, aligned_student)`` with matching last-dim.
        """

    # ------------------------------------------------------------------
    # Optional helpers (can be overridden)
    # ------------------------------------------------------------------

    def supports_model(self, model: nn.Module) -> bool:
        """Quick heuristic: does this adapter know how to handle *model*?

        Used by :class:`AdapterRegistry` during auto-detection.  Default
        implementation returns ``False``; concrete adapters override.
        """
        return False

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(modality={self.modality!r})"
