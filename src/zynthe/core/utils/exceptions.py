"""Exception hierarchy for Zynthe.

All public errors raised by Zynthe should derive from :class:`ZyntheError`.
This lets downstream code write ``except ZyntheError`` without catching
unrelated ``ValueError`` / ``RuntimeError`` raised by libraries like PyTorch.
"""

from __future__ import annotations

from typing import Any, Iterable, Optional


class ZyntheError(Exception):
    """Base class for every exception raised by Zynthe.

    Args:
        message: Human-readable description of the failure.
        context: Optional dict of relevant context (will be shown in __str__).
        cause: Optional original exception.
    """

    def __init__(
        self,
        message: str = "",
        *,
        context: Optional[dict] = None,
        cause: Optional[BaseException] = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.context: dict[str, Any] = dict(context or {})
        if cause is not None:
            self.__cause__ = cause

    def __str__(self) -> str:
        if not self.context:
            return self.message
        ctx = ", ".join(f"{k}={v!r}" for k, v in self.context.items())
        return f"{self.message} [{ctx}]"


class ConfigError(ZyntheError):
    """Configuration is invalid (missing fields, bad types, unknown keys)."""


class DistillationError(ZyntheError):
    """A distillation step failed (forward, loss, hook, optimizer)."""


class AdapterError(ZyntheError):
    """No suitable adapter was found for a model, or an adapter mis-handled I/O."""


class PreflightError(ZyntheError):
    """Pre-train checks blocked the run (OOM, shape mismatch, missing data)."""


class QuantizationError(ZyntheError):
    """Quantization flow (PTQ / QAT / export) failed."""


class RegistryError(ZyntheError):
    """A name was not found in a registry (distiller / pipeline / adapter)."""


def format_missing_layers(missing: Iterable[str]) -> str:
    """Render a list of missing layer names for error messages."""
    missing_list = list(missing)
    if not missing_list:
        return ""
    if len(missing_list) == 1:
        return f"missing layer {missing_list[0]!r}"
    head = ", ".join(repr(name) for name in missing_list[:-1])
    return f"missing layers {head} and {missing_list[-1]!r}"
