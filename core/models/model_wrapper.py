import contextlib
import logging
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional

import torch
from torch import nn


logger = logging.getLogger(__name__)


@dataclass
class ForwardContext:
	"""Configuration bundle controlling forward execution."""

	use_grad: bool = False
	autocast: bool = False
	autocast_dtype: torch.dtype = torch.float16
	training_mode: Optional[bool] = None


class ModelWrapper:
	"""Enterprise wrapper providing unified model utilities."""

	def __init__(
		self,
		model: nn.Module,
		*,
		device: Optional[torch.device] = None,
		tokenizer: Optional[Any] = None,
		name: Optional[str] = None,
	) -> None:
		self.model = model
		self.tokenizer = tokenizer
		self.name = name or model.__class__.__name__
		self.device = device or torch.device("cpu")
		self.model.to(self.device)
		logger.info("Model '%s' moved to device %s", self.name, self.device)

	# ------------------------------------------------------------------
	# Core execution helpers
	# ------------------------------------------------------------------

	def forward(self, *args: Any, context: Optional[ForwardContext] = None, **kwargs: Any) -> Any:
		ctx = context or ForwardContext()
		mgr_stack = self._build_forward_managers(ctx)

		with contextlib.ExitStack() as stack:
			for mgr in mgr_stack:
				stack.enter_context(mgr)
			if ctx.training_mode is not None:
				train_mode = self.model.training
				if ctx.training_mode != train_mode:
					self.model.train(ctx.training_mode)
					stack.callback(lambda: self.model.train(train_mode))
			return self.model(*args, **kwargs)

	__call__ = forward

	def generate(self, *args: Any, **kwargs: Any) -> Any:
		if not hasattr(self.model, "generate"):
			raise AttributeError("Underlying model does not implement `generate`")
		generate_fn = getattr(self.model, "generate")
		return generate_fn(*args, **kwargs)

	def save(self, path: str, *, use_safetensors: bool = False) -> None:
		try:
			if hasattr(self.model, "save_pretrained"):
				self.model.save_pretrained(path, safe_serialization=use_safetensors)  # type: ignore[attr-defined]
				if self.tokenizer is not None and hasattr(self.tokenizer, "save_pretrained"):
					self.tokenizer.save_pretrained(path)  # type: ignore[attr-defined]
			else:
				torch.save(self.model.state_dict(), path)
		except Exception as exc:  # pragma: no cover - IO heavy
			logger.error("Failed to save model '%s': %s", self.name, exc)
			raise

	# ------------------------------------------------------------------
	# Model management helpers
	# ------------------------------------------------------------------

	def to(self, device: torch.device) -> None:
		self.device = device
		self.model.to(device)
		logger.info("Model '%s' moved to %s", self.name, device)

	def freeze(self, modules: Optional[Iterable[str]] = None) -> None:
		if modules is None:
			for param in self.model.parameters():
				param.requires_grad = False
			return

		named_modules = dict(self.model.named_parameters())
		for module in modules:
			param = named_modules.get(module)
			if param is not None:
				param.requires_grad = False

	def unfreeze(self, modules: Optional[Iterable[str]] = None) -> None:
		target = modules or [name for name, _ in self.model.named_parameters()]
		named_modules = dict(self.model.named_parameters())
		for module in target:
			param = named_modules.get(module)
			if param is not None:
				param.requires_grad = True

	def quantize(self, dtype: torch.dtype = torch.qint8) -> nn.Module:
		try:
			quantized = torch.quantization.quantize_dynamic(
				self.model,
				{nn.Linear},
				dtype=dtype,
			)
			quantized.to(self.device)
			logger.info("Model '%s' quantized to %s", self.name, dtype)
			return quantized
		except Exception as exc:  # pragma: no cover - backend specific
			logger.warning("Quantization failed for '%s': %s", self.name, exc)
			return self.model

	# ------------------------------------------------------------------
	# Diagnostics
	# ------------------------------------------------------------------

	def summary(self) -> Dict[str, Any]:
		param_total = sum(p.numel() for p in self.model.parameters())
		param_trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
		return {
			"name": self.name,
			"class": self.model.__class__.__name__,
			"parameters": int(param_total),
			"trainable_parameters": int(param_trainable),
			"device": str(self.device),
		}

	# ------------------------------------------------------------------
	# Internal utilities
	# ------------------------------------------------------------------

	def _build_forward_managers(self, ctx: ForwardContext) -> Iterable[contextlib.AbstractContextManager[Any]]:
		managers = []
		if not ctx.use_grad:
			managers.append(torch.no_grad())
		if ctx.autocast and torch.cuda.is_available():
			managers.append(torch.cuda.amp.autocast(dtype=ctx.autocast_dtype))
		return managers
