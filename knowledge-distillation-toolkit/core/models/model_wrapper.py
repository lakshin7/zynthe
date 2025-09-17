import logging
from typing import Any, Optional
import torch

class ModelWrapper:
	"""
	Enterprise-level wrapper for PyTorch/HuggingFace models.
	Provides unified interface for forward, save, quantize, and device management.
	"""
	def __init__(self, model: torch.nn.Module, device: Optional[torch.device] = None, tokenizer: Optional[Any] = None):
		self.model = model
		self.tokenizer = tokenizer
		self.device = device or torch.device("cpu")
		self.logger = logging.getLogger(self.__class__.__name__)
		self.model.to(self.device)
		self.logger.info(f"Model moved to device: {self.device}")

	def forward(self, *args, **kwargs) -> Any:
		self.model.eval()
		with torch.no_grad():
			return self.model(*args, **kwargs)

	def save(self, path: str):
		try:
			if hasattr(self.model, "save_pretrained"):
				self.model.save_pretrained(path)
				self.logger.info(f"Model saved to {path}")
				if self.tokenizer is not None and hasattr(self.tokenizer, "save_pretrained"):
					self.tokenizer.save_pretrained(path)
					self.logger.info(f"Tokenizer saved to {path}")
			else:
				torch.save(self.model.state_dict(), path)
				self.logger.info(f"Model state_dict saved to {path}")
		except Exception as e:
			self.logger.error(f"Failed to save model: {e}")
			raise

	def quantize(self, dtype: torch.dtype = torch.qint8) -> torch.nn.Module:
		try:
			quantized = torch.quantization.quantize_dynamic(
				self.model, {torch.nn.Linear}, dtype=dtype
			)
			quantized.to(self.device)
			self.logger.info(f"Model quantized with dtype {dtype}")
			return quantized
		except Exception as e:
			self.logger.warning(f"Quantization failed: {e}")
			return self.model

	def to(self, device: torch.device):
		self.device = device
		self.model.to(device)
		self.logger.info(f"Model moved to device: {device}")

	def summary(self) -> dict:
		param_count = sum(p.numel() for p in self.model.parameters())
		return {
			"class": self.model.__class__.__name__,
			"parameters": param_count,
			"device": str(self.device),
		}
