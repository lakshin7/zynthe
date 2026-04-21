"""Quantization-aware training runner."""

from __future__ import annotations

import copy
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import torch
from torch import nn
from torch.optim import AdamW

from core.models.model_loader import ModelLoader
from data.dataloaders import create_dataloaders

from .ptq import PTQRunner, _estimate_model_size, _resolve_device

LOG = logging.getLogger(__name__)


class QATRunner:
	"""Applies lightweight quantization-aware fine-tuning to the student model."""

	def __init__(self, cfg: Dict[str, Any]):
		self.cfg = cfg
		self.quant_cfg = cfg.get("quantization", {}) or {}
		self.device = _resolve_device(self.quant_cfg.get("device") or cfg.get("runtime", {}).get("device"))
		self.backend = (self.quant_cfg.get("backend") or "fbgemm").lower()
		self.epochs = int(self.quant_cfg.get("qat_epochs", 1))
		self.max_steps = int(self.quant_cfg.get("max_steps", 200))
		self.learning_rate = float(self.quant_cfg.get("learning_rate", 5e-5))
		self.weight_decay = float(self.quant_cfg.get("weight_decay", 0.0))
		self.eval_interval = int(self.quant_cfg.get("eval_interval", 0))
		self.use_amp = bool(self.quant_cfg.get("amp", False)) and self.device.type == "cuda"
		self.export_dir = self._prepare_output_dir()
		self.history: list[Dict[str, Any]] = []

	def _prepare_output_dir(self) -> Path:
		root = Path(self.quant_cfg.get("output_dir") or self.cfg.get("output_root", "experiments"))
		root = root.expanduser().resolve()
		root.mkdir(parents=True, exist_ok=True)
		timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%S")
		out_dir = root / f"qat_{timestamp}"
		out_dir.mkdir(parents=True, exist_ok=True)
		return out_dir

	def run(self) -> Dict[str, Any]:
		LOG.info("Starting QAT runner (backend=%s, device=%s)", self.backend, self.device)
		loader = ModelLoader(self.cfg, device=self.device)
		bundle = loader.load(use_agent=False, return_bundle=True)
		if isinstance(bundle, tuple):
			_, student, tokenizer = bundle
		else:
			student = bundle.student
			tokenizer = bundle.tokenizer

		train_loader, val_loader = create_dataloaders(self.cfg, tokenizer)

		prepared_model = self._prepare_model(student)
		quantized_model = self._train(prepared_model, train_loader, val_loader)

		eval_metrics = self._evaluate(quantized_model, val_loader, torch.device("cpu"))

		summary = self._summarize(student, quantized_model, eval_metrics)
		self._export(quantized_model, tokenizer)
		self._write_summary(summary)

		return summary

	def _prepare_model(self, model: nn.Module) -> nn.Module:
		torch.backends.quantized.engine = self.backend
		clone = copy.deepcopy(model)
		clone.to(self.device)
		clone.train()
		try:
			setattr(clone, "qconfig", torch.quantization.get_default_qat_qconfig(self.backend))
			prepared = torch.quantization.prepare_qat(clone, inplace=False)
			LOG.info("Model prepared for QAT using backend '%s'", self.backend)
			return prepared
		except Exception as exc:
			LOG.warning("QAT preparation failed (%s). Proceeding without fake quant modules.", exc)
			return clone

	def _train(
		self,
		model: nn.Module,
		train_loader,
		val_loader,
	) -> nn.Module:
		optimizer = AdamW(model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
		scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)

		global_step = 0
		for epoch in range(self.epochs):
			model.train()
			for batch in train_loader:
				if self.max_steps and global_step >= self.max_steps:
					break
				optimizer.zero_grad()
				batch_on_device = self._move_batch_to_device(batch)
				with torch.amp.autocast("cuda", enabled=self.use_amp):
					outputs = model(**batch_on_device)
					loss = self._resolve_loss(outputs, batch_on_device)
				scaler.scale(loss).backward()
				scaler.step(optimizer)
				scaler.update()
				global_step += 1
			if self.eval_interval and (epoch + 1) % self.eval_interval == 0:
				metrics = self._evaluate(model, val_loader, self.device)
				self.history.append({"epoch": epoch + 1, "metrics": metrics})

		try:
			model_cpu = copy.deepcopy(model).to(torch.device("cpu"))
			quantized = torch.quantization.convert(model_cpu.eval(), inplace=False)
			return quantized
		except Exception as exc:
			LOG.warning("QAT convert failed (%s). Falling back to dynamic PTQ.", exc)
			quantized, strategy = PTQRunner.quantize_model(
				model=model.cpu(),
				strategy="dynamic",
				device="cpu",
				dtype=torch.qint8,
				backend=self.backend,
				target_modules=None,
				fallback="float16",
				calibration_loader=None,
				calibration_cfg=None,
			)
			LOG.info("Fallback PTQ strategy used: %s", strategy)
			return quantized

	def _move_batch_to_device(self, batch: Dict[str, Any]) -> Dict[str, Any]:
		out: Dict[str, Any] = {}
		for key, value in batch.items():
			if torch.is_tensor(value):
				out[key] = value.to(self.device)
			elif isinstance(value, list) and value and torch.is_tensor(value[0]):
				out[key] = torch.stack([v.to(self.device) for v in value])
			else:
				out[key] = value
		return out

	def _resolve_loss(self, outputs: Any, batch: Dict[str, Any]) -> torch.Tensor:
		if outputs is None:
			raise RuntimeError("Model returned no outputs during QAT")
		if isinstance(outputs, dict):
			if outputs.get("loss") is not None:
				return outputs["loss"]
			logits = outputs.get("logits")
		else:
			loss = getattr(outputs, "loss", None)
			if loss is not None:
				return loss
			logits = getattr(outputs, "logits", None)
		labels = batch.get("labels")
		if logits is None or labels is None:
			raise RuntimeError("Cannot compute QAT loss without logits and labels")
		labels = labels.to(logits.device)
		criterion = nn.CrossEntropyLoss()
		return criterion(logits, labels)

	def _evaluate(
		self,
		model: nn.Module,
		loader,
		device: torch.device,
	) -> Dict[str, Any]:
		model = model.to(device)
		model.eval()
		correct = 0
		total = 0
		with torch.inference_mode():
			for batch in loader:
				batch_on_device = {}
				for key, value in batch.items():
					if torch.is_tensor(value):
						batch_on_device[key] = value.to(device)
					elif isinstance(value, list) and value and torch.is_tensor(value[0]):
						batch_on_device[key] = torch.stack([v.to(device) for v in value])
					else:
						batch_on_device[key] = value
				labels = batch_on_device.get("labels")
				if labels is None:
					continue
				outputs = model(**batch_on_device)
				logits = None
				if isinstance(outputs, dict):
					logits = outputs.get("logits")
				else:
					logits = getattr(outputs, "logits", None)
				if logits is None:
					continue
				preds = torch.argmax(logits, dim=-1)
				correct += int((preds == labels).sum().item())
				total += int(labels.numel())
		accuracy = (correct / total) if total else 0.0
		return {"accuracy": accuracy, "samples": total}

	def _summarize(
		self,
		baseline: nn.Module,
		quantized: nn.Module,
		metrics: Dict[str, Any],
	) -> Dict[str, Any]:
		before_bytes = _estimate_model_size(baseline)
		after_bytes = _estimate_model_size(quantized)
		summary = {
			"backend": self.backend,
			"device": str(self.device),
			"epochs": self.epochs,
			"max_steps": self.max_steps,
			"learning_rate": self.learning_rate,
			"size_before_bytes": before_bytes,
			"size_after_bytes": after_bytes,
			"size_delta_bytes": before_bytes - after_bytes,
			"eval_metrics": metrics,
			"history": self.history,
			"export_dir": str(self.export_dir),
		}
		LOG.info(
			"QAT reduced model from %.2f MB to %.2f MB (Δ %.2f MB)",
			before_bytes / (1024 ** 2),
			after_bytes / (1024 ** 2),
			(before_bytes - after_bytes) / (1024 ** 2),
		)
		return summary

	def _export(self, model: nn.Module, tokenizer) -> None:
		export_model = model.to(torch.device("cpu"))
		try:
			if hasattr(export_model, "save_pretrained"):
				export_model.save_pretrained(self.export_dir)  # type: ignore[attr-defined]
			else:
				torch.save(export_model.state_dict(), self.export_dir / "qat_model.bin")
		except Exception as exc:
			LOG.warning("Failed to export QAT model via save_pretrained: %s", exc)
			torch.save(export_model.state_dict(), self.export_dir / "qat_model.bin")

		if tokenizer is not None and hasattr(tokenizer, "save_pretrained"):
			try:
				tokenizer.save_pretrained(self.export_dir)  # type: ignore[attr-defined]
			except Exception as exc:
				LOG.warning("Failed to export tokenizer: %s", exc)

	def _write_summary(self, summary: Dict[str, Any]) -> None:
		path = self.export_dir / "qat_summary.json"
		try:
			with path.open("w", encoding="utf-8") as handle:
				json.dump(summary, handle, indent=2)
		except Exception as exc:
			LOG.warning("Failed to write QAT summary: %s", exc)
