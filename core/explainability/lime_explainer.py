from lime.lime_text import LimeTextExplainer
import torch
import numpy as np
from typing import Any, Mapping, Optional, Sequence, Union

class LimeTextExplainerWrapper:
    def __init__(
        self,
        model,
        tokenizer,
        class_names: list,
        *,
        device: Optional[Union[str, torch.device]] = None,
    ):
        """
        model: Hugging Face text classification model
        tokenizer: HF tokenizer corresponding to the model
        class_names: list of label names for classification
        """
        self.model = model
        self.tokenizer = tokenizer
        self.class_names = class_names
        self._original_device = next(model.parameters()).device
        if device is not None:
            target_device = torch.device(device)
        else:
            target_device = self._original_device
        if target_device != self._original_device:
            self.model.to(target_device)
        self.device = target_device
        self.explainer = LimeTextExplainer(class_names=class_names)

    @staticmethod
    def _normalize_text(sample: Any) -> str:
        """Coerce batched samples into plain text for LIME."""
        if sample is None:
            return ""
        if isinstance(sample, str):
            return sample
        if isinstance(sample, Mapping):
            for key in ("text", "content", "sentence", "prompt", "input"):
                if key in sample:
                    candidate = LimeTextExplainerWrapper._normalize_text(sample[key])
                    if candidate:
                        return candidate
            try:
                joined = " ".join(
                    str(value) for value in sample.values() if value is not None
                )
                if joined:
                    return joined
            except Exception:
                pass
            return str(sample)
        if isinstance(sample, Sequence) and not isinstance(sample, (bytes, bytearray)):
            parts = [LimeTextExplainerWrapper._normalize_text(item) for item in sample]
            joined = " ".join(part for part in parts if part)
            if joined:
                return joined
        return str(sample)

    def _predict_proba(self, texts):
        self.model.eval()
        if isinstance(texts, str):
            normalised = [self._normalize_text(texts)]
        else:
            try:
                iterator = list(texts)
            except TypeError:
                iterator = [texts]
            normalised = [self._normalize_text(item) or "" for item in iterator]
        inputs = self.tokenizer(normalised, padding=True, truncation=True, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        return probs.cpu().numpy()

    def explain(self, text: str, num_features: int = 10):
        """
        Generate LIME explanation for a single input text
        """
        text_instance = self._normalize_text(text) or ""
        explanation = self.explainer.explain_instance(
            text_instance=text_instance,
            classifier_fn=self._predict_proba,
            num_features=num_features
        )
        return explanation

    def visualize(self, explanation, num_features: int = 10, label: int | None = None):
        """
        Display LIME explanation in text format for the requested label.
        """
        try:
            target = label if label is not None else explanation.available_labels()[0]
        except AttributeError:
            target = label if label is not None else 0
        return explanation.as_list(label=target)[:num_features]

    def restore(self):
        """Move model back to its original device if changed."""
        current_device = next(self.model.parameters()).device
        if current_device != self._original_device:
            self.model.to(self._original_device)