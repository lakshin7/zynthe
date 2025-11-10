from typing import Any, List, Mapping, Sequence, Tuple, Union

import matplotlib.pyplot as plt
import shap
from shap import maskers
import torch
from transformers import PreTrainedModel, PreTrainedTokenizerFast

class SHAPExplainer:
    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerFast,
        device: Union[str, torch.device] = 'cpu',
        background_size: int = 10,
    ):
        self.tokenizer = tokenizer
        self.background_size = background_size
        self._original_device = next(model.parameters()).device
        self.device = torch.device(device)
        self.model = model
        try:
            self.model.to(self.device)  # type: ignore[call-arg]
        except TypeError:
            self.model = self.model.to(self.device)  # type: ignore[call-arg]
        self.model.eval()

    @staticmethod
    def _normalize_text(sample: Any) -> str:
        if sample is None:
            return ""
        if isinstance(sample, str):
            return sample
        if isinstance(sample, Mapping):
            for key in ("text", "content", "sentence", "prompt", "input"):
                if key in sample:
                    candidate = SHAPExplainer._normalize_text(sample[key])
                    if candidate:
                        return candidate
            try:
                joined = " ".join(str(value) for value in sample.values() if value is not None)
                if joined:
                    return joined
            except Exception:
                pass
            return str(sample)
        if isinstance(sample, Sequence) and not isinstance(sample, (bytes, bytearray)):
            parts = [SHAPExplainer._normalize_text(item) for item in sample]
            joined = " ".join(part for part in parts if part)
            if joined:
                return joined
        return str(sample)

    def _normalize_batch(self, texts: Union[str, Sequence[Any]]) -> List[str]:
        if isinstance(texts, str):
            return [self._normalize_text(texts)]
        try:
            iterator = list(texts)
        except TypeError:
            iterator = [texts]
        normalised = [self._normalize_text(item) for item in iterator]
        return [text if text else "" for text in normalised]

    def encode(self, texts):
        normalised = self._normalize_batch(texts)
        return self.tokenizer(normalised, padding=True, truncation=True, return_tensors='pt').to(self.device)

    def predict(self, texts):
        inputs = self.encode(texts)
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits if hasattr(outputs, 'logits') else outputs[0]
            return logits.cpu().numpy()

    def explain(self, texts):
        if not texts:
            return None
        normalised = self._normalize_batch(texts)
        if not normalised:
            return None
        masker = maskers.Text()
        explainer = shap.Explainer(self.predict, masker, algorithm='permutation', seed=42)
        shap_values = explainer(normalised)
        return shap_values

    def visualize(self, shap_values, show=True, save_path=None):
        if shap_values is None:
            return None
        try:
            if show:
                shap.plots.bar(shap_values, show=True)
            if save_path:
                shap.plots.bar(shap_values, show=False)
                plt.savefig(save_path, dpi=150, bbox_inches="tight")
                plt.close()
        except Exception:
            shap.summary_plot(getattr(shap_values, 'values', shap_values), show=show, plot_type="bar")
            if save_path:
                shap.summary_plot(getattr(shap_values, 'values', shap_values), show=False, plot_type="bar")
                plt.savefig(save_path, dpi=150, bbox_inches="tight")
                plt.close()
        if not show:
            plt.close()
        return shap_values

    def summarize(self, shap_values, top_k: int = 10) -> List[List[Tuple[str, float]]]:
        """Return top token contributions per example."""
        summaries: List[List[Tuple[str, float]]] = []
        if shap_values is None:
            return summaries
        try:
            num_items = len(shap_values)
        except TypeError:
            return summaries
        for idx in range(num_items):
            try:
                tokens = shap_values.data[idx]
                contributions = shap_values.values[idx]
            except (AttributeError, IndexError, KeyError, TypeError):
                continue
            if tokens is None or contributions is None:
                continue
            pairs = list(zip(tokens, contributions))
            pairs.sort(key=lambda item: abs(float(item[1])), reverse=True)
            summaries.append([(str(token), float(score)) for token, score in pairs[:top_k]])
        return summaries

    def restore(self):
        """Restore model to its original device."""
        current_device = next(self.model.parameters()).device
        if current_device != self._original_device:
            try:
                self.model.to(self._original_device)  # type: ignore[call-arg]
            except TypeError:
                self.model = self.model.to(self._original_device)  # type: ignore[call-arg]