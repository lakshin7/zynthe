"""Lightweight inference utilities for student models.

Provides a simple, transparent wrapper that mirrors training-time tokenizer
settings (max_length, padding/truncation) and returns probabilities and labels.
"""
from __future__ import annotations
from typing import List, Dict, Any, Optional
import torch
import torch.nn.functional as F


class StudentInference:
    def __init__(self, model, tokenizer, config: Optional[Dict[str, Any]] = None, device: Optional[torch.device] = None):
        self.model = model.eval()
        self.tokenizer = tokenizer
        self.config = config or {}
        self.device = device or next(model.parameters()).device
        self.model.to(self.device)

        # Inference parameters
        self.max_length = (
            self.config.get('model', {}).get('max_length', 128)
        )

        # Label mapping if available
        cfg = getattr(self.model, 'config', None)
        self.id2label = getattr(cfg, 'id2label', None) if cfg is not None else None

    @torch.no_grad()
    def predict(self, texts: List[str], batch_size: int = 8) -> List[Dict[str, Any]]:
        """Return predictions with logits and probabilities.

        Output schema per item:
          {
            'text': str,
            'label_id': int,
            'label': Optional[str],
            'prob': float,
            'probs': List[float]
          }
        """
        results: List[Dict[str, Any]] = []
        if not texts:
            return results

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            enc = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors='pt'
            )
            enc = {k: v.to(self.device) for k, v in enc.items()}
            outputs = self.model(**enc)
            logits = outputs.logits if hasattr(outputs, 'logits') else outputs[0]
            probs = F.softmax(logits, dim=-1)
            top_prob, top_idx = probs.max(dim=-1)

            for j, text in enumerate(batch_texts):
                label_id = int(top_idx[j].item())
                label_name = self.id2label.get(label_id) if isinstance(self.id2label, dict) else None
                results.append({
                    'text': text,
                    'label_id': label_id,
                    'label': label_name,
                    'prob': float(top_prob[j].item()),
                    'probs': probs[j].detach().cpu().tolist(),
                })

        return results


__all__ = ['StudentInference']
