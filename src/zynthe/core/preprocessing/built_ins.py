"""Built-in dataset adapters and model preprocessors.

Provides sensible defaults for common datasets and model families so
pipelines can bootstrap without custom glue code. The registry is
metadata-aware and supports fallbacks for unknown sources.
"""

from __future__ import annotations

from typing import Any, Dict, List

import torch

try:  # torchvision is optional in text-focused setups
    from torchvision import transforms
except Exception:  # pragma: no cover - optional dependency
    transforms = None

from .registry import ModelPreprocessor, PreprocessRegistry, SampleAdapter


# ---------------------- DATASET ADAPTERS ----------------------
class IMDBAdapter(SampleAdapter):
    def adapt(self, raw: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "text": raw.get("text") or raw.get("review"),
            "label": int(raw.get("label", 0)),
        }


class SST2Adapter(SampleAdapter):
    def adapt(self, raw: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "text": raw.get("sentence") or raw.get("text"),
            "label": int(raw.get("label", 0)),
        }


class Sentiment140Adapter(SampleAdapter):
    def adapt(self, raw: Dict[str, Any]) -> Dict[str, Any]:
        mapping = {-1: 0, 0: 1, 4: 2}
        raw_label = raw.get("sentiment")
        try:
            key = int(raw_label) if raw_label is not None else 0
        except (TypeError, ValueError):
            key = 0
        return {
            "text": raw.get("text"),
            "label": mapping.get(key, 0),
        }


class MNLIAdapter(SampleAdapter):
    def adapt(self, raw: Dict[str, Any]) -> Dict[str, Any]:
        premise = raw.get("premise")
        hypothesis = raw.get("hypothesis")
        text = f"Premise: {premise} [SEP] Hypothesis: {hypothesis}"
        return {
            "text": text,
            "label": int(raw.get("label", 0)),
        }


class QQPAdapter(SampleAdapter):
    def adapt(self, raw: Dict[str, Any]) -> Dict[str, Any]:
        q1 = raw.get("question1") or raw.get("question1_text")
        q2 = raw.get("question2") or raw.get("question2_text")
        text = f"Question A: {q1}\nQuestion B: {q2}" if q1 and q2 else q1 or q2
        return {
            "text": text,
            "label": int(raw.get("label", raw.get("is_duplicate", 0))),
        }


class QNLIAdapter(SampleAdapter):
    def adapt(self, raw: Dict[str, Any]) -> Dict[str, Any]:
        question = raw.get("question")
        sentence = raw.get("sentence") or raw.get("context")
        text = f"Question: {question}\nSentence: {sentence}"
        return {
            "text": text,
            "label": int(raw.get("label", 0)),
        }


class COLAAdapter(SampleAdapter):
    def adapt(self, raw: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "text": raw.get("sentence") or raw.get("text"),
            "label": int(raw.get("label", 0)),
        }


class AGNewsAdapter(SampleAdapter):
    def adapt(self, raw: Dict[str, Any]) -> Dict[str, Any]:
        title = raw.get("title")
        description = raw.get("description") or raw.get("text")
        text = f"Title: {title}\nBody: {description}" if title else description
        return {
            "text": text,
            "label": int(raw.get("label", raw.get("category", 0))),
        }


class CIFAR10Adapter(SampleAdapter):
    def adapt(self, raw: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "image": raw.get("image"),
            "label": int(raw.get("label", 0)),
        }


class GenericTextAdapter(SampleAdapter):
    """Fallback adapter that best-effort normalizes text + label fields."""

    TEXT_KEYS = ["text", "sentence", "review", "content", "prompt", "document"]
    LABEL_KEYS = ["label", "labels", "target", "sentiment", "class", "category"]

    def adapt(self, raw: Dict[str, Any]) -> Dict[str, Any]:
        text_val = None
        for key in self.TEXT_KEYS:
            val = raw.get(key)
            if val:
                text_val = val
                break
        if text_val is None:
            raise ValueError("GenericTextAdapter could not locate a text field in sample")

        label_val = 0
        for key in self.LABEL_KEYS:
            if key in raw and raw[key] is not None:
                try:
                    label_val = int(raw[key])
                except (ValueError, TypeError):
                    label_val = hash(str(raw[key])) % 1_000_000
                break

        return {
            "text": text_val,
            "label": label_val,
            "raw": raw,
        }


# ---------------------- MODEL PREPROCESSORS ----------------------
class BertPreprocessor(ModelPreprocessor):
    def prepare(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        max_len = self.config.get("model", {}).get("max_length", 128)
        enc = self.tokenizer(
            sample["text"],
            truncation=True,
            padding="max_length",
            max_length=max_len,
            return_tensors="pt",
        )
        item = {k: v.squeeze(0) for k, v in enc.items()}
        item["labels"] = torch.tensor(sample["label"], dtype=torch.long)
        return item


class T5Preprocessor(ModelPreprocessor):
    def prepare(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        max_len = self.config.get("model", {}).get("max_length", 128)
        prefix = self.config.get("preprocessing", {}).get("task_prefix", "classify: ")
        text = prefix + sample["text"]
        enc = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=max_len,
            return_tensors="pt",
        )
        item = {k: v.squeeze(0) for k, v in enc.items()}
        item["labels"] = torch.tensor(sample["label"], dtype=torch.long)
        return item


class GenericPreprocessor(BertPreprocessor):
    """Alias for BERT-style preprocessing used as generic fallback."""


class GPTLikePreprocessor(ModelPreprocessor):
    """Causal language modeling preprocessor with mask-aware labels."""

    def prepare(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        text = sample["text"]
        model_cfg = self.config.get("model", {})
        max_len = model_cfg.get("max_length", 512)
        pad_to_max = self.config.get("preprocessing", {}).get("pad_to_max_length", True)
        padding = "max_length" if pad_to_max else False

        enc = self.tokenizer(
            text,
            truncation=True,
            max_length=max_len,
            padding=padding,
            return_tensors="pt",
        )
        input_ids = enc["input_ids"].squeeze(0)
        attention_mask = enc["attention_mask"].squeeze(0)
        labels = input_ids.clone()
        if pad_to_max:
            labels = labels.masked_fill(attention_mask == 0, -100)
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


class ViTPreprocessor(ModelPreprocessor):
    """Vision Transformer preprocessor that builds torchvision pipelines."""

    def __init__(self, tokenizer, config: Dict[str, Any]):  # tokenizer may be None for vision
        super().__init__(tokenizer, config)
        self.image_size = self.config.get("model", {}).get("image_size", 224)
        self.normalize_mean = self.config.get("preprocessing", {}).get("image_mean", [0.485, 0.456, 0.406])
        self.normalize_std = self.config.get("preprocessing", {}).get("image_std", [0.229, 0.224, 0.225])
        aug_cfg = self.config.get("preprocessing", {}).get("augment", {})

        if transforms:
            steps: List[Any] = [transforms.Resize((self.image_size, self.image_size))]
            if aug_cfg.get("random_flip", True):
                steps.append(transforms.RandomHorizontalFlip())
            if aug_cfg.get("random_crop", False):
                steps.append(transforms.RandomResizedCrop(self.image_size, scale=(0.9, 1.0)))
            if aug_cfg.get("color_jitter", False):
                steps.append(
                    transforms.ColorJitter(
                        brightness=0.1,
                        contrast=0.1,
                        saturation=0.1,
                        hue=0.05,
                    )
                )
            steps.append(transforms.ToTensor())
            steps.append(transforms.Normalize(self.normalize_mean, self.normalize_std))
            self.tx = transforms.Compose(steps)
        else:
            self.tx = None

    def prepare(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        img = sample.get("image")
        if img is None:
            raise ValueError("ViTPreprocessor requires 'image' key in sample")
        if self.tx:
            pixel_values = self.tx(img)
        else:
            if isinstance(img, torch.Tensor):
                pixel_values = img.float()
            else:
                raise RuntimeError("Torchvision not available and image is not a tensor")
        return {
            "pixel_values": pixel_values,
            "labels": torch.tensor(sample.get("label", 0), dtype=torch.long),
        }


# ---------------------- REGISTRATION ----------------------


def register_defaults() -> None:
    """Register built-in adapters and preprocessors with metadata."""

    PreprocessRegistry.register_dataset(
        "imdb",
        IMDBAdapter(),
        {"task": "sentiment", "modality": "text", "num_labels": 2},
    )
    PreprocessRegistry.register_dataset(
        "sst2",
        SST2Adapter(),
        {"task": "sentiment", "modality": "text", "num_labels": 2},
    )
    PreprocessRegistry.register_dataset(
        "glue/sst2",
        SST2Adapter(),
        {"task": "sentiment", "modality": "text", "alias": "sst2"},
    )
    PreprocessRegistry.register_dataset(
        "sentiment140",
        Sentiment140Adapter(),
        {"task": "sentiment", "modality": "text", "num_labels": 3},
    )
    PreprocessRegistry.register_dataset(
        "mnli",
        MNLIAdapter(),
        {"task": "nli", "modality": "text", "num_labels": 3},
    )
    PreprocessRegistry.register_dataset(
        "qqp",
        QQPAdapter(),
        {"task": "paraphrase", "modality": "text", "num_labels": 2},
    )
    PreprocessRegistry.register_dataset(
        "qnli",
        QNLIAdapter(),
        {"task": "qa", "modality": "text", "num_labels": 2},
    )
    PreprocessRegistry.register_dataset(
        "cola",
        COLAAdapter(),
        {"task": "acceptability", "modality": "text", "num_labels": 2},
    )
    PreprocessRegistry.register_dataset(
        "ag_news",
        AGNewsAdapter(),
        {"task": "topic", "modality": "text", "num_labels": 4},
    )
    PreprocessRegistry.register_dataset(
        "cifar10",
        CIFAR10Adapter(),
        {"task": "vision-classification", "modality": "vision", "num_labels": 10},
    )
    PreprocessRegistry.register_dataset(
        "cifar100",
        CIFAR10Adapter(),
        {"task": "vision-classification", "modality": "vision", "num_labels": 100},
    )
    PreprocessRegistry.register_dataset(
        "stl10",
        CIFAR10Adapter(),
        {"task": "vision-classification", "modality": "vision", "num_labels": 10},
    )
    PreprocessRegistry.register_dataset(
        "imagenet",
        CIFAR10Adapter(),
        {"task": "vision-classification", "modality": "vision"},
    )
    PreprocessRegistry.register_dataset(
        "image_folder",
        CIFAR10Adapter(),
        {"task": "vision-classification", "modality": "vision"},
    )
    PreprocessRegistry.register_dataset(
        PreprocessRegistry.GENERIC_DATASET_KEY,
        GenericTextAdapter(),
        {"task": "text-classification", "modality": "text", "fallback": True},
    )

    PreprocessRegistry.register_model(
        "bert",
        lambda tok, cfg: BertPreprocessor(tok, cfg),
        {"tasks": ["sequence-classification", "token-classification"]},
    )
    PreprocessRegistry.register_model(
        "t5",
        lambda tok, cfg: T5Preprocessor(tok, cfg),
        {"tasks": ["text2text", "seq2seq"]},
    )
    PreprocessRegistry.register_model(
        "gpt2",
        lambda tok, cfg: GPTLikePreprocessor(tok, cfg),
        {"tasks": ["causal-lm", "instruction-tuning"]},
    )
    PreprocessRegistry.register_model(
        "generic",
        lambda tok, cfg: GenericPreprocessor(tok, cfg),
        {"tasks": ["sequence-classification"], "fallback": True},
    )
    PreprocessRegistry.register_model(
        "vit",
        lambda tok, cfg: ViTPreprocessor(tok, cfg),
        {"tasks": ["vision-classification"]},
    )
