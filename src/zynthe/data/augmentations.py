
"""Lightweight text augmentation utilities used while training datasets.

The goal is to provide reproducible augmentations without adding heavyweight
dependencies. The :class:`TextAugmenter` below operates on tokenized text using
simple heuristics – random deletions, swaps, synonym substitutions, and noise
insertions. Each operation is optional and controlled through a configuration
dataclass, allowing the training configuration file to toggle augmentations per
split.

Example configuration snippet::

	preprocessing:
	  augment:
		enable: true
		apply_prob: 0.4
		dropout_prob: 0.1
		swap_prob: 0.05
		synonym_prob: 0.2
		noise_prob: 0.05
		reserved_tokens: ["[PAD]", "[CLS]", "<extra_id_0>"]

The augmenter is intentionally conservative: augmentations are only applied to
the ``text`` field by default and special tokens are protected. When WordNet is
available locally (``nltk`` with the corpus downloaded) synonym replacement is
enabled; otherwise the operation degrades gracefully without raising errors.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Iterable, List, Mapping, Optional, Sequence, Tuple, Union
NumberLike = Union[int, float, str]


def _coerce_float(value: object, fallback: float) -> float:
	if isinstance(value, (int, float)):
		return float(value)
	if isinstance(value, str):
		try:
			return float(value)
		except ValueError:
			return fallback
	return fallback


def _coerce_int(value: object, fallback: int) -> int:
	if isinstance(value, bool):  # bool is subclass of int
		return int(value)
	if isinstance(value, int):
		return value
	if isinstance(value, float):
		return int(value)
	if isinstance(value, str):
		try:
			return int(float(value))
		except ValueError:
			return fallback
	return fallback


def _coerce_sequence(value: object, *, fallback: Sequence[str] = ()) -> Tuple[str, ...]:
	if isinstance(value, str):
		return (value,)
	if isinstance(value, Iterable):
		return tuple(str(v) for v in value)
	return tuple(fallback)


def _coerce_optional_int(value: object) -> Optional[int]:
	if value is None:
		return None
	if isinstance(value, bool):
		return int(value)
	if isinstance(value, (int, float)):
		return int(value)
	if isinstance(value, str):
		try:
			return int(float(value))
		except ValueError:
			return None
	return None

try:  # Optional WordNet support
	from nltk.corpus import wordnet  # type: ignore
except Exception:  # pragma: no cover - optional dependency path
	wordnet = None


def _ensure_random(seed: Optional[int] = None) -> random.Random:
	rng = random.Random()
	if seed is not None:
		rng.seed(seed)
	return rng


@dataclass
class AugmentationConfig:
	"""Configuration for :class:`TextAugmenter`."""

	enable: bool = False
	apply_prob: float = 0.25
	dropout_prob: float = 0.0
	swap_prob: float = 0.0
	synonym_prob: float = 0.0
	noise_prob: float = 0.0
	max_ops_per_sample: int = 2
	reserved_tokens: Sequence[str] = ()
	min_tokens: int = 3
	apply_to_fields: Sequence[str] = ("text",)
	random_seed: Optional[int] = None

	@classmethod
	def from_mapping(cls, cfg: Mapping[str, object]) -> "AugmentationConfig":
		data = dict(cfg)
		reserved_val = data.get("reserved_tokens")
		fields_val = data.get("apply_to_fields")
		return cls(
			enable=bool(data.get("enable", data.get("enabled", False))),
			apply_prob=_coerce_float(data.get("apply_prob", data.get("prob", 0.25)), 0.25),
			dropout_prob=_coerce_float(data.get("dropout_prob", 0.0), 0.0),
			swap_prob=_coerce_float(data.get("swap_prob", 0.0), 0.0),
			synonym_prob=_coerce_float(data.get("synonym_prob", 0.0), 0.0),
			noise_prob=_coerce_float(data.get("noise_prob", data.get("random_noise_prob", 0.0)), 0.0),
			max_ops_per_sample=_coerce_int(data.get("max_ops_per_sample", 2), 2),
			reserved_tokens=_coerce_sequence(reserved_val, fallback=()),
			min_tokens=_coerce_int(data.get("min_tokens", 3), 3),
			apply_to_fields=_coerce_sequence(fields_val, fallback=("text",)),
			random_seed=_coerce_optional_int(data.get("random_seed")),
		)


class TextAugmenter:
	"""Apply stochastic text augmentations to dataset samples."""

	_NOISE_TOKENS: Sequence[str] = ("[MASK]", "<extra_id_0>", "<extra_id_1>", "<NOISE>")

	def __init__(self, config: AugmentationConfig):
		self.config = config
		self._rng = _ensure_random(config.random_seed)

	def __bool__(self) -> bool:  # pragma: no cover - convenience
		return self.config.enable

	def _should_apply(self) -> bool:
		return self._rng.random() < max(0.0, min(1.0, self.config.apply_prob))

	def __call__(self, sample: Mapping[str, object]) -> dict:
		if not self.config.enable or not self._should_apply():
			return dict(sample)

		augmented = dict(sample)
		applied_ops: List[str] = []
		for field in self.config.apply_to_fields:
			value = augmented.get(field)
			if not isinstance(value, str):
				continue
			new_text, ops = self._augment_text(value)
			augmented[field] = new_text
			applied_ops.extend(ops)

		augmented["__augmented__"] = True
		if applied_ops:
			augmented["augmentation_ops"] = applied_ops
		return augmented

	# ------------------------------------------------------------------
	def _augment_text(self, text: str) -> tuple[str, List[str]]:
		tokens = text.split()
		ops_applied: List[str] = []
		if len(tokens) < self.config.min_tokens:
			return text, ops_applied

		ops = [
			("dropout", self.config.dropout_prob, self._random_deletion),
			("swap", self.config.swap_prob, self._random_swap),
			("synonym", self.config.synonym_prob, self._synonym_replace),
			("noise", self.config.noise_prob, self._insert_noise),
		]
		self._rng.shuffle(ops)

		remaining = max(0, self.config.max_ops_per_sample)
		for name, prob, op in ops:
			if remaining == 0:
				break
			if prob <= 0 or self._rng.random() > prob:
				continue
			new_tokens = op(tokens)
			if new_tokens is None or new_tokens == tokens:
				continue
			tokens = new_tokens
			ops_applied.append(name)
			remaining -= 1

		return " ".join(tokens), ops_applied

	def _random_deletion(self, tokens: List[str]) -> Optional[List[str]]:
		keep = [
			t
			for t in tokens
			if (t in self.config.reserved_tokens) or (self._rng.random() > self.config.dropout_prob)
		]
		if len(keep) < self.config.min_tokens:
			return None
		return keep

	def _random_swap(self, tokens: List[str]) -> Optional[List[str]]:
		if len(tokens) < 2:
			return None
		idx1 = self._rng.randrange(len(tokens))
		idx2 = self._rng.randrange(len(tokens))
		if idx1 == idx2:
			return None
		swapped = list(tokens)
		swapped[idx1], swapped[idx2] = swapped[idx2], swapped[idx1]
		return swapped

	def _synonym_replace(self, tokens: List[str]) -> Optional[List[str]]:
		if wordnet is None:
			return None
		candidates = [i for i, t in enumerate(tokens) if self._is_replaceable(t)]
		if not candidates:
			return None
		idx = self._rng.choice(candidates)
		synonyms = self._fetch_synonyms(tokens[idx])
		if not synonyms:
			return None
		replaced = list(tokens)
		replaced[idx] = self._rng.choice(synonyms)
		return replaced

	def _insert_noise(self, tokens: List[str]) -> Optional[List[str]]:
		if not tokens:
			return None
		noise_token = self._rng.choice(self._NOISE_TOKENS)
		position = self._rng.randrange(len(tokens) + 1)
		noisy = list(tokens)
		noisy.insert(position, noise_token)
		return noisy

	# Helpers -----------------------------------------------------------------
	def _is_replaceable(self, token: str) -> bool:
		if token in self.config.reserved_tokens:
			return False
		if not token or token.isdigit():
			return False
		if token.startswith("<") and token.endswith(">"):
			return False
		if token.startswith("[") and token.endswith("]"):
			return False
		return True

	def _fetch_synonyms(self, token: str) -> List[str]:
		if wordnet is None:
			return []
		synsets = wordnet.synsets(token)
		lemmas = {lemma.name().replace("_", " ") for syn in synsets for lemma in syn.lemmas()}
		lemmas.discard(token)
		return [lemma for lemma in lemmas if self._is_replaceable(lemma)]


def build_text_augmenter(config: Mapping[str, object], *, split: str = "train") -> Optional[TextAugmenter]:
	"""Create a :class:`TextAugmenter` from a nested configuration."""

	preprocessing = config.get("preprocessing", {}) if isinstance(config, Mapping) else {}
	augment_cfg: dict = {}
	if isinstance(preprocessing, Mapping):
		augment_cfg = preprocessing.get("augment") or {}

	if not augment_cfg:
		return None

	aug_config = AugmentationConfig.from_mapping(augment_cfg)
	if not aug_config.enable:
		return None

	apply_on_eval = bool(augment_cfg.get("apply_on_eval", False))
	if split != "train" and not apply_on_eval:
		return None

	return TextAugmenter(aug_config)


__all__ = ["AugmentationConfig", "TextAugmenter", "build_text_augmenter"]
