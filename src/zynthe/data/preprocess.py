"""Basic preprocessing and text normalization utilities for JSONL datasets."""

from __future__ import annotations

import html
import re
import unicodedata
from dataclasses import dataclass
from typing import (
    Dict,
    Iterable,
    List,
    Mapping,
    MutableMapping,
    Optional,
    Sequence,
    Tuple,
    Literal,
    cast,
)

NormalizationForm = Literal["NFC", "NFD", "NFKC", "NFKD"]


HTML_TAG_RE = re.compile(r"<[^>]+>")
URL_RE = re.compile(r"(https?://\S+|www\.\S+)", re.IGNORECASE)
WHITESPACE_RE = re.compile(r"\s+")
CONTROL_RE = re.compile(r"[\u0000-\u001f\u007f]")


def _strip_emojis(text: str) -> str:
    return "".join(ch for ch in text if unicodedata.category(ch) != "So")


def _coerce_int(value: object, default: int) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        try:
            return int(float(value))
        except ValueError:
            return default
    return default


@dataclass
class PreprocessConfig:
    """Configuration flags controlling :func:`apply_preprocess_pipeline`."""

    lowercase: bool = False
    strip_html: bool = True
    remove_urls: bool = True
    replace_url_token: str = "<URL>"
    normalize_whitespace: bool = True
    remove_emojis: bool = False
    unicode_form: Optional[NormalizationForm] = "NFKC"
    min_characters: int = 1
    max_characters: Optional[int] = None
    drop_duplicates: bool = True
    dedupe_field: str = "text"
    clean_prompt_completion: bool = True
    extra_text_fields: Sequence[str] = ()


@dataclass
class PreprocessStatistics:
    total_samples: int = 0
    kept_samples: int = 0
    removed_empty: int = 0
    removed_short: int = 0
    truncated: int = 0
    removed_duplicates: int = 0

    def to_dict(self) -> Dict[str, int]:
        return {
            "total_samples": self.total_samples,
            "kept_samples": self.kept_samples,
            "removed_empty": self.removed_empty,
            "removed_short": self.removed_short,
            "truncated": self.truncated,
            "removed_duplicates": self.removed_duplicates,
        }


def _coerce_normalization_form(value: object) -> Optional[NormalizationForm]:
    if isinstance(value, str):
        candidate = value.upper()
        if candidate in {"NFC", "NFD", "NFKC", "NFKD"}:
            return cast(NormalizationForm, candidate)
    return None


def build_preprocess_config(
    config: Mapping[str, object], *, split: str = "train"
) -> PreprocessConfig:
    preprocessing = config.get("preprocessing", {}) if isinstance(config, Mapping) else {}
    basic_cfg: dict = {}
    if isinstance(preprocessing, Mapping):
        basic_cfg = preprocessing.get("basic") or {}

    extra_fields = basic_cfg.get("extra_text_fields", [])
    if isinstance(extra_fields, str):
        extra_fields = [extra_fields]
    extra_fields_tuple = tuple(str(field) for field in extra_fields)

    max_chars_key = "max_characters_eval" if split != "train" else "max_characters"

    max_chars_value = basic_cfg.get(max_chars_key)
    max_characters = _coerce_int(max_chars_value, 0) if max_chars_value is not None else None
    if max_characters is not None and max_characters <= 0:
        max_characters = None

    return PreprocessConfig(
        lowercase=bool(basic_cfg.get("lowercase", False)),
        strip_html=bool(basic_cfg.get("strip_html", True)),
        remove_urls=bool(basic_cfg.get("remove_urls", True)),
        replace_url_token=str(basic_cfg.get("replace_url_token", "<URL>")),
        normalize_whitespace=bool(basic_cfg.get("normalize_whitespace", True)),
        remove_emojis=bool(basic_cfg.get("remove_emojis", False)),
        unicode_form=_coerce_normalization_form(basic_cfg.get("unicode_form", "NFKC")),
        min_characters=_coerce_int(basic_cfg.get("min_characters", 1), 1),
        max_characters=max_characters,
        drop_duplicates=bool(basic_cfg.get("drop_duplicates", True)),
        dedupe_field=str(basic_cfg.get("dedupe_field", "text")),
        clean_prompt_completion=bool(basic_cfg.get("clean_prompt_completion", True)),
        extra_text_fields=extra_fields_tuple,
    )


def clean_text(text: object, cfg: PreprocessConfig) -> str:
    if not isinstance(text, str):
        text = str(text) if text is not None else ""

    if cfg.unicode_form:
        text = unicodedata.normalize(cfg.unicode_form, text)

    text = html.unescape(text)
    text = CONTROL_RE.sub(" ", text)

    if cfg.strip_html:
        text = HTML_TAG_RE.sub(" ", text)

    if cfg.remove_urls:
        text = URL_RE.sub(cfg.replace_url_token, text)

    if cfg.remove_emojis:
        text = _strip_emojis(text)

    if cfg.lowercase:
        text = text.lower()

    if cfg.normalize_whitespace:
        text = WHITESPACE_RE.sub(" ", text)

    return text.strip()


def _clean_field(record: MutableMapping[str, object], field: str, cfg: PreprocessConfig) -> None:
    if field in record and record[field] is not None:
        record[field] = clean_text(str(record[field]), cfg)


def preprocess_record(record: Mapping[str, object], cfg: PreprocessConfig) -> Dict[str, object]:
    cleaned: Dict[str, object] = dict(record)
    primary_text = cleaned.get("text")
    if primary_text is None:
        for candidate in (cleaned.get("prompt"), cleaned.get("sentence"), cleaned.get("input")):
            if isinstance(candidate, str) and candidate.strip():
                primary_text = candidate
                break
    normalized_primary = primary_text if isinstance(primary_text, str) else str(primary_text or "")
    cleaned["text"] = clean_text(normalized_primary, cfg)

    for field in cfg.extra_text_fields:
        _clean_field(cleaned, field, cfg)

    if cfg.clean_prompt_completion:
        _clean_field(cleaned, "prompt", cfg)
        _clean_field(cleaned, "completion", cfg)

    return cleaned


def apply_preprocess_pipeline(
    samples: Iterable[Mapping[str, object]],
    cfg: PreprocessConfig,
) -> Tuple[List[Dict[str, object]], PreprocessStatistics]:
    stats = PreprocessStatistics(total_samples=0)
    dedupe_set = set()
    processed: List[Dict[str, object]] = []

    for sample in samples:
        stats.total_samples += 1
        cleaned = preprocess_record(sample, cfg)
        text = str(cleaned.get("text", ""))
        if not text:
            stats.removed_empty += 1
            continue
        if cfg.max_characters and len(text) > cfg.max_characters:
            text = text[: cfg.max_characters]
            cleaned["text"] = text
            stats.truncated += 1
        if cfg.min_characters and len(text) < cfg.min_characters:
            stats.removed_short += 1
            continue

        dedupe_key = cleaned.get(cfg.dedupe_field, text) if cfg.drop_duplicates else None
        if cfg.drop_duplicates and dedupe_key in dedupe_set:
            stats.removed_duplicates += 1
            continue
        if cfg.drop_duplicates and dedupe_key is not None:
            dedupe_set.add(dedupe_key)

        processed.append(cleaned)

    stats.kept_samples = len(processed)
    return processed, stats


__all__ = [
    "PreprocessConfig",
    "PreprocessStatistics",
    "build_preprocess_config",
    "clean_text",
    "preprocess_record",
    "apply_preprocess_pipeline",
]
