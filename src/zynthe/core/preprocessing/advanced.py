"""Advanced preprocessing pipeline inspired by QLoRA fine-tuning.

Adds optional steps:
 - Basic cleaning (blank / ultra-short filtering)
 - Deduplication (exact text or prompt+completion pairs)
 - Instruction/Response templating for conversational / completion style data
 - Special token management (PAD/EOS/custom additional tokens)
 - Token length profiling (lightweight statistics)
 - Truncation policy (head or tail)
 - Optional curriculum ordering (short -> long) - disabled by default

Configuration (under config['preprocessing']['advanced']):
  enable: bool
  min_completion_len: int (default 2)
  deduplicate: bool
  instruction_prefix: str
  response_prefix: str
  add_eos: bool
  eos_token: Optional[str]
  add_pad_token: bool
  custom_special_tokens: List[str]
  max_seq_len: int
  trunc_strategy: 'head' | 'tail'
  curriculum: bool
  profile_lengths: bool

This pipeline operates on raw sample dicts BEFORE adapter normalization.
Safe fallbacks: if keys 'prompt'/'completion' absent, falls back to existing 'text'.
"""
from __future__ import annotations
from typing import List, Dict, Any
import logging

LOG = logging.getLogger(__name__)


def _is_valid(raw: Dict[str, Any], min_len: int) -> bool:
    # Prefer prompt/completion if present, else text
    prompt = (raw.get('prompt') or raw.get('text') or '').strip()
    completion = (raw.get('completion') or '').strip()
    # If completion field exists, enforce length constraint
    if raw.get('completion') is not None and len(completion) < min_len:
        return False
    if not prompt and not completion:
        return False
    return True


def _hash(raw: Dict[str, Any]) -> str:
    if raw.get('prompt') is not None and raw.get('completion') is not None:
        return raw['prompt'].strip() + '\n' + raw['completion'].strip()
    return (raw.get('text') or '').strip()


def _apply_template(raw: Dict[str, Any], cfg: Dict[str, Any]) -> str:
    instruction_prefix = cfg.get('instruction_prefix', '### Instruction:\n')
    response_prefix = cfg.get('response_prefix', '\n### Response:\n')
    eos_token = cfg.get('eos_token') or '<|endoftext|>'
    add_eos = cfg.get('add_eos', True)

    if raw.get('prompt') is not None and raw.get('completion') is not None:
        prompt = raw['prompt'].strip()
        completion = raw['completion'].strip()
        text = f"{instruction_prefix}{prompt}{response_prefix}{completion}" + (eos_token if add_eos else '')
        return text
    # Fallback single text
    base = (raw.get('text') or '').strip()
    if add_eos:
        base = base + eos_token
    return base


def _truncate(tokenizer, text: str, max_seq_len: int, strategy: str) -> str:
    """Truncate tokenized text to max_seq_len with stable EOS handling.

    - Preserves existing EOS if already present to avoid duplicates
    - Appends EOS at the end (not the beginning) when needed
    - Uses decode with skip_special_tokens=False to keep special tokens
    """
    if max_seq_len <= 0:
        return text
    enc = tokenizer(text, add_special_tokens=False)
    ids = enc['input_ids']
    if len(ids) <= max_seq_len:
        return text

    eos_id = getattr(tokenizer, 'eos_token_id', None)

    # Helper: ensure a single EOS at the end if eos_id is available
    def ensure_trailing_eos(token_ids):
        if eos_id is None:
            return token_ids
        if len(token_ids) == 0 or token_ids[-1] != eos_id:
            token_ids = token_ids[:-1] + [eos_id] if len(token_ids) >= 1 else [eos_id]
        return token_ids

    if strategy == 'tail':
        # Keep the tail portion. Reserve space for EOS if needed.
        kept = ids[-max_seq_len:]
        if eos_id is not None:
            kept = ensure_trailing_eos(kept)
    else:  # head
        kept = ids[:max_seq_len]
        if eos_id is not None:
            kept = ensure_trailing_eos(kept)

    return tokenizer.decode(kept, skip_special_tokens=False)


def run_advanced_pipeline(raw_samples: List[Dict[str, Any]], tokenizer, config: Dict[str, Any], role: str = 'train') -> List[Dict[str, Any]]:
    adv_cfg = config.get('preprocessing', {}).get('advanced', {})
    if not adv_cfg.get('enable', False):
        return raw_samples  # no-op

    LOG.info("[ADV-PRE] Starting advanced preprocessing pipeline (%s) on %d samples", role, len(raw_samples))

    min_len = adv_cfg.get('min_completion_len', 2)
    deduplicate = adv_cfg.get('deduplicate', True)
    max_seq_len = adv_cfg.get('max_seq_len', config.get('model', {}).get('max_length', 128))
    trunc_strategy = adv_cfg.get('trunc_strategy', 'head')
    curriculum = adv_cfg.get('curriculum', False)
    profile_lengths = adv_cfg.get('profile_lengths', True)

    # ----- cleaning -----
    cleaned = [r for r in raw_samples if _is_valid(r, min_len)]
    LOG.info("[ADV-PRE] Cleaned invalid/short samples: %d -> %d", len(raw_samples), len(cleaned))

    # ----- deduplication -----
    if deduplicate:
        seen = set()
        unique = []
        for r in cleaned:
            h = _hash(r)
            if h in seen:
                continue
            seen.add(h)
            unique.append(r)
        LOG.info("[ADV-PRE] Deduplicated: %d -> %d", len(cleaned), len(unique))
    else:
        unique = cleaned

    # ----- Special tokens mgmt -----
    if adv_cfg.get('add_pad_token', True) and tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '<pad>'})
        LOG.info("[ADV-PRE] Added <pad> token to tokenizer; new vocab size=%d", len(tokenizer))
    custom_tokens = adv_cfg.get('custom_special_tokens', [])
    if custom_tokens:
        added = tokenizer.add_special_tokens({'additional_special_tokens': custom_tokens})
        if added:
            LOG.info("[ADV-PRE] Added %d custom special tokens", added)

    # ----- Formatting & truncation -----
    processed = []
    for r in unique:
        text = _apply_template(r, adv_cfg)
        text = _truncate(tokenizer, text, max_seq_len, trunc_strategy)
        # Replace/augment sample with unified 'text'
        out = dict(r)
        out['text'] = text
        processed.append(out)

    # ----- Length profiling -----
    if profile_lengths and processed:
        lengths = []
        for r in processed[:1000]:  # cap profiling cost
            ids = tokenizer(r['text']).input_ids
            lengths.append(len(ids))
        lengths_sorted = sorted(lengths)
        mean_len = sum(lengths) / len(lengths)
        p95 = lengths_sorted[int(0.95 * len(lengths_sorted))]
        p99 = lengths_sorted[int(0.99 * len(lengths_sorted))]
        over_max = sum(1 for length in lengths if length > max_seq_len)
        LOG.info("[ADV-PRE] Length stats (%s) count=%d mean=%.1f max=%d p95=%d p99=%d >max=%d", role, len(lengths), mean_len, max(lengths), p95, p99, over_max)

    # ----- Optional curriculum -----
    if curriculum:
        processed.sort(key=lambda x: len(tokenizer(x['text']).input_ids))
        LOG.info("[ADV-PRE] Curriculum ordering applied (short->long)")

    LOG.info("[ADV-PRE] Finished advanced preprocessing pipeline; final samples=%d", len(processed))
    return processed


__all__ = ["run_advanced_pipeline"]
