"""Utilities for tracking and aggregating scalar metrics during training."""

from __future__ import annotations

import math
import statistics
import time
from collections import deque
from collections.abc import Iterable as IterableABC
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple, Union


Number = Union[int, float]
ReducerSpec = Union[str, Sequence[str], Mapping[str, Sequence[str]]]


def safe_divide(numerator: Number, denominator: Number, *, default: float = 0.0) -> float:
    """Divide two numbers without raising on zero denominators."""

    if denominator == 0:
        return default
    return float(numerator) / float(denominator)


def _ensure_iterable(obj: object) -> Iterable[Any]:
    if isinstance(obj, IterableABC) and not isinstance(obj, (str, bytes)):
        return obj
    return [obj]


def _coerce_step(value: object) -> Optional[int]:
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return None


def _coerce_float(value: object) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


@dataclass
class MetricHistory:
    """Fixed-length buffer that stores metric values and optional steps."""

    maxlen: Optional[int] = None
    values: Deque[float] = field(default_factory=deque)
    steps: Deque[Optional[int]] = field(default_factory=deque)

    def add(self, value: Number, step: Optional[int] = None) -> None:
        self.values.append(float(value))
        self.steps.append(step)
        if self.maxlen and len(self.values) > self.maxlen:
            self.values.popleft()
            self.steps.popleft()

    @property
    def count(self) -> int:
        return len(self.values)

    def mean(self) -> Optional[float]:
        if not self.values:
            return None
        return statistics.fmean(self.values)

    def last(self) -> Optional[float]:
        return self.values[-1] if self.values else None

    def max(self) -> Optional[float]:
        return max(self.values) if self.values else None

    def min(self) -> Optional[float]:
        return min(self.values) if self.values else None

    def sum(self) -> Optional[float]:
        return sum(self.values) if self.values else None

    def to_list(self) -> List[Tuple[Optional[int], float]]:
        return list(zip(self.steps, self.values))


class MetricTracker:
    """Tracks scalar metrics with configurable aggregation helpers."""

    _REDUCER_ALIASES = {
        "avg": "mean",
        "average": "mean",
        "latest": "last",
    }

    def __init__(
        self,
        *,
        window: Optional[int] = None,
        default_reducers: Sequence[str] = ("mean", "last"),
    ) -> None:
        self.window = window
        self.default_reducers = tuple(default_reducers)
        self._metrics: Dict[str, MetricHistory] = {}

    def _history(self, name: str) -> MetricHistory:
        history = self._metrics.get(name)
        if history is None:
            history = MetricHistory(maxlen=self.window)
            self._metrics[name] = history
        return history

    def update(self, metrics: Mapping[str, Number], *, step: Optional[int] = None) -> None:
        for key, value in metrics.items():
            if value is None:
                continue
            self._history(key).add(value, step)

    def record(self, name: str, value: Number, *, step: Optional[int] = None) -> None:
        self.update({name: value}, step=step)

    def get(self, name: str, reducer: str = "mean") -> Optional[float]:
        history = self._metrics.get(name)
        if not history:
            return None
        reducer_fn = self._get_reducer(reducer)
        return reducer_fn(history)

    def summary(
        self,
        reducers: Optional[ReducerSpec] = None,
        *,
        prefix: str = "",
        include: Optional[Iterable[str]] = None,
    ) -> Dict[str, float]:
        metrics = list(include) if include else list(self._metrics.keys())
        reducer_map = self._normalise_reducers(reducers, metrics)

        summary: Dict[str, float] = {}
        for name in metrics:
            history = self._metrics.get(name)
            if not history or history.count == 0:
                continue
            for reducer in reducer_map.get(name, self.default_reducers):
                reducer_fn = self._get_reducer(reducer)
                value = reducer_fn(history)
                if value is None:
                    continue
                if isinstance(value, float) and not math.isfinite(value):
                    continue
                summary[f"{prefix}{name}_{reducer}"] = float(value)
        return summary

    def history(self, name: str) -> List[Tuple[Optional[int], float]]:
        history = self._metrics.get(name)
        return history.to_list() if history else []

    def reset(self, *names: str) -> None:
        if names:
            for name in names:
                self._metrics.pop(name, None)
        else:
            self._metrics.clear()

    def state_dict(self) -> Dict[str, Dict[str, object]]:
        return {
            name: {
                "maxlen": history.maxlen,
                "steps": list(history.steps),
                "values": list(history.values),
            }
            for name, history in self._metrics.items()
        }

    def load_state_dict(self, state: Mapping[str, Mapping[str, object]]) -> None:
        self._metrics.clear()
        for name, payload in state.items():
            maxlen = payload.get("maxlen")
            steps_iter = list(_ensure_iterable(payload.get("steps", [])))
            values_iter = list(_ensure_iterable(payload.get("values", [])))
            history = MetricHistory(maxlen=maxlen if isinstance(maxlen, int) else None)
            for step_obj, value_obj in zip(steps_iter, values_iter):
                value = _coerce_float(value_obj)
                if value is None:
                    continue
                history.add(value, _coerce_step(step_obj))
            self._metrics[name] = history

    def _normalise_reducers(
        self,
        reducers: Optional[ReducerSpec],
        metrics: Sequence[str],
    ) -> Dict[str, Tuple[str, ...]]:
        if reducers is None:
            return {name: tuple(self.default_reducers) for name in metrics}
        if isinstance(reducers, (str, bytes)):
            reducer_list = (reducers,)
            return {name: reducer_list for name in metrics}
        if isinstance(reducers, Mapping):
            normalised: Dict[str, Tuple[str, ...]] = {}
            for name in metrics:
                spec = reducers.get(name)
                if spec is None:
                    normalised[name] = tuple(self.default_reducers)
                elif isinstance(spec, (str, bytes)):
                    normalised[name] = (spec,)
                else:
                    normalised[name] = tuple(spec)
            return normalised
        return {name: tuple(reducers) for name in metrics}

    def _get_reducer(self, name: str):
        canonical = self._REDUCER_ALIASES.get(name, name)
        if canonical == "mean":
            return MetricHistory.mean
        if canonical == "last":
            return MetricHistory.last
        if canonical == "max":
            return MetricHistory.max
        if canonical == "min":
            return MetricHistory.min
        if canonical == "sum":
            return MetricHistory.sum
        if canonical == "count":
            return lambda history: float(history.count)
        raise ValueError(f"Unknown reducer: {name}")


def merge_metric_summaries(
    *summaries: Mapping[str, Number],
    reducer: str = "last",
) -> Dict[str, float]:
    """Merge multiple summary dictionaries into one."""

    bucket: Dict[str, List[float]] = {}
    for summary in summaries:
        for key, value in summary.items():
            if value is None:
                continue
            bucket.setdefault(key, []).append(float(value))

    result: Dict[str, float] = {}
    reducer = reducer.lower()
    for key, values in bucket.items():
        if reducer == "last":
            result[key] = values[-1]
        elif reducer in {"mean", "avg", "average"}:
            result[key] = statistics.fmean(values)
        elif reducer == "max":
            result[key] = max(values)
        elif reducer == "min":
            result[key] = min(values)
        elif reducer == "sum":
            result[key] = sum(values)
        else:
            raise ValueError(f"Unsupported reducer for merge: {reducer}")
    return result


def _convert_duration(seconds: float, unit: str) -> float:
    unit = unit.lower()
    if unit in {"s", "sec", "secs", "second", "seconds"}:
        return seconds
    if unit in {"ms", "millisecond", "milliseconds"}:
        return seconds * 1_000.0
    if unit in {"us", "microsecond", "microseconds"}:
        return seconds * 1_000_000.0
    raise ValueError(f"Unsupported unit: {unit}")


@contextmanager
def record_time(
    tracker: MetricTracker,
    name: str,
    *,
    step: Optional[int] = None,
    unit: str = "s",
    record_on_error: bool = True,
) -> Iterator[None]:
    """Context manager that records execution time in the provided tracker."""

    start = time.perf_counter()
    try:
        yield
    except Exception:
        if record_on_error:
            duration = _convert_duration(time.perf_counter() - start, unit)
            tracker.record(name, duration, step=step)
        raise
    else:
        duration = _convert_duration(time.perf_counter() - start, unit)
        tracker.record(name, duration, step=step)
