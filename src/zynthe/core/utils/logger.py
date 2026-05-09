"""Utility helpers for consistent logging configuration across the project.

The goal is to provide a single entry-point for configuring loggers so that
all command-line tools, background workers, and notebooks emit logs using the
same formatting. The helpers below build on ``logging`` but add quality-of-life
features such as:

* opt-in stdout handlers with a consistent formatter
* optional file logging with automatic directory creation
* lightweight contextual logging via ``LoggerAdapter``
* a context manager that times code blocks and logs the duration

The module deliberately avoids any heavy third-party dependencies so it can be
used in minimal environments such as unit tests.
"""

from __future__ import annotations

import logging
import os
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, Mapping, MutableMapping, Optional, Union, cast


LevelType = Union[int, str]

DEFAULT_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
DEFAULT_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def _resolve_level(level: Optional[LevelType]) -> int:
    """Return a numeric logging level from string/int input."""

    if level is None:
        env_level = os.getenv("ZYNTHÉ_LOG_LEVEL") or os.getenv("LOG_LEVEL")
        level = env_level if env_level else logging.INFO

    if isinstance(level, int):
        return level

    candidate = level.upper()
    if candidate.isdigit():
        return int(candidate)

    resolved = logging.getLevelName(candidate)
    if isinstance(resolved, int):
        return resolved

    raise ValueError(f"Unsupported log level: {level}")


def _is_stream_handler_to_stdout(handler: logging.Handler) -> bool:
    return isinstance(handler, logging.StreamHandler) and getattr(handler, "stream", None) is sys.stdout


@dataclass
class LoggerConfig:
    """Configuration options used by :func:`configure_logger`."""

    name: str = "zynthe"
    level: Optional[LevelType] = None
    console: bool = True
    propagate: bool = False
    fmt: str = DEFAULT_LOG_FORMAT
    datefmt: str = DEFAULT_DATE_FORMAT
    file_path: Optional[Union[str, Path]] = None
    extra_handlers: Iterable[logging.Handler] = ()
    clear_handlers: bool = False


_CONFIGURED: Dict[str, logging.Logger] = {}


def configure_logger(config: Optional[LoggerConfig] = None, **overrides: object) -> logging.Logger:
    """Configure and return a logger according to ``LoggerConfig``."""

    base_config = config or LoggerConfig()
    if overrides:
        base_config = replace(base_config, **overrides)  # type: ignore[arg-type]

    logger = logging.getLogger(base_config.name)
    logger.setLevel(_resolve_level(base_config.level))
    logger.propagate = base_config.propagate

    if base_config.clear_handlers:
        logger.handlers.clear()

    formatter = logging.Formatter(base_config.fmt, base_config.datefmt)

    if base_config.console and not any(_is_stream_handler_to_stdout(h) for h in logger.handlers):
        stdout_handler = logging.StreamHandler(sys.stdout)
        stdout_handler.setFormatter(formatter)
        stdout_handler.setLevel(logger.level)
        logger.addHandler(stdout_handler)

    if base_config.file_path:
        file_path = Path(base_config.file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)

        already_configured = any(
            isinstance(handler, logging.FileHandler) and Path(handler.baseFilename) == file_path
            for handler in logger.handlers
        )
        if not already_configured:
            file_handler = logging.FileHandler(file_path, encoding="utf-8")
            file_handler.setFormatter(formatter)
            file_handler.setLevel(logger.level)
            logger.addHandler(file_handler)

    for handler in base_config.extra_handlers:
        logger.addHandler(handler)

    _CONFIGURED[base_config.name] = logger
    return logger


class ContextLogger(logging.LoggerAdapter):
    """Logger adapter that appends structured contextual data to messages."""

    def __init__(self, logger: logging.Logger, context: Optional[Mapping[str, Any]] = None):
        super().__init__(logger, dict(context or {}))

    def bind(self, **context: Any) -> "ContextLogger":
        base_extra = cast(Mapping[str, Any], self.extra or {})
        current: Dict[str, Any] = dict(base_extra)
        current.update(context)
        return ContextLogger(self.logger, current)

    def process(self, msg: Any, kwargs: MutableMapping[str, Any]) -> tuple[Any, MutableMapping[str, Any]]:
        if self.extra:
            context_str = " ".join(f"{key}={value}" for key, value in sorted(self.extra.items()))
            msg = f"{msg} | {context_str}"
        return msg, kwargs


def get_logger(
    name: str = "zynthe",
    *,
    level: Optional[LevelType] = None,
    context: Optional[Mapping[str, Any]] = None,
    **overrides: object,
) -> Union[logging.Logger, ContextLogger]:
    """Return a configured logger, optionally bound with context."""

    logger = configure_logger(LoggerConfig(name=name, level=level), **overrides)
    if context:
        return ContextLogger(logger, context)
    return logger


def with_context(logger: logging.Logger, **context: Any) -> ContextLogger:
    """Return a ``ContextLogger`` bound with additional context."""

    if isinstance(logger, ContextLogger):
        return logger.bind(**context)
    return ContextLogger(logger, context)


@contextmanager
def log_duration(
    logger: logging.Logger,
    message: str,
    *,
    level: int = logging.INFO,
    success_message: Optional[str] = None,
    error_message: Optional[str] = None,
) -> Iterator[None]:
    """Log how long a code block takes."""

    start = time.perf_counter()
    logger.log(level, message)
    try:
        yield
    except Exception:
        elapsed = time.perf_counter() - start
        text = error_message or f"{message} failed after {elapsed:.3f}s"
        logger.exception(text)
        raise
    else:
        elapsed = time.perf_counter() - start
        text = success_message or f"{message} completed in {elapsed:.3f}s"
        logger.log(level, text)

