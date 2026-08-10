"""Explicit application logging with context and secret redaction."""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path
from typing import Any

from loguru import logger as _loguru_logger


_SECRET_PATTERNS = (
    re.compile(r"(?i)(bearer\s+)[^\s,;]+"),
    re.compile(
        r"(?i)((?:api[_-]?key|password|secret|token)\s*[:=]\s*)[^\s,;]+"
    ),
)
_SENSITIVE_EXTRA = re.compile(
    r"(?i)(?:api[_-]?key|authorization|password|secret|token)"
)


def mask_sensitive(value: Any) -> Any:
    """Recursively remove credentials from messages and structured context."""

    if isinstance(value, dict):
        return {
            key: (
                "[REDACTED]"
                if _SENSITIVE_EXTRA.search(str(key))
                else mask_sensitive(item)
            )
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple, set, frozenset)):
        return [mask_sensitive(item) for item in value]
    if isinstance(value, str):
        result = value
        for pattern in _SECRET_PATTERNS:
            result = pattern.sub(lambda match: match.group(1) + "[REDACTED]", result)
        return result
    return value


def _redact_record(record: dict[str, Any]) -> None:
    record["message"] = mask_sensitive(str(record["message"]))
    record["extra"] = mask_sensitive(dict(record["extra"]))


logger = _loguru_logger.patch(_redact_record)


def _default_log_directory() -> Path:
    configured = os.environ.get("ATHENA_LOG_DIR")
    if configured:
        return Path(configured).expanduser()
    return Path(__file__).resolve().parents[1] / "logs"


def init_logger(
    *,
    log_directory: str | os.PathLike[str] | None = None,
    level: str = "INFO",
) -> Path:
    """Configure sinks explicitly and return the resolved log directory."""

    directory = Path(log_directory) if log_directory is not None else _default_log_directory()
    directory = directory.resolve()
    directory.mkdir(parents=True, exist_ok=True)

    _loguru_logger.remove()
    logger.add(
        sys.stderr,
        format=(
            "<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
            "<level>{message}</level> | {extra}"
        ),
        level=level,
        colorize=True,
    )
    logger.add(
        directory / "athena_execution.log",
        format=(
            "{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | "
            "{name}:{function}:{line} | {message} | {extra}"
        ),
        level=level,
        rotation="10 MB",
        retention="5 days",
        compression="zip",
        encoding="utf-8",
    )
    return directory


def bind_context(**identifiers: Any):
    """Bind only non-empty correlation identifiers to a logger instance."""

    return logger.bind(
        **{
            key: value
            for key, value in identifiers.items()
            if value is not None and str(value).strip()
        }
    )


__all__ = ["bind_context", "init_logger", "logger", "mask_sensitive"]
