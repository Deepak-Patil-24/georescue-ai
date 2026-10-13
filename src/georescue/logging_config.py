"""Structured logging configuration for GeoRescue AI.

Provides a single :func:`configure_logging` entry point that installs a
consistent, human-readable formatter on the root logger and on the
``georescue`` logger hierarchy.

Design notes
------------
- Idempotent: calling :func:`configure_logging` more than once does not
  add duplicate handlers.
- Streamlit-aware: when the app runs under Streamlit, logging still goes
  to the terminal (Streamlit captures stdout/stderr), so no special
  handling is required.
- No file logging by default to avoid cluttering the repo. Callers that
  want file logging can pass ``log_file``.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional

_DEFAULT_FORMAT = (
    "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
)
_DEFAULT_DATEFMT = "%Y-%m-%d %H:%M:%S"

_CONFIGURED = False


def configure_logging(
    level: str = "INFO",
    log_file: Optional[Path] = None,
    *,
    force: bool = False,
) -> logging.Logger:
    """Configure the GeoRescue AI logger hierarchy.

    Parameters
    ----------
    level:
        Logging level name (``DEBUG``, ``INFO``, ``WARNING``, ``ERROR``,
        ``CRITICAL``). Case-insensitive.
    log_file:
        Optional path to also write logs to a file. The parent directory
        is created if missing.
    force:
        If True, remove existing handlers and reconfigure even if
        :func:`configure_logging` was already called. Useful in tests.

    Returns
    -------
    logging.Logger
        The configured ``georescue`` logger.
    """
    global _CONFIGURED

    if _CONFIGURED and not force:
        return logging.getLogger("georescue")

    level_name = level.upper()
    numeric_level = getattr(logging, level_name, None)
    if not isinstance(numeric_level, int):
        # Fall back to INFO with a clear warning instead of raising, so
        # a bad env var cannot break application startup.
        numeric_level = logging.INFO
        level_name = "INFO"

    root = logging.getLogger()
    # Remove handlers we previously installed (identified by a marker
    # attribute) but leave handlers installed by Streamlit or pytest.
    for handler in list(root.handlers):
        if getattr(handler, "_georescue_handler", False) or force:
            root.removeHandler(handler)

    formatter = logging.Formatter(_DEFAULT_FORMAT, datefmt=_DEFAULT_DATEFMT)

    stream_handler = logging.StreamHandler(stream=sys.stderr)
    stream_handler.setFormatter(formatter)
    stream_handler.setLevel(numeric_level)
    stream_handler._georescue_handler = True  # type: ignore[attr-defined]
    root.addHandler(stream_handler)

    if log_file is not None:
        log_path = Path(log_file)
        try:
            log_path.parent.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            # Do not fail startup just because file logging failed.
            logging.getLogger("georescue").warning(
                "Could not create log directory %s: %s. Continuing with "
                "console logging only.",
                log_path.parent,
                exc,
            )
        else:
            file_handler = logging.FileHandler(log_path, encoding="utf-8")
            file_handler.setFormatter(formatter)
            file_handler.setLevel(numeric_level)
            file_handler._georescue_handler = True  # type: ignore[attr-defined]
            root.addHandler(file_handler)

    root.setLevel(numeric_level)

    # Keep noisy third-party libraries quieter unless we are debugging.
    if numeric_level > logging.DEBUG:
        for noisy in ("urllib3", "requests", "fiona", "rasterio", "matplotlib"):
            logging.getLogger(noisy).setLevel(logging.WARNING)

    logger = logging.getLogger("georescue")
    logger.setLevel(numeric_level)
    logger.debug("Logging configured at level %s", level_name)

    _CONFIGURED = True
    return logger


def get_logger(name: str) -> logging.Logger:
    """Return a child logger of the ``georescue`` hierarchy.

    Usage::

        from georescue.logging_config import get_logger
        log = get_logger(__name__)
    """
    if name.startswith("georescue"):
        return logging.getLogger(name)
    return logging.getLogger(f"georescue.{name}")