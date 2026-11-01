"""Structured logging setup shared across the package.

Logging is configured once via :func:`get_logger`. Level can be overridden with
the ``LAGGED_DSA_LOG_LEVEL`` environment variable (e.g. ``DEBUG``).
"""

from __future__ import annotations

import logging
import os

_CONFIGURED = False


def _configure_root() -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return
    level_name = os.environ.get("LAGGED_DSA_LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    root = logging.getLogger("lagged_indexer_dsa")
    root.setLevel(level)
    root.addHandler(handler)
    root.propagate = False
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Return a namespaced logger under the ``lagged_indexer_dsa`` root."""
    _configure_root()
    return logging.getLogger(f"lagged_indexer_dsa.{name}")
