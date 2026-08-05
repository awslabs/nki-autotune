"""Console logging shared by fixed developer workload drivers."""

from __future__ import annotations

import logging


def configure_console_logging() -> None:
    """Show timestamped workflow progress when a driver owns the process."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s", datefmt="%H:%M:%S")


__all__ = ["configure_console_logging"]
