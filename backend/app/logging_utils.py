"""Logging helpers for the Lao tutor backend."""

from __future__ import annotations

import logging
import logging.config
from pathlib import Path
from typing import Optional

_LOGGING_CONFIGURED = False
_CURRENT_LOG_DIR: Optional[Path] = None


def configure_logging(
    log_dir: Optional[Path] = None,
    level: int = logging.INFO,
    force: bool = False,
) -> None:
    """Configure application-wide logging with both console and file handlers."""
    global _LOGGING_CONFIGURED, _CURRENT_LOG_DIR
    target_dir = (log_dir or Path("logs")).resolve()
    if _LOGGING_CONFIGURED and not force and _CURRENT_LOG_DIR == target_dir:
        return

    target_dir.mkdir(parents=True, exist_ok=True)
    log_file = target_dir / "app.log"

    logging_config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "standard": {
                "format": "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
            }
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "level": level,
                "formatter": "standard",
            },
            "file": {
                "class": "logging.handlers.RotatingFileHandler",
                "level": level,
                "formatter": "standard",
                "filename": str(log_file),
                "maxBytes": 5 * 1024 * 1024,
                "backupCount": 5,
                "encoding": "utf-8",
            },
        },
        "root": {
            "level": level,
            "handlers": ["console", "file"],
        },
    }

    logging.config.dictConfig(logging_config)
    _LOGGING_CONFIGURED = True
    _CURRENT_LOG_DIR = target_dir



def get_logger(name: str) -> logging.Logger:
    """Return a module-specific logger, configuring the logging system on first use."""
    if not logging.getLogger().handlers:
        configure_logging()
    return logging.getLogger(name)
