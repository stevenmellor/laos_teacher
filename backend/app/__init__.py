"""Backend application package for the Lao language tutor."""

from .logging_utils import get_logger

logger = get_logger(__name__)
logger.debug("Backend app package initialised")

__all__ = [
    "config",
    "logging_utils",
    "main",
]
