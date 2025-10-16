"""Middleware utilities for the Lao tutor backend."""

from ..logging_utils import get_logger

logger = get_logger(__name__)
logger.debug("Middleware package initialised")

__all__ = [
    "request_logging",
]
