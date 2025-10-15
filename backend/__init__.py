"""Backend package for the Lao tutor service."""

from .app.logging_utils import get_logger

logger = get_logger(__name__)
logger.debug("Backend package imported")

__all__ = ["logger"]
