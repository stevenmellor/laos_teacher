"""Request logging middleware."""
from __future__ import annotations

import time

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.responses import Response

from ..logging_utils import get_logger

logger = get_logger(__name__)
logger.debug("Request logging middleware module loaded")


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Log each HTTP request lifecycle with timing metadata."""

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        start = time.perf_counter()
        client_host = request.client.host if request.client else "unknown"
        logger.info(
            "HTTP request started",
            extra={
                "method": request.method,
                "path": request.url.path,
                "client": client_host,
            },
        )
        try:
            response = await call_next(request)
        except Exception:
            duration = time.perf_counter() - start
            logger.exception(
                "HTTP request failed",
                extra={
                    "method": request.method,
                    "path": request.url.path,
                    "client": client_host,
                    "duration_ms": round(duration * 1000, 2),
                },
            )
            raise

        duration = time.perf_counter() - start
        logger.info(
            "HTTP request completed",
            extra={
                "method": request.method,
                "path": request.url.path,
                "client": client_host,
                "status_code": response.status_code,
                "duration_ms": round(duration * 1000, 2),
            },
        )
        return response


__all__ = ["RequestLoggingMiddleware"]
