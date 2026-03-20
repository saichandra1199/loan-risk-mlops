"""Request middleware: logging, request ID injection, latency tracking."""

from __future__ import annotations

import time
import uuid

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from loan_risk.logging_setup import get_logger

logger = get_logger(__name__)


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Log every request with method, path, status, and latency.

    Also injects a unique X-Request-ID header so responses can be
    correlated with logs in downstream systems.
    """

    async def dispatch(self, request: Request, call_next) -> Response:
        request_id = request.headers.get("X-Request-ID") or f"req_{uuid.uuid4().hex[:16]}"
        start = time.perf_counter()

        # Attach request_id to structlog context for this request
        import structlog
        structlog.contextvars.bind_contextvars(request_id=request_id)

        try:
            response = await call_next(request)
        except Exception as exc:
            latency_ms = (time.perf_counter() - start) * 1000
            logger.error(
                "request_error",
                method=request.method,
                path=request.url.path,
                latency_ms=round(latency_ms, 2),
                error=str(exc),
            )
            raise
        finally:
            structlog.contextvars.unbind_contextvars("request_id")

        latency_ms = (time.perf_counter() - start) * 1000

        logger.info(
            "request_complete",
            method=request.method,
            path=request.url.path,
            status_code=response.status_code,
            latency_ms=round(latency_ms, 2),
        )

        response.headers["X-Request-ID"] = request_id
        return response
