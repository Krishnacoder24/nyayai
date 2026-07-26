"""
Timing middleware. adds an X-Process-Time header (seconds, as a plain
float string) to every response - a cheap way to spot a slow endpoint
without reaching for a real profiler first.
"""

import time

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request


class TimingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        start = time.perf_counter()
        response = await call_next(request)
        elapsed = time.perf_counter() - start
        response.headers["X-Process-Time"] = f"{elapsed:.4f}"
        return response