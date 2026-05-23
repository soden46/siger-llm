from __future__ import annotations

import os
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Callable

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response


@dataclass(frozen=True)
class APISecuritySettings:
    max_body_bytes: int = 1_000_000
    rate_limit_requests: int = 120
    rate_limit_window_seconds: int = 60
    trusted_proxy_header: str = "x-forwarded-for"
    require_api_key: bool = False
    exempt_paths: tuple[str, ...] = ("/health", "/v1/status")

    @classmethod
    def from_env(cls) -> "APISecuritySettings":
        return cls(
            max_body_bytes=_env_int("SIGER_MAX_BODY_BYTES", 1_000_000),
            rate_limit_requests=_env_int("SIGER_RATE_LIMIT_REQUESTS", 120),
            rate_limit_window_seconds=_env_int("SIGER_RATE_LIMIT_WINDOW_SECONDS", 60),
            trusted_proxy_header=os.environ.get("SIGER_TRUSTED_PROXY_HEADER", "x-forwarded-for").lower(),
            require_api_key=_env_bool("SIGER_REQUIRE_API_KEY", False),
        )


class InMemoryRateLimiter:
    """Small per-process sliding-window limiter.

    Use a reverse proxy or managed WAF for real DDoS protection. This limiter
    protects local CPU/GPU resources from accidental floods and simple abuse.
    """

    def __init__(self, limit: int, window_seconds: int):
        self.limit = max(1, limit)
        self.window_seconds = max(1, window_seconds)
        self.events: dict[str, deque[float]] = defaultdict(deque)

    def allow(self, key: str) -> tuple[bool, int]:
        now = time.monotonic()
        bucket = self.events[key]
        cutoff = now - self.window_seconds
        while bucket and bucket[0] < cutoff:
            bucket.popleft()
        if len(bucket) >= self.limit:
            return False, 0
        bucket.append(now)
        return True, self.limit - len(bucket)


class APIProtectionMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, settings: APISecuritySettings | None = None):
        super().__init__(app)
        self.settings = settings or APISecuritySettings.from_env()
        self.rate_limiter = InMemoryRateLimiter(
            self.settings.rate_limit_requests,
            self.settings.rate_limit_window_seconds,
        )

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        if not self._is_exempt(request):
            content_length = request.headers.get("content-length")
            if content_length:
                try:
                    if int(content_length) > self.settings.max_body_bytes:
                        return self._json_error(413, "Request body too large")
                except ValueError:
                    return self._json_error(400, "Invalid Content-Length")

            key = self._rate_limit_key(request)
            allowed, remaining = self.rate_limiter.allow(key)
            if not allowed:
                response = self._json_error(429, "Rate limit exceeded")
                response.headers["Retry-After"] = str(self.settings.rate_limit_window_seconds)
                return response

            response = await call_next(request)
            response.headers["X-RateLimit-Limit"] = str(self.settings.rate_limit_requests)
            response.headers["X-RateLimit-Remaining"] = str(remaining)
        else:
            response = await call_next(request)

        self._set_security_headers(response)
        return response

    def _is_exempt(self, request: Request) -> bool:
        return request.url.path in self.settings.exempt_paths

    def _rate_limit_key(self, request: Request) -> str:
        api_key = request.headers.get("x-siger-api-key")
        if api_key:
            return f"key:{api_key[:12]}"

        forwarded = request.headers.get(self.settings.trusted_proxy_header)
        if forwarded:
            return f"ip:{forwarded.split(',')[0].strip()}"

        host = request.client.host if request.client else "unknown"
        return f"ip:{host}"

    def _set_security_headers(self, response: Response) -> None:
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        response.headers.setdefault("X-Frame-Options", "DENY")
        response.headers.setdefault("Referrer-Policy", "no-referrer")
        response.headers.setdefault("Permissions-Policy", "camera=(), microphone=(), geolocation=()")
        response.headers.setdefault("Cache-Control", "no-store")

    def _json_error(self, status_code: int, detail: str) -> JSONResponse:
        response = JSONResponse({"detail": detail}, status_code=status_code)
        self._set_security_headers(response)
        return response


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if not value:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}
