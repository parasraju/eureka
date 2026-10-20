"""Global rate limiting for API calls.

Provides per-provider rate limiting with monotonic timestamps, concurrency
caps, and an optional acquisition timeout. The limiter never holds its lock
while sleeping and never re-enters ``acquire`` recursively.
"""

import asyncio
import time
from collections import defaultdict, deque
from typing import Any

DEFAULT_TIMEOUT = 120.0


class RateLimiter:
    """Rate limiter with per-second/per-minute quotas and a concurrency cap.

    ``acquire`` reserves a slot; ``release`` (or the async context manager)
    gives the concurrency slot back. Rate-limit timestamps are recorded at
    acquisition time and pruned lazily using ``time.monotonic``.
    """

    def __init__(
        self,
        calls_per_minute: int | None = None,
        calls_per_second: float | None = None,
        max_concurrency: int | None = None,
        timeout: float = DEFAULT_TIMEOUT,
    ):
        self.calls_per_minute = calls_per_minute
        self.calls_per_second = calls_per_second
        self.max_concurrency = max_concurrency
        self.timeout = timeout
        self._calls_min: defaultdict[str, deque[float]] = defaultdict(deque)
        self._calls_sec: defaultdict[str, deque[float]] = defaultdict(deque)
        self._semaphore: asyncio.Semaphore | None = (
            asyncio.Semaphore(max_concurrency) if max_concurrency is not None else None
        )
        self._lock = asyncio.Lock()

    async def __aenter__(self) -> "RateLimiter":
        await self.acquire("")
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        self.release("")

    async def acquire(self, provider_name: str) -> None:
        """Acquire a slot, waiting for quotas/concurrency as needed.

        Raises ``asyncio.TimeoutError`` if a slot cannot be acquired within
        ``self.timeout`` seconds.
        """
        await asyncio.wait_for(self._acquire(provider_name), timeout=self.timeout)

    async def _acquire(self, provider_name: str) -> None:
        semaphore = self._semaphore
        if semaphore is not None:
            await semaphore.acquire()
        try:
            while True:
                wait_for = await self._wait_time(provider_name)
                if wait_for <= 0:
                    break
                # Never sleep while holding the lock; re-check after waking.
                await asyncio.sleep(wait_for)
            await self._record(provider_name)
        except BaseException:
            if semaphore is not None:
                semaphore.release()
            raise

    async def _wait_time(self, provider_name: str) -> float:
        async with self._lock:
            now = time.monotonic()
            wait_for = 0.0

            if self.calls_per_second is not None:
                bucket = self._calls_sec[provider_name]
                while bucket and now - bucket[0] >= 1.0:
                    bucket.popleft()
                if self.calls_per_second > 0 and len(bucket) >= self.calls_per_second:
                    wait_for = max(wait_for, 1.0 - (now - bucket[0]) + 1e-6)

            if self.calls_per_minute is not None:
                bucket = self._calls_min[provider_name]
                while bucket and now - bucket[0] >= 60.0:
                    bucket.popleft()
                if self.calls_per_minute > 0 and len(bucket) >= self.calls_per_minute:
                    wait_for = max(wait_for, 60.0 - (now - bucket[0]) + 1e-6)

            return wait_for

    async def _record(self, provider_name: str) -> None:
        async with self._lock:
            now = time.monotonic()
            if self.calls_per_second is not None:
                self._calls_sec[provider_name].append(now)
            if self.calls_per_minute is not None:
                self._calls_min[provider_name].append(now)

    def release(self, provider_name: str) -> None:
        """Release a concurrency slot."""
        if self._semaphore is not None:
            self._semaphore.release()


_global_limiter: RateLimiter | None = None


def set_global_limiter(limiter: RateLimiter | None) -> None:
    """Set the global rate limiter for all providers."""
    global _global_limiter
    _global_limiter = limiter


def get_global_limiter() -> RateLimiter | None:
    """Get the current global rate limiter."""
    return _global_limiter


async def rate_limited_complete(
    provider,
    req,
    rate_limiter: RateLimiter | None = None,
) -> Any:
    """Call ``provider.complete(req)`` once, under an optional rate limiter."""
    limiter = rate_limiter or _global_limiter
    if limiter is None:
        return await provider.complete(req)
    async with limiter:
        return await provider.complete(req)


async def rate_limited_embed(
    provider,
    texts: list[str],
    model: str,
    rate_limiter: RateLimiter | None = None,
) -> Any:
    """Call ``provider.embed(texts, model)`` once, under an optional rate limiter."""
    limiter = rate_limiter or _global_limiter
    if limiter is None:
        return await provider.embed(texts, model)
    async with limiter:
        return await provider.embed(texts, model)
