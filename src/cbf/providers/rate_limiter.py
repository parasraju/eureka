"""Global rate limiting for API calls.

Provides per-provider rate limiting with Retry-After support,
concurrency caps, and integration with tenacity-based retries.
"""

import asyncio
import time
from collections import defaultdict
from typing import DefaultDict, Optional


class RateLimiter:
    """Token bucket rate limiter per provider."""

    def __init__(self, calls_per_minute: int | None = None, calls_per_second: float | None = None, max_concurrency: int | None = None):
        self.calls_per_minute = calls_per_minute
        self.calls_per_second = calls_per_second
        self.max_concurrency = max_concurrency
        self._calls_min: DefaultDict[str, list[float]] = defaultdict(list)
        self._calls_sec: DefaultDict[str, list[float]] = defaultdict(list)
        self._semaphore: asyncio.Semaphore | None = None
        self._lock = asyncio.Lock()

    async def acquire(self, provider_name: str) -> None:
        """Acquire a slot, blocking if rate limits would be exceeded."""
        async with self._lock:
            if self.max_concurrency is not None:
                if self._semaphore is None:
                    self._semaphore = asyncio.Semaphore(self.max_concurrency)
                await self._semaphore.acquire()

            now = time.time()

            # Enforce per-second limit
            if self.calls_per_second is not None:
                bucket = self._calls_sec[provider_name]
                # Remove timestamps older than 1 second
                self._calls_sec[provider_name] = [t for t in bucket if now - t < 1.0]
                if len(self._calls_sec[provider_name]) >= self.calls_per_second * 1.0:
                    # Wait until a slot opens
                    await asyncio.sleep(0.01)
                    # Re-check after waiting
                    await self.acquire(provider_name)
                    return

            # Enforce per-minute limit
            if self.calls_per_minute is not None:
                bucket = self._calls_min[provider_name]
                # Remove timestamps older than 60 seconds
                self._calls_min[provider_name] = [t for t in bucket if now - t < 60.0]
                if len(self._calls_min[provider_name]) >= self.calls_per_minute:
                    # Calculate wait time
                    oldest = min(self._calls_min[provider_name])
                    wait_time = 60.0 - (now - oldest) + 0.01
                    if wait_time > 0:
                        await asyncio.sleep(wait_time)
                    # Re-check after waiting
                    await self.acquire(provider_name)
                    return

    def release(self, provider_name: str) -> None:
        """Release a concurrency slot."""
        if self._semaphore is not None:
            self._semaphore.release()


# Global rate limiter instance - can be configured per provider
_global_limiter: RateLimiter | None = None


def set_global_limiter(limiter: RateLimiter) -> None:
    """Set the global rate limiter for all providers."""
    global _global_limiter
    _global_limiter = limiter


def get_global_limiter() -> RateLimiter | None:
    """Get the current global rate limiter."""
    return _global_limiter


async def rate_limited_complete(
    provider,
    req,
    rate_limiter: Optional[RateLimiter] = None,
) -> any:
    """Wrap provider.complete with rate limiting.

    Should be called instead of provider.complete() directly when
    rate limiting is configured.
    """
    limiter = rate_limiter or _global_limiter
    if limiter is None:
        return await provider.complete(req)

    async with limiter:
        return await provider.complete(req)


async def rate_limited_embed(
    provider,
    texts: list[str],
    model: str,
    rate_limiter: Optional[RateLimiter] = None,
) -> any:
    """Wrap provider.embed with rate limiting."""
    limiter = rate_limiter or _global_limiter
    if limiter is None:
        return await provider.embed(texts, model)

    async with limiter:
        return await provider.embed(texts, model)