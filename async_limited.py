import asyncio
import time
import threading


class RateLimiter:
    """Synchronous rate limiter to avoid overwhelming the API."""

    def __init__(self, min_interval: float):
        self.min_interval = min_interval
        self.last_request = 0
        self.lock = threading.Lock()

    def wait(self):
        with self.lock:
            elapsed = time.time() - self.last_request
            if elapsed < self.min_interval:
                time.sleep(self.min_interval - elapsed)
            self.last_request = time.time()


class AsyncRateLimiter:
    """Async rate limiter to avoid overwhelming the API."""

    def __init__(self, min_interval: float):
        self.min_interval = min_interval
        self.last_request = 0
        self.lock = asyncio.Lock()

    async def wait(self):
        async with self.lock:
            elapsed = time.time() - self.last_request
            if elapsed < self.min_interval:
                await asyncio.sleep(self.min_interval - elapsed)
            self.last_request = time.time()
