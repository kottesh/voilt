from __future__ import annotations

import json
from typing import AsyncGenerator
from contextlib import asynccontextmanager

import redis.asyncio as aioredis

from server.core.config import ServerSettings

settings = ServerSettings()

QUEUE_KEY = "violation_queue"

_redis: aioredis.Redis | None = None


async def get_redis() -> aioredis.Redis:
    global _redis
    if _redis is None:
        _redis = aioredis.from_url(
            settings.REDIS_URL,
            encoding="utf-8",
            decode_responses=True,
        )
    return _redis


async def enqueue(job: dict) -> int:
    """Push a job dict to the left of the queue. Returns queue length."""
    r = await get_redis()
    return await r.lpush(QUEUE_KEY, json.dumps(job))


async def dequeue(timeout: int = 5) -> dict | None:
    """
    Blocking right-pop with timeout (seconds).
    Returns None if queue is empty after timeout.
    """
    r = await get_redis()
    result = await r.brpop(QUEUE_KEY, timeout=timeout)
    if result is None:
        return None
    _, raw = result
    return json.loads(raw)


async def queue_length() -> int:
    r = await get_redis()
    return await r.llen(QUEUE_KEY)