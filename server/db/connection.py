from contextlib import asynccontextmanager
from typing import AsyncGenerator

import asyncpg
from asyncpg import Connection, Pool

from server.core.config import get_settings

settings = get_settings()

_pool: Pool | None = None


async def get_pool() -> Pool:
    global _pool
    if _pool is None:
        _pool = await asyncpg.create_pool(
            dsn=settings.DATABASE_URL,
            min_size=2,
            max_size=10,
            command_timeout=30,
            ssl="require",  
        )
    return _pool


async def close_pool() -> None:
    global _pool
    if _pool is not None:
        await _pool.close()
        _pool = None


@asynccontextmanager
async def get_connection() -> AsyncGenerator[Connection, None]:
    pool = await get_pool()
    async with pool.acquire() as conn:
        yield conn


@asynccontextmanager
async def get_transaction() -> AsyncGenerator[Connection, None]:
    pool = await get_pool()
    async with pool.acquire() as conn:
        async with conn.transaction():
            yield conn
