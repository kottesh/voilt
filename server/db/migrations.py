import asyncio
import logging
from typing import Callable, Awaitable

from asyncpg import Connection

from server.db.connection import get_connection, get_transaction

logger = logging.getLogger(__name__)

MigrationFn = Callable[[Connection], Awaitable[None]]


async def _bootstrap() -> None:
    async with get_connection() as conn:
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS migrations (
                id          SERIAL PRIMARY KEY,
                name        TEXT UNIQUE NOT NULL,
                executed_at TIMESTAMPTZ DEFAULT NOW()
            );
        """)


async def run_migration(name: str, migration_fn: MigrationFn) -> None:
    async with get_transaction() as tx:
        row = await tx.fetchrow(
            "SELECT 1 FROM migrations WHERE name = $1", name
        )
        if row:
            logger.info("Skipping '%s' — already ran", name)
            return

        logger.info("Running migration '%s'", name)
        await migration_fn(tx)
        await tx.execute(
            "INSERT INTO migrations (name) VALUES ($1)", name
        )
        logger.info("OK '%s'", name)


async def m001_create_violations(tx: Connection) -> None:
    await tx.execute("""
        CREATE TABLE IF NOT EXISTS violations (
            id               UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            number_plate     VARCHAR(20),
            confidence_level FLOAT        NOT NULL,
            status           VARCHAR(20)  NOT NULL DEFAULT 'pending',
            evidence_image   TEXT,
            camera_id        VARCHAR(50),
            captured_at      TIMESTAMPTZ  NOT NULL,
            mailed_at        TIMESTAMPTZ,
            created_at       TIMESTAMPTZ  NOT NULL DEFAULT NOW()
        );
    """)

    # Index for fast status-based queries (dashboard / worker polling)
    await tx.execute("""
        CREATE INDEX IF NOT EXISTS idx_violations_status
            ON violations (status);
    """)

    # Index for plate lookups
    await tx.execute("""
        CREATE INDEX IF NOT EXISTS idx_violations_number_plate
            ON violations (number_plate);
    """)


MIGRATIONS: list[tuple[str, MigrationFn]] = [
    ("001_create_violations", m001_create_violations),
]


async def run_all() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)s  %(message)s",
    )
    logger.info("Bootstrapping migrations table…")
    await _bootstrap()

    for name, fn in MIGRATIONS:
        await run_migration(name, fn)

    logger.info("All migrations complete.")


if __name__ == "__main__":
    asyncio.run(run_all())
