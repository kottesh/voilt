"""CRUD helpers for the violations table."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from asyncpg import Connection


async def insert_violation(
    conn: Connection,
    *,
    number_plate: str | None,
    confidence_level: float,
    status: str,
    evidence_image: str | None,
    camera_id: str | None,
    captured_at: datetime,
) -> dict[str, Any]:
    """Insert a confirmed violation row and return the full record."""
    row = await conn.fetchrow(
        """
        INSERT INTO violations
            (number_plate, confidence_level, status,
             evidence_image, camera_id, captured_at)
        VALUES ($1, $2, $3, $4, $5, $6)
        RETURNING *
        """,
        number_plate,
        confidence_level,
        status,
        evidence_image,
        camera_id,
        captured_at,
    )
    return dict(row)


async def get_violation(conn: Connection, violation_id: uuid.UUID) -> dict | None:
    row = await conn.fetchrow("SELECT * FROM violations WHERE id = $1", violation_id)
    return dict(row) if row else None


async def list_violations(
    conn: Connection,
    status: str | None = None,
    limit: int = 50,
    offset: int = 0,
) -> list[dict]:
    if status:
        rows = await conn.fetch(
            "SELECT * FROM violations"
            " WHERE status = $1 ORDER BY created_at DESC LIMIT $2 OFFSET $3",
            status,
            limit,
            offset,
        )
    else:
        rows = await conn.fetch(
            "SELECT * FROM violations ORDER BY created_at DESC LIMIT $1 OFFSET $2",
            limit,
            offset,
        )
    return [dict(r) for r in rows]


async def update_mailed_at(
    conn: Connection,
    violation_id: uuid.UUID,
    mailed_at: datetime,
) -> None:
    await conn.execute(
        "UPDATE violations SET mailed_at = $1, status = 'mailed' WHERE id = $2",
        mailed_at,
        violation_id,
    )
