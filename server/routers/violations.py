from __future__ import annotations

from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel

from server.db.connection import get_transaction
from server.db.crud import list_violations
from server.services.storage import get_image_url

router = APIRouter(prefix="/violations", tags=["violations"])


class ViolationResponse(BaseModel):
    id: str
    number_plate: Optional[str]
    confidence_level: float
    evidence_image_url: Optional[str]
    camera_id: Optional[str]
    captured_at: str
    status: str
    created_at: str


class ViolationsListResponse(BaseModel):
    violations: List[ViolationResponse]
    total: int
    limit: int
    offset: int


@router.get(
    "",
    response_model=ViolationsListResponse,
    summary="List violations with pagination",
)
async def list_violations_endpoint(
    status: Optional[str] = Query(None, description="Filter by status"),
    limit: int = Query(50, ge=1, le=100, description="Number of items to return"),
    offset: int = Query(0, ge=0, description="Number of items to skip"),
):
    """
    Get a paginated list of violations.

    - **status**: Filter by violation status (pending, confirmed, mailed, etc.)
    - **limit**: Maximum number of records to return (1-100)
    - **offset**: Number of records to skip for pagination
    """
    async with get_transaction() as conn:
        violations = await list_violations(
            conn,
            status=status,
            limit=limit,
            offset=offset,
        )

        # Get total count for pagination
        if status:
            total_row = await conn.fetchrow(
                "SELECT COUNT(*) FROM violations WHERE status = $1", status
            )
        else:
            total_row = await conn.fetchrow("SELECT COUNT(*) FROM violations")

        total = total_row[0] if total_row else 0

        # Convert to response model
        violation_responses = []
        for violation in violations:
            evidence_image_url = get_image_url(violation.get("evidence_image"))
            violation_responses.append(
                ViolationResponse(
                    id=str(violation["id"]),
                    number_plate=violation.get("number_plate"),
                    confidence_level=float(violation["confidence_level"]),
                    evidence_image_url=evidence_image_url,
                    camera_id=violation.get("camera_id"),
                    captured_at=violation["captured_at"].isoformat(),
                    status=violation["status"],
                    created_at=violation["created_at"].isoformat(),
                )
            )

        return ViolationsListResponse(
            violations=violation_responses,
            total=total,
            limit=limit,
            offset=offset,
        )
