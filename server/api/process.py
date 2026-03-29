"""POST /process — dequeue one job, run vision analysis, persist if confidence ≥ threshold."""

from __future__ import annotations

import base64
import logging
from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel

from server.core.config import ServerSettings
from server.db.connection import get_transaction
from server.db.crud import insert_violation
from server.services.vision import VisionResult, analyze_image
from server.worker.queue import dequeue, queue_length

logger = logging.getLogger(__name__)
settings = ServerSettings()

router = APIRouter(prefix="/process", tags=["process"])


class VisionDetail(BaseModel):
    is_violation: bool
    confidence: float
    number_plate: str | None
    violation_type: str | None


class ProcessResponse(BaseModel):
    job_id: str
    action: str  # "saved" | "skipped" | "no_job"
    vision: VisionDetail | None
    violation_id: str | None  # UUID of the saved DB row, if any
    queue_remaining: int


def _read_image(job: dict) -> bytes:
    """Return raw image bytes from the job payload (base64 embedded or file path)."""
    if b64 := job.get("image_b64"):
        return base64.b64decode(b64)
    if path := job.get("image_path"):
        p = Path(path)
        if p.exists():
            return p.read_bytes()
    raise ValueError(f"Job {job.get('job_id')} has no usable image source.")


@router.post(
    "",
    response_model=ProcessResponse,
    summary="Dequeue one job, run vision analysis, persist if violation confirmed",
)
async def process_next() -> ProcessResponse:
    """
    Pulls the next job from the violation queue and runs the full pipeline:

    1. **Dequeue** — BRPOP with a 5-second timeout; returns `no_job` if empty.
    2. **Vision analysis** — sends the image to the configured vision model.
    3. **Confidence gate** — only writes to the DB if
       confidence >= `CONFIDENCE_THRESHOLD` (default 0.9).
    4. **Persist** — inserts a row into `violations` with status `confirmed`.
    5. **Skip** — if below threshold, logs and returns `skipped` (nothing written).

    Designed to be called by a worker loop or an external scheduler.
    """

    job = await dequeue(timeout=5)
    remaining = await queue_length()

    if job is None:
        return ProcessResponse(
            job_id="",
            action="no_job",
            vision=None,
            violation_id=None,
            queue_remaining=remaining,
        )

    job_id: str = job.get("job_id", "unknown")
    camera_id: str | None = job.get("camera_id")
    captured_at = datetime.fromisoformat(job["captured_at"])

    logger.info("Processing job %s (camera=%s)", job_id, camera_id)

    try:
        image_bytes = _read_image(job)
    except (ValueError, KeyError) as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc

    try:
        result: VisionResult = await analyze_image(image_bytes)
    except Exception as exc:
        logger.exception("Vision model failed for job %s", job_id)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Vision model error: {exc}",
        ) from exc

    vision_detail = VisionDetail(
        is_violation=result.is_violation,
        confidence=result.confidence,
        number_plate=result.number_plate,
        violation_type=result.violation_type,
    )

    logger.info(
        "Job %s — violation=%s confidence=%.3f plate=%s",
        job_id,
        result.is_violation,
        result.confidence,
        result.number_plate,
    )

    if not result.is_violation or result.confidence < settings.CONFIDENCE_THRESHOLD:
        logger.info(
            "Job %s skipped (confidence %.3f < threshold %.3f)",
            job_id,
            result.confidence,
            settings.CONFIDENCE_THRESHOLD,
        )
        return ProcessResponse(
            job_id=job_id,
            action="skipped",
            vision=vision_detail,
            violation_id=None,
            queue_remaining=remaining,
        )

    async with get_transaction() as tx:
        row = await insert_violation(
            tx,
            number_plate=result.number_plate,
            confidence_level=result.confidence,
            status="confirmed",
            evidence_image=job.get("image_path"),
            camera_id=camera_id,
            captured_at=captured_at,
        )

    violation_id = str(row["id"])
    logger.info("Job %s saved as violation %s", job_id, violation_id)

    return ProcessResponse(
        job_id=job_id,
        action="saved",
        vision=vision_detail,
        violation_id=violation_id,
        queue_remaining=remaining,
    )
