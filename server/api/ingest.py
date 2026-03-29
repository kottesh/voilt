"""POST /ingest — receive an evidence image and push it onto the violation queue."""

from __future__ import annotations

import base64
import logging
import uuid
from datetime import UTC, datetime
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status
from pydantic import BaseModel

from server.core.config import ServerSettings
from server.worker.queue import enqueue

logger = logging.getLogger(__name__)
settings = ServerSettings()

router = APIRouter(prefix="/ingest", tags=["ingest"])


class IngestResponse(BaseModel):
    job_id: str
    queue_length: int
    message: str


def _save_image(image_bytes: bytes, job_id: str) -> str:
    """Persist the raw image to disk; return the relative path stored in DB."""
    storage = Path(settings.IMAGE_STORAGE_PATH)
    storage.mkdir(parents=True, exist_ok=True)
    dest = storage / f"{job_id}.jpg"
    dest.write_bytes(image_bytes)
    return str(dest)


@router.post(
    "",
    status_code=status.HTTP_202_ACCEPTED,
    response_model=IngestResponse,
    summary="Upload a violation image onto the processing queue",
)
async def ingest_image(
    file: UploadFile = File(..., description="Evidence image captured by the edge camera"),
    camera_id: str | None = Form(None, description="Camera or sensor identifier"),
    captured_at: datetime | None = Form(
        None,
        description="UTC timestamp of capture; defaults to now if omitted",
    ),
) -> IngestResponse:
    """
    Accepts a multipart image upload from the edge layer.

    1. Validates the file is an image.
    2. Saves it to the configured storage path.
    3. Pushes a lightweight job payload onto the Redis queue.
    4. Returns 202 immediately — processing is async.
    """
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Expected an image file, got '{file.content_type}'.",
        )

    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty.",
        )

    job_id = str(uuid.uuid4())
    image_path = _save_image(image_bytes, job_id)

    job = {
        "job_id": job_id,
        "image_path": image_path,
        # also embed base64 so the worker doesn't need to re-read from disk
        "image_b64": base64.b64encode(image_bytes).decode(),
        "camera_id": camera_id,
        "captured_at": (captured_at or datetime.now(UTC)).isoformat(),
    }

    q_len = await enqueue(job)
    logger.info("Job %s enqueued (queue length now %d)", job_id, q_len)

    return IngestResponse(
        job_id=job_id,
        queue_length=q_len,
        message="Image accepted and queued for processing.",
    )
