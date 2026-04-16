"""POST /ingest — receive violation events with evidence images from edge."""

from __future__ import annotations

import json
import logging
import uuid
from datetime import UTC, datetime
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status
from pydantic import BaseModel

from server.core.config import get_settings
from server.worker.queue import enqueue

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/ingest", tags=["ingest"])


class IngestResponse(BaseModel):
    job_id: str
    queue_length: int
    message: str


def _save_evidence(image_bytes: bytes, job_id: str, index: int) -> str:
    """Persist an evidence image to disk; return the relative path."""
    settings = get_settings()
    storage = Path(settings.IMAGE_STORAGE_PATH)
    storage.mkdir(parents=True, exist_ok=True)
    dest = storage / f"{job_id}_evidence_{index}.jpg"
    dest.write_bytes(image_bytes)
    return str(dest)


@router.post(
    "",
    status_code=status.HTTP_202_ACCEPTED,
    response_model=IngestResponse,
    summary="Upload violation event with evidence images from edge",
)
async def ingest_event(
    event_json: str = Form(..., description="JSON-serialized violation event from edge"),
    evidence_0: UploadFile | None = File(None, description="Evidence image 0"),
    evidence_1: UploadFile | None = File(None, description="Evidence image 1"),
    evidence_2: UploadFile | None = File(None, description="Evidence image 2"),
    evidence_3: UploadFile | None = File(None, description="Evidence image 3"),
) -> IngestResponse:
    """
    Accepts a multipart upload from the edge layer containing:
    - event_json: Serialized event data (required)
    - evidence_N: Image files (optional, where N is 0, 1, 2, 3)

    1. Parses the event JSON
    2. Saves evidence images to storage
    3. Adds evidence paths to the job payload
    4. Pushes a job payload onto the Redis queue
    5. Returns 202 immediately — processing is async
    """
    try:
        event_data = json.loads(event_json)
    except json.JSONDecodeError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=f"Invalid event_json: {exc}",
        ) from exc

    job_id = event_data.get("event_id") or str(uuid.uuid4())
    camera_id = event_data.get("camera_id")
    captured_at = event_data.get("captured_at")

    # Save uploaded evidence images
    evidence_files = [evidence_0, evidence_1, evidence_2, evidence_3]
    saved_paths: list[str] = []

    for index, evidence_file in enumerate(evidence_files):
        if evidence_file is not None and evidence_file.filename:
            try:
                image_bytes = await evidence_file.read()
                if image_bytes:  # Only save non-empty files
                    path = _save_evidence(image_bytes, job_id, index)
                    saved_paths.append(path)
                    logger.info(
                        "Saved evidence %d for job %s: %s (%d bytes)",
                        index,
                        job_id,
                        path,
                        len(image_bytes),
                    )
            except Exception as exc:
                logger.warning(
                    "Failed to save evidence %d for job %s: %s",
                    index,
                    job_id,
                    exc,
                )

    # Use the first saved evidence path as the primary image for processing
    primary_image_path = saved_paths[0] if saved_paths else None

    job = {
        "job_id": job_id,
        "camera_id": camera_id,
        "captured_at": captured_at or datetime.now(UTC).isoformat(),
        "event": event_data,
        "image_path": primary_image_path,  # Primary evidence for vision processing
        "evidence_paths": saved_paths,  # All evidence paths
    }
    logger.info(
        "Received event %s (camera=%s, evidence_count=%d)",
        job_id,
        camera_id,
        len(saved_paths),
    )

    q_len = await enqueue(job)
    logger.info("Job %s enqueued (queue length now %d)", job_id, q_len)

    return IngestResponse(
        job_id=job_id,
        queue_length=q_len,
        message="Event accepted and queued for processing.",
    )
