"""Verification service using Falcon-based multi-stage pipeline."""

from __future__ import annotations

import logging
from io import BytesIO

from PIL import Image

from server.services.falcon_engine import extract_plate_text as falcon_extract_plate
from server.services.verification_pipeline import get_verifier
from shared.schemas import VerificationStatus, VerifiedEvent, ViolationEvent

logger = logging.getLogger(__name__)


async def verify_event(
    event: ViolationEvent,
    evidence_image: bytes | None = None,
) -> VerifiedEvent:
    """Multi-stage verification using Falcon-Perception and OCR.

    Args:
        event: Violation event from edge
        evidence_image: Evidence image bytes (annotated frame or crop)

    Returns:
        VerifiedEvent with verification results and OCR plate text
    """
    if evidence_image is None:
        # No image provided, cannot do vision verification
        logger.warning(
            "No evidence image for event %s, skipping vision verification", event.event_id
        )
        return VerifiedEvent(
            event=event,
            status=VerificationStatus.NEEDS_REVIEW,
            verification_score=event.max_confidence,
            plate_text=None,
            plate_confidence=None,
            reason="No evidence image provided",
        )

    # Load image
    try:
        image = Image.open(BytesIO(evidence_image)).convert("RGB")
    except Exception as exc:
        logger.exception("Failed to load evidence image: %s", exc)
        return VerifiedEvent(
            event=event,
            status=VerificationStatus.REJECTED,
            verification_score=0.0,
            plate_text=None,
            plate_confidence=None,
            reason=f"Invalid image: {exc}",
        )

    # Run multi-stage verification
    verifier = get_verifier(profile="balanced")
    result = await verifier.verify(image, event)

    # Determine status
    if result.accepted:
        status = VerificationStatus.ACCEPTED
    elif result.confidence >= 0.3:
        status = VerificationStatus.NEEDS_REVIEW
    else:
        status = VerificationStatus.REJECTED

    # Extract plate text (try to find plate crop in evidence)
    plate_text, plate_confidence = await extract_plate_text_from_event(event, evidence_image)

    logger.info(
        "Event %s verified: status=%s score=%.2f plate=%s",
        event.event_id,
        status.value,
        result.confidence,
        plate_text,
    )

    return VerifiedEvent(
        event=event,
        status=status,
        verification_score=result.confidence,
        plate_text=plate_text,
        plate_confidence=plate_confidence,
        reason=result.reason,
    )


async def extract_plate_text_from_event(
    event: ViolationEvent, evidence_image: bytes
) -> tuple[str | None, float | None]:
    """Extract license plate text from event evidence.

    Args:
        event: Violation event with evidence metadata
        evidence_image: Image bytes to extract from

    Returns:
        Tuple of (plate_text, confidence)
    """
    # Look for plate crop in evidence
    plate_crop_uri = None
    for evidence in event.evidence:
        if evidence.kind == "plate_crop":
            plate_crop_uri = evidence.uri
            break

    if plate_crop_uri:
        # TODO: Load plate crop from URI (file path or URL)
        # For now, use the full evidence image
        logger.info("Using full evidence image for OCR (plate crop URI: %s)", plate_crop_uri)

    try:
        # Use Falcon-OCR for extraction
        image = Image.open(BytesIO(evidence_image)).convert("RGB")
        plate_text, confidence = await falcon_extract_plate(image)
        return plate_text, confidence
    except Exception as exc:
        logger.exception("Failed to extract plate text: %s", exc)
        return None, None
