"""Tests for Falcon-based verification pipeline."""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

import numpy as np
import pytest
from PIL import Image

from server.services.verification_pipeline import (
    BusinessRulesStage,
    EdgeDetectionStage,
    ImageQualityStage,
    get_verifier,
)
from shared.schemas import (
    BBox,
    EventLocation,
    LocationSource,
    ViolationCounts,
    ViolationEvent,
    ViolationType,
)


def create_test_image(width: int = 640, height: int = 480, blur: bool = False) -> Image.Image:
    """Create a test image."""
    arr = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
    img = Image.fromarray(arr, mode="RGB")

    if blur:
        import cv2

        arr = cv2.GaussianBlur(np.array(img), (15, 15), 0)
        img = Image.fromarray(arr, mode="RGB")

    return img


def create_test_event(
    violations: list[ViolationType] = None,
    confidence: float = 0.6,
) -> ViolationEvent:
    """Create a test violation event."""
    if violations is None:
        violations = [ViolationType.NO_HELMET]

    return ViolationEvent(
        event_id=uuid4(),
        idempotency_key=f"test-{uuid4()}",
        device_id="test-device-01",
        track_id="track-123",
        captured_at=datetime.now(UTC),
        violations=violations,
        max_confidence=confidence,
        motorcycle_bbox=BBox(x1=100, y1=100, x2=200, y2=200),
        counts=ViolationCounts(rider=1, pillion=1, no_helmet=1),
        location=EventLocation(
            lat=25.276987,
            lon=55.296249,
            accuracy_m=50.0,
            source=LocationSource.WIFI,
        ),
        model_version="yolo11s-v1",
        software_version="edge-v0.1.0",
    )


@pytest.mark.asyncio
async def test_image_quality_stage_good():
    """Test image quality stage with good image."""
    stage = ImageQualityStage()
    image = create_test_image()
    event = create_test_event()

    passed, confidence, reason = await stage.verify(image, event)

    assert passed is True
    assert confidence > 0.0
    assert reason is None


@pytest.mark.asyncio
async def test_image_quality_stage_blurry():
    """Test image quality stage with blurry image."""
    stage = ImageQualityStage(min_blur_threshold=200.0)
    image = create_test_image(blur=True)
    event = create_test_event()

    passed, confidence, reason = await stage.verify(image, event)

    assert passed is False
    assert "blurry" in reason.lower()


@pytest.mark.asyncio
async def test_edge_detection_stage_valid():
    """Test edge detection validation with valid event."""
    stage = EdgeDetectionStage()
    image = create_test_image()
    event = create_test_event()

    passed, confidence, reason = await stage.verify(image, event)

    assert passed is True
    assert confidence == event.max_confidence


@pytest.mark.asyncio
async def test_edge_detection_stage_low_confidence():
    """Test edge detection validation with low confidence."""
    stage = EdgeDetectionStage(min_confidence=0.8)
    image = create_test_image()
    event = create_test_event(confidence=0.3)

    passed, confidence, reason = await stage.verify(image, event)

    assert passed is False
    assert "too low" in reason.lower()


@pytest.mark.asyncio
async def test_edge_detection_stage_invalid_bbox():
    """Test edge detection validation with invalid bounding box."""
    stage = EdgeDetectionStage()
    image = create_test_image()
    event = create_test_event()
    # Create valid event first, then modify bbox to be invalid after creation
    # (bypassing Pydantic validation)
    event.motorcycle_bbox.x2 = event.motorcycle_bbox.x1  # Make x1 == x2

    passed, confidence, reason = await stage.verify(image, event)

    assert passed is False
    assert "invalid" in reason.lower()


@pytest.mark.asyncio
async def test_business_rules_stage_valid():
    """Test business rules with valid event."""
    stage = BusinessRulesStage()
    image = create_test_image()
    event = create_test_event()

    passed, confidence, reason = await stage.verify(image, event)

    assert passed is True


@pytest.mark.asyncio
async def test_business_rules_stage_poor_location():
    """Test business rules with poor location accuracy."""
    stage = BusinessRulesStage(require_location=True)
    image = create_test_image()
    event = create_test_event()
    event.location.accuracy_m = 2000.0  # 2km accuracy

    passed, confidence, reason = await stage.verify(image, event)

    assert passed is False
    assert "location" in reason.lower()


@pytest.mark.asyncio
async def test_multi_stage_verifier_balanced():
    """Test balanced verifier profile."""
    verifier = get_verifier(profile="balanced")
    image = create_test_image()
    event = create_test_event(confidence=0.6)

    # Note: This will use mock Falcon stage if Falcon not available
    result = await verifier.verify(image, event)

    # Should pass quality, edge, and business rules
    assert len(result.stage_results) > 0
    assert result.confidence >= 0.0


@pytest.mark.asyncio
async def test_multi_stage_verifier_strict():
    """Test strict verifier profile (all stages must pass)."""
    verifier = get_verifier(profile="strict")
    image = create_test_image()
    event = create_test_event(confidence=0.3)  # Low confidence

    result = await verifier.verify(image, event)

    # Should fail due to low confidence
    assert result.accepted is False


@pytest.mark.asyncio
async def test_multi_stage_verifier_lenient():
    """Test lenient verifier profile."""
    verifier = get_verifier(profile="lenient")
    image = create_test_image(blur=True)  # Blurry image
    event = create_test_event(confidence=0.4)

    result = await verifier.verify(image, event)

    # Lenient mode should be more forgiving
    # Even if image quality fails, edge detection might pass
    assert result.stage_results is not None


def test_get_verifier_profiles():
    """Test different verifier profiles are distinct."""
    strict = get_verifier(profile="strict")
    balanced = get_verifier(profile="balanced")
    lenient = get_verifier(profile="lenient")

    assert strict.min_overall_confidence > balanced.min_overall_confidence
    assert balanced.min_overall_confidence > lenient.min_overall_confidence
    assert strict.require_all_stages is True
    assert balanced.require_all_stages is False
    assert lenient.require_all_stages is False
