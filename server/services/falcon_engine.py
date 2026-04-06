"""Falcon Perception and OCR engine for traffic violation verification.

This module provides:
1. Falcon-OCR for license plate text extraction
2. Falcon-Perception for violation verification (helmet detection, rider counting)
3. Unified inference pipeline with caching and batching
"""

from __future__ import annotations

import logging
import sys
from functools import lru_cache
from pathlib import Path
from typing import Literal

import torch
from PIL import Image

logger = logging.getLogger(__name__)

# Add falcon-perception to path
FALCON_PATH = Path("/tmp/falcon-perception")
if FALCON_PATH.exists():
    sys.path.insert(0, str(FALCON_PATH))

try:
    from falcon_perception import load_and_prepare_model, setup_torch_config
    from falcon_perception.data import ImageProcessor
    from falcon_perception.paged_inference import PagedInferenceEngine
    from falcon_perception.paged_ocr_inference import OCRInferenceEngine

    FALCON_AVAILABLE = True
except ImportError as exc:
    logger.warning("Falcon Perception not available: %s", exc)
    FALCON_AVAILABLE = False


class FalconEngineConfig:
    """Configuration for Falcon engines."""

    # Model identifiers
    OCR_MODEL: str = "tiiuae/Falcon-OCR"
    PERCEPTION_MODEL: str = "tiiuae/Falcon-Perception"

    # Device config
    DEVICE: str | None = None  # Auto-detect
    DTYPE: Literal["bfloat16", "float32"] = "float32"

    # Performance
    COMPILE: bool = True
    CUDAGRAPH: bool = True

    # OCR settings
    OCR_USE_LAYOUT: bool = False  # Simple plates don't need layout detection

    # Perception settings
    DETECTION_TASK: Literal["detection", "segmentation"] = "detection"
    CONFIDENCE_THRESHOLD: float = 0.25


@lru_cache(maxsize=1)
def get_falcon_ocr_engine() -> OCRInferenceEngine | None:
    """Get or create the shared Falcon-OCR engine."""
    if not FALCON_AVAILABLE:
        logger.error("Falcon Perception library not available")
        return None

    try:
        setup_torch_config()

        logger.info("Loading Falcon-OCR model: %s", FalconEngineConfig.OCR_MODEL)
        model, tokenizer, model_args = load_and_prepare_model(
            hf_model_id=FalconEngineConfig.OCR_MODEL,
            device=FalconEngineConfig.DEVICE,
            dtype=FalconEngineConfig.DTYPE,
            compile=FalconEngineConfig.COMPILE,
        )

        image_processor = ImageProcessor(patch_size=16, merge_size=1)

        engine = OCRInferenceEngine(
            model,
            tokenizer,
            image_processor,
            capture_cudagraph=FalconEngineConfig.CUDAGRAPH,
        )

        logger.info("Falcon-OCR engine initialized successfully")
        return engine

    except Exception as exc:
        logger.exception("Failed to initialize Falcon-OCR engine: %s", exc)
        return None


@lru_cache(maxsize=1)
def get_falcon_perception_engine() -> PagedInferenceEngine | None:
    """Get or create the shared Falcon-Perception engine."""
    if not FALCON_AVAILABLE:
        logger.error("Falcon Perception library not available")
        return None

    try:
        setup_torch_config()

        logger.info("Loading Falcon-Perception model: %s", FalconEngineConfig.PERCEPTION_MODEL)
        model, tokenizer, model_args = load_and_prepare_model(
            hf_model_id=FalconEngineConfig.PERCEPTION_MODEL,
            device=FalconEngineConfig.DEVICE,
            dtype=FalconEngineConfig.DTYPE,
            compile=FalconEngineConfig.COMPILE,
        )

        image_processor = ImageProcessor(patch_size=16, merge_size=1)

        engine = PagedInferenceEngine(
            model,
            tokenizer,
            image_processor,
            model_args,
            capture_cudagraph=FalconEngineConfig.CUDAGRAPH,
        )

        logger.info("Falcon-Perception engine initialized successfully")
        return engine

    except Exception as exc:
        logger.exception("Failed to initialize Falcon-Perception engine: %s", exc)
        return None


async def extract_plate_text(image: Image.Image | bytes) -> tuple[str | None, float]:
    """Extract license plate text from image using Falcon-OCR.

    Args:
        image: PIL Image or raw bytes

    Returns:
        Tuple of (plate_text, confidence_score)
    """
    engine = get_falcon_ocr_engine()
    if engine is None:
        logger.error("OCR engine not available")
        return None, 0.0

    if isinstance(image, bytes):
        from io import BytesIO

        image = Image.open(BytesIO(image)).convert("RGB")

    try:
        # Use plain OCR mode (no layout detection for simple plates)
        with torch.inference_mode():
            texts = engine.generate_plain(images=[image], use_tqdm=False)

        plate_text = texts[0].strip() if texts else None

        # Confidence is approximated from text quality
        # (Falcon-OCR doesn't provide per-character confidence)
        confidence = _estimate_plate_confidence(plate_text) if plate_text else 0.0

        logger.info("Extracted plate text: %s (conf=%.2f)", plate_text, confidence)
        return plate_text, confidence

    except Exception as exc:
        logger.exception("OCR extraction failed: %s", exc)
        return None, 0.0


async def verify_violation(
    image: Image.Image | bytes, violation_type: str
) -> tuple[bool, float, str | None]:
    """Verify violation using Falcon-Perception vision queries.

    Args:
        image: PIL Image or raw bytes
        violation_type: "no_helmet" or "triple_riding"

    Returns:
        Tuple of (is_violation, confidence, reason)
    """
    engine = get_falcon_perception_engine()
    if engine is None:
        logger.error("Perception engine not available")
        return False, 0.0, "Engine unavailable"

    if isinstance(image, bytes):
        from io import BytesIO

        image = Image.open(BytesIO(image)).convert("RGB")

    try:
        # Build natural language query based on violation type
        query = _build_verification_query(violation_type)

        with torch.inference_mode():
            # Run detection
            results = engine.generate(
                images=[image],
                queries=[query],
                task=FalconEngineConfig.DETECTION_TASK,
                use_tqdm=False,
            )

        result = results[0] if results else None
        if not result:
            return False, 0.0, "No detection result"

        # Parse results and determine violation
        is_violation, confidence, reason = _parse_perception_result(result, violation_type)

        logger.info(
            "Verification: %s -> violation=%s (conf=%.2f) reason=%s",
            violation_type,
            is_violation,
            confidence,
            reason,
        )
        return is_violation, confidence, reason

    except Exception as exc:
        logger.exception("Verification failed: %s", exc)
        return False, 0.0, f"Error: {exc}"


def _build_verification_query(violation_type: str) -> str:
    """Build natural language query for violation verification."""
    queries = {
        "no_helmet": "Detect all persons on motorcycles without helmets",
        "triple_riding": "Detect all motorcycles with three or more riders",
        "wrong_side": "Detect motorcycles driving on the wrong side of the road",
        "red_light": "Detect motorcycles crossing during red light",
        # Generic fallback
        "default": "Detect motorcycles and persons",
    }
    return queries.get(violation_type, queries["default"])


def _parse_perception_result(result: dict, violation_type: str) -> tuple[bool, float, str | None]:
    """Parse Falcon-Perception result and determine violation status.

    Args:
        result: Detection result from engine
        violation_type: Type of violation to check

    Returns:
        Tuple of (is_violation, confidence, reason)
    """
    # Extract detections
    detections = result.get("detections", [])
    if not detections:
        return False, 0.0, "No detections found"

    # For no_helmet: check if any person detected without helmet
    if violation_type == "no_helmet":
        no_helmet_count = sum(
            1
            for d in detections
            if "no" in d.get("label", "").lower() and "helmet" in d.get("label", "").lower()
        )
        if no_helmet_count > 0:
            max_conf = max(d.get("score", 0.0) for d in detections)
            return True, max_conf, f"Detected {no_helmet_count} person(s) without helmet"
        return False, 0.0, "No helmet violations detected"

    # For triple_riding: count riders on motorcycles
    elif violation_type == "triple_riding":
        rider_count = sum(1 for d in detections if "rider" in d.get("label", "").lower())
        if rider_count >= 3:
            max_conf = max(d.get("score", 0.0) for d in detections)
            return True, max_conf, f"Detected {rider_count} riders (≥3)"
        return False, 0.0, f"Only {rider_count} rider(s) detected"

    # Generic
    else:
        if detections:
            max_conf = max(d.get("score", 0.0) for d in detections)
            return True, max_conf, f"Detected {len(detections)} object(s)"
        return False, 0.0, "No violations detected"


def _estimate_plate_confidence(plate_text: str | None) -> float:
    """Estimate confidence score from plate text quality.

    Heuristics:
    - Length check (typical plates are 6-10 chars)
    - Alphanumeric ratio
    - No special characters
    """
    if not plate_text:
        return 0.0

    score = 0.5  # Base score

    # Length check
    length = len(plate_text)
    if 6 <= length <= 10:
        score += 0.2
    elif 4 <= length <= 12:
        score += 0.1

    # Alphanumeric check
    alnum_ratio = sum(c.isalnum() for c in plate_text) / max(len(plate_text), 1)
    score += 0.3 * alnum_ratio

    # Penalize if too many special chars
    if alnum_ratio < 0.7:
        score -= 0.2

    return min(1.0, max(0.0, score))


async def batch_extract_plates(
    images: list[Image.Image | bytes],
) -> list[tuple[str | None, float]]:
    """Batch extract plate text from multiple images.

    Args:
        images: List of PIL Images or bytes

    Returns:
        List of (plate_text, confidence) tuples
    """
    engine = get_falcon_ocr_engine()
    if engine is None:
        logger.error("OCR engine not available")
        return [(None, 0.0)] * len(images)

    # Convert all to PIL Images
    pil_images = []
    for img in images:
        if isinstance(img, bytes):
            from io import BytesIO

            img = Image.open(BytesIO(img)).convert("RGB")
        pil_images.append(img)

    try:
        with torch.inference_mode():
            texts = engine.generate_plain(images=pil_images, use_tqdm=False)

        results = []
        for text in texts:
            plate_text = text.strip() if text else None
            confidence = _estimate_plate_confidence(plate_text) if plate_text else 0.0
            results.append((plate_text, confidence))

        return results

    except Exception as exc:
        logger.exception("Batch OCR extraction failed: %s", exc)
        return [(None, 0.0)] * len(images)
