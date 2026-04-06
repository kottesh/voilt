"""Multi-stage verification pipeline for traffic violations.

Verification Pipeline:
1. Edge Detection Quality Check
2. Falcon-Perception Vision Verification
3. Confidence Thresholding
4. Business Rules Validation
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal

import cv2
import numpy as np
from PIL import Image

from server.services.falcon_engine import verify_violation
from shared.schemas import ViolationEvent, ViolationType

logger = logging.getLogger(__name__)


@dataclass
class VerificationResult:
    """Result of multi-stage verification."""

    accepted: bool
    confidence: float
    stage_results: dict[str, dict]
    reason: str | None = None


class VerificationStage:
    """Base class for verification stages."""

    def __init__(self, name: str):
        self.name = name

    async def verify(
        self, image: Image.Image, event: ViolationEvent
    ) -> tuple[bool, float, str | None]:
        """Run verification stage.

        Returns:
            Tuple of (passed, confidence, reason)
        """
        raise NotImplementedError


class ImageQualityStage(VerificationStage):
    """Check if image quality is sufficient for verification."""

    def __init__(
        self,
        min_blur_threshold: float = 100.0,
        min_brightness: int = 20,
        max_brightness: int = 235,
    ):
        super().__init__("image_quality")
        self.min_blur_threshold = min_blur_threshold
        self.min_brightness = min_brightness
        self.max_brightness = max_brightness

    async def verify(
        self, image: Image.Image, event: ViolationEvent
    ) -> tuple[bool, float, str | None]:
        """Check image quality metrics."""
        try:
            # Convert to numpy array
            img_array = np.array(image)
            if len(img_array.shape) == 3:
                gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            else:
                gray = img_array

            # 1. Blur detection (Laplacian variance)
            blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()

            # 2. Brightness check
            brightness = np.mean(gray)

            # Determine pass/fail
            reasons = []
            if blur_score < self.min_blur_threshold:
                reasons.append(f"Too blurry (score={blur_score:.1f})")

            if brightness < self.min_brightness:
                reasons.append(f"Too dark (brightness={brightness:.1f})")
            elif brightness > self.max_brightness:
                reasons.append(f"Too bright (brightness={brightness:.1f})")

            if reasons:
                return False, 0.0, "; ".join(reasons)

            # Confidence based on quality metrics
            blur_conf = min(1.0, blur_score / 300.0)
            brightness_conf = 1.0 - abs(brightness - 127.5) / 127.5  # Best at mid-brightness
            confidence = (blur_conf + brightness_conf) / 2

            return True, confidence, None

        except Exception as exc:
            logger.exception("Image quality check failed: %s", exc)
            return False, 0.0, f"Quality check error: {exc}"


class EdgeDetectionStage(VerificationStage):
    """Validate edge detection results against minimum requirements."""

    def __init__(
        self,
        min_confidence: float = 0.15,
        require_motorcycle: bool = True,
    ):
        super().__init__("edge_detection")
        self.min_confidence = min_confidence
        self.require_motorcycle = require_motorcycle

    async def verify(
        self, image: Image.Image, event: ViolationEvent
    ) -> tuple[bool, float, str | None]:
        """Validate edge detection metadata."""
        try:
            # Check confidence threshold
            if event.max_confidence < self.min_confidence:
                return (
                    False,
                    event.max_confidence,
                    f"Confidence too low ({event.max_confidence:.2f} < {self.min_confidence})",
                )

            # Motorcycle should be detected
            if self.require_motorcycle:
                bbox = event.motorcycle_bbox
                if bbox.x1 >= bbox.x2 or bbox.y1 >= bbox.y2:
                    return False, 0.0, "Invalid motorcycle bounding box"

            # Check violation-specific requirements
            for violation in event.violations:
                if violation == ViolationType.NO_HELMET:
                    if event.counts.no_helmet == 0:
                        return False, 0.0, "No helmet violation claimed but count is 0"
                elif violation == ViolationType.TRIPLE_RIDING:
                    total_riders = event.counts.rider + event.counts.pillion
                    if total_riders < 3:
                        return (
                            False,
                            0.0,
                            f"Triple riding claimed but only {total_riders} riders",
                        )

            return True, event.max_confidence, None

        except Exception as exc:
            logger.exception("Edge detection validation failed: %s", exc)
            return False, 0.0, f"Validation error: {exc}"


class FalconVisionStage(VerificationStage):
    """Re-verify violation using Falcon-Perception vision model."""

    def __init__(self, min_confidence: float = 0.4):
        super().__init__("falcon_vision")
        self.min_confidence = min_confidence

    async def verify(
        self, image: Image.Image, event: ViolationEvent
    ) -> tuple[bool, float, str | None]:
        """Run Falcon-Perception verification."""
        try:
            # Verify each claimed violation
            max_conf = 0.0
            verified_violations = []
            reasons = []

            for violation in event.violations:
                is_violation, conf, reason = await verify_violation(image, violation.value)

                if is_violation and conf >= self.min_confidence:
                    verified_violations.append(violation.value)
                    max_conf = max(max_conf, conf)
                else:
                    reasons.append(f"{violation.value}: {reason or 'not confirmed'}")

            if not verified_violations:
                return (
                    False,
                    max_conf,
                    "; ".join(reasons) if reasons else "No violations confirmed by vision model",
                )

            return True, max_conf, f"Verified: {', '.join(verified_violations)}"

        except Exception as exc:
            logger.exception("Falcon vision verification failed: %s", exc)
            return False, 0.0, f"Vision error: {exc}"


class BusinessRulesStage(VerificationStage):
    """Apply business logic rules for final decision."""

    def __init__(
        self,
        require_location: bool = False,
        max_track_age_seconds: int = 300,
    ):
        super().__init__("business_rules")
        self.require_location = require_location
        self.max_track_age_seconds = max_track_age_seconds

    async def verify(
        self, image: Image.Image, event: ViolationEvent
    ) -> tuple[bool, float, str | None]:
        """Apply business rules."""
        try:
            # Location requirement
            if self.require_location:
                if event.location.accuracy_m > 1000.0:  # More than 1km accuracy is too coarse
                    return (
                        False,
                        0.5,
                        f"Location too inaccurate ({event.location.accuracy_m:.0f}m)",
                    )

            # Time-based checks could go here
            # (e.g., event age, duplicate detection window)

            # Evidence availability
            if not event.evidence:
                logger.warning("No evidence files attached to event %s", event.event_id)

            return True, 1.0, None

        except Exception as exc:
            logger.exception("Business rules check failed: %s", exc)
            return False, 0.0, f"Rules error: {exc}"


class MultiStageVerifier:
    """Multi-stage verification pipeline coordinator."""

    def __init__(
        self,
        stages: list[VerificationStage] | None = None,
        min_overall_confidence: float = 0.5,
        require_all_stages: bool = False,
    ):
        """Initialize verifier with stages.

        Args:
            stages: List of verification stages (default: all stages)
            min_overall_confidence: Minimum weighted average confidence
            require_all_stages: If True, all stages must pass; if False, majority
        """
        self.stages = stages or self._default_stages()
        self.min_overall_confidence = min_overall_confidence
        self.require_all_stages = require_all_stages

    def _default_stages(self) -> list[VerificationStage]:
        """Get default verification stages."""
        return [
            ImageQualityStage(),
            EdgeDetectionStage(),
            FalconVisionStage(min_confidence=0.4),
            BusinessRulesStage(),
        ]

    async def verify(self, image: Image.Image, event: ViolationEvent) -> VerificationResult:
        """Run multi-stage verification pipeline.

        Args:
            image: Evidence image (annotated frame or crop)
            event: Violation event from edge

        Returns:
            VerificationResult with aggregated decision
        """
        stage_results = {}
        passed_count = 0
        total_confidence = 0.0
        rejection_reasons = []

        for stage in self.stages:
            logger.info("Running verification stage: %s", stage.name)

            passed, confidence, reason = await stage.verify(image, event)

            stage_results[stage.name] = {
                "passed": passed,
                "confidence": confidence,
                "reason": reason,
            }

            if passed:
                passed_count += 1
                total_confidence += confidence
            else:
                if reason:
                    rejection_reasons.append(f"{stage.name}: {reason}")
                logger.warning("Stage %s failed: %s", stage.name, reason)

                # Early exit if all stages required
                if self.require_all_stages:
                    return VerificationResult(
                        accepted=False,
                        confidence=confidence,
                        stage_results=stage_results,
                        reason="; ".join(rejection_reasons),
                    )

        # Calculate weighted average confidence
        avg_confidence = total_confidence / len(self.stages) if self.stages else 0.0

        # Decision logic
        if self.require_all_stages:
            # All must pass
            accepted = passed_count == len(self.stages)
        else:
            # Majority must pass AND confidence threshold met
            majority = passed_count > len(self.stages) / 2
            conf_ok = avg_confidence >= self.min_overall_confidence
            accepted = majority and conf_ok

        reason = None if accepted else "; ".join(rejection_reasons)

        return VerificationResult(
            accepted=accepted,
            confidence=avg_confidence,
            stage_results=stage_results,
            reason=reason,
        )


# Singleton verifier instance
_default_verifier: MultiStageVerifier | None = None


def get_verifier(
    profile: Literal["strict", "balanced", "lenient"] = "balanced",
) -> MultiStageVerifier:
    """Get configured verifier instance.

    Profiles:
        - strict: All stages must pass, high confidence threshold
        - balanced: Majority pass, medium confidence
        - lenient: Any stage pass, low confidence
    """
    global _default_verifier

    if profile == "strict":
        return MultiStageVerifier(
            min_overall_confidence=0.7,
            require_all_stages=True,
        )
    elif profile == "lenient":
        return MultiStageVerifier(
            min_overall_confidence=0.3,
            require_all_stages=False,
        )
    else:  # balanced
        if _default_verifier is None:
            _default_verifier = MultiStageVerifier(
                min_overall_confidence=0.5,
                require_all_stages=False,
            )
        return _default_verifier
