"""Detector adapter interfaces for edge inference."""

from __future__ import annotations

from typing import Protocol

from edge.types import Detection, FrameInput
from shared.schemas import BBox


class Detector(Protocol):
    """Inference adapter protocol."""

    def detect(self, frame: FrameInput) -> list[Detection]:
        """Run inference and return detections for one frame."""

        raise NotImplementedError


class StubDetector:
    """Deterministic detector for tests and local development."""

    def __init__(self, detections: list[Detection] | None = None) -> None:
        self._detections = detections or []

    def detect(self, frame: FrameInput) -> list[Detection]:
        del frame
        return list(self._detections)


class YoloDetector:
    """YOLO runtime detector for realtime frame inference."""

    def __init__(self, model_path: str, conf: float = 0.25) -> None:
        import ultralytics

        self._model = ultralytics.YOLO(model_path)
        self._conf = conf
        self._labels = {
            "helmet",
            "motorcycle",
            "no_helmet",
            "number_plate",
            "pillion",
            "rider",
        }

    def detect(self, frame: FrameInput) -> list[Detection]:
        if frame.image is None:
            return []
        results = self._model.predict(source=frame.image, conf=self._conf, verbose=False)
        detections: list[Detection] = []
        if not results:
            return detections
        result = results[0]
        names = result.names
        boxes = result.boxes
        if boxes is None:
            return detections
        for box in boxes:
            class_index = int(box.cls.item())
            label = str(names[class_index])
            if label not in self._labels:
                continue
            confidence = float(box.conf.item())
            xyxy = box.xyxy[0].tolist()
            detections.append(
                Detection(
                    label=label,
                    confidence=confidence,
                    bbox=BBox(
                        x1=float(xyxy[0]),
                        y1=float(xyxy[1]),
                        x2=float(xyxy[2]),
                        y2=float(xyxy[3]),
                    ),
                )
            )
        return detections
