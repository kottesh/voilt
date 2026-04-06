"""Realtime edge runner for camera/video processing."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass

import cv2

from edge.camera import VideoSource, VideoSourceConfig
from edge.config import EdgeSettings
from edge.detector import Detector
from edge.location import LocationProvider
from edge.pipeline import EdgePipeline
from edge.queue import SQLiteEventQueue
from edge.types import Detection, FrameInput
from edge.uploader import EventUploader


@dataclass
class RuntimeStats:
    """Minimal runtime counters for observability."""

    frames: int = 0
    emitted: int = 0
    uploaded: int = 0
    failed_uploads: int = 0
    started_at: float = time.time()


def _draw_detections(frame: FrameInput, detections: list[Detection]) -> None:
    if frame.image is None:
        return
    for detection in detections:
        x1 = int(detection.bbox.x1)
        y1 = int(detection.bbox.y1)
        x2 = int(detection.bbox.x2)
        y2 = int(detection.bbox.y2)
        color = (0, 255, 0)
        if detection.label in {"no_helmet", "pillion", "rider"}:
            color = (0, 165, 255)
        if detection.label == "motorcycle":
            color = (255, 255, 0)
        cv2.rectangle(frame.image, (x1, y1), (x2, y2), color, 2)
        caption = f"{detection.label}:{detection.confidence:.2f}"
        cv2.putText(
            frame.image,
            caption,
            (x1, max(12, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            cv2.LINE_AA,
        )


def run_realtime(
    *,
    settings: EdgeSettings,
    detector: Detector,
    source: str,
    headless: bool,
    upload: bool,
) -> None:
    """Run continuous edge loop over camera or video source."""

    if not headless and not os.environ.get("DISPLAY"):
        headless = True
        print("no DISPLAY env var detected, enabling headless mode")

    queue = SQLiteEventQueue(settings.queue_db_path)
    location_provider = LocationProvider()
    location_provider.update_from_network(lat=12.97, lon=77.59, accuracy_m=120.0, source="ip")
    pipeline = EdgePipeline(
        settings=settings,
        detector=detector,
        queue=queue,
        location_provider=location_provider,
    )
    uploader = EventUploader(settings.ingest_url)
    stats = RuntimeStats(started_at=time.time())
    frame_delay_seconds = 1.0 / settings.max_fps

    capture = VideoSource(
        VideoSourceConfig(
            source=source,
            frame_width=settings.frame_width,
            frame_height=settings.frame_height,
        )
    )
    try:
        for frame in capture.frames():
            loop_start = time.time()
            result = pipeline.process_frame_with_details(frame)
            emitted = result.enqueued_events
            stats.frames += 1
            stats.emitted += emitted

            if upload:
                sent, failed = uploader.upload_once(queue=queue, batch_size=10)
                stats.uploaded += sent
                stats.failed_uploads += failed

            if not headless and frame.image is not None:
                _draw_detections(frame, result.detections)
                try:
                    cv2.imshow("voilt-edge", frame.image)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q"):
                        break
                except cv2.error:
                    print("display error, falling back to headless mode")
                    headless = True

            if stats.frames % 30 == 0:
                elapsed = max(1e-6, time.time() - stats.started_at)
                fps = stats.frames / elapsed
                print(
                    f"frames={stats.frames} fps={fps:.2f} emitted={stats.emitted} "
                    f"queue={queue.size()} uploaded={stats.uploaded} failed={stats.failed_uploads}"
                )

            loop_elapsed = time.time() - loop_start
            if loop_elapsed < frame_delay_seconds:
                time.sleep(frame_delay_seconds - loop_elapsed)
    finally:
        capture.close()
        if not headless:
            cv2.destroyAllWindows()
