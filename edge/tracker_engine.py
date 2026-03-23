"""Simple motorcycle tracker using IoU matching."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime

from edge.tracking import TrackState
from edge.types import Detection
from shared.schemas import BBox


def iou(a: BBox, b: BBox) -> float:
    """Compute intersection-over-union for two boxes."""

    inter_x1 = max(a.x1, b.x1)
    inter_y1 = max(a.y1, b.y1)
    inter_x2 = min(a.x2, b.x2)
    inter_y2 = min(a.y2, b.y2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    if inter_area == 0.0:
        return 0.0
    a_area = (a.x2 - a.x1) * (a.y2 - a.y1)
    b_area = (b.x2 - b.x1) * (b.y2 - b.y1)
    union = a_area + b_area - inter_area
    if union <= 0.0:
        return 0.0
    return inter_area / union


@dataclass(frozen=True)
class TrackMatch:
    """Association output for one current-frame motorcycle."""

    track_id: str
    motorcycle: Detection
    detection_index: int


class SimpleTracker:
    """Matches current motorcycles to previous tracks by IoU."""

    def __init__(self, iou_threshold: float = 0.2, max_age_seconds: float = 30.0) -> None:
        self._iou_threshold = iou_threshold
        self._max_age_seconds = max_age_seconds
        self._next_id = 1
        self._tracks: dict[str, TrackState] = {}

    @property
    def tracks(self) -> dict[str, TrackState]:
        """Return mutable track map used by pipeline."""

        return self._tracks

    def update(self, motorcycles: list[Detection]) -> list[TrackMatch]:
        """Assign track IDs for current motorcycles."""

        self._evict_stale()
        matches: list[TrackMatch] = []
        used_track_ids: set[str] = set()
        for index, motorcycle in enumerate(motorcycles):
            matched_id = self._match_existing(motorcycle, used_track_ids)
            if matched_id is None:
                matched_id = f"moto-{self._next_id}"
                self._next_id += 1
                self._tracks[matched_id] = TrackState(
                    track_id=matched_id,
                    motorcycle_bbox=motorcycle.bbox,
                )
            used_track_ids.add(matched_id)
            matches.append(
                TrackMatch(
                    track_id=matched_id,
                    motorcycle=motorcycle,
                    detection_index=index,
                )
            )
        return matches

    def _evict_stale(self) -> None:
        """Remove tracks not seen within max_age_seconds."""

        now = datetime.now(UTC)
        stale_ids = [
            track_id
            for track_id, track in self._tracks.items()
            if (now - track.last_seen).total_seconds() > self._max_age_seconds
        ]
        for track_id in stale_ids:
            del self._tracks[track_id]

    def _match_existing(self, motorcycle: Detection, used_track_ids: set[str]) -> str | None:
        best_track_id: str | None = None
        best_iou = 0.0
        for track_id, track in self._tracks.items():
            if track_id in used_track_ids:
                continue
            overlap = iou(track.motorcycle_bbox, motorcycle.bbox)
            if overlap > best_iou:
                best_iou = overlap
                best_track_id = track_id
        if best_track_id is None or best_iou < self._iou_threshold:
            return None
        return best_track_id
