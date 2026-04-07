"""Storage service for managing file paths and URLs."""

from __future__ import annotations

import os
from pathlib import Path
from urllib.parse import quote

from server.core.config import get_settings


def get_image_url(image_path: str | None) -> str | None:
    """
    Convert a local image path to a URL that can be served by the frontend.

    If image_path is None, returns None.
    If image_path is already a URL (starts with http), returns it as-is.
    Otherwise, treats it as a local path relative to IMAGE_STORAGE_PATH and
    returns a URL path that can be served by a static file server.
    """
    if not image_path:
        return None

    # If it's already a URL, return as-is
    if image_path.startswith(("http://", "https://")):
        return image_path

    settings = get_settings()
    storage_path = Path(settings.IMAGE_STORAGE_PATH)

    # Make sure the path is within the storage directory for security
    try:
        image_path_obj = Path(image_path)
        # Resolve to absolute path and check if it's within storage
        resolved_path = (storage_path / image_path_obj).resolve()
        storage_path_resolved = storage_path.resolve()

        # Check if the resolved path is within the storage directory
        if not str(resolved_path).startswith(str(storage_path_resolved)):
            # Security violation - path tries to escape storage
            return None

        # Return a URL path that can be served by static file server
        # We'll assume the frontend can access images at /images/{filename}
        # In a real deployment, this would be configured by your web server
        relative_path = resolved_path.relative_to(storage_path_resolved)
        return f"/images/{quote(str(relative_path))}"

    except (ValueError, OSError):
        # If there's any issue with the path, return None
        return None


# Keep the existing InMemoryStore class for backward compatibility
from dataclasses import dataclass
from shared.schemas import VerifiedEvent, ViolationEvent


@dataclass
class InMemoryStore:
    """Simple in-memory persistence for tests and early development."""

    raw_events: list[ViolationEvent]
    verified_events: list[VerifiedEvent]
    idempotency_index: dict[str, str]
    upload_dir: Path

    def __init__(self, upload_dir: str = "server/uploads") -> None:
        self.raw_events = []
        self.verified_events = []
        self.idempotency_index = {}
        self.upload_dir = Path(upload_dir)
        self.upload_dir.mkdir(parents=True, exist_ok=True)

    def store_raw(self, event: ViolationEvent) -> bool:
        existing_event_id = self.idempotency_index.get(event.idempotency_key)
        if existing_event_id is not None:
            return False
        self.raw_events.append(event)
        self.idempotency_index[event.idempotency_key] = str(event.event_id)
        return True

    def store_verified(self, event: VerifiedEvent) -> None:
        self.verified_events.append(event)

    def save_evidence(self, event_id: str, filename: str, content: bytes) -> str:
        event_dir = self.upload_dir / event_id
        event_dir.mkdir(parents=True, exist_ok=True)
        target = event_dir / filename
        target.write_bytes(content)
        return str(target)
