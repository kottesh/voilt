"""Storage service for managing file paths and URLs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote

from shared.schemas import VerifiedEvent, ViolationEvent


def get_image_url(image_path: str | None) -> str | None:
    """
    Convert a local image path to a URL that can be served by the frontend.

    If image_path is None, returns None.
    If image_path is already a URL (starts with http), returns it as-is.
    Otherwise, treats it as a local path and returns a URL path that can be
    served by the static file server mounted at /images (which serves the storage/ directory).
    """
    if not image_path:
        return None

    # If it's already a URL, return as-is
    if image_path.startswith(("http://", "https://")):
        return image_path

    try:
        image_path_obj = Path(image_path)

        # Check if the file exists
        if not image_path_obj.exists():
            return None

        # The server mounts the 'storage' directory at '/images'
        # So we need to remove the 'storage/' prefix from the path
        path_str = str(image_path_obj)

        # Handle both absolute and relative paths
        if path_str.startswith("storage/"):
            # Remove 'storage/' prefix since it's mounted at /images
            relative_path = path_str[len("storage/") :]
        elif image_path_obj.is_absolute():
            # For absolute paths, try to make them relative to storage directory
            storage_path = Path("storage").resolve()
            resolved_path = image_path_obj.resolve()
            try:
                relative_path = str(resolved_path.relative_to(storage_path))
            except ValueError:
                # Path is not within storage directory
                return None
        else:
            # Path is already relative, use as-is
            relative_path = path_str

        # Return URL path with proper encoding
        return f"/images/{quote(relative_path)}"

    except (ValueError, OSError):
        # If there's any issue with the path, return None
        return None


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