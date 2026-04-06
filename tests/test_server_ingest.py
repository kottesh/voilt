"""Tests for the ingest API endpoint."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient

from server.app import app


def test_ingest_accepts_image_file() -> None:
    """POST /ingest with an event_json returns 202 and a job_id."""
    client = TestClient(app)
    event_payload = {
        "event_id": "test-event-1",
        "camera_id": "cam-1",
        "captured_at": "2024-01-01T12:00:00Z",
    }
    with patch("server.api.ingest.enqueue", new_callable=AsyncMock, return_value=1):
        response = client.post(
            "/ingest",
            data={"event_json": json.dumps(event_payload)},
        )

    assert response.status_code == 202
    body = response.json()
    assert body["job_id"]
    assert body["queue_length"] == 1
    assert "queued" in body["message"].lower()


def test_ingest_rejects_invalid_json() -> None:
    """POST /ingest with invalid JSON returns 422."""
    client = TestClient(app)
    response = client.post(
        "/ingest",
        data={"event_json": "not valid json"},
    )
    assert response.status_code == 422


def test_ingest_rejects_missing_event_json() -> None:
    """POST /ingest without event_json returns 422."""
    client = TestClient(app)
    response = client.post(
        "/ingest",
        data={},
    )
    assert response.status_code == 422


def test_health_returns_ok() -> None:
    """GET /health returns 200 with status ok."""
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
