"""Tests for the ingest API endpoint."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient

from server.app import app


def test_ingest_accepts_image_file() -> None:
    """POST /ingest with an image file returns 202 and a job_id."""
    client = TestClient(app)
    with patch("server.api.ingest.enqueue", new_callable=AsyncMock, return_value=1):
        response = client.post(
            "/ingest",
            files={"file": ("frame.jpg", b"fakejpgdata", "image/jpeg")},
            data={"camera_id": "cam-1"},
        )

    assert response.status_code == 202
    body = response.json()
    assert body["job_id"]
    assert body["queue_length"] == 1
    assert "queued" in body["message"].lower()


def test_ingest_rejects_non_image() -> None:
    """POST /ingest with a non-image file returns 415."""
    client = TestClient(app)
    response = client.post(
        "/ingest",
        files={"file": ("data.txt", b"hello", "text/plain")},
    )
    assert response.status_code == 415


def test_ingest_rejects_empty_file() -> None:
    """POST /ingest with an empty file returns 400."""
    client = TestClient(app)
    with patch("server.api.ingest.enqueue", new_callable=AsyncMock, return_value=1):
        response = client.post(
            "/ingest",
            files={"file": ("frame.jpg", b"", "image/jpeg")},
        )
    assert response.status_code == 400


def test_health_returns_ok() -> None:
    """GET /health returns 200 with status ok."""
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
