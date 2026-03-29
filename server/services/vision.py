"""Vision model service.

Sends the evidence image to the configured vision model (GPT-4o by default)
and returns:
  - whether a traffic violation is present
  - the confidence level (0.0 – 1.0)
  - the extracted number plate string (or None)
"""

from __future__ import annotations

import base64
import json
import logging
from pathlib import Path

import httpx
from pydantic import BaseModel

from server.core.config import ServerSettings

logger = logging.getLogger(__name__)
settings = ServerSettings()

SYSTEM_PROMPT = """You are a traffic-violation analysis system.
You will receive an image from a traffic camera.

Respond ONLY with a valid JSON object — no markdown, no explanation — in this exact shape:
{
  "is_violation": true | false,
  "confidence": 0.0 to 1.0,
  "number_plate": "PLATE STRING or null",
  "violation_type": "brief label e.g. red-light, speeding, wrong-lane or null"
}

Rules:
- confidence reflects how certain you are that a traffic violation occurred.
- If no plate is visible, set number_plate to null.
- If no violation is visible, set is_violation to false and confidence to 0.0.
"""


class VisionResult(BaseModel):
    is_violation: bool
    confidence: float
    number_plate: str | None
    violation_type: str | None


def _image_to_data_url(image_bytes: bytes, media_type: str = "image/jpeg") -> str:
    b64 = base64.b64encode(image_bytes).decode()
    return f"data:{media_type};base64,{b64}"


def _load_image(source: str | bytes | Path) -> str:
    """Accept raw bytes, a base64 string, or a file path — return a data URL."""
    if isinstance(source, Path):
        source = source.read_bytes()
    if isinstance(source, bytes):
        return _image_to_data_url(source)
    # assume already a base64 string
    try:
        base64.b64decode(source, validate=True)
        return f"data:image/jpeg;base64,{source}"
    except Exception:
        raise ValueError("image must be raw bytes, base64 string, or a Path") from None


async def analyze_image(image: str | bytes | Path) -> VisionResult:
    """
    Call the vision model and return a structured VisionResult.

    Args:
        image: evidence image as raw bytes, base64-encoded string, or Path.

    Returns:
        VisionResult with is_violation, confidence, number_plate, violation_type.

    Raises:
        httpx.HTTPStatusError: on non-2xx from the API.
        ValueError: if the model returns unparseable JSON.
    """
    data_url = _load_image(image)

    payload = {
        "model": settings.VISION_MODEL,
        "max_tokens": 256,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": SYSTEM_PROMPT},
                    {
                        "type": "image_url",
                        "image_url": {"url": data_url, "detail": "high"},
                    },
                ],
            }
        ],
    }

    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.post(
            "https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {settings.VISION_API_KEY}"},
            json=payload,
        )
        response.raise_for_status()

    raw_text = response.json()["choices"][0]["message"]["content"].strip()

    try:
        data = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        logger.error("Vision model returned non-JSON: %s", raw_text)
        raise ValueError(f"Vision model returned non-JSON response: {raw_text}") from exc

    return VisionResult(
        is_violation=bool(data.get("is_violation", False)),
        confidence=float(data.get("confidence", 0.0)),
        number_plate=data.get("number_plate"),
        violation_type=data.get("violation_type"),
    )
