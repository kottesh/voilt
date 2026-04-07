"""Vision model service using Falcon-Perception and Falcon-OCR.

Primary vision engine: Falcon-Perception + Falcon-OCR
Fallback: Gemini 3 Flash Preview, LitAI, local Gemma, or OpenAI GPT-4o Vision
"""

from __future__ import annotations

import base64
import json
import logging
from pathlib import Path

import httpx
from pydantic import BaseModel

from server.core.config import get_settings
from server.services.falcon_engine import FALCON_AVAILABLE

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a traffic-violation analysis system.
You will receive an image from a traffic camera.

Respond ONLY with a valid JSON object — no markdown, no explanation — in this exact shape:
{
  "is_violation": <boolean>,
  "confidence": <number between 0.0 and 1.0>,
  "number_plate": <string or null>,
  "violation_type": <string or null>
}

Rules:
- is_violation must be true or false (boolean, not string)
- confidence must be a number between 0.0 and 1.0
- number_plate must be either a string with the plate text, or null (NOT the string "null")
- violation_type should be a brief label like "red-light", "speeding", "wrong-lane",
  "no helmet", "triple riding" or null
- If no plate is visible, set number_plate to null (not "null" or empty string)
- If no violation is visible, set is_violation to false and confidence to 0.0

Example valid response:
{"is_violation": true, "confidence": 0.85, "number_plate": "ABC123", "violation_type": "red-light"}
{"is_violation": false, "confidence": 0.0, "number_plate": null, "violation_type": null}
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
    """Analyze image using Falcon-Perception + OCR (primary), Gemma 4, or OpenAI (fallbacks).

    Args:
        image: evidence image as raw bytes, base64-encoded string, or Path.

    Returns:
        VisionResult with is_violation, confidence, number_plate, violation_type.

    Raises:
        httpx.HTTPStatusError: on non-2xx from the API.
        ValueError: if the model returns unparseable JSON.
    """
    settings = get_settings()

    # Try Falcon first if available
    if FALCON_AVAILABLE:
        try:
            return await _analyze_with_falcon(image)
        except Exception as exc:
            logger.warning("Falcon analysis failed, falling back to next backend: %s", exc)

    # Try Gemini API
    if settings.GEMINI_API_KEY:
        try:
            return await _analyze_with_gemini_flash(image)
        except Exception as exc:
            logger.warning("Gemini analysis failed, falling back to next backend: %s", exc)

    # Try LitAI (cloud Gemma)
    if settings.LITAI_API_KEY and settings.LITAI_BILLING:
        try:
            return await _analyze_with_litai(image)
        except Exception as exc:
            logger.warning("LitAI analysis failed, falling back to next backend: %s", exc)

    # Try local Gemma 4
    try:
        return await _analyze_with_gemma(image)
    except Exception as exc:
        logger.warning("Local Gemma analysis failed, falling back to OpenAI: %s", exc)

    # Fallback to OpenAI if API key configured
    if settings.VISION_API_KEY:
        return await _analyze_with_openai(image)

    # No vision backend available
    raise RuntimeError(
        "No vision backend available. "
        "Install Falcon-Perception, configure GEMINI_API_KEY, LITAI, "
        "use local Gemma, or configure VISION_API_KEY"
    )


async def _analyze_with_falcon(image: str | bytes | Path) -> VisionResult:
    """Analyze image using Falcon-Perception + Falcon-OCR."""
    from io import BytesIO

    from PIL import Image

    from server.services.falcon_engine import (
        extract_plate_text,
        get_falcon_ocr_engine,
        get_falcon_perception_engine,
        verify_violation,
    )

    if get_falcon_perception_engine() is None or get_falcon_ocr_engine() is None:
        raise RuntimeError("Falcon engines not available")

    # Load image
    if isinstance(image, Path):
        image = image.read_bytes()
    if isinstance(image, str):
        # Assume base64
        image = base64.b64decode(image)

    pil_image = Image.open(BytesIO(image)).convert("RGB")

    # 1. Check for violations (Falcon-Perception)
    # Try common violation types
    is_violation = False
    max_confidence = 0.0
    detected_violation_type = None

    for v_type in ["no_helmet", "triple_riding"]:
        is_v, conf, _ = await verify_violation(pil_image, v_type)
        if is_v and conf > max_confidence:
            is_violation = True
            max_confidence = conf
            detected_violation_type = v_type

    # 2. Extract plate text (Falcon-OCR)
    plate_text, plate_conf = await extract_plate_text(pil_image)

    logger.info(
        "Falcon analysis: violation=%s (type=%s, conf=%.2f), plate=%s (conf=%.2f)",
        is_violation,
        detected_violation_type,
        max_confidence,
        plate_text,
        plate_conf or 0.0,
    )

    return VisionResult(
        is_violation=is_violation,
        confidence=max_confidence,
        number_plate=plate_text,
        violation_type=detected_violation_type,
    )


async def _analyze_with_gemini_flash(image: str | bytes | Path) -> VisionResult:
    """Analyze image using Google Gemini 3 Flash Preview.

    Uses the google-genai SDK with thinking capabilities.
    """
    from google import genai
    from google.genai import types

    settings = get_settings()
    if not settings.GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY must be configured")

    # Load image as bytes
    if isinstance(image, Path):
        image = image.read_bytes()
    if isinstance(image, str):
        image = base64.b64decode(image)

    model_name = settings.GEMINI_MODEL

    try:
        # Initialize Gemini client
        client = genai.Client(api_key=settings.GEMINI_API_KEY)

        # Build content with text prompt and image
        contents = [
            types.Content(
                role="user",
                parts=[
                    types.Part.from_text(text=SYSTEM_PROMPT),
                    types.Part.from_bytes(
                        data=image,
                        mime_type="image/jpeg",
                    ),
                ],
            ),
        ]

        # Configure generation with thinking enabled
        generate_content_config = types.GenerateContentConfig(
            temperature=0.15,
            max_output_tokens=1024,  # Increased to prevent truncation
            thinking_config=types.ThinkingConfig(
                thinking_level="HIGH",
            ),
        )

        # Generate content (non-streaming)
        response = client.models.generate_content(
            model=model_name,
            contents=contents,
            config=generate_content_config,
        )

        # Check if response was truncated
        if response.candidates:
            finish_reason = response.candidates[0].finish_reason
            if finish_reason and finish_reason.name not in ("STOP", "MAX_TOKENS"):
                logger.warning(
                    "Gemini response finish_reason=%s, may be incomplete", finish_reason.name
                )

        # Extract text from response parts (filter out thinking if needed)
        # response.text automatically concatenates all text parts
        raw_text = ""
        if response.candidates and response.candidates[0].content.parts:
            # Try to extract from parts explicitly
            text_parts = []
            for part in response.candidates[0].content.parts:
                if hasattr(part, "text") and part.text:
                    text_parts.append(part.text)
            raw_text = "".join(text_parts).strip()

        # Fallback to response.text if parts extraction failed
        if not raw_text:
            raw_text = response.text.strip()

        # Log response details
        logger.info(
            "Gemini response: candidates=%d, parts=%d, text_len=%d, finish=%s",
            len(response.candidates) if response.candidates else 0,
            len(response.candidates[0].content.parts) if response.candidates else 0,
            len(raw_text),
            response.candidates[0].finish_reason.name if response.candidates else "N/A",
        )

    except Exception as exc:
        logger.error("Gemini API call failed (model=%s): %s", model_name, exc)
        raise RuntimeError(f"Gemini analysis failed: {exc}") from exc

    logger.info(
        "Gemini raw response text (model=%s): %s",
        model_name,
        raw_text[:500] if len(raw_text) > 500 else raw_text,
    )

    # Parse JSON response
    try:
        data = json.loads(raw_text)
    except json.JSONDecodeError:
        # Try to clean up markdown formatting
        cleaned_text = raw_text.replace("```json", "").replace("```", "").strip()
        try:
            data = json.loads(cleaned_text)
        except json.JSONDecodeError as exc:
            logger.error(
                "Gemini returned non-JSON (len=%d): %s",
                len(raw_text),
                raw_text[:500],  # Show more context
            )
            raise ValueError(f"Gemini model returned non-JSON response: {raw_text[:200]}") from exc

    # Validate required fields and types
    if "is_violation" not in data:
        raise ValueError("Gemini response missing 'is_violation' field")
    if "confidence" not in data:
        raise ValueError("Gemini response missing 'confidence' field")

    is_violation = data["is_violation"]
    if not isinstance(is_violation, bool):
        raise ValueError(f"is_violation must be boolean, got {type(is_violation).__name__}")

    confidence = data["confidence"]
    if not isinstance(confidence, (int, float)):
        raise ValueError(f"confidence must be number, got {type(confidence).__name__}")
    confidence = float(confidence)
    if not 0.0 <= confidence <= 1.0:
        logger.warning("confidence %.2f out of range [0.0, 1.0], clamping", confidence)
        confidence = max(0.0, min(1.0, confidence))

    # Handle number_plate - reject string "null"
    number_plate = data.get("number_plate")
    if isinstance(number_plate, str) and number_plate.lower() == "null":
        logger.warning("Gemini returned string 'null' for number_plate, converting to None")
        number_plate = None
    elif number_plate is not None and not isinstance(number_plate, str):
        logger.warning("number_plate is not string or null, converting: %s", number_plate)
        number_plate = str(number_plate) if number_plate else None

    # Handle violation_type - reject string "null"
    violation_type = data.get("violation_type")
    if isinstance(violation_type, str) and violation_type.lower() == "null":
        logger.warning("Gemini returned string 'null' for violation_type, converting to None")
        violation_type = None
    elif violation_type is not None and not isinstance(violation_type, str):
        logger.warning("violation_type is not string or null, converting: %s", violation_type)
        violation_type = str(violation_type) if violation_type else None

    logger.info(
        "Gemini analysis (model=%s): violation=%s, conf=%.2f, plate=%s, type=%s",
        model_name,
        is_violation,
        confidence,
        number_plate,
        violation_type,
    )

    return VisionResult(
        is_violation=is_violation,
        confidence=confidence,
        number_plate=number_plate,
        violation_type=violation_type,
    )


async def _analyze_with_litai(image: str | bytes | Path) -> VisionResult:
    """Analyze image using LitAI cloud vision model.

    Note: Uses vision-capable models like Llama-3.2-11B-Vision-Instruct.
    Text-only models (Gemma) will not work for image analysis.
    """
    from litai import LLM

    settings = get_settings()
    if not settings.LITAI_API_KEY or not settings.LITAI_BILLING:
        raise RuntimeError("LITAI_API_KEY and LITAI_BILLING must be configured")

    # Load and encode image as base64
    if isinstance(image, Path):
        image = image.read_bytes()
    if isinstance(image, str):
        image = base64.b64decode(image)

    # Convert to base64 data URL
    image_b64 = base64.b64encode(image).decode("utf-8")
    data_url = f"data:image/jpeg;base64,{image_b64}"

    # Get model from config (defaults to Llama Vision)
    model_name = settings.LITAI_MODEL

    # Check if model supports vision
    if "vision" not in model_name.lower() and "gemma-4" not in model_name.lower():
        logger.warning(
            "LitAI model '%s' may not support vision. Use a vision-capable model.", model_name
        )

    # Initialize LitAI LLM
    llm = LLM(
        model=model_name,
        api_key=settings.LITAI_API_KEY,
    )

    # For vision models, we might need to pass the image differently
    # Try different approaches based on model type
    if "llama" in model_name.lower() and "vision" in model_name.lower():
        # Llama Vision models - try multimodal format
        prompt = (
            f"{SYSTEM_PROMPT}\n\n"
            f"[Image: {data_url[:50]}...]\n\n"
            "Analyze the traffic camera image above."
        )
    else:
        # Gemma or other - text with base64
        prompt = (
            f"{SYSTEM_PROMPT}\n\n"
            f"Image data (base64): {image_b64[:100]}...\n\n"
            "Analyze this traffic camera image."
        )

    try:
        response = llm.chat(prompt)
        raw_text = response.strip()
    except Exception as exc:
        logger.error("LitAI API call failed (model=%s): %s", model_name, exc)
        raise RuntimeError(f"LitAI analysis failed: {exc}") from exc

    # Parse JSON response
    try:
        data = json.loads(raw_text)
    except json.JSONDecodeError:
        # Try to clean up markdown formatting
        raw_text = raw_text.replace("```json", "").replace("```", "").strip()
        try:
            data = json.loads(raw_text)
        except json.JSONDecodeError as exc:
            logger.error("LitAI returned non-JSON: %s", raw_text[:200])
            raise ValueError("LitAI model returned non-JSON response") from exc

    logger.info(
        "LitAI analysis (model=%s): violation=%s, conf=%.2f, plate=%s",
        model_name,
        data.get("is_violation"),
        data.get("confidence"),
        data.get("number_plate"),
    )

    return VisionResult(
        is_violation=bool(data.get("is_violation", False)),
        confidence=float(data.get("confidence", 0.0)),
        number_plate=data.get("number_plate"),
        violation_type=data.get("violation_type"),
    )


async def _analyze_with_gemma(image: str | bytes | Path) -> VisionResult:
    """Analyze image using Google Gemma 4."""
    from functools import lru_cache
    from io import BytesIO

    from PIL import Image
    from transformers import AutoModelForMultimodalLM, AutoProcessor

    settings = get_settings()
    GEMMA_MODEL = "google/gemma-4-E2B-it"
    GEMMA_CACHE = settings.VISION_MODEL_CACHE if settings.VISION_MODEL_CACHE else None

    @lru_cache(maxsize=1)
    def _load_gemma_model():
        logger.info("Loading Gemma 4 model: %s (cache: %s)", GEMMA_MODEL, GEMMA_CACHE)
        processor = AutoProcessor.from_pretrained(
            GEMMA_MODEL,
            cache_dir=GEMMA_CACHE,
        )
        model = AutoModelForMultimodalLM.from_pretrained(
            GEMMA_MODEL,
            dtype="auto",
            device_map="auto",
            cache_dir=GEMMA_CACHE,
        )
        logger.info("Gemma 4 model loaded successfully")
        return processor, model

    # Load image
    if isinstance(image, Path):
        image = image.read_bytes()
    if isinstance(image, str):
        image = base64.b64decode(image)

    pil_image = Image.open(BytesIO(image)).convert("RGB")

    processor, model = _load_gemma_model()

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": pil_image},
                {
                    "type": "text",
                    "text": SYSTEM_PROMPT,
                },
            ],
        }
    ]

    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        add_generation_prompt=True,
    ).to(model.device)
    input_len = inputs["input_ids"].shape[-1]

    outputs = model.generate(
        **inputs,
        max_new_tokens=256,
        temperature=1.0,
        top_p=0.95,
        top_k=64,
    )
    response = processor.decode(outputs[0][input_len:], skip_special_tokens=False)
    parsed = processor.parse_response(response)

    raw_text = parsed.get("text", "").strip()

    try:
        data = json.loads(raw_text)
    except json.JSONDecodeError:
        raw_text = raw_text.replace("```json", "").replace("```", "")
        try:
            data = json.loads(raw_text)
        except json.JSONDecodeError as exc:
            logger.error("Gemma returned non-JSON: %s", raw_text[:200])
            raise ValueError("Gemma model returned non-JSON response") from exc

    logger.info(
        "Gemma analysis: violation=%s, conf=%.2f, plate=%s",
        data.get("is_violation"),
        data.get("confidence"),
        data.get("number_plate"),
    )

    return VisionResult(
        is_violation=bool(data.get("is_violation", False)),
        confidence=float(data.get("confidence", 0.0)),
        number_plate=data.get("number_plate"),
        violation_type=data.get("violation_type"),
    )


async def _analyze_with_openai(image: str | bytes | Path) -> VisionResult:
    """Fallback: Call OpenAI GPT-4o Vision API."""
    settings = get_settings()
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
