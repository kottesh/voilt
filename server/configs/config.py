"""Server runtime configuration."""

from __future__ import annotations

from pydantic_settings import SettingsConfigDict

from shared.config import BaseAppSettings


class ServerSettings(BaseAppSettings):
    """Settings for ingest API and persistence."""

    DATABASE_URL: str = "postgresql://localhost/voilt_test"  # Override in production via env
    REDIS_URL: str = "redis://localhost:6379"
    VISION_API_KEY: str = ""
    VISION_MODEL: str = "gpt-4o"
    VISION_MODEL_CACHE: str = "/home/anush/.cache/huggingface/hub/models--google--gemma-4-E2B-it"
    LITAI_API_KEY: str = ""
    LITAI_BILLING: str = ""
    LITAI_MODEL: str = "meta-llama/Llama-3.2-11B-Vision-Instruct"  # Vision-capable model
    NVIDIA_API_KEY: str = ""
    NVIDIA_MODEL: str = "mistralai/mistral-large-3-675b-instruct-2512"
    SMTP_HOST: str = "smtp.gmail.com"
    SMTP_PORT: int = 587
    SMTP_USER: str = ""
    SMTP_PASSWORD: str = ""
    MAIL_FROM: str = ""
    MAIL_TO: str = ""
    CONFIDENCE_THRESHOLD: float = 0.9
    IMAGE_STORAGE_PATH: str = "storage/evidence"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
    )
