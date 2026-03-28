"""Server runtime configuration."""

from __future__ import annotations

from pydantic_settings import SettingsConfigDict

from shared.config import BaseAppSettings


class ServerSettings(BaseAppSettings):
    """Settings for ingest API and persistence."""

    DATABASE_URL: str
    REDIS_URL: str = "redis://localhost:6379"
    VISION_API_KEY: str = ""
    VISION_MODEL: str = "gpt-4o"
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
