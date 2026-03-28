"""Server core configuration exports."""

from functools import lru_cache

from server.configs.config import ServerSettings


@lru_cache
def get_settings() -> ServerSettings:
    """Return cached server settings."""
    return ServerSettings()
