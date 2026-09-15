"""Environment-driven settings shared by all services (prefix ``ST_``)."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


def _default_data_dir() -> Path:
    return Path(os.environ.get("ST_DATA_DIR", Path.cwd() / ".data"))


class CommonSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="ST_", env_file=".env", extra="ignore")

    data_dir: Path = _default_data_dir()
    redis_url: str = "redis://localhost:6379/0"
    log_level: str = "INFO"
    log_json: bool = True

    @property
    def blobs_dir(self) -> Path:
        return self.data_dir / "blobs"

    @property
    def models_dir(self) -> Path:
        return self.data_dir / "models"


@lru_cache
def get_common_settings() -> CommonSettings:
    return CommonSettings()
