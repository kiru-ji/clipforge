"""Configuration management with validation and defaults."""

from __future__ import annotations

import os
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, field_validator


class Settings(BaseSettings):
    """All pipeline configuration from .env + environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Work directory
    work_dir: Path = Field(default=Path("/shared/clipforge"))
    max_concurrent_jobs: int = Field(default=2, ge=1, le=8)

    # ASR
    whisper_model: str = Field(default="medium")
    whisper_device: str = Field(default="cuda")
    whisper_compute: str = Field(default="int8_float16")

    # LLM
    llm_backend: str = Field(default="ollama")
    ollama_base_url: str = Field(default="http://localhost:11434/v1")
    ollama_model: str = Field(default="qwen2.5:7b")
    gemini_api_key: str = Field(default="")

    # Publishing
    youtube_client_secrets_file: str = Field(default="")
    youtube_token_file: Path = Field(default=Path("/shared/clipforge/state/youtube_token.json"))
    instagram_access_token: str = Field(default="")
    instagram_user_id: str = Field(default="")
    upload_post_api_key: str = Field(default="")

    # Telegram
    telegram_bot_token: str = Field(default="")
    telegram_chat_id: str = Field(default="")
    approval_required: bool = Field(default=True)

    # Scrapling
    scrapling_headless: bool = Field(default=True)
    scrapling_stealth: bool = Field(default=True)

    # Pipeline
    clip_min_duration: int = Field(default=15, ge=5, le=120)
    clip_max_duration: int = Field(default=60, ge=10, le=300)
    max_clips_per_video: int = Field(default=10, ge=1, le=30)
    reframe_mode: str = Field(default="auto")
    burn_subtitles: bool = Field(default=True)
    nvenc_preset: str = Field(default="p4")
    auto_cleanup_hours: int = Field(default=24, ge=1)

    @field_validator("work_dir")
    @classmethod
    def ensure_work_dir(cls, v: Path) -> Path:
        v.mkdir(parents=True, exist_ok=True)
        for sub in ["downloads", "transcripts", "clips", "output", "metadata", "state"]:
            (v / sub).mkdir(parents=True, exist_ok=True)
        return v

    @property
    def downloads_dir(self) -> Path:
        return self.work_dir / "downloads"

    @property
    def transcripts_dir(self) -> Path:
        return self.work_dir / "transcripts"

    @property
    def clips_dir(self) -> Path:
        return self.work_dir / "clips"

    @property
    def output_dir(self) -> Path:
        return self.work_dir / "output"

    @property
    def metadata_dir(self) -> Path:
        return self.work_dir / "metadata"

    @property
    def state_dir(self) -> Path:
        return self.work_dir / "state"

    @property
    def has_gemini(self) -> bool:
        return bool(self.gemini_api_key)

    @property
    def has_youtube(self) -> bool:
        return bool(self.youtube_client_secrets_file)

    @property
    def has_instagram(self) -> bool:
        return bool(self.instagram_access_token and self.instagram_user_id)

    @property
    def has_tiktok(self) -> bool:
        return bool(self.upload_post_api_key)

    @property
    def has_telegram(self) -> bool:
        return bool(self.telegram_bot_token and self.telegram_chat_id)


def get_settings() -> Settings:
    """Load settings from .env file and environment."""
    return Settings()
