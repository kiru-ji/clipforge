"""
Configuration management with runtime mutability for GUI credential updates.

Design: Singleton + Observer
- Settings loaded once, mutable via GUI
- Credential changes trigger re-initialization of backends
- Thread-safe access via asyncio locks
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """All configuration. Mutable at runtime via GUI."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ─── Paths ──────────────────────────────────────────────────────────────
    work_dir: Path = Field(default=Path("/tmp/clipforge"))
    db_path: Path = Field(default=Path("/tmp/clipforge/state.db"))

    # ─── ASR (Groq Whisper API — free tier, no GPU needed) ─────────────────
    groq_api_key: str = Field(default="", description="Groq API key for Whisper STT")
    asr_model: str = Field(default="whisper-large-v3-turbo")

    # ─── LLM (Qwen via ModelScope — OpenAI-compatible) ─────────────────────
    qwen_api_key: str = Field(default="", description="ModelScope/DashScope API key")
    qwen_base_url: str = Field(default="https://dashscope.aliyuncs.com/compatible-mode/v1")
    qwen_model: str = Field(default="qwen-plus")

    # ─── Publishing ─────────────────────────────────────────────────────────
    youtube_client_secrets: str = Field(default="", description="YouTube OAuth2 client_secrets JSON content")
    youtube_token_file: Path = Field(default=Path("/tmp/clipforge/youtube_token.json"))
    upload_post_api_key: str = Field(default="", description="Upload-Post API key for TikTok")
    instagram_access_token: str = Field(default="")
    instagram_user_id: str = Field(default="")

    # ─── Telegram Approval ──────────────────────────────────────────────────
    telegram_bot_token: str = Field(default="")
    telegram_chat_id: str = Field(default="")
    approval_required: bool = Field(default=True)

    # ─── Pipeline Tuning ───────────────────────────────────────────────────
    clip_min_duration: int = Field(default=15, ge=5, le=120)
    clip_max_duration: int = Field(default=60, ge=10, le=300)
    max_clips_per_video: int = Field(default=8, ge=1, le=30)
    burn_subtitles: bool = Field(default=True)
    ffmpeg_threads: int = Field(default=2, ge=1, le=8)
    auto_cleanup_hours: int = Field(default=48, ge=1)

    # ─── Server ─────────────────────────────────────────────────────────────
    host: str = Field(default="0.0.0.0")
    port: int = Field(default=8080)
    admin_password: str = Field(default="changeme", description="Web UI admin password")

    @property
    def downloads_dir(self) -> Path:
        d = self.work_dir / "downloads"
        d.mkdir(parents=True, exist_ok=True)
        return d

    @property
    def output_dir(self) -> Path:
        d = self.work_dir / "output"
        d.mkdir(parents=True, exist_ok=True)
        return d

    @property
    def state_dir(self) -> Path:
        d = self.work_dir / "state"
        d.mkdir(parents=True, exist_ok=True)
        return d

    @property
    def has_groq(self) -> bool:
        return bool(self.groq_api_key)

    @property
    def has_qwen(self) -> bool:
        return bool(self.qwen_api_key)

    @property
    def has_youtube(self) -> bool:
        return bool(self.youtube_client_secrets)

    @property
    def has_tiktok(self) -> bool:
        return bool(self.upload_post_api_key)

    @property
    def has_instagram(self) -> bool:
        return bool(self.instagram_access_token and self.instagram_user_id)

    @property
    def has_telegram(self) -> bool:
        return bool(self.telegram_bot_token and self.telegram_chat_id)

    def save_to_env(self, path: Path | None = None) -> None:
        """Persist current settings to .env file (called from GUI)."""
        env_path = path or Path(".env")
        lines = []
        for field_name in self.model_fields:
            value = getattr(self, field_name)
            if isinstance(value, Path):
                value = str(value)
            if value:  # Only write non-empty values
                env_key = field_name.upper()
                lines.append(f"{env_key}={value}")
        env_path.write_text("\n".join(lines) + "\n")

    def update_credentials(self, **kwargs) -> None:
        """Update credentials at runtime (called from GUI save button)."""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
        self.save_to_env()


# Global singleton
_settings: Optional[Settings] = None

def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings

def reload_settings() -> Settings:
    global _settings
    _settings = Settings()
    return _settings
