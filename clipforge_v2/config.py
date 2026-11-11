"""
Configuration management.

Fixes applied (vs audit):
- J12: save_to_env() quotes multi-line values (YouTube JSON no longer corrupts .env)
- J15: writes ALL fields including falsy ones (approval_required=False, cleared keys)
- J16: chmod 600 on .env after write
- J19: clearing a key now persists (writes empty string)
- J17: update_credentials uses model_validate for pydantic constraint enforcement
- J20/P44: .gitignore created automatically
- J21/P51: work_dir defaults to ~/.clipforge (survives reboot, not /tmp)
- J22: thread-safe via threading.Lock (not asyncio lock — config is sync)
- J25: mkdir side effects moved to explicit ensure_dirs() call, not property getters
- J26: extra="forbid" catches typo'd env vars
- J27: get_settings() uses double-checked locking
- J11/P21: env_file resolved relative to this file's parent (repo root), not CWD
- P22: validates clip_min_duration <= clip_max_duration
- J14: admin_password is actually enforced (see app.py auth middleware)
- J23/J24: dead properties removed
"""

from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Optional

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Resolve .env relative to this file (repo root), not CWD — fixes J11/P21
_REPO_ROOT = Path(__file__).resolve().parent.parent
_ENV_PATH = _REPO_ROOT / ".env"


class Settings(BaseSettings):
    """All configuration. Thread-safe singleton."""

    model_config = SettingsConfigDict(
        env_file=str(_ENV_PATH),
        env_file_encoding="utf-8",
        extra="forbid",  # J26: catch typo'd env vars
    )

    # ─── Paths (J21/P51: home dir, not /tmp) ────────────────────────────────
    work_dir: Path = Field(default=Path.home() / ".clipforge")

    # ─── ASR ────────────────────────────────────────────────────────────────
    groq_api_key: str = Field(default="")
    asr_model: str = Field(default="whisper-large-v3-turbo")

    # ─── LLM ────────────────────────────────────────────────────────────────
    qwen_api_key: str = Field(default="")
    qwen_base_url: str = Field(default="https://dashscope.aliyuncs.com/compatible-mode/v1")
    qwen_model: str = Field(default="qwen-plus")

    # ─── Publishing ─────────────────────────────────────────────────────────
    youtube_client_secrets: str = Field(default="")
    upload_post_api_key: str = Field(default="")
    instagram_access_token: str = Field(default="")
    instagram_user_id: str = Field(default="")

    # ─── Telegram ───────────────────────────────────────────────────────────
    telegram_bot_token: str = Field(default="")
    telegram_chat_id: str = Field(default="")
    approval_required: bool = Field(default=True)

    # ─── Pipeline ───────────────────────────────────────────────────────────
    clip_min_duration: int = Field(default=15, ge=5, le=120)
    clip_max_duration: int = Field(default=60, ge=10, le=300)
    max_clips_per_video: int = Field(default=8, ge=1, le=30)
    burn_subtitles: bool = Field(default=True)
    ffmpeg_threads: int = Field(default=2, ge=1, le=8)
    auto_cleanup_hours: int = Field(default=48, ge=1)
    max_source_duration_min: int = Field(default=180, ge=10, description="Reject videos longer than this")

    # ─── Server ─────────────────────────────────────────────────────────────
    host: str = Field(default="127.0.0.1")  # J1/J3: bind localhost by default, use SSH tunnel
    port: int = Field(default=8090)  # P49: 8080 is taken on the VPS
    admin_password: str = Field(default="", description="Required for web UI access. Empty = first-run setup.")

    @model_validator(mode="after")
    def _validate_durations(self) -> "Settings":
        """P22: ensure clip_min_duration <= clip_max_duration."""
        if self.clip_min_duration > self.clip_max_duration:
            raise ValueError(
                f"clip_min_duration ({self.clip_min_duration}) must be <= "
                f"clip_max_duration ({self.clip_max_duration})"
            )
        return self

    @property
    def downloads_dir(self) -> Path:
        return self.work_dir / "downloads"

    @property
    def output_dir(self) -> Path:
        return self.work_dir / "output"

    @property
    def state_dir(self) -> Path:
        return self.work_dir / "state"

    def ensure_dirs(self) -> None:
        """Create all work directories. Call once at startup, not in property getters (J25)."""
        for d in [self.work_dir, self.downloads_dir, self.output_dir, self.state_dir]:
            d.mkdir(parents=True, exist_ok=True)

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

    def save_to_env(self) -> None:
        """
        Persist settings to .env file.
        J12: multi-line values are quoted so they don't corrupt the file.
        J15: writes ALL fields including falsy/empty (so clearing works).
        J16: chmod 600 after write.
        """
        lines = []
        for field_name in self.model_fields:
            value = getattr(self, field_name)
            if isinstance(value, Path):
                value = str(value)
            elif isinstance(value, bool):
                value = "true" if value else "false"
            elif not isinstance(value, str):
                value = str(value)
            env_key = field_name.upper()
            # J12: quote values containing newlines, spaces, or special chars
            if "\n" in value or " " in value or any(c in value for c in '"\'#$'):
                escaped = value.replace("\\", "\\\\").replace('"', '\\"')
                lines.append(f'{env_key}="{escaped}"')
            else:
                lines.append(f"{env_key}={value}")

        _ENV_PATH.parent.mkdir(parents=True, exist_ok=True)
        _ENV_PATH.write_text("\n".join(lines) + "\n")
        os.chmod(_ENV_PATH, 0o600)  # J16: not world-readable

    def update_credentials(self, **kwargs) -> "Settings":
        """
        Update credentials at runtime with validation.
        J17: uses model_validate to enforce pydantic constraints.
        Returns a NEW Settings instance (immutable replacement, not in-place mutation).
        """
        current = self.model_dump()
        current.update(kwargs)
        new_settings = Settings(**current)
        new_settings.save_to_env()
        return new_settings


# ─── Thread-safe singleton (J22, J27) ───────────────────────────────────────

_settings: Optional[Settings] = None
_lock = threading.Lock()


def get_settings() -> Settings:
    """Double-checked locking singleton."""
    global _settings
    if _settings is None:
        with _lock:
            if _settings is None:
                _settings = Settings()
                _settings.ensure_dirs()
    return _settings


def reload_settings() -> Settings:
    """Force reload from .env. Returns new instance and updates singleton."""
    global _settings
    with _lock:
        _settings = Settings()
        _settings.ensure_dirs()
    return _settings


def update_settings(**kwargs) -> Settings:
    """Update credentials and update the global singleton atomically."""
    global _settings
    with _lock:
        if _settings is None:
            _settings = Settings()
        _settings = _settings.update_credentials(**kwargs)
        return _settings


def ensure_gitignore() -> None:
    """P44/J20: create .gitignore to prevent committing secrets."""
    gitignore = _REPO_ROOT / ".gitignore"
    if not gitignore.exists():
        gitignore.write_text(
            ".env\n"
            "__pycache__/\n"
            "*.pyc\n"
            ".venv/\n"
            "venv/\n"
            "*.egg-info/\n"
            ".pytest_cache/\n"
            f"{Path.home() / '.clipforge'}/\n"
        )
