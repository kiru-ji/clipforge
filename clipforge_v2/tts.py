"""
ElevenLabs TTS backend (wired 12 Nov 2026 — NOT invoked by any pipeline stage).

Purpose: narration/voice-over synthesis for ClipForge output (e.g. 'Kitaab
Adalat' episodes). Voice casting is PENDING, so:
- settings.elevenlabs_voice_id stays empty until ANY picks a voice;
- synthesize() refuses to run without an explicit voice_id (guard against
  burning account characters on the wrong voice);
- nothing in pipeline.py/app.py calls this module yet — it is wiring only.

API: https://api.elevenlabs.io/v1 (header 'xi-api-key'), verified against
ElevenLabs OpenAPI docs Nov 2026:
- POST /v1/text-to-speech/{voice_id}?output_format=mp3_44100_128
  body: {"text", "model_id", "voice_settings"?} -> audio/mpeg bytes
- GET  /v1/voices                     -> voice library
- GET  /v1/user/subscription          -> free health/credits probe (no chars burned)
"""

from __future__ import annotations

from pathlib import Path

import httpx
import structlog

from .contracts import TTSBackend

logger = structlog.get_logger(__name__)

_BASE_URL = "https://api.elevenlabs.io/v1"
_TIMEOUT = 120.0


class VoiceNotCastedError(RuntimeError):
    """Raised when synthesize() is called before a voice_id has been chosen."""


class ElevenLabsTTSBackend(TTSBackend):
    """ElevenLabs text-to-speech via the v1 REST API (httpx, async)."""

    def __init__(self, api_key: str, voice_id: str = "", model_id: str = "eleven_multilingual_v2"):
        self.api_key = api_key
        self.voice_id = voice_id  # empty = voice casting pending
        self.model_id = model_id

    def _headers(self) -> dict:
        return {
            "xi-api-key": self.api_key,
            "Accept": "audio/mpeg",
            "Content-Type": "application/json",
        }

    async def synthesize(self, text: str, output_path: Path, voice_id: str | None = None) -> Path:
        """Synthesize text to an MP3 file.

        GUARD: raises VoiceNotCastedError unless a voice_id is provided or
        configured — prevents accidental character burn while casting pending.
        """
        voice = voice_id or self.voice_id
        if not voice:
            raise VoiceNotCastedError(
                "No ElevenLabs voice_id configured (voice casting pending). "
                "Pass voice_id explicitly or set ELEVENLABS_VOICE_ID in .env."
            )
        if not text or not text.strip():
            raise ValueError("text must be non-empty")

        url = f"{_BASE_URL}/text-to-speech/{voice}"
        params = {"output_format": "mp3_44100_128"}
        payload = {"text": text, "model_id": self.model_id}

        async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
            r = await client.post(url, headers=self._headers(), params=params, json=payload)
            if r.status_code == 401:
                raise RuntimeError("ElevenLabs API key invalid (401)")
            if r.status_code == 429:
                raise RuntimeError("ElevenLabs rate limited / quota exhausted (429)")
            if r.status_code != 200:
                raise RuntimeError(f"ElevenLabs error {r.status_code}: {r.text[:300]}")
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_bytes(r.content)

        logger.info("elevenlabs_synthesized", chars=len(text), out=str(output_path))
        return output_path

    async def list_voices(self) -> list[dict]:
        """Voice library (for the casting session — read-only, burns no chars)."""
        async with httpx.AsyncClient(timeout=30) as client:
            r = await client.get(f"{_BASE_URL}/voices", headers=self._headers())
            r.raise_for_status()
            return r.json().get("voices", [])

    async def health_check(self) -> bool:
        """Free subscription probe — validates the key WITHOUT consuming characters."""
        try:
            async with httpx.AsyncClient(timeout=15) as client:
                r = await client.get(
                    f"{_BASE_URL}/user/subscription",
                    headers={"xi-api-key": self.api_key},
                )
                if r.status_code == 401:
                    logger.error("elevenlabs_health_invalid_key")
                    return False
                return r.status_code == 200
        except httpx.ConnectError:
            logger.error("elevenlabs_health_network_down")
            return False


def get_tts_backend(settings) -> ElevenLabsTTSBackend | None:
    """Factory: build the backend from Settings, or None when unconfigured.

    Deliberately NOT referenced from pipeline.py yet (wiring-only milestone).
    """
    if not settings.has_elevenlabs:
        return None
    return ElevenLabsTTSBackend(
        api_key=settings.elevenlabs_api_key,
        voice_id=settings.elevenlabs_voice_id,
        model_id=settings.elevenlabs_model_id,
    )
