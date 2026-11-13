"""
Groq Whisper ASR backend.

Fixes applied (vs audit):
- J29/P24: tenacity retry with max_attempts=3 + exponential backoff (no infinite recursion)
- J30: ffmpeg chunk failure raises RuntimeError (not silent empty transcript)
- J31: async subprocess for chunking (no event-loop blocking)
- J32: safe int parsing of retry-after header with fallback
- J33: try/finally ensures chunk temp files are always cleaned up
- J34/P25: offset uses exact segment_time (600s), not API-reported duration
- J36/P26: language preserved from first chunk, not hardcoded "en"
- P27/J37: extract_audio() called by pipeline before transcribe (16kHz mono mp3)
- J38: comment corrected (25MB Groq limit, 24MB threshold)
- J29: health_check distinguishes invalid-key from network-down
"""

from __future__ import annotations

import asyncio
import shutil
import tempfile
from pathlib import Path

import httpx
import structlog
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from .contracts import ASRBackend, Transcript, TranscriptSegment, WordTimestamp

logger = structlog.get_logger(__name__)

# Groq file size limit is 25MB; we use 24MB as safe threshold
_MAX_FILE_SIZE_BYTES = 24 * 1024 * 1024
_CHUNK_SEGMENT_SECONDS = 600  # 10 minutes per chunk


class RateLimitError(Exception):
    """Raised when Groq returns 429 after all retries exhausted."""


class GroqASRBackend(ASRBackend):
    """Groq-hosted Whisper API with word-level timestamps."""

    def __init__(self, api_key: str, model: str = "whisper-large-v3-turbo",
                 local_address: str | None = None):
        self.api_key = api_key
        self.model = model
        self.base_url = "https://api.groq.com/openai/v1/audio/transcriptions"
        # Optional egress pinning: bind outbound sockets to one source IP.
        # On this VPS the default route exits via a WireGuard VPN whose IP Groq
        # rejects with 403 "Access denied. Please check your network settings.";
        # binding to the host's own address (policy-routed direct) returns 200.
        self._transport = (
            httpx.AsyncHTTPTransport(local_address=local_address) if local_address else None
        )

    def _client(self, timeout: float) -> httpx.AsyncClient:
        kw: dict = {"timeout": timeout}
        if self._transport is not None:
            kw["transport"] = self._transport
        return httpx.AsyncClient(**kw)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=5, min=5, max=60),
        retry=retry_if_exception_type(RateLimitError),
        reraise=True,
    )
    async def _transcribe_file(self, audio_path: Path) -> dict:
        """Transcribe a single file (<24MB) via Groq API with retry."""
        async with self._client(300) as client:
            with open(audio_path, "rb") as f:
                response = await client.post(
                    self.base_url,
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    files={"file": (audio_path.name, f, "audio/mpeg")},
                    data={
                        "model": self.model,
                        "response_format": "verbose_json",
                        "timestamp_granularities[]": ["word", "segment"],
                    },
                )

            if response.status_code == 429:
                # J32: safe parsing of retry-after
                retry_after = response.headers.get("retry-after", "30")
                try:
                    wait_secs = int(retry_after)
                except (ValueError, TypeError):
                    wait_secs = 30
                logger.warning("groq_rate_limited", retry_after=wait_secs)
                await asyncio.sleep(min(wait_secs, 60))
                raise RateLimitError(f"Rate limited, retry after {wait_secs}s")

            if response.status_code == 401:
                raise RuntimeError("Groq API key is invalid (401)")

            if response.status_code != 200:
                raise RuntimeError(f"Groq API error {response.status_code}: {response.text[:300]}")

            return response.json()

    async def transcribe(self, audio_path: Path, max_duration_hint: float = 0) -> Transcript:
        """Transcribe audio file. Auto-chunks if >24MB."""
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        file_size = audio_path.stat().st_size
        logger.info("groq_transcribe_start", file=str(audio_path), size_mb=round(file_size / 1e6, 1))

        if file_size <= _MAX_FILE_SIZE_BYTES:
            data = await self._transcribe_file(audio_path)
            return self._parse_response(data)
        else:
            return await self._transcribe_chunked(audio_path)

    async def _transcribe_chunked(self, audio_path: Path) -> Transcript:
        """Split large files into chunks and transcribe sequentially."""
        chunk_dir = Path(tempfile.mkdtemp(prefix="clipforge_chunks_"))
        try:
            # J31: async subprocess (no event-loop blocking)
            proc = await asyncio.create_subprocess_exec(
                "ffmpeg", "-y", "-i", str(audio_path),
                "-f", "segment", "-segment_time", str(_CHUNK_SEGMENT_SECONDS),
                "-c:a", "libmp3lame", "-b:a", "128k", "-ar", "16000", "-ac", "1",
                str(chunk_dir / "chunk_%03d.mp3"),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            _, stderr = await asyncio.wait_for(proc.communicate(), timeout=600)

            # J30: check returncode — ffmpeg failure must NOT produce silent empty transcript
            if proc.returncode != 0:
                raise RuntimeError(f"FFmpeg chunk split failed: {stderr.decode()[-300:]}")

            chunks = sorted(chunk_dir.glob("chunk_*.mp3"))
            if not chunks:
                raise RuntimeError("FFmpeg produced zero chunks")

            all_segments: list[TranscriptSegment] = []
            detected_language = "en"
            # J34/P25: use exact segment time for offset, not API-reported duration
            offset = 0.0

            for i, chunk in enumerate(chunks):
                logger.info("transcribing_chunk", chunk=i + 1, total=len(chunks))
                data = await self._transcribe_file(chunk)

                if i == 0:
                    detected_language = data.get("language", "en")

                for seg in data.get("segments", []):
                    words = tuple(
                        WordTimestamp(start=w["start"] + offset, end=w["end"] + offset, word=w["word"].strip())
                        for w in seg.get("words", [])
                    )
                    all_segments.append(TranscriptSegment(
                        start=seg["start"] + offset,
                        end=seg["end"] + offset,
                        text=seg["text"].strip(),
                        words=words,
                    ))

                # J34: advance by exact segment time, not API duration
                offset += _CHUNK_SEGMENT_SECONDS

            logger.info("groq_chunked_done", chunks=len(chunks), segments=len(all_segments))
            return Transcript(
                segments=tuple(all_segments),
                language=detected_language,  # J36: preserved, not hardcoded
                duration=offset,
            )
        finally:
            # J33: always clean up, even on exception
            shutil.rmtree(chunk_dir, ignore_errors=True)

    @staticmethod
    def _parse_response(data: dict) -> Transcript:
        """Parse Groq verbose_json response into Transcript DTO."""
        segments = []
        for seg in data.get("segments", []):
            words = tuple(
                WordTimestamp(start=w["start"], end=w["end"], word=w["word"].strip())
                for w in seg.get("words", [])
            )
            segments.append(TranscriptSegment(
                start=seg["start"],
                end=seg["end"],
                text=seg["text"].strip(),
                words=words,
            ))
        return Transcript(
            segments=tuple(segments),
            language=data.get("language", "en"),
            duration=data.get("duration", 0.0),
        )

    async def health_check(self) -> bool:
        """J29: distinguish invalid key from network down."""
        try:
            async with self._client(10) as client:
                r = await client.get(
                    "https://api.groq.com/openai/v1/models",
                    headers={"Authorization": f"Bearer {self.api_key}"},
                )
                if r.status_code == 401:
                    logger.error("groq_health_invalid_key")
                    return False
                return r.status_code == 200
        except httpx.ConnectError:
            logger.error("groq_health_network_down")
            return False
        except Exception as e:
            logger.error("groq_health_unknown", error=str(e))
            return False
