"""
Concrete ASR backend: Groq Whisper API.

Why Groq over local whisper.cpp on CPU-only VPS:
- whisper.cpp medium on ARM CPU: ~0.3× realtime (30min audio = 100min processing)
- Groq whisper-large-v3-turbo: ~30× realtime (30min audio = ~1min processing)
- Free tier is rate-limited but sufficient for personal automation
- Word timestamps via timestamp_granularities=["word"]
- Zero GPU, zero model downloads, zero CUDA
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import httpx
import structlog

from .contracts import ASRBackend, Transcript, TranscriptSegment, WordTimestamp

logger = structlog.get_logger(__name__)


class GroqASRBackend(ASRBackend):
    """Groq-hosted Whisper API with word-level timestamps."""

    def __init__(self, api_key: str, model: str = "whisper-large-v3-turbo"):
        self.api_key = api_key
        self.model = model
        self.base_url = "https://api.groq.com/openai/v1/audio/transcriptions"

    async def transcribe(self, audio_path: Path) -> Transcript:
        """Transcribe audio via Groq API with word timestamps."""
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        file_size_mb = audio_path.stat().st_size / 1e6
        logger.info("groq_transcribe_start", file=str(audio_path), size_mb=round(file_size_mb, 1))

        # Groq has a 25MB file limit. Split if needed.
        if file_size_mb > 24:
            return await self._transcribe_chunked(audio_path)

        async with httpx.AsyncClient(timeout=300) as client:
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
                retry_after = int(response.headers.get("retry-after", "30"))
                logger.warning("groq_rate_limited", retry_after=retry_after)
                await asyncio.sleep(retry_after)
                return await self.transcribe(audio_path)

            if response.status_code != 200:
                raise RuntimeError(f"Groq API error {response.status_code}: {response.text[:300]}")

            data = response.json()

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

        transcript = Transcript(
            segments=tuple(segments),
            language=data.get("language", "en"),
            duration=data.get("duration", 0.0),
        )

        logger.info("groq_transcribe_done", segments=len(segments), duration=transcript.duration)
        return transcript

    async def _transcribe_chunked(self, audio_path: Path) -> Transcript:
        """Split large files into 20MB chunks and transcribe sequentially."""
        import subprocess, tempfile

        chunk_dir = Path(tempfile.mkdtemp(prefix="clipforge_chunks_"))
        # Split into 10-minute chunks (~20MB each for typical audio)
        subprocess.run([
            "ffmpeg", "-y", "-i", str(audio_path),
            "-f", "segment", "-segment_time", "600",
            "-c:a", "libmp3lame", "-b:a", "128k",
            str(chunk_dir / "chunk_%03d.mp3"),
        ], capture_output=True, timeout=600)

        chunks = sorted(chunk_dir.glob("chunk_*.mp3"))
        all_segments = []
        offset = 0.0

        for chunk in chunks:
            result = await self.transcribe(chunk)
            for seg in result.segments:
                all_segments.append(TranscriptSegment(
                    start=seg.start + offset,
                    end=seg.end + offset,
                    text=seg.text,
                    words=tuple(
                        WordTimestamp(start=w.start + offset, end=w.end + offset, word=w.word)
                        for w in seg.words
                    ),
                ))
            offset += result.duration
            chunk.unlink()

        chunk_dir.rmdir()
        return Transcript(segments=tuple(all_segments), language="en", duration=offset)

    async def health_check(self) -> bool:
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                r = await client.get(
                    "https://api.groq.com/openai/v1/models",
                    headers={"Authorization": f"Bearer {self.api_key}"},
                )
                return r.status_code == 200
        except Exception:
            return False
