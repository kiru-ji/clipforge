"""
Abstract base classes for all pluggable backends.

Design Pattern: Strategy + Factory
- Each backend (ASR, LLM, Publisher) has an abstract interface
- Concrete implementations are swapped via config without changing pipeline code
- Factory creates the right backend from settings at runtime
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Optional


# ─── Data Transfer Objects (immutable value objects) ────────────────────────

@dataclass(frozen=True)
class TranscriptSegment:
    """A single segment of transcription with optional word-level timestamps."""
    start: float
    end: float
    text: str
    words: tuple[WordTimestamp, ...] = ()

@dataclass(frozen=True)
class WordTimestamp:
    """Word-level timestamp for precise subtitle alignment."""
    start: float
    end: float
    word: str

@dataclass(frozen=True)
class Transcript:
    """Complete transcription result."""
    segments: tuple[TranscriptSegment, ...]
    language: str
    duration: float

@dataclass(frozen=True)
class ViralMoment:
    """A detected viral-worthy moment in the video."""
    start: float
    end: float
    title: str
    hook: str
    viral_score: int  # 1-10
    reason: str

    @property
    def duration(self) -> float:
        return self.end - self.start

@dataclass(frozen=True)
class ClipMetadata:
    """Platform-specific metadata for a clip."""
    youtube_title: str
    youtube_description: str
    instagram_caption: str
    tiktok_caption: str
    hashtags: tuple[str, ...]
    thumbnail_text: str

@dataclass(frozen=True)
class PublishResult:
    """Result of publishing to a platform."""
    platform: str
    status: str  # "published", "skipped", "error"
    url: str = ""
    error: str = ""


# ─── Job State Machine ──────────────────────────────────────────────────────

class JobStatus(str, Enum):
    """Job lifecycle states (finite state machine)."""
    QUEUED = "queued"
    DOWNLOADING = "downloading"
    TRANSCRIBING = "transcribing"
    DETECTING_MOMENTS = "detecting_moments"
    EXTRACTING_CLIPS = "extracting_clips"
    GENERATING_METADATA = "generating_metadata"
    PUBLISHING = "publishing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

@dataclass
class JobProgress:
    """Observable progress for a job (Observer pattern)."""
    job_id: str
    status: JobStatus
    progress_pct: float = 0.0
    message: str = ""
    clips_generated: int = 0
    total_clips: int = 0
    error: str = ""


# ─── Abstract Backends (Strategy Pattern) ───────────────────────────────────

class ASRBackend(ABC):
    """Abstract Speech-to-Text backend."""

    @abstractmethod
    async def transcribe(self, audio_path: Path) -> Transcript:
        """Transcribe audio file to text with word-level timestamps."""
        ...

    @abstractmethod
    async def health_check(self) -> bool:
        """Check if the backend is available."""
        ...


class LLMBackend(ABC):
    """Abstract LLM backend for moment detection and metadata generation."""

    @abstractmethod
    async def detect_moments(
        self, transcript: Transcript, min_duration: int, max_duration: int, max_clips: int
    ) -> list[ViralMoment]:
        """Identify viral-worthy moments from transcript."""
        ...

    @abstractmethod
    async def generate_metadata(
        self, moment: ViralMoment, transcript_excerpt: str, source_title: str
    ) -> ClipMetadata:
        """Generate platform-specific metadata for a clip."""
        ...

    @abstractmethod
    async def health_check(self) -> bool:
        ...


class VideoProcessor(ABC):
    """Abstract video processing backend."""

    @abstractmethod
    async def download(self, url: str, output_dir: Path) -> Path:
        """Download video from URL."""
        ...

    @abstractmethod
    async def extract_clip(
        self, source: Path, start: float, end: float, output: Path
    ) -> Path:
        """Extract a time-range clip from source video."""
        ...

    @abstractmethod
    async def reframe_vertical(self, source: Path, output: Path) -> Path:
        """Reframe landscape video to 9:16 vertical."""
        ...

    @abstractmethod
    async def burn_subtitles(self, video: Path, srt: Path, output: Path) -> Path:
        """Burn SRT subtitles into video."""
        ...


class PublisherBackend(ABC):
    """Abstract publishing backend for a single platform."""

    @property
    @abstractmethod
    def platform_name(self) -> str:
        ...

    @abstractmethod
    async def publish(self, video_path: Path, metadata: ClipMetadata) -> PublishResult:
        ...

    @abstractmethod
    async def is_configured(self) -> bool:
        ...


# ─── Observer Pattern for Job Progress ──────────────────────────────────────

class ProgressObserver(ABC):
    """Observer interface for job progress updates."""

    @abstractmethod
    async def on_progress(self, progress: JobProgress) -> None:
        ...


class ProgressSubject:
    """Subject that notifies observers of progress changes."""

    def __init__(self):
        self._observers: list[ProgressObserver] = []

    def attach(self, observer: ProgressObserver) -> None:
        self._observers.append(observer)

    def detach(self, observer: ProgressObserver) -> None:
        self._observers.remove(observer)

    async def notify(self, progress: JobProgress) -> None:
        await asyncio.gather(*(o.on_progress(progress) for o in self._observers))
