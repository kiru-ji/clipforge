"""
Abstract base classes and data transfer objects for ClipForge.

Design Patterns:
- Strategy: swappable backends via abstract interfaces
- Observer: progress notifications (with return_exceptions for safety)
- State Machine: JobStatus with guarded transitions

Fixes applied (vs audit):
- P40/J7: WordTimestamp defined BEFORE TranscriptSegment (no forward-ref)
- J5: notify() uses return_exceptions=True so one flaky observer can't kill the job
- J6: detach() is safe for never-attached observers
- J8: removed unused imports (field, Any, Optional)
- J10: viral_score clamped to 1-10 in ViralMoment constructor
- J41: ViralMoment validates start>=0, end>start
- P41: PublishResult.status is now a proper Enum
- J9: CANCELLED is now emitted by the job manager
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from pathlib import Path


# ─── Data Transfer Objects (immutable value objects) ────────────────────────

@dataclass(frozen=True)
class WordTimestamp:
    """Word-level timestamp for precise subtitle alignment."""
    start: float
    end: float
    word: str


@dataclass(frozen=True)
class TranscriptSegment:
    """A single segment of transcription with optional word-level timestamps."""
    start: float
    end: float
    text: str
    words: tuple[WordTimestamp, ...] = ()


@dataclass(frozen=True)
class Transcript:
    """Complete transcription result."""
    segments: tuple[TranscriptSegment, ...]
    language: str
    duration: float


@dataclass(frozen=True)
class ViralMoment:
    """A detected viral-worthy moment. Immutable + validated."""
    start: float
    end: float
    title: str
    hook: str
    viral_score: int  # clamped to 1-10
    reason: str

    def __post_init__(self):
        if self.start < 0:
            raise ValueError(f"start must be >= 0, got {self.start}")
        if self.end <= self.start:
            raise ValueError(f"end ({self.end}) must be > start ({self.start})")
        object.__setattr__(self, "viral_score", max(1, min(10, self.viral_score)))

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


class PublishStatus(str, Enum):
    """Result of a publish attempt."""
    PUBLISHED = "published"
    SKIPPED = "skipped"
    ERROR = "error"


@dataclass(frozen=True)
class PublishResult:
    """Result of publishing to a platform."""
    platform: str
    status: PublishStatus
    url: str = ""
    error: str = ""


# ─── Job State Machine ──────────────────────────────────────────────────────

class JobStatus(str, Enum):
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


_VALID_TRANSITIONS: dict[JobStatus, frozenset[JobStatus]] = {
    JobStatus.QUEUED: frozenset({JobStatus.DOWNLOADING, JobStatus.CANCELLED, JobStatus.FAILED}),
    JobStatus.DOWNLOADING: frozenset({JobStatus.TRANSCRIBING, JobStatus.FAILED, JobStatus.CANCELLED}),
    JobStatus.TRANSCRIBING: frozenset({JobStatus.DETECTING_MOMENTS, JobStatus.FAILED, JobStatus.CANCELLED}),
    JobStatus.DETECTING_MOMENTS: frozenset({JobStatus.EXTRACTING_CLIPS, JobStatus.FAILED, JobStatus.CANCELLED}),
    JobStatus.EXTRACTING_CLIPS: frozenset({JobStatus.GENERATING_METADATA, JobStatus.FAILED, JobStatus.CANCELLED}),
    JobStatus.GENERATING_METADATA: frozenset({JobStatus.PUBLISHING, JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED}),
    JobStatus.PUBLISHING: frozenset({JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED}),
    JobStatus.COMPLETED: frozenset(),
    JobStatus.FAILED: frozenset(),
    JobStatus.CANCELLED: frozenset(),
}


@dataclass
class JobProgress:
    """Observable progress for a job."""
    job_id: str
    status: JobStatus
    progress_pct: float = 0.0
    message: str = ""
    clips_generated: int = 0
    total_clips: int = 0
    error: str = ""


# ─── Abstract Backends (Strategy Pattern) ───────────────────────────────────

class ASRBackend(ABC):
    @abstractmethod
    async def transcribe(self, audio_path: Path, max_duration_hint: float = 0) -> Transcript:
        ...

    @abstractmethod
    async def health_check(self) -> bool:
        ...


class LLMBackend(ABC):
    @abstractmethod
    async def detect_moments(
        self, transcript: Transcript, min_duration: int, max_duration: int, max_clips: int
    ) -> list[ViralMoment]:
        ...

    @abstractmethod
    async def generate_metadata(
        self, moment: ViralMoment, transcript_excerpt: str, source_title: str
    ) -> ClipMetadata:
        ...

    @abstractmethod
    async def health_check(self) -> bool:
        ...


class VideoProcessor(ABC):
    @abstractmethod
    async def download(self, url: str, job_download_dir: Path) -> tuple[Path, str]:
        """Returns (video_path, source_title)."""

    @abstractmethod
    async def extract_and_reframe(
        self, source: Path, start: float, end: float, output: Path,
        transcript_segments: tuple[TranscriptSegment, ...], burn_subs: bool,
    ) -> Path:
        """Extract clip, reframe to 9:16, optionally burn subtitles — single encode pass."""

    @abstractmethod
    async def extract_audio(self, video_path: Path, output: Path) -> Path:
        """Extract 16kHz mono MP3 audio (for ASR upload efficiency)."""


class PublisherBackend(ABC):
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


# ─── Observer Pattern ───────────────────────────────────────────────────────

class ProgressObserver(ABC):
    @abstractmethod
    async def on_progress(self, progress: JobProgress) -> None:
        ...


class ProgressSubject:
    """Subject that notifies observers. return_exceptions prevents cascade failures."""

    def __init__(self):
        self._observers: list[ProgressObserver] = []

    def attach(self, observer: ProgressObserver) -> None:
        if observer not in self._observers:
            self._observers.append(observer)

    def detach(self, observer: ProgressObserver) -> None:
        try:
            self._observers.remove(observer)
        except ValueError:
            pass

    async def notify(self, progress: JobProgress) -> None:
        await asyncio.gather(
            *(o.on_progress(progress) for o in list(self._observers)),
            return_exceptions=True,
        )
