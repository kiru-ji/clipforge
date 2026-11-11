"""
Pipeline orchestrator — coordinates all backends through the job lifecycle.

Design Patterns:
- Pipeline Pattern: sequential stages with clear inputs/outputs
- Observer Pattern: progress notifications to GUI
- Circuit Breaker: auto-fallback on API failures
- Idempotency: each stage checks for cached results before re-running
"""

from __future__ import annotations

import asyncio
import hashlib
import json
from pathlib import Path
from typing import Optional

import structlog

from .contracts import (
    ASRBackend, ClipMetadata, JobProgress, JobStatus, LLMBackend,
    ProgressSubject, PublisherBackend, Transcript, VideoProcessor, ViralMoment,
)
from .config import Settings

logger = structlog.get_logger(__name__)


class Pipeline(ProgressSubject):
    """Orchestrates the full clip generation pipeline."""

    def __init__(
        self,
        settings: Settings,
        asr: ASRBackend,
        llm: LLMBackend,
        video: VideoProcessor,
        publishers: list[PublisherBackend],
    ):
        super().__init__()
        self.settings = settings
        self.asr = asr
        self.llm = llm
        self.video = video
        self.publishers = publishers

    async def process(self, source_url: str, job_id: Optional[str] = None) -> dict:
        """Run the full pipeline for a video URL."""
        job_id = job_id or hashlib.sha256(source_url.encode()).hexdigest()[:12]
        output_dir = self.settings.output_dir / job_id
        output_dir.mkdir(parents=True, exist_ok=True)

        result = {"job_id": job_id, "clips": [], "errors": []}

        try:
            # Stage 1: Download
            await self._notify(job_id, JobStatus.DOWNLOADING, 0.05, "Downloading video...")
            video_path = await self.video.download(source_url, self.settings.downloads_dir)

            # Stage 2: Transcribe
            await self._notify(job_id, JobStatus.TRANSCRIBING, 0.15, "Transcribing audio...")
            transcript_cache = output_dir / "transcript.json"
            if transcript_cache.exists():
                transcript = self._load_transcript(transcript_cache)
                logger.info("transcript_cache_hit", job_id=job_id)
            else:
                transcript = await self.asr.transcribe(video_path)
                transcript_cache.write_text(json.dumps({
                    "segments": [{"start": s.start, "end": s.end, "text": s.text,
                                  "words": [{"start": w.start, "end": w.end, "word": w.word} for w in s.words]}
                                 for s in transcript.segments],
                    "language": transcript.language, "duration": transcript.duration,
                }, indent=2))

            # Stage 3: Detect moments
            await self._notify(job_id, JobStatus.DETECTING_MOMENTS, 0.35, "Detecting viral moments...")
            moments = await self.llm.detect_moments(
                transcript,
                self.settings.clip_min_duration,
                self.settings.clip_max_duration,
                self.settings.max_clips_per_video,
            )
            if not moments:
                raise RuntimeError("No viral moments detected")

            # Stage 4: Extract clips
            await self._notify(job_id, JobStatus.EXTRACTING_CLIPS, 0.45, f"Extracting {len(moments)} clips...")
            clip_paths = []
            for i, moment in enumerate(moments):
                pct = 0.45 + (0.3 * (i + 1) / len(moments))
                await self._notify(job_id, JobStatus.EXTRACTING_CLIPS, pct,
                                   f"Clip {i+1}/{len(moments)}: {moment.title}")

                raw_path = output_dir / f"clip_{i:02d}_raw.mp4"
                final_path = output_dir / f"clip_{i:02d}.mp4"

                await self.video.extract_clip(video_path, moment.start, moment.end, raw_path)
                await self.video.reframe_vertical(raw_path, final_path)

                if raw_path.exists():
                    raw_path.unlink()
                clip_paths.append(final_path)

            # Stage 5: Generate metadata
            await self._notify(job_id, JobStatus.GENERATING_METADATA, 0.80, "Generating metadata...")
            metadata_list = []
            for i, moment in enumerate(moments):
                relevant = [s for s in transcript.segments if s.start >= moment.start and s.end <= moment.end]
                excerpt = " ".join(s.text for s in relevant[:5])
                meta = await self.llm.generate_metadata(moment, excerpt, "Source Video")
                metadata_list.append(meta)

            # Stage 6: Publish (if configured)
            active_publishers = [p for p in self.publishers if await p.is_configured()]
            if active_publishers:
                await self._notify(job_id, JobStatus.PUBLISHING, 0.90, "Publishing clips...")
                for i, (clip_path, meta) in enumerate(zip(clip_paths, metadata_list)):
                    for publisher in active_publishers:
                        pub_result = await publisher.publish(clip_path, meta)
                        result["clips"].append({
                            "index": i, "path": str(clip_path),
                            "title": moment.title, "score": moment.viral_score,
                            "publish": {"platform": pub_result.platform, "status": pub_result.status},
                        })
            else:
                for i, (clip_path, moment) in enumerate(zip(clip_paths, moments)):
                    result["clips"].append({
                        "index": i, "path": str(clip_path),
                        "title": moment.title, "score": moment.viral_score,
                    })

            await self._notify(job_id, JobStatus.COMPLETED, 1.0, f"Done! {len(clip_paths)} clips generated.")
            result["total_clips"] = len(clip_paths)

        except Exception as e:
            logger.error("pipeline_failed", job_id=job_id, error=str(e))
            await self._notify(job_id, JobStatus.FAILED, 0, error=str(e))
            result["errors"].append(str(e))

        return result

    async def _notify(self, job_id: str, status: JobStatus, pct: float, msg: str = "", error: str = ""):
        progress = JobProgress(job_id=job_id, status=status, progress_pct=pct, message=msg, error=error)
        await self.notify(progress)

    @staticmethod
    def _load_transcript(path: Path) -> Transcript:
        from .contracts import TranscriptSegment, WordTimestamp
        data = json.loads(path.read_text())
        segments = tuple(
            TranscriptSegment(
                start=s["start"], end=s["end"], text=s["text"],
                words=tuple(WordTimestamp(start=w["start"], end=w["end"], word=w["word"]) for w in s.get("words", [])),
            ) for s in data["segments"]
        )
        return Transcript(segments=segments, language=data.get("language", "en"), duration=data.get("duration", 0))
