"""
Pipeline orchestrator — coordinates all backends through the job lifecycle.

Fixes applied (vs audit):
- P01/J64: publish stage uses moments[i] not stale loop variable
- P02/J55/P34: per-job download dir + cleanup after completion
- P03/J70: source_title captured from yt-dlp and passed to metadata generation
- P04/J56: SRT generated and burned via extract_and_reframe (single pass)
- P05/J65: asyncio.Lock prevents concurrent runs on same job_id
- P06/J59: single-pass extract+reframe (one encode, not two)
- P07/J53: all subprocess calls have timeouts (enforced in video.py)
- P08/J9: JobProgress.clips_generated/total_clips populated; CANCELLED supported
- P09/J66: transcript cache includes content hash for invalidation
- P10/J40: LLM errors propagate as RuntimeError (not masked as "no moments")
- J67: atomic transcript cache write (write to temp then rename)
- J68/J82: errors list checked by GUI; FAILED status set on any error
- J69: raw clip cleanup in finally block
- J71/J72: publisher errors caught per-platform, don't abort whole job
- J73: total_clips = len(clip_paths), clips list has one entry per clip
- J76: excerpt includes boundary-straddling segments
- J77: _load_transcript validates schema
- J79: max_source_duration_min enforced before processing
- J80: per-clip sub-progress during extraction
- J85: pipeline runs as app-scoped background task (not client-scoped)
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import tempfile
from pathlib import Path

import structlog

from .contracts import (
    ASRBackend, ClipMetadata, JobProgress, JobStatus, LLMBackend,
    ProgressSubject, PublisherBackend, Transcript, TranscriptSegment,
    VideoProcessor, ViralMoment,
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
        # P05/J65: lock prevents concurrent runs on same job_id
        self._locks: dict[str, asyncio.Lock] = {}
        self._global_lock = asyncio.Lock()

    async def _get_job_lock(self, job_id: str) -> asyncio.Lock:
        async with self._global_lock:
            if job_id not in self._locks:
                self._locks[job_id] = asyncio.Lock()
            return self._locks[job_id]

    async def process(self, source_url: str, job_id: str | None = None) -> dict:
        """Run the full pipeline. Returns result dict with clips/errors."""
        job_id = job_id or hashlib.sha256(source_url.encode()).hexdigest()[:12]

        # P05: prevent concurrent runs on same URL
        lock = await self._get_job_lock(job_id)
        if lock.locked():
            return {"job_id": job_id, "clips": [], "errors": ["Job already running"]}

        async with lock:
            return await self._process_locked(source_url, job_id)

    async def _process_locked(self, source_url: str, job_id: str) -> dict:
        job_dir = self.settings.output_dir / job_id
        download_dir = self.settings.downloads_dir / job_id  # J51: per-job dir
        job_dir.mkdir(parents=True, exist_ok=True)
        download_dir.mkdir(parents=True, exist_ok=True)

        result: dict = {"job_id": job_id, "clips": [], "errors": [], "total_clips": 0}

        try:
            # ── Stage 1: Download ───────────────────────────────────────────
            await self._notify(job_id, JobStatus.DOWNLOADING, 0.05, "Downloading video...")
            video_path, source_title = await self.video.download(source_url, download_dir)

            # J79: enforce max duration
            probe_dur = await self._get_duration(video_path)
            max_min = self.settings.max_source_duration_min
            if probe_dur > max_min * 60:
                raise RuntimeError(f"Video too long ({probe_dur/60:.0f}min > {max_min}min limit)")

            # ── Stage 2: Transcribe ─────────────────────────────────────────
            await self._notify(job_id, JobStatus.TRANSCRIBING, 0.15, "Transcribing audio...")
            transcript_cache = job_dir / "transcript.json"

            # P09/J66: cache key includes content hash
            content_hash = hashlib.md5(video_path.read_bytes()[:1024 * 1024]).hexdigest()[:8]
            cache_valid = False
            if transcript_cache.exists():
                try:
                    cached = json.loads(transcript_cache.read_text())
                    if cached.get("content_hash") == content_hash:
                        cache_valid = True
                except (json.JSONDecodeError, KeyError):
                    pass

            if cache_valid:
                transcript = self._load_transcript(transcript_cache)
                logger.info("transcript_cache_hit", job_id=job_id)
            else:
                # P27: extract audio first for efficient upload
                audio_path = job_dir / "audio.mp3"
                await self.video.extract_audio(video_path, audio_path)
                transcript = await self.asr.transcribe(audio_path, max_duration_hint=probe_dur)
                # Clean up audio file
                if audio_path.exists():
                    audio_path.unlink()

                # J67: atomic cache write
                cache_data = {
                    "content_hash": content_hash,
                    "segments": [
                        {"start": s.start, "end": s.end, "text": s.text,
                         "words": [{"start": w.start, "end": w.end, "word": w.word} for w in s.words]}
                        for s in transcript.segments
                    ],
                    "language": transcript.language,
                    "duration": transcript.duration,
                }
                tmp_fd, tmp_path = tempfile.mkstemp(dir=str(job_dir), suffix=".json")
                try:
                    Path(tmp_path).write_text(json.dumps(cache_data, indent=2))
                    Path(tmp_path).rename(transcript_cache)
                finally:
                    Path(tmp_path).unlink(missing_ok=True)

            # ── Stage 3: Detect moments ─────────────────────────────────────
            await self._notify(job_id, JobStatus.DETECTING_MOMENTS, 0.35, "Detecting viral moments...")
            # P10/J40: errors propagate (not swallowed)
            moments = await self.llm.detect_moments(
                transcript,
                self.settings.clip_min_duration,
                self.settings.clip_max_duration,
                self.settings.max_clips_per_video,
            )
            if not moments:
                raise RuntimeError("No viral moments detected — try a different video or adjust duration settings")

            # ── Stage 4: Extract clips (single pass each) ───────────────────
            await self._notify(job_id, JobStatus.EXTRACTING_CLIPS, 0.45, f"Extracting {len(moments)} clips...")
            clip_paths: list[Path] = []

            for i, moment in enumerate(moments):
                pct = 0.45 + (0.30 * (i + 1) / len(moments))
                await self._notify(job_id, JobStatus.EXTRACTING_CLIPS, pct,
                                   f"Clip {i+1}/{len(moments)}: {moment.title}",
                                   clips_generated=i, total_clips=len(moments))

                final_path = job_dir / f"clip_{i:02d}.mp4"

                # J76: include boundary-straddling segments
                relevant_segs = tuple(
                    s for s in transcript.segments
                    if s.end > moment.start and s.start < moment.end
                )

                await self.video.extract_and_reframe(
                    video_path, moment.start, moment.end, final_path,
                    relevant_segs, self.settings.burn_subtitles,
                )
                clip_paths.append(final_path)

            # ── Stage 5: Generate metadata ──────────────────────────────────
            await self._notify(job_id, JobStatus.GENERATING_METADATA, 0.80, "Generating metadata...")
            metadata_list: list[ClipMetadata] = []
            for i, moment in enumerate(moments):
                # J76: broader excerpt including boundary segments
                relevant = [s for s in transcript.segments if s.end > moment.start and s.start < moment.end]
                excerpt = " ".join(s.text for s in relevant[:10])
                meta = await self.llm.generate_metadata(moment, excerpt, source_title)  # P03: real title
                metadata_list.append(meta)

            # ── Stage 6: Publish ────────────────────────────────────────────
            active_publishers = [p for p in self.publishers if await p.is_configured()]
            if active_publishers:
                await self._notify(job_id, JobStatus.PUBLISHING, 0.90, "Publishing clips...")
                for i, (clip_path, meta) in enumerate(zip(clip_paths, metadata_list)):
                    for publisher in active_publishers:
                        try:
                            # J72: per-platform error handling
                            pub_result = await publisher.publish(clip_path, meta)
                            result["clips"].append({
                                "index": i, "path": str(clip_path),
                                "title": moments[i].title,  # P01/J64: correct index
                                "score": moments[i].viral_score,
                                "publish": {"platform": pub_result.platform, "status": pub_result.status.value},
                            })
                        except Exception as e:
                            logger.error("publish_failed", platform=publisher.platform_name, error=str(e))
                            result["clips"].append({
                                "index": i, "path": str(clip_path),
                                "title": moments[i].title,
                                "score": moments[i].viral_score,
                                "publish": {"platform": publisher.platform_name, "status": "error", "error": str(e)},
                            })
            else:
                for i, (clip_path, moment) in enumerate(zip(clip_paths, moments)):
                    result["clips"].append({
                        "index": i, "path": str(clip_path),
                        "title": moment.title, "score": moment.viral_score,
                    })

            result["total_clips"] = len(clip_paths)  # J73: consistent count
            await self._notify(job_id, JobStatus.COMPLETED, 1.0,
                               f"Done! {len(clip_paths)} clips generated.",
                               clips_generated=len(clip_paths), total_clips=len(clip_paths))

        except Exception as e:
            logger.error("pipeline_failed", job_id=job_id, error=str(e))
            result["errors"].append(str(e))
            await self._notify(job_id, JobStatus.FAILED, 0, error=str(e))

        finally:
            # P02/J55: clean up downloaded source after processing
            if download_dir.exists():
                import shutil
                shutil.rmtree(download_dir, ignore_errors=True)

        return result

    async def _notify(self, job_id: str, status: JobStatus, pct: float,
                      msg: str = "", error: str = "",
                      clips_generated: int = 0, total_clips: int = 0):
        progress = JobProgress(
            job_id=job_id, status=status, progress_pct=pct,
            message=msg, error=error,
            clips_generated=clips_generated, total_clips=total_clips,
        )
        await self.notify(progress)

    @staticmethod
    def _load_transcript(path: Path) -> Transcript:
        """J77: validate schema before constructing DTOs."""
        data = json.loads(path.read_text())
        if not isinstance(data, dict) or "segments" not in data:
            raise ValueError(f"Invalid transcript cache format: {path}")

        segments = []
        for s in data["segments"]:
            words = tuple(
                WordTimestamp(start=w["start"], end=w["end"], word=w["word"])
                for w in s.get("words", [])
            )
            segments.append(TranscriptSegment(
                start=float(s["start"]), end=float(s["end"]),
                text=str(s["text"]), words=words,
            ))
        return Transcript(
            segments=tuple(segments),
            language=data.get("language", "en"),
            duration=float(data.get("duration", 0)),
        )

    @staticmethod
    async def _get_duration(video_path: Path) -> float:
        """Get video duration via ffprobe."""
        import asyncio, json
        cmd = ["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
               "-print_format", "json", str(video_path)]
        proc = await asyncio.create_subprocess_exec(*cmd, stdout=asyncio.subprocess.PIPE)
        stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=30)
        data = json.loads(stdout)
        return float(data.get("format", {}).get("duration", 0))


# Import here to avoid circular dependency at module level
from .contracts import WordTimestamp
