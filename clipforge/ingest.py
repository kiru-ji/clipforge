"""Video ingestion: download from URL or accept local file."""

from __future__ import annotations

import hashlib
import subprocess
import json
from pathlib import Path
from typing import Optional

import structlog

logger = structlog.get_logger(__name__)


class Ingestor:
    """Download videos via yt-dlp or validate local files."""

    def __init__(self, downloads_dir: Path):
        self.downloads_dir = downloads_dir
        self.downloads_dir.mkdir(parents=True, exist_ok=True)

    def ingest(self, source: str, job_id: Optional[str] = None) -> dict:
        """
        Ingest a video from URL or local path.

        Returns dict with:
            - video_path: Path to the downloaded/local video
            - metadata: dict with title, duration, channel, etc.
            - job_id: unique identifier for this job
        """
        source_path = Path(source)

        if source_path.exists() and source_path.is_file():
            return self._ingest_local(source_path, job_id)
        else:
            return self._ingest_url(source, job_id)

    def _ingest_local(self, path: Path, job_id: Optional[str]) -> dict:
        """Validate and register a local video file."""
        if not path.suffix.lower() in {".mp4", ".mkv", ".webm", ".mov", ".avi"}:
            raise ValueError(f"Unsupported video format: {path.suffix}")

        metadata = self._probe_video(path)
        jid = job_id or self._make_job_id(str(path))

        logger.info("local_video_ingested", path=str(path), duration=metadata.get("duration"), job_id=jid)
        return {
            "video_path": path,
            "metadata": metadata,
            "job_id": jid,
        }

    def _ingest_url(self, url: str, job_id: Optional[str]) -> dict:
        """Download video via yt-dlp with robust error handling."""
        jid = job_id or self._make_job_id(url)
        output_template = str(self.downloads_dir / f"{jid}.%(ext)s")

        # First get metadata without downloading
        meta_cmd = [
            "yt-dlp", "--dump-json", "--no-download",
            "--no-warnings", url
        ]
        try:
            result = subprocess.run(meta_cmd, capture_output=True, text=True, timeout=60)
            if result.returncode != 0:
                raise RuntimeError(f"yt-dlp metadata failed: {result.stderr[:500]}")
            metadata = json.loads(result.stdout)
        except (subprocess.TimeoutExpired, json.JSONDecodeError) as e:
            logger.warning("metadata_fetch_failed", url=url, error=str(e))
            metadata = {"title": "unknown", "duration": 0}

        # Download with NVENC-friendly format selection
        dl_cmd = [
            "yt-dlp",
            "-f", "bestvideo[height<=1080][ext=mp4]+bestaudio[ext=m4a]/best[height<=1080][ext=mp4]/best",
            "--merge-output-format", "mp4",
            "-o", output_template,
            "--no-playlist",
            "--restrict-filenames",
            "--no-warnings",
            "--socket-timeout", "30",
            "--retries", "3",
            url,
        ]

        logger.info("download_started", url=url, job_id=jid)
        try:
            result = subprocess.run(dl_cmd, capture_output=True, text=True, timeout=3600)
            if result.returncode != 0:
                raise RuntimeError(f"yt-dlp download failed: {result.stderr[:500]}")
        except subprocess.TimeoutExpired:
            raise RuntimeError(f"Download timed out after 1 hour: {url}")

        # Find the downloaded file
        downloaded = list(self.downloads_dir.glob(f"{jid}.*"))
        if not downloaded:
            raise RuntimeError(f"No file found after download: {jid}")

        video_path = downloaded[0]
        probe = self._probe_video(video_path)
        metadata.update(probe)

        logger.info("download_complete", path=str(video_path), size_mb=video_path.stat().st_size / 1e6, job_id=jid)
        return {
            "video_path": video_path,
            "metadata": metadata,
            "job_id": jid,
        }

    def _probe_video(self, path: Path) -> dict:
        """Get video properties via ffprobe."""
        cmd = [
            "ffprobe", "-v", "quiet", "-print_format", "json",
            "-show_format", "-show_streams", str(path)
        ]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            data = json.loads(result.stdout)
            fmt = data.get("format", {})
            streams = data.get("streams", [])
            video_stream = next((s for s in streams if s.get("codec_type") == "video"), {})
            audio_stream = next((s for s in streams if s.get("codec_type") == "audio"), {})

            return {
                "duration": float(fmt.get("duration", 0)),
                "width": int(video_stream.get("width", 0)),
                "height": int(video_stream.get("height", 0)),
                "fps": eval(video_stream.get("r_frame_rate", "0/1")) if "/" in video_stream.get("r_frame_rate", "") else float(video_stream.get("r_frame_rate", 0)),
                "codec": video_stream.get("codec_name", "unknown"),
                "has_audio": bool(audio_stream),
                "size_bytes": int(fmt.get("size", 0)),
            }
        except Exception as e:
            logger.warning("ffprobe_failed", path=str(path), error=str(e))
            return {"duration": 0, "width": 0, "height": 0}

    @staticmethod
    def _make_job_id(source: str) -> str:
        """Generate deterministic job ID from source."""
        return hashlib.sha256(source.encode()).hexdigest()[:12]
