"""
Video processing via FFmpeg (CPU-only, optimized for ARM VPS).

No GPU dependencies. Uses libx264 for encoding, which works on any CPU.
Process pool for parallel clip extraction (OS concept: process isolation).
"""

from __future__ import annotations

import asyncio
import json
import subprocess
from pathlib import Path

import structlog

from .contracts import VideoProcessor

logger = structlog.get_logger(__name__)


class FFmpegVideoProcessor(VideoProcessor):
    """CPU-only FFmpeg video processor for ARM/x86 VPS."""

    def __init__(self, threads: int = 2):
        self.threads = threads

    async def download(self, url: str, output_dir: Path) -> Path:
        """Download video via yt-dlp."""
        output_template = str(output_dir / "%(id)s.%(ext)s")
        cmd = [
            "yt-dlp",
            "-f", "bestvideo[height<=1080][ext=mp4]+bestaudio[ext=m4a]/best[height<=1080][ext=mp4]/best",
            "--merge-output-format", "mp4",
            "-o", output_template,
            "--no-playlist", "--restrict-filenames",
            "--socket-timeout", "30", "--retries", "3",
            url,
        ]
        proc = await asyncio.create_subprocess_exec(*cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
        _, stderr = await proc.communicate()
        if proc.returncode != 0:
            raise RuntimeError(f"Download failed: {stderr.decode()[-300:]}")

        # Find downloaded file
        files = sorted(output_dir.glob("*.mp4"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not files:
            raise RuntimeError("No file found after download")
        logger.info("downloaded", path=str(files[0]), size_mb=files[0].stat().st_size / 1e6)
        return files[0]

    async def extract_clip(self, source: Path, start: float, end: float, output: Path) -> Path:
        """Extract a precise time-range clip."""
        cmd = [
            "ffmpeg", "-y", "-ss", str(start), "-to", str(end),
            "-i", str(source),
            "-c:v", "libx264", "-preset", "fast", "-crf", "23",
            "-c:a", "aac", "-b:a", "128k",
            "-threads", str(self.threads),
            "-movflags", "+faststart",
            str(output),
        ]
        proc = await asyncio.create_subprocess_exec(*cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
        _, stderr = await proc.communicate()
        if proc.returncode != 0:
            raise RuntimeError(f"Clip extraction failed: {stderr.decode()[-300:]}")
        return output

    async def reframe_vertical(self, source: Path, output: Path) -> Path:
        """Reframe landscape to 9:16 with center crop + scale."""
        probe = await self._probe(source)
        w, h = probe["width"], probe["height"]

        if w > h:  # Landscape → center crop
            crop_w = int(h * 9 / 16)
            x = max(0, (w - crop_w) // 2)
            vf = f"crop={crop_w}:{h}:{x}:0,scale=1080:1920:flags=lanczos"
        elif w == h:  # Square → blur background
            vf = (
                f"split[bg][fg];"
                f"[bg]scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920,boxblur=20:5[b];"
                f"[fg]scale=1080:1920:force_original_aspect_ratio=decrease[f];"
                f"[b][f]overlay=(W-w)/2:(H-h)/2"
            )
        else:  # Already vertical
            vf = "scale=1080:1920:force_original_aspect_ratio=decrease,pad=1080:1920:(ow-iw)/2:(oh-ih)/2"

        cmd = [
            "ffmpeg", "-y", "-i", str(source),
            "-vf", vf,
            "-c:v", "libx264", "-preset", "fast", "-crf", "23",
            "-c:a", "aac", "-b:a", "128k",
            "-threads", str(self.threads),
            "-movflags", "+faststart",
            str(output),
        ]
        proc = await asyncio.create_subprocess_exec(*cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
        _, stderr = await proc.communicate()
        if proc.returncode != 0:
            raise RuntimeError(f"Reframe failed: {stderr.decode()[-300:]}")
        return output

    async def burn_subtitles(self, video: Path, srt: Path, output: Path) -> Path:
        """Burn SRT subtitles into video."""
        escaped = str(srt).replace("\\", "/").replace(":", "\\:")
        cmd = [
            "ffmpeg", "-y", "-i", str(video),
            "-vf", f"subtitles='{escaped}':force_style='FontSize=22,PrimaryColour=&HFFFFFF,OutlineColour=&H000000,Outline=2,MarginV=40'",
            "-c:v", "libx264", "-preset", "fast", "-crf", "23",
            "-c:a", "copy",
            "-threads", str(self.threads),
            "-movflags", "+faststart",
            str(output),
        ]
        proc = await asyncio.create_subprocess_exec(*cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
        _, stderr = await proc.communicate()
        if proc.returncode != 0:
            logger.warning("subtitle_burn_failed", error=stderr.decode()[-200:])
            # Return original video without subs
            import shutil
            shutil.copy2(video, output)
        return output

    @staticmethod
    async def _probe(path: Path) -> dict:
        cmd = ["ffprobe", "-v", "quiet", "-select_streams", "v:0",
               "-show_entries", "stream=width,height", "-print_format", "json", str(path)]
        proc = await asyncio.create_subprocess_exec(*cmd, stdout=asyncio.subprocess.PIPE)
        stdout, _ = await proc.communicate()
        data = json.loads(stdout)
        stream = data.get("streams", [{}])[0]
        return {"width": int(stream.get("width", 1920)), "height": int(stream.get("height", 1080))}
