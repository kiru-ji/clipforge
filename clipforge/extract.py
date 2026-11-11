"""Video extraction, reframing, and encoding with FFmpeg + NVENC."""

from __future__ import annotations

import subprocess
import json
from pathlib import Path
from typing import Optional

import structlog

logger = structlog.get_logger(__name__)


class ClipExtractor:
    """Extract, reframe, and encode clips using FFmpeg with NVENC acceleration."""

    def __init__(
        self,
        clips_dir: Path,
        output_dir: Path,
        nvenc_preset: str = "p4",
        burn_subtitles: bool = True,
    ):
        self.clips_dir = clips_dir
        self.output_dir = output_dir
        self.nvenc_preset = nvenc_preset
        self.burn_subtitles = burn_subtitles
        self.clips_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Probe NVENC availability once
        self._has_nvenc = self._check_nvenc()
        if not self._has_nvenc:
            logger.warning("nvenc_not_available_using_cpu")

    def _check_nvenc(self) -> bool:
        """Test if NVENC encoding works."""
        try:
            result = subprocess.run(
                [
                    "ffmpeg", "-hide_banner", "-f", "lavfi",
                    "-i", "testsrc=size=256x256:rate=1:duration=1",
                    "-c:v", "h264_nvenc", "-preset", "p4",
                    "-f", "null", "-",
                ],
                capture_output=True, timeout=10,
            )
            return result.returncode == 0
        except Exception:
            return False

    def extract_clip(
        self,
        video_path: Path,
        moment: dict,
        job_id: str,
        index: int,
        transcript_segments: Optional[list] = None,
    ) -> Path:
        """
        Extract a single clip, reframe to 9:16, optionally burn subtitles.

        Args:
            video_path: source video
            moment: dict with start, end, title
            job_id: for file naming
            index: clip index within the job
            transcript_segments: optional segments for subtitle generation

        Returns:
            Path to the final output clip
        """
        start = moment["start"]
        end = moment["end"]
        duration = end - start
        safe_title = "".join(c if c.isalnum() or c in " -_" else "_" for c in moment.get("title", f"clip_{index}"))[:40]
        clip_name = f"{job_id}_clip{index:02d}_{safe_title}"

        raw_path = self.clips_dir / f"{clip_name}_raw.mp4"
        output_path = self.output_dir / f"{clip_name}.mp4"

        # Step 1: Extract raw clip
        self._extract_raw(video_path, raw_path, start, end)

        # Step 2: Get video dimensions for reframing decision
        probe = self._probe_dimensions(raw_path)
        width, height = probe["width"], probe["height"]

        # Step 3: Reframe to 9:16
        if width > height:  # Landscape → needs reframing
            self._reframe_to_vertical(raw_path, output_path, width, height)
        elif width == height:  # Square → pad to 9:16
            self._pad_to_vertical(raw_path, output_path, width)
        else:  # Already vertical
            self._copy_with_encode(raw_path, output_path)

        # Step 4: Burn subtitles if enabled and transcript available
        if self.burn_subtitles and transcript_segments:
            srt_path = self._generate_srt(transcript_segments, start, end, self.clips_dir / f"{clip_name}.srt")
            if srt_path.exists():
                self._burn_subtitles(output_path, srt_path)

        # Cleanup raw
        if raw_path.exists() and raw_path != output_path:
            raw_path.unlink()

        size_mb = output_path.stat().st_size / 1e6
        logger.info("clip_extracted", path=str(output_path), size_mb=round(size_mb, 1), duration=round(duration, 1))
        return output_path

    def _extract_raw(self, source: Path, dest: Path, start: float, end: float):
        """Cut a precise segment from the source video."""
        cmd = [
            "ffmpeg", "-y", "-ss", str(start), "-to", str(end),
            "-i", str(source),
            "-c:v", "h264_nvenc" if self._has_nvenc else "libx264",
            "-preset", self.nvenc_preset if self._has_nvenc else "fast",
            "-c:a", "aac", "-b:a", "128k",
            "-movflags", "+faststart",
            str(dest),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if result.returncode != 0:
            raise RuntimeError(f"FFmpeg extract failed: {result.stderr[-300:]}")

    def _reframe_to_vertical(self, source: Path, dest: Path, src_w: int, src_h: int):
        """
        Reframe landscape video to 9:16 vertical.

        Strategy: center-crop with smart positioning.
        For face tracking, we'd use MediaPipe/YOLOv8 — simplified here to center crop
        which works well for single-speaker content.
        """
        # Target: 1080x1920 (9:16)
        target_w, target_h = 1080, 1920

        # Calculate crop region (center of frame, maintaining aspect ratio)
        crop_h = src_h
        crop_w = int(src_h * 9 / 16)
        if crop_w > src_w:
            crop_w = src_w
            crop_h = int(src_w * 16 / 9)

        x_offset = max(0, (src_w - crop_w) // 2)
        y_offset = max(0, (src_h - crop_h) // 2)

        vf = f"crop={crop_w}:{crop_h}:{x_offset}:{y_offset},scale={target_w}:{target_h}:flags=lanczos"

        encoder = "h264_nvenc" if self._has_nvenc else "libx264"
        preset = self.nvenc_preset if self._has_nvenc else "fast"

        cmd = [
            "ffmpeg", "-y", "-i", str(source),
            "-vf", vf,
            "-c:v", encoder, "-preset", preset,
            "-c:a", "aac", "-b:a", "128k",
            "-movflags", "+faststart",
            str(dest),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if result.returncode != 0:
            raise RuntimeError(f"FFmpeg reframe failed: {result.stderr[-300:]}")

    def _pad_to_vertical(self, source: Path, dest: Path, size: int):
        """Pad square video to 9:16 with blurred background."""
        target_w, target_h = 1080, 1920

        vf = (
            f"split[bg][fg];"
            f"[bg]scale={target_w}:{target_h}:force_original_aspect_ratio=increase,"
            f"crop={target_w}:{target_h},boxblur=20:5[bg_blur];"
            f"[fg]scale={target_w}:{target_h}:force_original_aspect_ratio=decrease[fg_scaled];"
            f"[bg_blur][fg_scaled]overlay=(W-w)/2:(H-h)/2"
        )

        encoder = "h264_nvenc" if self._has_nvenc else "libx264"
        preset = self.nvenc_preset if self._has_nvenc else "fast"

        cmd = [
            "ffmpeg", "-y", "-i", str(source),
            "-filter_complex", vf,
            "-c:v", encoder, "-preset", preset,
            "-c:a", "aac", "-b:a", "128k",
            "-movflags", "+faststart",
            str(dest),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if result.returncode != 0:
            raise RuntimeError(f"FFmpeg pad failed: {result.stderr[-300:]}")

    def _copy_with_encode(self, source: Path, dest: Path):
        """Re-encode an already-vertical video with NVENC."""
        encoder = "h264_nvenc" if self._has_nvenc else "libx264"
        preset = self.nvenc_preset if self._has_nvenc else "fast"

        cmd = [
            "ffmpeg", "-y", "-i", str(source),
            "-c:v", encoder, "-preset", preset,
            "-c:a", "aac", "-b:a", "128k",
            "-movflags", "+faststart",
            str(dest),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if result.returncode != 0:
            raise RuntimeError(f"FFmpeg encode failed: {result.stderr[-300:]}")

    def _generate_srt(self, segments: list, clip_start: float, clip_end: float, srt_path: Path) -> Path:
        """Generate SRT subtitle file for a clip's time range."""
        lines = []
        idx = 1
        for seg in segments:
            seg_start = seg["start"] - clip_start
            seg_end = seg["end"] - clip_start
            if seg_end < 0 or seg_start > (clip_end - clip_start):
                continue
            seg_start = max(0, seg_start)
            seg_end = min(clip_end - clip_start, seg_end)

            lines.append(str(idx))
            lines.append(f"{self._format_srt_time(seg_start)} --> {self._format_srt_time(seg_end)}")
            lines.append(seg["text"])
            lines.append("")
            idx += 1

        srt_path.write_text("\n".join(lines))
        return srt_path

    @staticmethod
    def _format_srt_time(seconds: float) -> str:
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = int(seconds % 60)
        ms = int((seconds % 1) * 1000)
        return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

    def _burn_subtitles(self, video_path: Path, srt_path: Path):
        """Burn SRT subtitles into the video."""
        temp_path = video_path.with_suffix(".tmp.mp4")
        encoder = "h264_nvenc" if self._has_nvenc else "libx264"
        preset = self.nvenc_preset if self._has_nvenc else "fast"

        # Escape special characters in path for FFmpeg filter
        escaped_srt = str(srt_path).replace("\\", "/").replace(":", "\\:")

        cmd = [
            "ffmpeg", "-y", "-i", str(video_path),
            "-vf", f"subtitles='{escaped_srt}':force_style='FontSize=22,PrimaryColour=&HFFFFFF,OutlineColour=&H000000,Outline=2,MarginV=40'",
            "-c:v", encoder, "-preset", preset,
            "-c:a", "copy",
            "-movflags", "+faststart",
            str(temp_path),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if result.returncode == 0 and temp_path.exists():
            temp_path.replace(video_path)
        else:
            logger.warning("subtitle_burn_failed_keeping_video_without_subs", error=result.stderr[-200:] if result.stderr else "unknown")
            if temp_path.exists():
                temp_path.unlink()

    @staticmethod
    def _probe_dimensions(path: Path) -> dict:
        """Get video width and height."""
        cmd = [
            "ffprobe", "-v", "quiet", "-select_streams", "v:0",
            "-show_entries", "stream=width,height",
            "-print_format", "json", str(path),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        data = json.loads(result.stdout)
        stream = data.get("streams", [{}])[0]
        return {"width": int(stream.get("width", 1920)), "height": int(stream.get("height", 1080))}
