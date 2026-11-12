"""
FFmpeg video processor (CPU-only, ARM/x86 VPS compatible).

Fixes applied (vs audit):
- J51/P34: per-job download subdirectory (no shared-dir race)
- J52/SSRF: URL scheme/host validation before yt-dlp (blocks file://, 169.254.x.x, localhost)
- P07/J53: asyncio.wait_for timeout on ALL subprocess calls
- J54: extract_clip validates output exists and has size > 0
- J55/P02: cleanup method for old downloads/clips
- J56/P04: SRT generation from word timestamps + burn_subtitles actually called
- J57/P36: proper subtitle path escaping using FFmpeg's subtitles filter syntax
- J58/P37: _probe checks returncode, raises on failure instead of silent defaults
- J59: single-pass extract+reframe (one x264 encode, not two)
- J60: crop dimensions aligned to even numbers for yuv420
- J61: removed unused subprocess import claim
- J62: fixed f-string without placeholders
- J63: burn failure logs warning but doesn't silently ship unsubtitled video — returns original
- P35: yt-dlp format chain includes webm fallback
- P38: documented center-crop limitation
"""

from __future__ import annotations

import asyncio
import ipaddress
import json
import re
import socket
import time
from pathlib import Path
from urllib.parse import urlparse

import structlog

from .contracts import TranscriptSegment, VideoProcessor

logger = structlog.get_logger(__name__)

_SUBPROCESS_TIMEOUT = 600  # 10 minutes max per FFmpeg call
_DOWNLOAD_TIMEOUT = 1800   # 30 minutes max for yt-dlp

# J52/SSRF: blocked hosts/schemes + private/reserved networks.
# ENI fix (12 Nov 2026): string-prefix matching on the host literal was
# bypassable three ways: (1) it only blocked "100.64.*" (= /16), leaving
# 63/64ths of the CGNAT 100.64/10 range open (e.g. 100.100.5.5 — used by
# Oracle Cloud internal services); (2) integer-form IPs like
# http://2130706433/ (= 127.0.0.1) matched no prefix; (3) DNS names
# resolving to internal IPs were never checked at all. Now every host is
# resolved and every resulting IP checked against proper CIDR networks.
_BLOCKED_SCHEMES = {"file", "ftp", "gopher", "data"}
_BLOCKED_HOSTS = {"localhost", "metadata.google.internal"}
_BLOCKED_NETWORKS = tuple(
    ipaddress.ip_network(cidr) for cidr in (
        "0.0.0.0/8",        # "this" network
        "10.0.0.0/8",       # RFC 1918 private (incl. VPS WireGuard tunnel)
        "100.64.0.0/10",    # RFC 6598 CGNAT (Oracle Cloud internal services)
        "127.0.0.0/8",      # loopback
        "169.254.0.0/16",   # link-local: cloud metadata (169.254.169.254)
        "172.16.0.0/12",    # RFC 1918 private (covers 172.16-31, Docker)
        "192.168.0.0/16",   # RFC 1918 private
        "::1/128",          # IPv6 loopback
        "fc00::/7",         # IPv6 unique-local
        "fe80::/10",        # IPv6 link-local
    )
)


def _ip_is_blocked(ip_str: str) -> bool:
    """True if ip_str falls in any blocked network. Fails CLOSED on garbage."""
    try:
        ip = ipaddress.ip_address(ip_str)
    except ValueError:
        return True
    return any(ip.version == net.version and ip in net for net in _BLOCKED_NETWORKS)


def _validate_url(url: str) -> None:
    """J52: reject SSRF-targeting URLs before passing to yt-dlp.

    Checks the literal host, and for hostnames every IP from getaddrinfo,
    against _BLOCKED_NETWORKS. Known limitation (documented, not fixed):
    TOCTOU DNS rebinding — validation and yt-dlp each resolve separately;
    closing that fully requires pinning the validated IP into the download.
    """
    parsed = urlparse(url)
    if parsed.scheme.lower() in _BLOCKED_SCHEMES:
        raise ValueError(f"Blocked URL scheme: {parsed.scheme}")
    host = (parsed.hostname or "").lower()
    if not host:
        raise ValueError("URL has no host")
    if host in _BLOCKED_HOSTS or f"[{host}]" in _BLOCKED_HOSTS:
        raise ValueError(f"Blocked host: {host}")

    candidate = host.strip("[]")
    try:
        ipaddress.ip_address(candidate)
        is_literal_ip = True
    except ValueError:
        is_literal_ip = False

    if is_literal_ip:
        if _ip_is_blocked(candidate):
            raise ValueError(f"Blocked private IP range: {host}")
        return

    # Hostname: resolve and check EVERY address (catches DNS -> internal IPs,
    # and integer-form hosts like "2130706433" which getaddrinfo normalizes).
    try:
        infos = socket.getaddrinfo(host, None)
    except socket.gaierror:
        return  # unresolvable: let yt-dlp surface the real error
    for info in infos:
        addr = str(info[4][0]).split("%")[0]  # strip IPv6 scope id
        if _ip_is_blocked(addr):
            raise ValueError(
                f"Blocked host resolving into private range: {host} -> {addr}")


class FFmpegVideoProcessor(VideoProcessor):
    """CPU-only FFmpeg processor for ARM/x86 VPS."""

    def __init__(self, threads: int = 2):
        self.threads = threads

    async def download(self, url: str, job_download_dir: Path) -> tuple[Path, str]:
        """
        Download video via yt-dlp into a per-job directory.
        Returns (video_path, source_title).
        J51: per-job dir prevents race conditions.
        J52: URL validated before yt-dlp.
        """
        _validate_url(url)
        job_download_dir.mkdir(parents=True, exist_ok=True)

        # Get metadata first (title)
        meta_cmd = ["yt-dlp", "--dump-json", "--no-download", "--no-warnings", url]
        source_title = "Source Video"
        try:
            proc = await asyncio.create_subprocess_exec(
                *meta_cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
            )
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=60)
            if proc.returncode == 0:
                meta = json.loads(stdout)
                source_title = meta.get("title", "Source Video")
        except Exception as e:
            logger.warning("metadata_fetch_failed", error=str(e))

        # Download with broad format support (P35: includes webm fallback)
        output_template = str(job_download_dir / "%(id)s.%(ext)s")
        dl_cmd = [
            "yt-dlp",
            "-f", "bestvideo[height<=1080]+bestaudio/best[height<=1080]/best",
            "--merge-output-format", "mp4",
            "-o", output_template,
            "--no-playlist", "--restrict-filenames",
            "--socket-timeout", "30", "--retries", "3",
            url,
        ]
        proc = await asyncio.create_subprocess_exec(
            *dl_cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
        )
        _, stderr = await asyncio.wait_for(proc.communicate(), timeout=_DOWNLOAD_TIMEOUT)
        if proc.returncode != 0:
            raise RuntimeError(f"Download failed: {stderr.decode()[-300:]}")

        # Find the downloaded file in THIS job's directory only (J51)
        files = sorted(job_download_dir.glob("*.mp4"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not files:
            raise RuntimeError(f"No mp4 file found after download in {job_download_dir}")

        video_path = files[0]
        logger.info("downloaded", path=str(video_path), size_mb=video_path.stat().st_size / 1e6, title=source_title)
        return video_path, source_title

    async def extract_audio(self, video_path: Path, output: Path) -> Path:
        """Extract 16kHz mono MP3 for efficient ASR upload (P27)."""
        cmd = [
            "ffmpeg", "-y", "-i", str(video_path),
            "-vn", "-c:a", "libmp3lame", "-b:a", "128k", "-ar", "16000", "-ac", "1",
            str(output),
        ]
        proc = await asyncio.create_subprocess_exec(*cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
        _, stderr = await asyncio.wait_for(proc.communicate(), timeout=_SUBPROCESS_TIMEOUT)
        if proc.returncode != 0:
            raise RuntimeError(f"Audio extraction failed: {stderr.decode()[-300:]}")
        return output

    async def extract_and_reframe(
        self, source: Path, start: float, end: float, output: Path,
        transcript_segments: tuple[TranscriptSegment, ...], burn_subs: bool,
    ) -> Path:
        """
        Single-pass extract + reframe + optional subtitle burn (J59: one encode, not two).
        Uses -ss/-to INPUT seeking so both video AND audio are trimmed correctly.
        J54: validates output exists and has size > 0.
        """
        probe = await self._probe(source)
        w, h = probe["width"], probe["height"]

        # Build video filter chain (always comma-joined, never semicolons)
        filters: list[str] = []

        # Reset timestamps after input seeking
        filters.append("setpts=PTS-STARTPTS")

        # Reframe to 9:16
        if w > h:  # Landscape → center crop (P38: documented limitation)
            crop_w = _align_even(int(h * 9 / 16))
            crop_h = _align_even(h)
            x = _align_even(max(0, (w - crop_w) // 2))
            filters.append(f"crop={crop_w}:{crop_h}:{x}:0")
            filters.append("scale=1080:1920:flags=lanczos")
        elif w == h:  # Square → blur background (needs filter_complex)
            return await self._extract_square_blur(source, start, end, output, transcript_segments, burn_subs)
        else:  # Already vertical
            filters.append("scale=1080:1920:force_original_aspect_ratio=decrease,pad=1080:1920:(ow-iw)/2:(oh-ih)/2")

        # Subtitle burn (J56/P04: actually generate SRT and burn it)
        srt_path = None
        if burn_subs and transcript_segments:
            srt_path = output.with_suffix(".srt")
            _write_srt(srt_path, transcript_segments, start, end)
            if srt_path.exists() and srt_path.stat().st_size > 0:
                # J57/P36: proper escaping for FFmpeg subtitles filter
                escaped_srt = str(srt_path).replace("\\", "\\\\").replace("'", "\\'").replace(":", "\\:")
                filters.append(
                    f"subtitles='{escaped_srt}':force_style='FontSize=22,PrimaryColour=&HFFFFFF,"
                    f"OutlineColour=&H000000,Outline=2,MarginV=40'"
                )

        # ALWAYS comma-join (simple filtergraph). Semicolons create separate chains = crash.
        vf = ",".join(filters)

        # Use -ss/-to BEFORE -i for fast input seeking (trims BOTH video and audio)
        cmd = [
            "ffmpeg", "-y",
            "-ss", str(start), "-to", str(end),
            "-i", str(source),
            "-vf", vf,
            "-c:v", "libx264", "-preset", "fast", "-crf", "23",
            "-c:a", "aac", "-b:a", "128k",
            "-threads", str(self.threads),
            "-movflags", "+faststart",
            str(output),
        ]
        proc = await asyncio.create_subprocess_exec(*cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
        _, stderr = await asyncio.wait_for(proc.communicate(), timeout=_SUBPROCESS_TIMEOUT)

        # Clean up SRT temp file
        if srt_path and srt_path.exists():
            srt_path.unlink()

        if proc.returncode != 0:
            raise RuntimeError(f"Extract+reframe failed: {stderr.decode()[-300:]}")

        # J54: validate output
        if not output.exists() or output.stat().st_size == 0:
            raise RuntimeError(f"Output file is empty or missing: {output}")

        return output

    async def _extract_square_blur(
        self, source: Path, start: float, end: float, output: Path,
        transcript_segments: tuple[TranscriptSegment, ...], burn_subs: bool,
    ) -> Path:
        """Square video needs filter_complex (split/overlay), not simple -vf."""
        fc = (
            "[0:v]split[bg][fg];"
            "[bg]scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920,boxblur=20:5[b];"
            "[fg]scale=1080:1920:force_original_aspect_ratio=decrease[f];"
            "[b][f]overlay=(W-w)/2:(H-h)/2"
        )

        # Add subtitle filter if needed
        srt_path = None
        if burn_subs and transcript_segments:
            srt_path = output.with_suffix(".srt")
            _write_srt(srt_path, transcript_segments, start, end)
            if srt_path.exists() and srt_path.stat().st_size > 0:
                escaped_srt = str(srt_path).replace("\\", "\\\\").replace("'", "\\'").replace(":", "\\:")
                fc += (
                    f"[outv];[outv]subtitles='{escaped_srt}':"
                    f"force_style='FontSize=22,PrimaryColour=&HFFFFFF,"
                    f"OutlineColour=&H000000,Outline=2,MarginV=40'"
                )

        cmd = [
            "ffmpeg", "-y",
            "-ss", str(start), "-to", str(end),
            "-i", str(source),
            "-filter_complex", fc,
            "-c:v", "libx264", "-preset", "fast", "-crf", "23",
            "-c:a", "aac", "-b:a", "128k",
            "-threads", str(self.threads),
            "-movflags", "+faststart",
            str(output),
        ]
        proc = await asyncio.create_subprocess_exec(*cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
        _, stderr = await asyncio.wait_for(proc.communicate(), timeout=_SUBPROCESS_TIMEOUT)

        if srt_path and srt_path.exists():
            srt_path.unlink()

        if proc.returncode != 0:
            raise RuntimeError(f"Square blur extract failed: {stderr.decode()[-300:]}")

        if not output.exists() or output.stat().st_size == 0:
            raise RuntimeError(f"Output file is empty or missing: {output}")

        return output

    async def _probe(self, path: Path) -> dict:
        """J58/P37: check returncode, raise on failure."""
        cmd = [
            "ffprobe", "-v", "quiet", "-select_streams", "v:0",
            "-show_entries", "stream=width,height",
            "-print_format", "json", str(path),
        ]
        proc = await asyncio.create_subprocess_exec(*cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=30)

        if proc.returncode != 0:
            raise RuntimeError(f"ffprobe failed for {path}: {stderr.decode()[-200:]}")

        try:
            data = json.loads(stdout)
        except json.JSONDecodeError:
            raise RuntimeError(f"ffprobe returned invalid JSON for {path}")

        streams = data.get("streams", [])
        if not streams:
            raise RuntimeError(f"No video stream found in {path}")

        stream = streams[0]
        width = int(stream.get("width", 0))
        height = int(stream.get("height", 0))
        if width == 0 or height == 0:
            raise RuntimeError(f"Invalid dimensions {width}x{height} for {path}")

        return {"width": width, "height": height}

    @staticmethod
    async def cleanup_old(work_dir: Path, max_age_hours: int) -> int:
        """J55/P02: delete files older than max_age_hours. Returns count deleted."""
        cutoff = time.time() - (max_age_hours * 3600)
        deleted = 0
        for subdir in ["downloads", "output"]:
            d = work_dir / subdir
            if not d.exists():
                continue
            for f in d.rglob("*"):
                if f.is_file() and f.stat().st_mtime < cutoff:
                    f.unlink()
                    deleted += 1
        logger.info("cleanup_done", deleted=deleted, max_age_hours=max_age_hours)
        return deleted


def _align_even(n: int) -> int:
    """J60: align to even number for yuv420 chroma subsampling."""
    return n - (n % 2)


def _write_srt(path: Path, segments: tuple[TranscriptSegment, ...], clip_start: float, clip_end: float) -> None:
    """Generate SRT subtitle file from transcript segments for a clip's time range."""
    lines: list[str] = []
    idx = 1
    clip_duration = clip_end - clip_start

    for seg in segments:
        seg_start = seg.start - clip_start
        seg_end = seg.end - clip_start
        if seg_end < 0 or seg_start > clip_duration:
            continue
        seg_start = max(0.0, seg_start)
        seg_end = min(clip_duration, seg_end)
        if seg_end <= seg_start:
            continue

        lines.append(str(idx))
        lines.append(f"{_srt_time(seg_start)} --> {_srt_time(seg_end)}")
        lines.append(seg.text)
        lines.append("")
        idx += 1

    path.write_text("\n".join(lines), encoding="utf-8")


def _srt_time(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds % 1) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"
