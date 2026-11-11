"""CLI interface for ClipForge."""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import click
import structlog
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn

from .config import get_settings
from .ingest import Ingestor
from .transcribe import Transcriber
from .moments import MomentPicker
from .extract import ClipExtractor
from .metadata import MetadataGenerator
from .publisher import Publisher

console = Console()
logger = structlog.get_logger(__name__)


@click.group()
def cli():
    """ClipForge — AI Video Clipping & Auto-Posting Agent."""
    structlog.configure(
        processors=[
            structlog.stdlib.add_log_level,
            structlog.dev.ConsoleRenderer(),
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
    )


@cli.command()
@click.argument("source")
@click.option("--output", "-o", default=None, help="Output directory (default: /shared/clipforge/output)")
@click.option("--max-clips", "-n", default=None, type=int, help="Max clips to extract")
@click.option("--publish", is_flag=True, help="Auto-publish to configured platforms")
@click.option("--dry-run", is_flag=True, help="Show what would be done without executing")
def process(source: str, output: str | None, max_clips: int | None, publish: bool, dry_run: bool):
    """Process a video: ingest → transcribe → detect moments → extract clips."""
    settings = get_settings()

    if output:
        settings.output_dir = Path(output)
    if max_clips:
        settings.max_clips_per_video = max_clips

    console.print(f"\n[bold cyan]ClipForge[/] Processing: {source}\n")

    # Step 1: Ingest
    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}")) as progress:
        task = progress.add_task("Ingesting video...", total=None)
        ingestor = Ingestor(settings.downloads_dir)
        ingest_result = ingestor.ingest(source)
        progress.update(task, completed=True)

    job_id = ingest_result["job_id"]
    video_path = ingest_result["video_path"]
    metadata = ingest_result["metadata"]

    table = Table(title="Video Info")
    table.add_column("Property", style="cyan")
    table.add_column("Value", style="green")
    table.add_row("Job ID", job_id)
    table.add_row("Duration", f"{metadata.get('duration', 0):.0f}s")
    table.add_row("Resolution", f"{metadata.get('width', '?')}x{metadata.get('height', '?')}")
    table.add_row("Size", f"{metadata.get('size_bytes', 0) / 1e6:.1f} MB")
    console.print(table)

    if dry_run:
        console.print("\n[yellow]Dry run — stopping before transcription.[/]")
        return

    # Step 2: Transcribe
    console.print("\n[bold]Transcribing...[/]")
    transcriber = Transcriber(
        model_size=settings.whisper_model,
        device=settings.whisper_device,
        compute_type=settings.whisper_compute,
        transcripts_dir=settings.transcripts_dir,
    )
    transcript = transcriber.transcribe(video_path, job_id)
    console.print(f"  ✓ {len(transcript['segments'])} segments, language={transcript['language']}")

    # Step 3: Pick moments
    console.print("\n[bold]Detecting viral moments...[/]")
    picker = MomentPicker(
        backend=settings.llm_backend,
        ollama_url=settings.ollama_base_url,
        ollama_model=settings.ollama_model,
        gemini_api_key=settings.gemini_api_key,
        min_duration=settings.clip_min_duration,
        max_duration=settings.clip_max_duration,
        max_clips=settings.max_clips_per_video,
    )
    moments = picker.pick_moments(transcript, job_id)

    if not moments:
        console.print("[red]No viral moments detected. Try a different video or adjust settings.[/]")
        return

    moments_table = Table(title=f"Detected Moments ({len(moments)})")
    moments_table.add_column("#", style="dim")
    moments_table.add_column("Time", style="cyan")
    moments_table.add_column("Score", style="yellow")
    moments_table.add_column("Title", style="green")
    for i, m in enumerate(moments):
        moments_table.add_row(
            str(i + 1),
            f"{m['start']:.0f}-{m['end']:.0f}s",
            f"{m['viral_score']}/10",
            m["title"],
        )
    console.print(moments_table)

    # Step 4: Extract clips
    console.print("\n[bold]Extracting clips...[/]")
    extractor = ClipExtractor(
        clips_dir=settings.clips_dir,
        output_dir=settings.output_dir,
        nvenc_preset=settings.nvenc_preset,
        burn_subtitles=settings.burn_subtitles,
    )

    meta_gen = MetadataGenerator(
        backend=settings.llm_backend,
        ollama_url=settings.ollama_base_url,
        ollama_model=settings.ollama_model,
        gemini_api_key=settings.gemini_api_key,
        metadata_dir=settings.metadata_dir,
    )

    clip_paths = []
    all_metadata = []
    for i, moment in enumerate(moments):
        console.print(f"  Extracting clip {i+1}/{len(moments)}: {moment['title']}")

        # Get transcript excerpt for this moment
        relevant_segs = [
            s for s in transcript["segments"]
            if s["start"] >= moment["start"] and s["end"] <= moment["end"]
        ]
        excerpt = " ".join(s["text"] for s in relevant_segs[:5])

        # Generate metadata
        meta = meta_gen.generate(moment, excerpt, metadata.get("title", ""), job_id, i)
        all_metadata.append(meta)

        # Extract clip
        clip_path = extractor.extract_clip(
            video_path, moment, job_id, i,
            transcript_segments=relevant_segs,
        )
        clip_paths.append(clip_path)

    console.print(f"\n[bold green]✓ {len(clip_paths)} clips extracted to {settings.output_dir}[/]")

    # Step 5: Publish (optional)
    if publish:
        console.print("\n[bold]Publishing...[/]")
        pub = Publisher(
            youtube_secrets=settings.youtube_client_secrets_file,
            youtube_token_file=settings.youtube_token_file,
            instagram_token=settings.instagram_access_token,
            instagram_user_id=settings.instagram_user_id,
            upload_post_key=settings.upload_post_api_key,
        )
        for clip_path, meta in zip(clip_paths, all_metadata):
            results = asyncio.run(pub.publish_all(clip_path, meta))
            for r in results:
                status_icon = "✓" if r["status"] == "published" else "✗"
                console.print(f"  {status_icon} {r['platform']}: {r['status']}")

    # Save job summary
    summary = {
        "job_id": job_id,
        "source": source,
        "clips": [
            {"path": str(p), "metadata": m}
            for p, m in zip(clip_paths, all_metadata)
        ],
    }
    summary_path = settings.metadata_dir / f"{job_id}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    console.print(f"\n[dim]Summary saved to {summary_path}[/]")


@cli.command()
def status():
    """Show current configuration and API status."""
    settings = get_settings()

    table = Table(title="ClipForge Status")
    table.add_column("Component", style="cyan")
    table.add_column("Status", style="bold")
    table.add_column("Details")

    table.add_row("Work Dir", "✓", str(settings.work_dir))
    table.add_row("Whisper", "✓", f"{settings.whisper_model} / {settings.whisper_device} / {settings.whisper_compute}")
    table.add_row("LLM Backend", "✓", f"{settings.llm_backend} ({settings.ollama_model})")
    table.add_row("YouTube API", "✓" if settings.has_youtube else "✗", "Configured" if settings.has_youtube else "Not configured")
    table.add_row("Instagram API", "✓" if settings.has_instagram else "✗", "Configured" if settings.has_instagram else "Not configured")
    table.add_row("TikTok (Upload-Post)", "✓" if settings.has_tiktok else "✗", "Configured" if settings.has_tiktok else "Not configured")
    table.add_row("Telegram Approval", "✓" if settings.has_telegram else "✗", "Configured" if settings.has_telegram else "Not configured")
    table.add_row("NVENC", "checking...", "")

    console.print(table)


if __name__ == "__main__":
    cli()
