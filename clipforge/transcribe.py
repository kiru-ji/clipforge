"""Transcription with faster-whisper, optimized for 6GB VRAM."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import structlog

logger = structlog.get_logger(__name__)


class Transcriber:
    """
    Speech-to-text with word-level timestamps.

    Optimized for RTX 3050 6GB:
    - medium + int8_float16 ≈ 2.5GB VRAM (safe with YOLOv8n loaded)
    - large-v3-turbo + fp16 ≈ 4.5GB (too tight, OOM risk)
    - Falls back to CPU if GPU OOMs
    """

    def __init__(
        self,
        model_size: str = "medium",
        device: str = "cuda",
        compute_type: str = "int8_float16",
        transcripts_dir: Optional[Path] = None,
    ):
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self.transcripts_dir = transcripts_dir
        self._model = None

    def _load_model(self):
        """Lazy-load the whisper model."""
        if self._model is not None:
            return

        from faster_whisper import WhisperModel

        logger.info(
            "loading_whisper",
            model=self.model_size,
            device=self.device,
            compute=self.compute_type,
        )

        try:
            self._model = WhisperModel(
                self.model_size,
                device=self.device,
                compute_type=self.compute_type,
            )
        except Exception as e:
            if self.device == "cuda":
                logger.warning("gpu_load_failed_falling_back_to_cpu", error=str(e))
                self.device = "cpu"
                self.compute_type = "int8"
                self._model = WhisperModel(
                    self.model_size,
                    device="cpu",
                    compute_type="int8",
                )
            else:
                raise

    def transcribe(self, video_path: Path, job_id: str) -> dict:
        """
        Transcribe a video file.

        Returns dict with:
            - segments: list of {start, end, text, words}
            - language: detected language code
            - duration: total audio duration
            - transcript_path: path to saved JSON transcript
        """
        self._load_model()

        cache_path = self.transcripts_dir / f"{job_id}.json" if self.transcripts_dir else None
        if cache_path and cache_path.exists():
            logger.info("transcript_cache_hit", job_id=job_id)
            return json.loads(cache_path.read_text())

        logger.info("transcription_started", video=str(video_path), job_id=job_id)

        try:
            segments_gen, info = self._model.transcribe(
                str(video_path),
                beam_size=5,
                word_timestamps=True,
                vad_filter=True,
                vad_parameters=dict(min_silence_duration_ms=500),
            )

            segments = []
            for seg in segments_gen:
                words = []
                if seg.words:
                    words = [
                        {"start": w.start, "end": w.end, "word": w.word.strip()}
                        for w in seg.words
                    ]
                segments.append({
                    "start": round(seg.start, 3),
                    "end": round(seg.end, 3),
                    "text": seg.text.strip(),
                    "words": words,
                })

        except Exception as e:
            logger.error("transcription_failed", job_id=job_id, error=str(e))
            raise

        result = {
            "segments": segments,
            "language": info.language,
            "duration": round(info.duration, 3),
            "language_probability": round(info.language_probability, 3),
        }

        if cache_path:
            cache_path.write_text(json.dumps(result, indent=2))
            logger.info("transcript_saved", path=str(cache_path), segments=len(segments))

        logger.info(
            "transcription_complete",
            job_id=job_id,
            segments=len(segments),
            duration=info.duration,
            language=info.language,
        )
        return result
