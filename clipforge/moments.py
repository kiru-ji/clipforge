"""Viral moment detection using LLM (Ollama or Gemini)."""

from __future__ import annotations

import json
from typing import Optional

import structlog
from openai import OpenAI

logger = structlog.get_logger(__name__)

MOMENT_PICKER_PROMPT = """You are a viral content expert. Analyze this transcript and identify the {max_clips} most viral-worthy moments for short-form video (TikTok/Reels/Shorts).

Each moment must be:
- Between {min_dur} and {max_dur} seconds
- Self-contained (makes sense without context)
- Emotionally engaging (surprise, humor, insight, controversy, story)
- Has a clear hook in the first 3 seconds

TRANSCRIPT:
{transcript}

Respond with ONLY valid JSON array. Each element:
{{
  "start": <float seconds>,
  "end": <float seconds>,
  "title": "<catchy title under 60 chars>",
  "hook": "<first-line hook text>",
  "viral_score": <1-10 integer>,
  "reason": "<why this moment is viral-worthy, one sentence>"
}}

Sort by viral_score descending. Return exactly {max_clips} or fewer if not enough good moments exist."""


class MomentPicker:
    """Select viral moments from transcript using LLM."""

    def __init__(
        self,
        backend: str = "ollama",
        ollama_url: str = "http://localhost:11434/v1",
        ollama_model: str = "qwen2.5:7b",
        gemini_api_key: str = "",
        min_duration: int = 15,
        max_duration: int = 60,
        max_clips: int = 10,
    ):
        self.backend = backend
        self.min_duration = min_duration
        self.max_duration = max_duration
        self.max_clips = max_clips

        if backend == "gemini" and gemini_api_key:
            self.client = OpenAI(
                api_key=gemini_api_key,
                base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
            )
            self.model = "gemini-2.0-flash"
        else:
            self.client = OpenAI(api_key="ollama", base_url=ollama_url)
            self.model = ollama_model

    def pick_moments(self, transcript: dict, job_id: str) -> list[dict]:
        """
        Select viral moments from a transcript.

        Args:
            transcript: dict with 'segments' list from Transcriber
            job_id: for logging

        Returns:
            List of moment dicts sorted by viral_score desc
        """
        # Build condensed transcript text with timestamps
        lines = []
        for seg in transcript.get("segments", []):
            lines.append(f"[{seg['start']:.1f}s-{seg['end']:.1f}s] {seg['text']}")
        transcript_text = "\n".join(lines)

        # Truncate if too long (keep ~8k tokens worth)
        if len(transcript_text) > 30000:
            transcript_text = transcript_text[:30000] + "\n... [truncated]"

        prompt = MOMENT_PICKER_PROMPT.format(
            max_clips=self.max_clips,
            min_dur=self.min_duration,
            max_dur=self.max_duration,
            transcript=transcript_text,
        )

        logger.info("moment_picking_started", job_id=job_id, backend=self.backend, model=self.model)

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=4096,
            )
            content = response.choices[0].message.content.strip()
        except Exception as e:
            logger.error("llm_call_failed", job_id=job_id, error=str(e))
            raise

        # Parse JSON from response (handle markdown code blocks)
        if content.startswith("```"):
            content = content.split("\n", 1)[1].rsplit("```", 1)[0].strip()

        try:
            moments = json.loads(content)
        except json.JSONDecodeError:
            logger.warning("json_parse_failed_attempting_repair", job_id=job_id)
            # Try to extract JSON array from response
            import re
            match = re.search(r'\[.*\]', content, re.DOTALL)
            if match:
                moments = json.loads(match.group())
            else:
                logger.error("moment_picking_failed_no_json", job_id=job_id)
                return []

        # Validate and filter moments
        valid = []
        for m in moments:
            try:
                start = float(m["start"])
                end = float(m["end"])
                duration = end - start
                if self.min_duration <= duration <= self.max_duration:
                    valid.append({
                        "start": start,
                        "end": end,
                        "title": str(m.get("title", "Untitled"))[:60],
                        "hook": str(m.get("hook", "")),
                        "viral_score": int(m.get("viral_score", 5)),
                        "reason": str(m.get("reason", "")),
                    })
            except (KeyError, ValueError, TypeError) as e:
                logger.warning("invalid_moment_skipped", moment=m, error=str(e))

        valid.sort(key=lambda x: x["viral_score"], reverse=True)
        logger.info("moments_picked", job_id=job_id, count=len(valid))
        return valid[:self.max_clips]
