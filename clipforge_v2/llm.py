"""
Concrete LLM backend: Qwen via ModelScope/DashScope (OpenAI-compatible).

Uses the OpenAI SDK pointed at DashScope's compatible endpoint.
Same key works for moment detection AND metadata generation.
"""

from __future__ import annotations

import json
import re

import structlog
from openai import AsyncOpenAI

from .contracts import (
    ClipMetadata, LLMBackend, Transcript, ViralMoment,
)

logger = structlog.get_logger(__name__)

MOMENT_SYSTEM_PROMPT = """You are a viral content expert. Analyze transcripts and identify the most viral-worthy moments for short-form video (TikTok/Reels/Shorts).

Each moment must be:
- Between {min_dur} and {max_dur} seconds
- Self-contained (makes sense without context)
- Emotionally engaging (surprise, humor, insight, controversy, story)
- Has a clear hook in the first 3 seconds

Respond with ONLY valid JSON array. Each element:
{{"start": <float>, "end": <float>, "title": "<catchy title under 60 chars>", "hook": "<first-line hook>", "viral_score": <1-10>, "reason": "<one sentence why>"}}

Sort by viral_score descending. Return exactly {max_clips} or fewer if not enough good moments."""

METADATA_SYSTEM_PROMPT = """Generate social media metadata for this video clip. Respond with ONLY valid JSON:
{{"youtube_title": "<under 100 chars with emoji>", "youtube_description": "<2-3 lines>", "instagram_caption": "<engaging, under 2200 chars>", "tiktok_caption": "<punchy, under 150 chars>", "hashtags": ["<10-15 tags without #>"], "thumbnail_text": "<3-5 words>"}}

Rules: NO AI-sounding phrases. Natural language. Curiosity gap for YouTube. Encourage saves for Instagram."""


class QwenLLMBackend(LLMBackend):
    """Qwen via DashScope OpenAI-compatible API."""

    def __init__(self, api_key: str, base_url: str, model: str):
        self.client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        self.model = model

    async def detect_moments(
        self, transcript: Transcript, min_duration: int, max_duration: int, max_clips: int
    ) -> list[ViralMoment]:
        # Build condensed transcript
        lines = [f"[{s.start:.1f}s-{s.end:.1f}s] {s.text}" for s in transcript.segments]
        text = "\n".join(lines)[:30000]

        system = MOMENT_SYSTEM_PROMPT.format(min_dur=min_duration, max_dur=max_duration, max_clips=max_clips)

        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": f"TRANSCRIPT:\n{text}"},
                ],
                temperature=0.7,
                max_tokens=4096,
            )
            content = response.choices[0].message.content.strip()
        except Exception as e:
            logger.error("qwen_moment_detection_failed", error=str(e))
            return []

        # Parse JSON with repair
        moments = self._parse_json_array(content)
        valid = []
        for m in moments:
            try:
                start, end = float(m["start"]), float(m["end"])
                if min_duration <= (end - start) <= max_duration:
                    valid.append(ViralMoment(
                        start=start, end=end,
                        title=str(m.get("title", "Untitled"))[:60],
                        hook=str(m.get("hook", "")),
                        viral_score=int(m.get("viral_score", 5)),
                        reason=str(m.get("reason", "")),
                    ))
            except (KeyError, ValueError, TypeError):
                continue

        valid.sort(key=lambda x: x.viral_score, reverse=True)
        logger.info("moments_detected", count=len(valid))
        return valid[:max_clips]

    async def generate_metadata(
        self, moment: ViralMoment, transcript_excerpt: str, source_title: str
    ) -> ClipMetadata:
        prompt = f"CLIP: {moment.title}\nHOOK: {moment.hook}\nSOURCE: {source_title}\nEXCERPT: {transcript_excerpt[:2000]}"

        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": METADATA_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.8,
                max_tokens=2048,
            )
            content = response.choices[0].message.content.strip()
            data = self._parse_json_object(content)
        except Exception as e:
            logger.warning("metadata_gen_failed_using_fallback", error=str(e))
            data = {}

        return ClipMetadata(
            youtube_title=data.get("youtube_title", f"{moment.title} #shorts"),
            youtube_description=data.get("youtube_description", moment.reason),
            instagram_caption=data.get("instagram_caption", f"{moment.title}\n\n{moment.reason}"),
            tiktok_caption=data.get("tiktok_caption", moment.title[:150]),
            hashtags=tuple(data.get("hashtags", ["shorts", "viral", "trending"])),
            thumbnail_text=data.get("thumbnail_text", moment.title[:20]),
        )

    async def health_check(self) -> bool:
        try:
            r = await self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": "ping"}],
                max_tokens=5,
            )
            return bool(r.choices)
        except Exception:
            return False

    @staticmethod
    def _parse_json_array(text: str) -> list[dict]:
        if text.startswith("```"):
            text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
        match = re.search(r'\[.*\]', text, re.DOTALL)
        try:
            return json.loads(match.group() if match else text)
        except (json.JSONDecodeError, AttributeError):
            return []

    @staticmethod
    def _parse_json_object(text: str) -> dict:
        if text.startswith("```"):
            text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
        match = re.search(r'\{.*\}', text, re.DOTALL)
        try:
            return json.loads(match.group() if match else text)
        except (json.JSONDecodeError, AttributeError):
            return {}
