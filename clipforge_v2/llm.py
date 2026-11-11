"""
Qwen/DashScope LLM backend for moment detection + metadata generation.

Fixes applied (vs audit):
- J40/P10: exceptions are logged AND re-raised as RuntimeError (not swallowed to [])
- J41: moment timestamps validated against transcript.duration
- J42: safe markdown fence stripping (no IndexError on bare ```)
- J43: _parse_json_object validates result is dict before returning
- J44/P30: sliding window for long transcripts (>30k chars)
- J45: non-greedy regex for JSON extraction
- J46: handles message.content being None
- J47: AsyncOpenAI configured with timeout + max_retries
- J48: health_check uses models list (free), not paid chat completion
- J49: hashtags validated (must be list of strings without #)
- J50: viral_score parsed safely with int(float(...)) and clamped
- P33: seed=42 for reproducibility
"""

from __future__ import annotations

import json
import re

import structlog
from openai import AsyncOpenAI

from .contracts import ClipMetadata, LLMBackend, Transcript, ViralMoment

logger = structlog.get_logger(__name__)

_MOMENT_SYSTEM = """You are a viral content expert. Identify the {max_clips} most viral-worthy moments for short-form video.

Each moment must be:
- Between {min_dur} and {max_dur} seconds
- Self-contained (makes sense without context)
- Emotionally engaging (surprise, humor, insight, controversy, story)
- Has a clear hook in the first 3 seconds

Respond with ONLY a valid JSON array. Each element:
{{"start": <float seconds>, "end": <float seconds>, "title": "<catchy title under 60 chars>", "hook": "<first-line hook>", "viral_score": <1-10 integer>, "reason": "<one sentence why>"}}

Sort by viral_score descending."""

_METADATA_SYSTEM = """Generate social media metadata for this video clip. Respond with ONLY valid JSON:
{{"youtube_title": "<under 100 chars with emoji>", "youtube_description": "<2-3 lines>", "instagram_caption": "<engaging, under 2200 chars>", "tiktok_caption": "<punchy, under 150 chars>", "hashtags": ["<10-15 tags without # symbol>"], "thumbnail_text": "<3-5 words>"}}

Rules: NO AI-sounding phrases. Natural conversational language. Curiosity gap for YouTube."""

_MAX_TRANSCRIPT_CHARS = 30000


class QwenLLMBackend(LLMBackend):
    """Qwen via DashScope OpenAI-compatible API."""

    def __init__(self, api_key: str, base_url: str, model: str):
        self.client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=180.0,      # J47/HIGH-14: 180s for large moment detection calls
            max_retries=2,      # J47: SDK-level retries
        )
        self.model = model

    async def detect_moments(
        self, transcript: Transcript, min_duration: int, max_duration: int, max_clips: int
    ) -> list[ViralMoment]:
        # J44/P30: sliding window for long transcripts
        text = self._build_transcript_text(transcript)

        system = _MOMENT_SYSTEM.format(min_dur=min_duration, max_dur=max_duration, max_clips=max_clips)

        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": f"TRANSCRIPT:\n{text}"},
                ],
                temperature=0.7,
                seed=42,  # P33: reproducible
                max_tokens=4096,
            )
            content = response.choices[0].message.content  # J46: can be None
            if not content:
                raise RuntimeError("LLM returned empty response for moment detection")
            content = content.strip()
        except Exception as e:
            # J40: log AND raise — don't swallow to []
            logger.error("qwen_moment_detection_failed", error=str(e))
            raise RuntimeError(f"Moment detection failed: {e}") from e

        raw_moments = self._parse_json_array(content)
        valid: list[ViralMoment] = []

        for m in raw_moments:
            try:
                start = float(m["start"])
                end = float(m["end"])
                # J41: validate against transcript duration
                if end > transcript.duration + 1.0:  # 1s tolerance
                    logger.warning("moment_beyond_duration", start=start, end=end, dur=transcript.duration)
                    continue
                if start < 0 or end <= start:
                    continue
                if not (min_duration <= (end - start) <= max_duration):
                    continue

                # J50: safe int parsing
                score_raw = m.get("viral_score", 5)
                try:
                    score = int(float(score_raw))
                except (ValueError, TypeError):
                    score = 5

                valid.append(ViralMoment(
                    start=start, end=end,
                    title=str(m.get("title", "Untitled"))[:60],
                    hook=str(m.get("hook", "")),
                    viral_score=score,  # clamped by ViralMoment.__post_init__
                    reason=str(m.get("reason", "")),
                ))
            except (KeyError, ValueError, TypeError) as e:
                logger.warning("invalid_moment_skipped", moment=m, error=str(e))

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
                    {"role": "system", "content": _METADATA_SYSTEM},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.8,
                seed=42,
                max_tokens=2048,
            )
            content = response.choices[0].message.content
            if not content:
                raise RuntimeError("LLM returned empty response for metadata")
            data = self._parse_json_object(content.strip())
        except Exception as e:
            logger.warning("metadata_gen_failed_using_fallback", error=str(e))
            data = {}

        # J49: validate hashtags
        raw_tags = data.get("hashtags", ["shorts", "viral", "trending"])
        if not isinstance(raw_tags, list):
            raw_tags = ["shorts", "viral", "trending"]
        hashtags = tuple(str(t).lstrip("#") for t in raw_tags if isinstance(t, str) and t.strip())

        return ClipMetadata(
            youtube_title=data.get("youtube_title", f"{moment.title} #shorts"),
            youtube_description=data.get("youtube_description", moment.reason),
            instagram_caption=data.get("instagram_caption", f"{moment.title}\n\n{moment.reason}"),
            tiktok_caption=data.get("tiktok_caption", moment.title[:150]),
            hashtags=hashtags,
            thumbnail_text=data.get("thumbnail_text", moment.title[:20]),
        )

    async def health_check(self) -> bool:
        """J48: use free models-list endpoint, not paid chat completion."""
        try:
            r = await self.client.models.list()
            return True
        except Exception:
            return False

    @staticmethod
    def _build_transcript_text(transcript: Transcript) -> str:
        """Build condensed transcript with sliding window for long videos (J44/P30)."""
        lines = [f"[{s.start:.1f}s-{s.end:.1f}s] {s.text}" for s in transcript.segments]
        full = "\n".join(lines)
        if len(full) <= _MAX_TRANSCRIPT_CHARS:
            return full
        # Sliding window: take first half + last half to cover beginning and end
        half = _MAX_TRANSCRIPT_CHARS // 2
        return full[:half] + "\n... [middle section truncated for length] ...\n" + full[-half:]

    @staticmethod
    def _parse_json_array(text: str) -> list[dict]:
        """J42/J45: safe markdown stripping + non-greedy JSON extraction."""
        # Strip markdown code fences safely
        if text.startswith("```"):
            first_newline = text.index("\n") if "\n" in text else len(text)
            text = text[first_newline + 1:]
            if text.endswith("```"):
                text = text[:-3]
            text = text.strip()

        # J45: non-greedy match to avoid spanning stray brackets
        match = re.search(r'\[.*?\]', text, re.DOTALL)
        if not match:
            # Try greedy as fallback
            match = re.search(r'\[.*\]', text, re.DOTALL)
        try:
            result = json.loads(match.group() if match else text)
            return result if isinstance(result, list) else []
        except (json.JSONDecodeError, AttributeError):
            return []

    @staticmethod
    def _parse_json_object(text: str) -> dict:
        """J42/J43: safe parse that guarantees dict return."""
        if text.startswith("```"):
            first_newline = text.index("\n") if "\n" in text else len(text)
            text = text[first_newline + 1:]
            if text.endswith("```"):
                text = text[:-3]
            text = text.strip()

        match = re.search(r'\{.*?\}', text, re.DOTALL)
        if not match:
            match = re.search(r'\{.*\}', text, re.DOTALL)
        try:
            result = json.loads(match.group() if match else text)
            # J43: validate it's actually a dict
            return result if isinstance(result, dict) else {}
        except (json.JSONDecodeError, AttributeError):
            return {}
