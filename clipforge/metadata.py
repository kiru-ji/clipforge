"""Metadata generation: titles, descriptions, hashtags via LLM."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import structlog
from openai import OpenAI

logger = structlog.get_logger(__name__)

METADATA_PROMPT = """Generate social media metadata for this video clip.

CLIP TITLE: {title}
CLIP HOOK: {hook}
TRANSCRIPT EXCERPT: {transcript}
SOURCE VIDEO: {source_title}

Generate JSON with:
{{
  "youtube_title": "<title optimized for YouTube Shorts, under 100 chars, with emoji>",
  "youtube_description": "<2-3 line description with relevant keywords>",
  "instagram_caption": "<engaging caption with line breaks, under 2200 chars>",
  "tiktok_caption": "<short punchy caption under 150 chars>",
  "hashtags": ["<10-15 relevant hashtags without # symbol>"],
  "thumbnail_text": "<3-5 word text overlay suggestion>"
}}

Rules:
- NO AI-sounding phrases (no "dive into", "game-changer", "revolutionary")
- Use natural, conversational language
- Include 1-2 trending/relevant hashtags, rest should be niche-specific
- YouTube title should create curiosity gap
- Instagram caption should encourage saves/shares"""


class MetadataGenerator:
    """Generate platform-specific metadata for clips."""

    def __init__(
        self,
        backend: str = "ollama",
        ollama_url: str = "http://localhost:11434/v1",
        ollama_model: str = "qwen2.5:7b",
        gemini_api_key: str = "",
        metadata_dir: Optional[Path] = None,
    ):
        self.metadata_dir = metadata_dir

        if backend == "gemini" and gemini_api_key:
            self.client = OpenAI(
                api_key=gemini_api_key,
                base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
            )
            self.model = "gemini-2.0-flash"
        else:
            self.client = OpenAI(api_key="ollama", base_url=ollama_url)
            self.model = ollama_model

    def generate(self, moment: dict, transcript_excerpt: str, source_title: str, job_id: str, clip_index: int) -> dict:
        """Generate metadata for a single clip."""
        cache_path = self.metadata_dir / f"{job_id}_clip{clip_index:02d}_meta.json" if self.metadata_dir else None
        if cache_path and cache_path.exists():
            return json.loads(cache_path.read_text())

        prompt = METADATA_PROMPT.format(
            title=moment.get("title", ""),
            hook=moment.get("hook", ""),
            transcript=transcript_excerpt[:2000],
            source_title=source_title,
        )

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.8,
                max_tokens=2048,
            )
            content = response.choices[0].message.content.strip()
        except Exception as e:
            logger.error("metadata_generation_failed", job_id=job_id, clip=clip_index, error=str(e))
            return self._fallback_metadata(moment)

        # Parse JSON
        if content.startswith("```"):
            content = content.split("\n", 1)[1].rsplit("```", 1)[0].strip()

        try:
            import re
            match = re.search(r'\{.*\}', content, re.DOTALL)
            metadata = json.loads(match.group() if match else content)
        except (json.JSONDecodeError, AttributeError):
            logger.warning("metadata_json_parse_failed", job_id=job_id, clip=clip_index)
            metadata = self._fallback_metadata(moment)

        if cache_path:
            cache_path.write_text(json.dumps(metadata, indent=2))

        return metadata

    @staticmethod
    def _fallback_metadata(moment: dict) -> dict:
        """Generate basic metadata when LLM fails."""
        title = moment.get("title", "Amazing Moment")
        return {
            "youtube_title": f"{title} #shorts",
            "youtube_description": f"{moment.get('reason', 'Check this out!')}",
            "instagram_caption": f"{title}\n\n{moment.get('reason', '')}",
            "tiktok_caption": title[:150],
            "hashtags": ["shorts", "viral", "trending", "fyp", "clips"],
            "thumbnail_text": title[:20],
        }
