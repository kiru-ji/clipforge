"""
Publisher backends for YouTube, TikTok, Instagram.

Fixes applied (vs audit):
- P43/J4: publisher.py NOW EXISTS with real implementations
- Each publisher is a concrete PublisherBackend subclass
- YouTube: direct Data API v3 (100 uploads/day as of Nov 2, 2026)
- TikTok: Upload-Post API (10 free/month)
- Instagram: Graph API container/publish flow
- All publishers handle errors gracefully and return PublishResult
"""

from __future__ import annotations

from pathlib import Path

import httpx
import structlog

from .contracts import ClipMetadata, PublisherBackend, PublishResult, PublishStatus

logger = structlog.get_logger(__name__)


class YouTubePublisher(PublisherBackend):
    """YouTube Data API v3 direct upload."""

    def __init__(self, client_secrets_json: str, token_file: Path):
        self.client_secrets_json = client_secrets_json
        self.token_file = token_file

    @property
    def platform_name(self) -> str:
        return "youtube"

    async def is_configured(self) -> bool:
        return bool(self.client_secrets_json)

    async def publish(self, video_path: Path, metadata: ClipMetadata) -> PublishResult:
        if not await self.is_configured():
            return PublishResult(platform="youtube", status=PublishStatus.SKIPPED, error="Not configured")

        try:
            from googleapiclient.discovery import build
            from googleapiclient.http import MediaFileUpload
            from google.oauth2.credentials import Credentials
            import json

            # Load credentials
            creds = None
            if self.token_file.exists():
                creds = Credentials.from_authorized_user_file(str(self.token_file))

            if not creds or not creds.valid:
                # For first-time setup, user needs to run OAuth flow manually
                return PublishResult(
                    platform="youtube", status=PublishStatus.ERROR,
                    error="OAuth token not found. Run CLI auth first.",
                )

            youtube = build("youtube", "v3", credentials=creds)
            body = {
                "snippet": {
                    "title": metadata.youtube_title[:100],
                    "description": metadata.youtube_description,
                    "tags": list(metadata.hashtags)[:30],
                    "categoryId": "22",
                },
                "status": {"privacyStatus": "public", "selfDeclaredMadeForKids": False},
            }
            media = MediaFileUpload(str(video_path), mimetype="video/mp4", resumable=True)
            request = youtube.videos().insert(part="snippet,status", body=body, media_body=media)

            response = None
            while response is None:
                _, response = request.next_chunk()

            video_id = response["id"]
            url = f"https://youtube.com/shorts/{video_id}"
            logger.info("youtube_published", video_id=video_id)
            return PublishResult(platform="youtube", status=PublishStatus.PUBLISHED, url=url)

        except ImportError:
            return PublishResult(platform="youtube", status=PublishStatus.ERROR,
                                 error="google-api-python-client not installed")
        except Exception as e:
            logger.error("youtube_publish_failed", error=str(e))
            return PublishResult(platform="youtube", status=PublishStatus.ERROR, error=str(e)[:200])


class TikTokPublisher(PublisherBackend):
    """TikTok via Upload-Post API. Note: requires paid Basic plan ($24/mo) for TikTok posting."""

    def __init__(self, api_key: str):
        self.api_key = api_key

    @property
    def platform_name(self) -> str:
        return "tiktok"

    async def is_configured(self) -> bool:
        return bool(self.api_key)

    async def publish(self, video_path: Path, metadata: ClipMetadata) -> PublishResult:
        if not await self.is_configured():
            return PublishResult(platform="tiktok", status=PublishStatus.SKIPPED, error="Not configured")

        try:
            async with httpx.AsyncClient(timeout=120) as client:
                with open(video_path, "rb") as f:
                    resp = await client.post(
                        "https://api.upload-post.com/api/upload",
                        headers={"Authorization": f"Bearer {self.api_key}"},
                        files={"file": (video_path.name, f, "video/mp4")},
                        data={"title": metadata.tiktok_caption, "platforms": "tiktok"},
                    )
                if resp.status_code != 200:
                    return PublishResult(platform="tiktok", status=PublishStatus.ERROR,
                                         error=resp.text[:200])
                logger.info("tiktok_published")
                return PublishResult(platform="tiktok", status=PublishStatus.PUBLISHED)
        except Exception as e:
            logger.error("tiktok_publish_failed", error=str(e))
            return PublishResult(platform="tiktok", status=PublishStatus.ERROR, error=str(e)[:200])


class InstagramPublisher(PublisherBackend):
    """Instagram Graph API Reels publishing."""

    def __init__(self, access_token: str, user_id: str):
        self.access_token = access_token
        self.user_id = user_id

    @property
    def platform_name(self) -> str:
        return "instagram"

    async def is_configured(self) -> bool:
        return bool(self.access_token and self.user_id)

    async def publish(self, video_path: Path, metadata: ClipMetadata) -> PublishResult:
        if not await self.is_configured():
            return PublishResult(platform="instagram", status=PublishStatus.SKIPPED, error="Not configured")

        try:
            async with httpx.AsyncClient(timeout=120) as client:
                # Step 1: Create container
                create_resp = await client.post(
                    f"https://graph.instagram.com/v21.0/{self.user_id}/media",
                    data={
                        "media_type": "REELS",
                        "video_url": str(video_path),
                        "caption": metadata.instagram_caption,
                        "access_token": self.access_token,
                    },
                )
                create_data = create_resp.json()
                if "error" in create_data:
                    return PublishResult(platform="instagram", status=PublishStatus.ERROR,
                                         error=create_data["error"].get("message", ""))

                container_id = create_data["id"]

                # Step 2: Poll for processing
                import asyncio
                for _ in range(60):
                    status_resp = await client.get(
                        f"https://graph.instagram.com/v21.0/{container_id}",
                        params={"fields": "status_code", "access_token": self.access_token},
                    )
                    sd = status_resp.json()
                    if sd.get("status_code") == "FINISHED":
                        break
                    elif sd.get("status_code") == "ERROR":
                        return PublishResult(platform="instagram", status=PublishStatus.ERROR,
                                             error="Processing failed")
                    await asyncio.sleep(5)

                # Step 3: Publish
                pub_resp = await client.post(
                    f"https://graph.instagram.com/v21.0/{self.user_id}/media_publish",
                    data={"creation_id": container_id, "access_token": self.access_token},
                )
                pd = pub_resp.json()
                if "error" in pd:
                    return PublishResult(platform="instagram", status=PublishStatus.ERROR,
                                         error=pd["error"].get("message", ""))

                logger.info("instagram_published", media_id=pd.get("id"))
                return PublishResult(platform="instagram", status=PublishStatus.PUBLISHED)

        except Exception as e:
            logger.error("instagram_publish_failed", error=str(e))
            return PublishResult(platform="instagram", status=PublishStatus.ERROR, error=str(e)[:200])
