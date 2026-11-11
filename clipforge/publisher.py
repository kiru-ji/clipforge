"""Publishing to YouTube, Instagram, TikTok with proper API handling."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional

import httpx
import structlog

logger = structlog.get_logger(__name__)


class Publisher:
    """
    Multi-platform publisher with verified 2026 API constraints.

    YouTube: Direct API, 100 uploads/day, public by default (Nov 2, 2026 reversal)
    Instagram: Graph API, 100 posts/24h, requires App Review + resumable upload
    TikTok: Via Upload-Post API (10 free/month) — direct API requires audit
    """

    def __init__(
        self,
        youtube_secrets: str = "",
        youtube_token_file: Optional[Path] = None,
        instagram_token: str = "",
        instagram_user_id: str = "",
        upload_post_key: str = "",
    ):
        self.youtube_secrets = youtube_secrets
        self.youtube_token_file = youtube_token_file
        self.instagram_token = instagram_token
        self.instagram_user_id = instagram_user_id
        self.upload_post_key = upload_post_key

    async def publish_youtube(self, video_path: Path, metadata: dict) -> dict:
        """
        Upload to YouTube Shorts via Data API v3.

        As of Nov 2, 2026: unverified apps can upload publicly.
        Quota: 100 videos.insert/day in dedicated bucket.
        """
        if not self.youtube_secrets:
            logger.warning("youtube_not_configured")
            return {"platform": "youtube", "status": "skipped", "reason": "not configured"}

        try:
            from googleapiclient.discovery import build
            from googleapiclient.http import MediaFileUpload
            from google_auth_oauthlib.flow import InstalledAppFlow
            from google.oauth2.credentials import Credentials

            # Load or create credentials
            creds = None
            if self.youtube_token_file and self.youtube_token_file.exists():
                creds = Credentials.from_authorized_user_file(str(self.youtube_token_file))

            if not creds or not creds.valid:
                if creds and creds.expired and creds.refresh_token:
                    from google.auth.transport.requests import Request
                    creds.refresh(Request())
                else:
                    flow = InstalledAppFlow.from_client_secrets_file(
                        self.youtube_secrets,
                        ["https://www.googleapis.com/auth/youtube.upload"],
                    )
                    creds = flow.run_local_server(port=8080)

                if self.youtube_token_file:
                    self.youtube_token_file.parent.mkdir(parents=True, exist_ok=True)
                    self.youtube_token_file.write_text(creds.to_json())

            youtube = build("youtube", "v3", credentials=creds)

            body = {
                "snippet": {
                    "title": metadata.get("youtube_title", "Untitled Short"),
                    "description": metadata.get("youtube_description", ""),
                    "tags": metadata.get("hashtags", [])[:30],
                    "categoryId": "22",  # People & Blogs
                },
                "status": {
                    "privacyStatus": "public",
                    "selfDeclaredMadeForKids": False,
                },
            }

            media = MediaFileUpload(str(video_path), mimetype="video/mp4", resumable=True)
            request = youtube.videos().insert(part="snippet,status", body=body, media_body=media)

            response = None
            while response is None:
                status, response = request.next_chunk()
                if status:
                    logger.info("youtube_upload_progress", progress=f"{int(status.progress() * 100)}%")

            video_id = response["id"]
            logger.info("youtube_published", video_id=video_id)
            return {"platform": "youtube", "status": "published", "video_id": video_id, "url": f"https://youtube.com/shorts/{video_id}"}

        except ImportError:
            logger.error("google_api_libraries_not_installed")
            return {"platform": "youtube", "status": "error", "reason": "missing google-api-python-client"}
        except Exception as e:
            logger.error("youtube_publish_failed", error=str(e))
            return {"platform": "youtube", "status": "error", "reason": str(e)[:200]}

    async def publish_instagram(self, video_path: Path, metadata: dict) -> dict:
        """
        Publish Reel via Instagram Graph API.

        Requires: App Review approved, professional account, Facebook Page linked.
        Rate limit: 100 posts/24h rolling window.
        Uses container/publish two-step flow.
        """
        if not self.instagram_token or not self.instagram_user_id:
            return {"platform": "instagram", "status": "skipped", "reason": "not configured"}

        try:
            async with httpx.AsyncClient(timeout=120) as client:
                # Step 1: Create media container
                # NOTE: For local files, you'd need resumable upload session.
                # This implementation assumes video is at a public URL.
                # For self-hosted, set up a simple file server or use S3.
                create_resp = await client.post(
                    f"https://graph.instagram.com/v21.0/{self.instagram_user_id}/media",
                    data={
                        "media_type": "REELS",
                        "video_url": str(video_path),  # Must be public HTTPS URL
                        "caption": metadata.get("instagram_caption", ""),
                        "access_token": self.instagram_token,
                    },
                )
                create_data = create_resp.json()
                if "error" in create_data:
                    return {"platform": "instagram", "status": "error", "reason": create_data["error"].get("message", "")}

                container_id = create_data["id"]

                # Step 2: Wait for processing
                for _ in range(60):  # Max 5 minutes
                    status_resp = await client.get(
                        f"https://graph.instagram.com/v21.0/{container_id}",
                        params={"fields": "status_code", "access_token": self.instagram_token},
                    )
                    status_data = status_resp.json()
                    if status_data.get("status_code") == "FINISHED":
                        break
                    elif status_data.get("status_code") == "ERROR":
                        return {"platform": "instagram", "status": "error", "reason": "processing failed"}
                    time.sleep(5)

                # Step 3: Publish
                pub_resp = await client.post(
                    f"https://graph.instagram.com/v21.0/{self.instagram_user_id}/media_publish",
                    data={
                        "creation_id": container_id,
                        "access_token": self.instagram_token,
                    },
                )
                pub_data = pub_resp.json()
                if "error" in pub_data:
                    return {"platform": "instagram", "status": "error", "reason": pub_data["error"].get("message", "")}

                logger.info("instagram_published", media_id=pub_data.get("id"))
                return {"platform": "instagram", "status": "published", "media_id": pub_data.get("id")}

        except Exception as e:
            logger.error("instagram_publish_failed", error=str(e))
            return {"platform": "instagram", "status": "error", "reason": str(e)[:200]}

    async def publish_tiktok(self, video_path: Path, metadata: dict) -> dict:
        """
        Publish to TikTok via Upload-Post API (free tier: 10/month).

        Direct TikTok Content Posting API requires audit that rejects
        personal/hobbyist apps. Upload-Post is the realistic $0 path.
        """
        if not self.upload_post_key:
            return {"platform": "tiktok", "status": "skipped", "reason": "Upload-Post key not configured"}

        try:
            async with httpx.AsyncClient(timeout=120) as client:
                # Upload file
                with open(video_path, "rb") as f:
                    upload_resp = await client.post(
                        "https://api.upload-post.com/api/upload",
                        headers={"Authorization": f"Bearer {self.upload_post_key}"},
                        files={"file": (video_path.name, f, "video/mp4")},
                        data={
                            "title": metadata.get("tiktok_caption", ""),
                            "platforms": "tiktok",
                        },
                    )

                result = upload_resp.json()
                if upload_resp.status_code != 200:
                    return {"platform": "tiktok", "status": "error", "reason": result.get("error", "upload failed")}

                logger.info("tiktok_published_via_uploadpost", result=result)
                return {"platform": "tiktok", "status": "published", "details": result}

        except Exception as e:
            logger.error("tiktok_publish_failed", error=str(e))
            return {"platform": "tiktok", "status": "error", "reason": str(e)[:200]}

    async def publish_all(self, video_path: Path, metadata: dict) -> list[dict]:
        """Publish to all configured platforms."""
        results = []
        results.append(await self.publish_youtube(video_path, metadata))
        results.append(await self.publish_instagram(video_path, metadata))
        results.append(await self.publish_tiktok(video_path, metadata))
        return [r for r in results if r["status"] != "skipped"]
