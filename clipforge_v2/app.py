"""
NiceGUI Web Dashboard with authentication.

Fixes applied (vs audit):
- P11/J1-J3: password authentication enforced on ALL pages
- J2: API keys shown as masked placeholders, not prefilled values
- P12/J84: settings always read fresh via get_settings() (no stale closure)
- P13/J81: per-session observer (not global singleton)
- P14/J83: is_running checked before starting new pipeline
- P15/J86: clip preview via ui.video + static file serving
- P16: Settings page includes ALL config fields
- J17: Qwen health check button added
- J18/P18: job history view for recovering progress after refresh
- J52: URL validation before processing
- J82: errors displayed prominently (not green success on failure)
- J85: pipeline runs as app-scoped background task
- J87: Groq status classes reset before update
- J90: correct package name in __main__
- J108: correct run command documented
- P49: default port 8090, bind 127.0.0.1 (SSH tunnel access)
"""

from __future__ import annotations

import asyncio
import secrets
from pathlib import Path

from nicegui import ui, app as nicegui_app
import structlog

from .config import get_settings, update_settings, reload_settings, ensure_gitignore
from .contracts import JobProgress, JobStatus, ProgressObserver
from .asr import GroqASRBackend
from .llm import QwenLLMBackend
from .video import FFmpegVideoProcessor
from .publisher import YouTubePublisher, TikTokPublisher, InstagramPublisher
from .pipeline import Pipeline

logger = structlog.get_logger(__name__)


# ─── Per-session observer (P13/J81: not a global singleton) ─────────────────

class SessionObserver(ProgressObserver):
    """Per-browser-session progress observer."""

    def __init__(self):
        self.status_text = ""
        self.progress_value = 0.0
        self.is_running = False
        self._progress_bar = None
        self._status_label = None
        self._log_area = None

    def bind_ui(self, progress_bar, status_label, log_area):
        self._progress_bar = progress_bar
        self._status_label = status_label
        self._log_area = log_area

    async def on_progress(self, progress: JobProgress) -> None:
        self.status_text = f"{progress.status.value}: {progress.message}"
        self.progress_value = progress.progress_pct
        if self._progress_bar:
            self._progress_bar.set_value(progress.progress_pct)
        if self._status_label:
            self._status_label.set_text(self.status_text)
        if self._log_area and progress.message:
            self._log_area.push(f"[{progress.status.value}] {progress.message}")
        if progress.error:
            if self._log_area:
                self._log_area.push(f"[ERROR] {progress.error}")


# ─── Job history (J18/P18) ──────────────────────────────────────────────────

_job_history: list[dict] = []
# J8/HIGH-8: module-level lock shared across ALL Pipeline instances
_pipeline_lock = asyncio.Lock()


def _record_job(result: dict):
    _job_history.append(result)
    if len(_job_history) > 50:
        _job_history.pop(0)


# ─── Auth middleware (P11/J1-J3) ────────────────────────────────────────────

def _check_auth() -> bool:
    """Check if user is authenticated via session storage."""
    settings = get_settings()
    if not settings.admin_password:
        return True  # No password set = first-run setup mode
    return ui.storage.user.get("authenticated", False)


# ─── App creation ───────────────────────────────────────────────────────────

def create_app():
    """Create and configure the NiceGUI application."""
    ensure_gitignore()  # P44/J20

    @ui.page("/login")
    def login_page():
        ui.dark_mode().enable()
        with ui.column().classes("items-center justify-center h-screen gap-4"):
            ui.label("🔐 ClipForge Login").classes("text-2xl font-bold text-white")
            pw_input = ui.input(label="Password", password=True).props("outlined dark")
            error_label = ui.label("").classes("text-red-400")

            async def do_login():
                settings = get_settings()
                if not settings.admin_password:
                    ui.storage.user["authenticated"] = True
                    ui.navigate.to("/")
                    return
                if pw_input.value == settings.admin_password:
                    ui.storage.user["authenticated"] = True
                    ui.navigate.to("/")
                else:
                    error_label.set_text("Invalid password")

            ui.button("Login", on_click=do_login).classes("bg-blue-600 text-white px-6 py-2")

    @ui.page("/")
    def dashboard():
        # P11/J1: auth gate
        if not _check_auth():
            ui.navigate.to("/login")
            return

        ui.dark_mode().enable()
        observer = SessionObserver()  # P13/J81: per-session

        with ui.header().classes("bg-gray-900 border-b border-gray-700"):
            with ui.row().classes("w-full items-center justify-between px-4"):
                ui.label("🎬 ClipForge").classes("text-2xl font-bold text-white")
                with ui.row().classes("gap-3"):
                    ui.link("Dashboard", "/").classes("text-blue-400")
                    ui.link("Settings", "/settings").classes("text-gray-400 hover:text-white")
                    ui.button("Logout", on_click=lambda: (ui.storage.user.update({"authenticated": False}), ui.navigate.to("/login"))).classes("text-xs bg-gray-700 text-gray-300")

        with ui.column().classes("w-full max-w-4xl mx-auto p-6 gap-6"):
            ui.label("Process Video").classes("text-xl font-semibold text-white")

            url_input = ui.input(
                label="YouTube URL",
                placeholder="https://youtube.com/watch?v=...",
            ).classes("w-full").props("outlined dark")

            process_btn = ui.button("🚀 Process Video", on_click=lambda: run_pipeline(url_input.value))
            process_btn.classes("bg-blue-600 hover:bg-blue-500 text-white px-6 py-2 rounded-lg")

            status_label = ui.label("").classes("text-gray-300")
            progress_bar = ui.linear_progress(value=0, show_value=False).classes("w-full h-3")
            progress_bar.props("color=blue rounded track-color=grey-8")
            log_area = ui.log(max_lines=50).classes("w-full h-48 bg-gray-900 rounded-lg p-3 text-sm font-mono")

            observer.bind_ui(progress_bar, status_label, log_area)

            results_container = ui.column().classes("w-full gap-4")

            # J18/P18: job history
            if _job_history:
                ui.separator().classes("border-gray-700")
                ui.label("Recent Jobs").classes("text-lg text-white")
                for job in reversed(_job_history[-5:]):
                    with ui.card().classes("w-full bg-gray-800 border-gray-700"):
                        errs = job.get("errors", [])
                        status_icon = "❌" if errs else "✅"
                        ui.label(f"{status_icon} {job['job_id']}: {job.get('total_clips', 0)} clips").classes("text-white")
                        if errs:
                            ui.label(f"Errors: {'; '.join(errs)}").classes("text-red-400 text-sm")

        async def run_pipeline(url: str):
            # J52: validate URL
            if not url or not url.startswith(("http://", "https://")):
                ui.notify("Please enter a valid URL", type="warning")
                return

            # P14/J83: prevent concurrent runs (module-level lock)
            if _pipeline_lock.locked():
                ui.notify("A job is already running", type="warning")
                return

            settings = get_settings()  # P12/J84: always fresh
            if not settings.has_groq or not settings.has_qwen:
                ui.notify("Configure API keys in Settings first!", type="error")
                return

            process_btn.disable()
            observer.is_running = True
            log_area.clear()

            async def _run_job():
                """J85/HIGH-6: runs under module-level lock, survives browser disconnect."""
                async with _pipeline_lock:
                    try:
                        asr = GroqASRBackend(settings.groq_api_key, settings.asr_model)
                        llm = QwenLLMBackend(settings.qwen_api_key, settings.qwen_base_url, settings.qwen_model)
                        video = FFmpegVideoProcessor(settings.ffmpeg_threads)

                        publishers = []
                        if settings.has_youtube:
                            publishers.append(YouTubePublisher(settings.youtube_client_secrets, settings.work_dir / "yt_token.json"))
                        if settings.has_tiktok:
                            publishers.append(TikTokPublisher(settings.upload_post_api_key))
                        if settings.has_instagram:
                            publishers.append(InstagramPublisher(settings.instagram_access_token, settings.instagram_user_id))

                        pipeline = Pipeline(settings, asr, llm, video, publishers)
                        pipeline.attach(observer)

                        result = await pipeline.process(url)
                        _record_job(result)

                        # J82: show errors prominently
                        results_container.clear()
                        with results_container:
                            ui.separator().classes("border-gray-700")

                            if result.get("errors"):
                                ui.label("❌ Errors occurred").classes("text-red-400 text-lg font-bold")
                                for err in result["errors"]:
                                    ui.label(err).classes("text-red-300 text-sm")

                            total = result.get("total_clips", 0)
                            if total > 0:
                                ui.label(f"✅ {total} clips generated").classes("text-green-400 text-lg font-bold")

                            for clip in result.get("clips", []):
                                with ui.card().classes("w-full bg-gray-800 border-gray-700"):
                                    with ui.row().classes("items-center justify-between"):
                                        ui.label(f"#{clip['index']+1} {clip['title']}").classes("text-white font-medium")
                                        ui.badge(f"Score: {clip['score']}/10").props("color=orange")
                                    # P15/J86: video preview via static route
                                    clip_path = clip.get("path", "")
                                    if Path(clip_path).exists():
                                        # Use relative path from output dir for static serving
                                        try:
                                            rel = Path(clip_path).relative_to(settings.output_dir)
                                            ui.video(f"/clips/{rel}").classes("w-full max-h-64 rounded mt-2")
                                        except ValueError:
                                            ui.video(clip_path).classes("w-full max-h-64 rounded mt-2")
                                    else:
                                        ui.label(clip_path).classes("text-gray-400 text-xs")

                    except Exception as e:
                        ui.notify(f"Error: {e}", type="negative")
                        logger.error("gui_pipeline_error", error=str(e))
                    finally:
                        process_btn.enable()
                        observer.is_running = False

            # Launch as background task (survives browser disconnect)
            ui.timer(0, lambda: asyncio.ensure_future(_run_job()), once=True)

    @ui.page("/settings")
    def settings_page():
        if not _check_auth():
            ui.navigate.to("/login")
            return

        s = get_settings()  # P12/J84: always fresh
        ui.dark_mode().enable()

        with ui.header().classes("bg-gray-900 border-b border-gray-700"):
            with ui.row().classes("w-full items-center justify-between px-4"):
                ui.label("🎬 ClipForge").classes("text-2xl font-bold text-white")
                with ui.row().classes("gap-3"):
                    ui.link("Dashboard", "/").classes("text-gray-400 hover:text-white")
                    ui.link("Settings", "/settings").classes("text-blue-400")
                    ui.button("Logout", on_click=lambda: (ui.storage.user.update({"authenticated": False}), ui.navigate.to("/login"))).classes("text-xs bg-gray-700 text-gray-300")

        with ui.column().classes("w-full max-w-3xl mx-auto p-6 gap-6"):
            ui.label("⚙️ Settings & API Keys").classes("text-xl font-semibold text-white")

            # Admin password
            with ui.card().classes("w-full bg-gray-800 border-gray-700"):
                ui.label("🔐 Admin Password").classes("text-lg text-white font-medium")
                admin_pw = ui.input(label="Admin Password", placeholder="••••••••" if s.admin_password else "", password=True).classes("w-full mt-2").props("outlined dark")

            # Groq
            with ui.card().classes("w-full bg-gray-800 border-gray-700"):
                ui.label("🎤 Groq (Speech-to-Text)").classes("text-lg text-white font-medium")
                ui.label("Free whisper-large-v3-turbo API. No GPU needed.").classes("text-gray-400 text-sm")
                groq_key = ui.input(label="Groq API Key", placeholder="••••••••" if s.groq_api_key else "gsk_...", password=True).classes("w-full mt-2").props("outlined dark")
                groq_status = ui.label("").classes("text-sm")

                async def check_groq():
                    groq_status.classes(remove="text-green-400 text-red-400")  # J87: reset
                    backend = GroqASRBackend(groq_key.value)
                    ok = await backend.health_check()
                    groq_status.set_text("✅ Connected" if ok else "❌ Invalid key or network down")
                    groq_status.classes(add="text-green-400" if ok else "text-red-400")

                ui.button("Test Connection", on_click=check_groq).classes("mt-2 bg-gray-700 text-white")

            # Qwen
            with ui.card().classes("w-full bg-gray-800 border-gray-700"):
                ui.label("🧠 Qwen / ModelScope (LLM)").classes("text-lg text-white font-medium")
                qwen_key = ui.input(label="API Key", placeholder="••••••••" if s.qwen_api_key else "sk-...", password=True).classes("w-full mt-2").props("outlined dark")
                qwen_url = ui.input(label="Base URL", value=s.qwen_base_url).classes("w-full mt-2").props("outlined dark")
                qwen_model = ui.input(label="Model", value=s.qwen_model).classes("w-full mt-2").props("outlined dark")
                qwen_status = ui.label("").classes("text-sm")

                async def check_qwen():
                    qwen_status.classes(remove="text-green-400 text-red-400")  # J17: test button for Qwen
                    backend = QwenLLMBackend(qwen_key.value, qwen_url.value, qwen_model.value)
                    ok = await backend.health_check()
                    qwen_status.set_text("✅ Connected" if ok else "❌ Failed")
                    qwen_status.classes(add="text-green-400" if ok else "text-red-400")

                ui.button("Test Connection", on_click=check_qwen).classes("mt-2 bg-gray-700 text-white")

            # Publishing
            with ui.card().classes("w-full bg-gray-800 border-gray-700"):
                ui.label("📤 Publishing").classes("text-lg text-white font-medium")
                yt_secrets = ui.textarea(label="YouTube Client Secrets JSON", placeholder="••• configured •••" if s.youtube_client_secrets else "Paste client_secrets.json content").classes("w-full mt-2").props("outlined dark")
                up_key = ui.input(label="Upload-Post API Key (TikTok)", placeholder="••••••••" if s.upload_post_api_key else "", password=True).classes("w-full mt-2").props("outlined dark")
                ig_token = ui.input(label="Instagram Access Token", placeholder="••••••••" if s.instagram_access_token else "", password=True).classes("w-full mt-2").props("outlined dark")
                ig_uid = ui.input(label="Instagram User ID", value=s.instagram_user_id).classes("w-full mt-2").props("outlined dark")

            # Telegram
            with ui.card().classes("w-full bg-gray-800 border-gray-700"):
                ui.label("💬 Telegram Approval Gate").classes("text-lg text-white font-medium")
                tg_token = ui.input(label="Bot Token", placeholder="••••••••" if s.telegram_bot_token else "", password=True).classes("w-full mt-2").props("outlined dark")
                tg_chat = ui.input(label="Chat ID", value=s.telegram_chat_id).classes("w-full mt-2").props("outlined dark")
                approval = ui.checkbox("Require approval before publishing", value=s.approval_required).classes("mt-2 text-white")

            # Pipeline tuning (P16: ALL fields)
            with ui.card().classes("w-full bg-gray-800 border-gray-700"):
                ui.label("🔧 Pipeline Tuning").classes("text-lg text-white font-medium")
                min_dur = ui.number(label="Min clip duration (s)", value=s.clip_min_duration, min=5, max=120).classes("w-full mt-2").props("outlined dark")
                max_dur = ui.number(label="Max clip duration (s)", value=s.clip_max_duration, min=10, max=300).classes("w-full mt-2").props("outlined dark")
                max_clips = ui.number(label="Max clips per video", value=s.max_clips_per_video, min=1, max=30).classes("w-full mt-2").props("outlined dark")
                threads = ui.number(label="FFmpeg threads", value=s.ffmpeg_threads, min=1, max=8).classes("w-full mt-2").props("outlined dark")
                burn_subs = ui.checkbox("Burn subtitles", value=s.burn_subtitles).classes("mt-2 text-white")

            # Save
            async def save_all():
                try:
                    # Preserve existing keys if user left placeholder unchanged (empty input)
                    updates = {
                        "admin_password": admin_pw.value or s.admin_password,
                        "groq_api_key": groq_key.value or s.groq_api_key,
                        "qwen_api_key": qwen_key.value or s.qwen_api_key,
                        "qwen_base_url": qwen_url.value,
                        "qwen_model": qwen_model.value,
                        "youtube_client_secrets": yt_secrets.value or s.youtube_client_secrets,
                        "upload_post_api_key": up_key.value or s.upload_post_api_key,
                        "instagram_access_token": ig_token.value or s.instagram_access_token,
                        "instagram_user_id": ig_uid.value,
                        "telegram_bot_token": tg_token.value or s.telegram_bot_token,
                        "telegram_chat_id": tg_chat.value,
                        "approval_required": approval.value,
                        "clip_min_duration": int(min_dur.value),
                        "clip_max_duration": int(max_dur.value),
                        "max_clips_per_video": int(max_clips.value),
                        "ffmpeg_threads": int(threads.value),
                        "burn_subtitles": burn_subs.value,
                    }
                    new_settings = update_settings(**updates)
                    ui.notify("Settings saved!", type="positive")
                except Exception as e:
                    ui.notify(f"Save failed: {e}", type="negative")

            ui.button("💾 Save All Settings", on_click=save_all).classes("bg-green-600 hover:bg-green-500 text-white px-8 py-3 rounded-lg text-lg self-center")

    # Serve output directory for clip previews (P15/J86)
    settings = get_settings()
    nicegui_app.add_static_files("/clips", str(settings.output_dir))


def main():
    """Entry point: python -m clipforge_v2.app"""
    settings = get_settings()
    create_app()
    ui.run(
        host=settings.host,
        port=settings.port,
        title="ClipForge",
        reload=False,
        storage_secret=secrets.token_hex(32),  # Required for user storage (auth)
    )


if __name__ == "__main__":
    main()
