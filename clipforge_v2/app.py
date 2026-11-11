"""
NiceGUI Web Dashboard — credentials management + automation control.

Design: Event-driven reactive UI (not Streamlit-style rerun-everything).
- Settings page: enter API keys, save to .env, live health checks
- Dashboard: paste URL → process → watch progress in real-time
- Clips gallery: preview generated clips with metadata
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from nicegui import ui, app
import structlog

from .config import get_settings, reload_settings
from .contracts import JobProgress, JobStatus, ProgressObserver
from .asr import GroqASRBackend
from .llm import QwenLLMBackend
from .video import FFmpegVideoProcessor
from .pipeline import Pipeline

logger = structlog.get_logger(__name__)


class GUIProgressObserver(ProgressObserver):
    """Bridges pipeline progress to NiceGUI reactive state."""

    def __init__(self):
        self.status_text = ""
        self.progress_value = 0.0
        self.is_running = False
        self._ui_progress = None
        self._ui_status = None
        self._ui_log = None

    def bind_ui(self, progress_bar, status_label, log_area):
        self._ui_progress = progress_bar
        self._ui_status = status_label
        self._ui_log = log_area

    async def on_progress(self, progress: JobProgress) -> None:
        self.status_text = f"{progress.status.value}: {progress.message}"
        self.progress_value = progress.progress_pct
        if self._ui_progress:
            self._ui_progress.set_value(progress.progress_pct)
        if self._ui_status:
            self._ui_status.set_text(self.status_text)
        if self._ui_log and progress.message:
            self._ui_log.push(f"[{progress.status.value}] {progress.message}")


# Global observer
observer = GUIProgressObserver()


def create_app():
    """Create and configure the NiceGUI application."""
    settings = get_settings()

    @ui.page("/")
    def dashboard():
        ui.dark_mode().enable()
        ui.add_head_html('<style>body { font-family: Inter, system-ui, sans-serif; }</style>')

        with ui.header().classes("bg-gray-900 border-b border-gray-700"):
            with ui.row().classes("w-full items-center justify-between px-4"):
                ui.label("🎬 ClipForge").classes("text-2xl font-bold text-white")
                with ui.row().classes("gap-2"):
                    ui.link("Dashboard", "/").classes("text-blue-400 hover:text-blue-300")
                    ui.link("Settings", "/settings").classes("text-gray-400 hover:text-white")

        with ui.column().classes("w-full max-w-4xl mx-auto p-6 gap-6"):
            ui.label("Process Video").classes("text-xl font-semibold text-white")

            url_input = ui.input(
                label="YouTube URL",
                placeholder="https://youtube.com/watch?v=...",
            ).classes("w-full").props("outlined dark")

            with ui.row().classes("gap-4 items-center"):
                process_btn = ui.button("🚀 Process Video", on_click=lambda: run_pipeline(url_input.value))
                process_btn.classes("bg-blue-600 hover:bg-blue-500 text-white px-6 py-2 rounded-lg")

            # Progress section
            status_label = ui.label("").classes("text-gray-300")
            progress_bar = ui.linear_progress(value=0, show_value=False).classes("w-full h-3")
            progress_bar.props("color=blue rounded track-color=grey-8")

            log_area = ui.log(max_lines=50).classes("w-full h-48 bg-gray-900 rounded-lg p-3 text-sm font-mono")

            # Bind observer to UI elements
            observer.bind_ui(progress_bar, status_label, log_area)

            # Results area
            results_container = ui.column().classes("w-full gap-4")

        async def run_pipeline(url: str):
            if not url:
                ui.notify("Please enter a URL", type="warning")
                return
            if not settings.has_groq or not settings.has_qwen:
                ui.notify("Configure API keys in Settings first!", type="error")
                return

            process_btn.disable()
            observer.is_running = True
            log_area.clear()

            try:
                asr = GroqASRBackend(settings.groq_api_key, settings.asr_model)
                llm = QwenLLMBackend(settings.qwen_api_key, settings.qwen_base_url, settings.qwen_model)
                video = FFmpegVideoProcessor(settings.ffmpeg_threads)

                pipeline = Pipeline(settings, asr, llm, video, publishers=[])
                pipeline.attach(observer)

                result = await pipeline.process(url)

                # Show results
                results_container.clear()
                with results_container:
                    ui.separator().classes("border-gray-700")
                    ui.label(f"✅ {result.get('total_clips', 0)} clips generated").classes("text-green-400 text-lg font-bold")

                    for clip in result.get("clips", []):
                        with ui.card().classes("w-full bg-gray-800 border-gray-700"):
                            with ui.row().classes("items-center justify-between"):
                                ui.label(f"#{clip['index']+1} {clip['title']}").classes("text-white font-medium")
                                ui.badge(f"Score: {clip['score']}/10").props("color=orange")
                            ui.label(clip["path"]).classes("text-gray-400 text-xs")

            except Exception as e:
                ui.notify(f"Error: {e}", type="negative")
                logger.error("gui_pipeline_error", error=str(e))
            finally:
                process_btn.enable()
                observer.is_running = False

    @ui.page("/settings")
    def settings_page():
        s = get_settings()

        ui.dark_mode().enable()

        with ui.header().classes("bg-gray-900 border-b border-gray-700"):
            with ui.row().classes("w-full items-center justify-between px-4"):
                ui.label("🎬 ClipForge").classes("text-2xl font-bold text-white")
                with ui.row().classes("gap-2"):
                    ui.link("Dashboard", "/").classes("text-gray-400 hover:text-white")
                    ui.link("Settings", "/settings").classes("text-blue-400 hover:text-blue-300")

        with ui.column().classes("w-full max-w-3xl mx-auto p-6 gap-6"):
            ui.label("⚙️ Settings & API Keys").classes("text-xl font-semibold text-white")

            # Groq
            with ui.card().classes("w-full bg-gray-800 border-gray-700"):
                ui.label("🎤 Groq (Speech-to-Text)").classes("text-lg text-white font-medium")
                ui.label("Free whisper-large-v3-turbo API. No GPU needed.").classes("text-gray-400 text-sm")
                groq_key = ui.input(label="Groq API Key", value=s.groq_api_key, password=True).classes("w-full mt-2").props("outlined dark")
                groq_status = ui.label("").classes("text-sm")

                async def check_groq():
                    backend = GroqASRBackend(groq_key.value)
                    ok = await backend.health_check()
                    groq_status.set_text("✅ Connected" if ok else "❌ Invalid key")
                    groq_status.classes("text-green-400" if ok else "text-red-400")

                ui.button("Test Connection", on_click=check_groq).classes("mt-2 bg-gray-700 text-white")

            # Qwen / ModelScope
            with ui.card().classes("w-full bg-gray-800 border-gray-700"):
                ui.label("🧠 Qwen / ModelScope (LLM)").classes("text-lg text-white font-medium")
                ui.label("Moment detection + metadata generation.").classes("text-gray-400 text-sm")
                qwen_key = ui.input(label="API Key", value=s.qwen_api_key, password=True).classes("w-full mt-2").props("outlined dark")
                qwen_url = ui.input(label="Base URL", value=s.qwen_base_url).classes("w-full mt-2").props("outlined dark")
                qwen_model = ui.input(label="Model", value=s.qwen_model).classes("w-full mt-2").props("outlined dark")

            # Publishing
            with ui.card().classes("w-full bg-gray-800 border-gray-700"):
                ui.label("📤 Publishing").classes("text-lg text-white font-medium")
                yt_secrets = ui.textarea(label="YouTube Client Secrets JSON", value=s.youtube_client_secrets).classes("w-full mt-2").props("outlined dark")
                up_key = ui.input(label="Upload-Post API Key (TikTok)", value=s.upload_post_api_key, password=True).classes("w-full mt-2").props("outlined dark")

            # Save button
            async def save_all():
                s.update_credentials(
                    groq_api_key=groq_key.value,
                    qwen_api_key=qwen_key.value,
                    qwen_base_url=qwen_url.value,
                    qwen_model=qwen_model.value,
                    youtube_client_secrets=yt_secrets.value,
                    upload_post_api_key=up_key.value,
                )
                reload_settings()
                ui.notify("Settings saved!", type="positive")

            ui.button("💾 Save All Settings", on_click=save_all).classes("bg-green-600 hover:bg-green-500 text-white px-8 py-3 rounded-lg text-lg self-center")

    return app


def main():
    """Entry point: python -m clipforge.app"""
    settings = get_settings()
    create_app()
    ui.run(host=settings.host, port=settings.port, title="ClipForge", reload=False)


if __name__ == "__main__":
    main()
