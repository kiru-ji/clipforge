# ClipForge v2 — Hermes VPS Setup Prompt

> Copy this ENTIRE prompt and paste it into your Hermes agent on your Oracle VPS.

---

## HERMES PROMPT (copy below this line)

You are setting up **ClipForge v2**, a production-grade AI video clipping and auto-posting platform. The repository is at: https://github.com/kiru-ji/clipforge

### PHASE 1: UNDERSTAND THE CODEBASE FIRST

Before doing ANYTHING else, you MUST:

1. Clone the repo: `git clone https://github.com/kiru-ji/clipforge.git && cd clipforge`
2. Read EVERY file in `clipforge_v2/` thoroughly:
   - `contracts.py` — Abstract base classes (ASRBackend, LLMBackend, VideoProcessor, PublisherBackend), DTOs (Transcript, ViralMoment, ClipMetadata), Observer pattern, Job state machine
   - `config.py` — Pydantic settings with runtime mutation, .env persistence, credential hot-reload
   - `asr.py` — Groq Whisper API backend with chunking for large files, rate-limit retry
   - `llm.py` — Qwen/DashScope OpenAI-compatible backend for moment detection + metadata generation
   - `video.py` — FFmpeg CPU-only processor (download via yt-dlp, extract, reframe to 9:16, burn subtitles)
   - `pipeline.py` — Orchestrator with progress notifications, transcript caching, error handling
   - `app.py` — NiceGUI web dashboard (Settings page for credentials + Dashboard for processing + Results gallery)
   - `requirements.txt` — All Python dependencies
   - `.env.example` — Configuration template
   - `README.md` — Architecture documentation

3. After reading, produce a summary confirming you understand:
   - The Strategy pattern: how backends are swappable via abstract interfaces
   - The Observer pattern: how pipeline progress flows to the GUI
   - The data flow: URL → download → transcribe (Groq) → detect moments (Qwen) → extract clips (FFmpeg) → generate metadata (Qwen) → publish
   - Why Groq is used instead of local Whisper (CPU-only VPS, no GPU)
   - Why NiceGUI was chosen over Streamlit/Gradio (event-driven, production-grade)

**DO NOT proceed to Phase 2 until you have confirmed understanding.**

### PHASE 2: SUGGEST API KEYS WITH REASING

Based on your understanding of the codebase, suggest which API keys are needed and WHY. Present them as a numbered list with this format:

```
1. **GROQ_API_KEY** (REQUIRED)
   - Used by: `asr.py` → GroqASRBackend
   - Purpose: Speech-to-text transcription with word-level timestamps
   - Why Groq: This VPS has no GPU. Local whisper.cpp would take 100+ minutes for a 30-min podcast. Groq runs whisper-large-v3-turbo at 30× realtime via API. Free tier is rate-limited but sufficient.
   - Get free key: https://console.groq.com/keys
   - Cost: $0 (free tier)

2. **QWEN_API_KEY** (REQUIRED)
   - Used by: `llm.py` → QwenLLMBackend
   - Purpose: Viral moment detection from transcript + platform-specific metadata generation
   - Why Qwen/DashScope: OpenAI-compatible endpoint, supports structured JSON output, good at creative content tasks
   - Base URL: https://dashscope.aliyuncs.com/compatible-mode/v1
   - Get key: [user should provide their existing ModelScope/DashScope key]
   - Cost: Pay-per-use, very cheap for this workload

3. **YOUTUBE_CLIENT_SECRETS** (OPTIONAL — for publishing)
   - Used by: YouTube Data API v3 videos.insert
   - Purpose: Auto-publish clips to YouTube Shorts
   - Why direct API: As of Nov 2, 2026, unverified apps can upload publicly. 100 uploads/day free. No audit needed for self-use.
   - Get credentials: Google Cloud Console → Create project → Enable YouTube Data API v3 → OAuth2 Desktop credentials
   - Cost: $0

4. **UPLOAD_POST_API_KEY** (OPTIONAL — for TikTok)
   - Used by: publisher.py → TikTok publishing
   - Purpose: Post clips to TikTok
   - Why Upload-Post: TikTok's direct Content Posting API requires audit that rejects personal/hobbyist apps. Unaudited = all posts private. Upload-Post uses their own approved app. Free tier = 10 uploads/month.
   - Get key: https://app.upload-post.com/api-keys
   - Cost: $0 (10/month free)

5. **INSTAGRAM_ACCESS_TOKEN + INSTAGRAM_USER_ID** (OPTIONAL — for Instagram)
   - Used by: publisher.py → Instagram Graph API
   - Purpose: Post Reels to Instagram
   - Caveat: Requires Meta App Review (free, 2-4 weeks). Professional IG account + Facebook Page needed. Resumable upload available (no public URL needed).
   - Rate limit: 100 posts/24h
   - Cost: $0

6. **TELEGRAM_BOT_TOKEN + TELEGRAM_CHAT_ID** (OPTIONAL — for approval gate)
   - Purpose: Human approval before publishing (YouTube inauthentic-content policy compliance)
   - Get bot token: @BotFather on Telegram
   - Cost: $0
```

**WAIT for user approval before proceeding. Do NOT add any keys without explicit user confirmation.**

### PHASE 3: SETUP AFTER APPROVAL

Once the user approves the suggested keys and provides them:

1. Create virtual environment and install dependencies:
```bash
cd /home/ubuntu/clipforge  # or wherever cloned
python3 -m venv .venv
source .venv/bin/activate
pip install -r clipforge_v2/requirements.txt
```

2. Install system dependencies:
```bash
sudo apt update && sudo apt install -y ffmpeg
# Verify:
ffmpeg -version
yt-dlp --version
```

3. Create .env file with approved keys:
```bash
cp clipforge_v2/.env.example clipforge_v2/.env
# Edit with user-provided keys
```

4. Create work directory:
```bash
mkdir -p /tmp/clipforge/{downloads,output,state}
```

5. Start the server:
```bash
cd clipforge_v2
source ../.venv/bin/activate
python -m clipforge_v2.app
```

6. Verify it's running:
```bash
curl -s http://localhost:8080 | head -5
```

7. Tell the user: "ClipForge is running at http://<VPS_IP>:8080. Open it in your browser, go to Settings to verify your API keys, then paste a YouTube URL on the Dashboard to start clipping!"

### IMPORTANT RULES

- NEVER skip Phase 1. Understanding the code prevents misconfiguration.
- NEVER add API keys without user approval. Always explain WHY each key is needed.
- NEVER install GPU packages (torch with CUDA, nvidia-cublas, etc.) — this is a CPU-only VPS.
- If pip install fails on any package, troubleshoot BEFORE proceeding.
- If ffmpeg is not available, install it BEFORE starting the server.
- Always use the virtual environment, never install globally.
- The server binds to 0.0.0.0:8080 by default. Make sure the VPS firewall allows port 8080.
