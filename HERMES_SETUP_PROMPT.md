# ClipForge v2 — Hermes VPS Setup Prompt

> Copy this ENTIRE prompt and paste it into your Hermes agent on your Oracle VPS.

---

## HERMES PROMPT (copy below this line)

You are setting up **ClipForge v2**, a production-grade AI video clipping platform.
Repo: https://github.com/kiru-ji/clipforge

### PHASE 1: UNDERSTAND THE CODEBASE

Before doing ANYTHING else:

1. Clone: `git clone https://github.com/kiru-ji/clipforge.git && cd clipforge`
2. Read EVERY file in `clipforge_v2/`:
   - `contracts.py` — Abstract base classes, DTOs, Observer pattern, Job state machine
   - `config.py` — Pydantic settings, thread-safe singleton, atomic .env writes, chmod 600
   - `asr.py` — Groq Whisper API with tenacity retry, async chunking, rate-limit handling
   - `llm.py` — Qwen/DashScope backend, sliding window for long transcripts, safe JSON parsing
   - `video.py` — FFmpeg CPU processor, SSRF protection, single-pass extract+reframe, SRT generation
   - `pipeline.py` — Orchestrator with job locking, atomic cache, error propagation, cleanup
   - `publisher.py` — YouTube/TikTok/Instagram publisher implementations
   - `app.py` — NiceGUI dashboard with password auth, per-session observers, clip preview
   - `requirements.txt`, `.env.example`, `README.md`

3. Confirm you understand:
   - Strategy pattern: backends swappable via abstract interfaces
   - Observer pattern: per-session progress (not global singleton)
   - Data flow: URL → download → extract audio → transcribe (Groq) → detect moments (Qwen) → extract+reframe clips (FFmpeg) → generate metadata (Qwen) → publish
   - Security: auth middleware, SSRF protection, .env chmod 600, localhost binding
   - Why Groq not local Whisper: CPU-only VPS, no GPU

**DO NOT proceed until confirmed.**

### PHASE 2: SUGGEST API KEYS

Present these with reasoning. WAIT for user approval.

```
1. GROQ_API_KEY (REQUIRED)
   - Used by: asr.py → GroqASRBackend
   - Purpose: Speech-to-text with word timestamps
   - Why: No GPU on this VPS. Groq = 30× realtime free tier.
   - Get: https://console.groq.com/keys
   - Cost: $0

2. QWEN_API_KEY (REQUIRED)
   - Used by: llm.py → QwenLLMBackend
   - Purpose: Moment detection + metadata generation
   - Base URL: https://dashscope.aliyuncs.com/compatible-mode/v1
   - Get: User's existing ModelScope/DashScope key
   - Cost: Pay-per-use (very cheap)

3. ADMIN_PASSWORD (REQUIRED)
   - Purpose: Web UI authentication
   - Set to a strong password of your choice
   - Without this, the UI is inaccessible (first-run setup mode)

4. YOUTUBE_CLIENT_SECRETS (OPTIONAL)
   - Purpose: Auto-publish to YouTube Shorts (~6 uploads/day free)
   - Get: Google Cloud Console → YouTube Data API v3 → OAuth2 Desktop

5. UPLOAD_POST_API_KEY (OPTIONAL)
   - Purpose: TikTok publishing (10 free/month)
   - Get: https://app.upload-post.com/api-keys

6. INSTAGRAM_ACCESS_TOKEN + INSTAGRAM_USER_ID (OPTIONAL)
   - Purpose: Instagram Reels (requires App Review, 2-4 weeks)
```

**WAIT for approval before proceeding.**

### PHASE 3: DEPLOY

After user approves and provides keys:

```bash
cd /home/ubuntu/clipforge  # or wherever cloned

# Use python3.12 (NOT system python3.14)
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r clipforge_v2/requirements.txt

# System deps
sudo apt update && sudo apt install -y ffmpeg
ffmpeg -version  # verify

# Create .env from approved keys
cp clipforge_v2/.env.example .env
# Edit .env with user-provided values
chmod 600 .env

# Start server (RUN FROM REPO ROOT, not clipforge_v2/)
python -m clipforge_v2.app
```

Tell user: "ClipForge running at 127.0.0.1:8090. Access via SSH tunnel:
`ssh -L 8090:127.0.0.1:8090 user@vps-ip` then open http://localhost:8090"

### RULES

- NEVER skip Phase 1
- NEVER add keys without approval
- NEVER install GPU packages
- NEVER bind to 0.0.0.0 (use 127.0.0.1 + SSH tunnel)
- ALWAYS use python3.12 for venv
- ALWAYS run `python -m clipforge_v2.app` from repo ROOT
- ALWAYS chmod 600 on .env
