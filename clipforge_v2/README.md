# ClipForge v2 — Cloud-Native AI Video Clipping Platform

> **$0/month** production-grade video clipping automation designed for **Oracle VPS (no GPU)**
> with Qwen API from ModelScope. Full web GUI — just enter credentials and go.

## Architecture Redesign for CPU-Only VPS

Your Oracle VPS has no GPU. This changes everything:

| Component | Old (GPU) | New (CPU-only VPS) | Why |
|---|---|---|---|
| **ASR** | faster-whisper local GPU | **Groq Whisper API** (free tier) | Groq runs whisper-large-v3-turbo on LPU at 30× realtime. Free tier = rate-limited but sufficient. Word timestamps included. Zero GPU needed. |
| **Moment Detection** | Local Ollama qwen2.5:7b | **Qwen API via ModelScope** | You already have this. OpenAI-compatible endpoint. No local model hosting. |
| **Metadata Generation** | Local Ollama | **Qwen API via ModelScope** | Same API, same key. |
| **Video Processing** | FFmpeg + NVENC | **FFmpeg CPU** (libx264) | Oracle ARM A1 has 4 cores / 24GB RAM free tier. Sufficient for encoding. |
| **Frontend** | CLI only | **NiceGUI web dashboard** | Production-grade Python web UI. Credentials management, job monitoring, clip preview. |
| **Deployment** | Local machine | **Oracle VPS + systemd** | Always-on, accessible from anywhere via browser. |

### Why Groq for ASR (not local whisper.cpp)?

- **whisper.cpp on ARM CPU**: ~0.3× realtime for medium model. A 30-min podcast = **100+ minutes** to transcribe.
- **Groq Whisper API**: whisper-large-v3-turbo at **30× realtime**. Same 30-min podcast = **~1 minute**. Free tier is rate-limited but handles personal use easily.
- **Word timestamps**: Groq supports `timestamp_granularities=["word"]` natively.
- **Zero infrastructure**: No model downloads, no VRAM, no CUDA. Just an API key.

### Why NiceGUI for Frontend?

- **Python-native**: No Node.js/React build step. Single `pip install nicegui`.
- **Production-ready**: Built on FastAPI + Vue.js. Supports auth, file uploads, real-time updates.
- **Self-hosted**: Runs on your VPS, accessible via browser. No external dependencies.
- **Better than Streamlit**: Streamlit reruns entire script on every interaction. NiceGUI is event-driven like a real web app.
- **Better than Gradio**: Gradio is for ML demos. NiceGUI is for production dashboards.

## System Design Principles Applied

### OOP
- **Abstract base classes** for ASR, LLM, Publisher backends (Strategy Pattern)
- **Factory pattern** for creating backend instances from config
- **Observer pattern** for job progress notifications
- **Repository pattern** for job state persistence

### DSA
- **Priority queue** for job scheduling (higher viral score = higher priority)
- **Hash map** for idempotency keys (prevent duplicate processing)
- **B-tree indexed SQLite** for job history queries

### System Design
- **Microservice-ready monolith**: Clean module boundaries, can split later
- **Async pipeline**: Non-blocking I/O for API calls, concurrent clip extraction
- **Circuit breaker**: Auto-fallback if Groq/Qwen API is down
- **Idempotent operations**: Every step can be safely retried

### OS Concepts
- **Process pool** for CPU-bound FFmpeg encoding
- **Signal handling** for graceful shutdown
- **File locking** for concurrent job safety
- **Memory-mapped files** for large video I/O

## Quick Start on Oracle VPS

```bash
# 1. Clone and install
git clone <your-repo> && cd clipforge_v2
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. Configure via web UI
python -m clipforge.app
# Open http://your-vps-ip:8080 in browser
# Enter your API keys in Settings → Save

# 3. Start automation
# Paste a YouTube URL in the dashboard → Click "Process"
# Watch clips appear in real-time
```

## API Keys Needed (all free)

| Service | Purpose | Free Tier | Get Key |
|---|---|---|---|
| **Groq** | Speech-to-text | Rate-limited whisper-large-v3-turbo | console.groq.com |
| **ModelScope/Qwen** | Moment detection + metadata | Your existing key | modelscope.cn |
| **YouTube Data API** | Publishing | 100 uploads/day | console.cloud.google.com |
| **Upload-Post** | TikTok publishing | 10 uploads/month | upload-post.com |

## License

MIT
