# ClipForge v2 — Cloud-Native AI Video Clipping Platform

> **$0/month** production-grade video clipping automation for **CPU-only VPS**
> with Qwen API + Groq Whisper. Full web GUI with authentication.

## Architecture

| Component | Implementation | Why |
|---|---|---|
| **ASR** | Groq Whisper API (free tier) | 30× realtime on LPU. No GPU needed. Word timestamps. |
| **LLM** | Qwen via DashScope API | OpenAI-compatible. Your existing ModelScope key. |
| **Video** | FFmpeg CPU (libx264) | Works on any ARM/x86 VPS. Single-pass extract+reframe. |
| **Frontend** | NiceGUI web dashboard | Python-native, event-driven, auth built-in. |
| **Publishing** | YouTube direct + TikTok Upload-Post + Instagram Graph | Real publisher implementations. |

## What's Actually Implemented

- ✅ Password-protected web UI (binds localhost by default, SSH tunnel access)
- ✅ Groq Whisper STT with chunking, retry, rate-limit handling
- ✅ Qwen moment detection with sliding window for long videos
- ✅ Single-pass FFmpeg extract+reframe+subtitle burn (one encode)
- ✅ Per-job download directories (no race conditions)
- ✅ SSRF protection (URL validation before yt-dlp)
- ✅ Atomic transcript caching with content-hash invalidation
- ✅ Job locking (prevents concurrent runs on same URL)
- ✅ Per-session progress observers (multi-tab safe)
- ✅ Clip video preview in browser
- ✅ Job history view
- ✅ All settings configurable via GUI
- ✅ Auto-cleanup of old files
- ✅ .gitignore prevents secret commits
- ✅ .env chmod 600 (not world-readable)
- ✅ YouTube/TikTok/Instagram publisher backends
- ✅ tenacity retry with exponential backoff
- ✅ Proper error propagation (no silent failures)

## Quick Start

```bash
# On your VPS:
git clone https://github.com/kiru-ji/clipforge.git && cd clipforge
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r clipforge_v2/requirements.txt
sudo apt install -y ffmpeg  # if not already installed

# Start server (binds 127.0.0.1:8090 by default)
python -m clipforge_v2.app

# Access via SSH tunnel from your local machine:
ssh -L 8090:127.0.0.1:8090 user@your-vps-ip
# Then open http://localhost:8090 in your browser
```

## Security

- Server binds `127.0.0.1` by default (not `0.0.0.0`)
- Password authentication required on all pages
- API keys stored in `.env` with `chmod 600`
- `.gitignore` prevents committing secrets
- URL validation blocks SSRF attacks
- No credentials exposed in HTML (password fields only)

## API Keys Needed (all free)

| Service | Purpose | Free Tier |
|---|---|---|
| **Groq** | Speech-to-text | Rate-limited whisper-large-v3-turbo |
| **ModelScope/Qwen** | Moment detection + metadata | Your existing key |
| **YouTube Data API** | Publishing | ~6 uploads/day (1600 units/upload) |
| **Upload-Post** | TikTok publishing | 10 uploads/month |

## License

MIT
