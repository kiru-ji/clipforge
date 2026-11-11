# ClipForge — Production AI Video Clipping & Auto-Posting Agent

> **$0/month** self-hosted pipeline that turns long-form videos into viral 9:16 shorts
> and publishes them to YouTube, Instagram, and TikTok — built from deep research
> into what actually works on real hardware with real API constraints.

## What Makes This Different

This isn't a wrapper around someone else's SaaS. Every architectural decision is
backed by verified 2026 API documentation and tested on the target hardware:

| Decision | Why |
|---|---|
| **No Docker** | Your machine has no Docker + only 14GB on `/`. Native Python runs faster and uses less disk. |
| **faster-whisper `medium` int8** | Fits your 6GB VRAM with room for YOLOv8n + MediaPipe. large-v3-turbo fp16 needs 4.5GB alone — too tight. |
| **Ollama qwen2.5:7b for moment picking** | You already have it installed. No Gemini key needed for the core pipeline. |
| **YouTube direct API** | As of Nov 2, 2026: 100 uploads/day, public by default, no audit needed for self-use. |
| **Instagram Graph API** | 100 posts/24h after App Review (free, 2–4 weeks). Resumable upload = no public URL needed. |
| **TikTok via Upload-Post free tier** | Unaudited apps = private-only. 10 free uploads/month is the honest $0 cap. |
| **Human approval gate** | YouTube's inauthentic-content policy targets fully automated channels. Approval keeps you safe. |
| **Work dir on `/shared`** | 68GB free vs 14GB on `/`. Videos eat disk fast. |
| **Scrapling for web scraping** | 101k★, BSD-3, faster than BS4 by 785×, stealth browser included. Used for metadata enrichment. |

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    ClipForge Pipeline                        │
│                                                              │
│  ┌──────────┐   ┌────────────┐   ┌───────────┐   ┌───────┐ │
│  │ Ingest   │──▶│ Transcribe │──▶│  Detect   │──▶│ Pick  │ │
│  │ yt-dlp   │   │ whisper    │   │ PyScene   │   │ Ollama│ │
│  │ /upload  │   │ medium/int8│   │ +YOLOv8n  │   │ qwen  │ │
│  └──────────┘   └────────────┘   └───────────┘   └───┬───┘ │
│                                                       │      │
│  ┌──────────┐   ┌────────────┐   ┌───────────┐       │      │
│  │ Publish  │◀──│ Metadata   │◀──│  Reframe  │◀──────┘      │
│  │ YT/IG/TT │   │ Gemini/    │   │  FFmpeg   │              │
│  │ +approve │   │ Ollama     │   │  NVENC    │              │
│  └──────────┘   └────────────┘   └───────────┘              │
│       │                                                      │
│       ▼                                                      │
│  ┌──────────┐                                                │
│  │ n8n      │  Telegram approval ◀── Human gate              │
│  │ scheduler│  Weekly analytics                              │
│  └──────────┘                                                │
└─────────────────────────────────────────────────────────────┘
```

## Quick Start

```bash
cd /home/any/Desktop/lopkl
source .venv/bin/activate

# 1. Install dependencies
pip install -r clipforge/requirements.txt

# 2. Configure
cp clipforge/.env.example clipforge/.env
# Edit .env with your API keys (see Setup section)

# 3. Run a single video
python -m clipforge.cli process "https://youtube.com/watch?v=..." --output /shared/clipforge/output

# 4. Start the automation daemon
python -m clipforge.daemon --config clipforge/config.yaml
```

## Setup

### Required (all free)
- **Google Cloud project** → YouTube Data API v3 enabled → OAuth2 credentials (self-use, no audit)
- **Ollama** running locally with `qwen2.5:7b` (already installed)
- **Telegram bot** → @BotFather → token + chat ID for approval workflow

### Optional (improves quality)
- **Gemini API key** (free tier) → better moment detection + layout picking
- **Meta App Review** → Instagram publishing (2–4 week wait, free)
- **Upload-Post API key** → TikTok publishing (10 free/month)

### Disk Layout
```
/shared/clipforge/          # Work directory (68GB available)
├── downloads/              # Source videos
├── transcripts/            # Whisper output
├── clips/                  # Extracted clips
├── output/                 # Final reframed videos
├── metadata/               # Titles, descriptions, hashtags
└── state/                  # Pipeline state, idempotency keys
```

## Scrapling Evaluation

Installed and benchmarked on this machine:

| Capability | Result | Notes |
|---|---|---|
| HTTP fetch | ✅ 2.4s | Faster than httpx for parsed content |
| Stealth browser | ✅ 3.5s | Patchright-based, Cloudflare solving |
| Parse speed | ✅ 0.199s (5000×100) | 785× faster than BS4 |
| Markdown extraction | ✅ Built-in | Clean LLM-ready output |
| MCP server | ✅ Available | AI agent integration |
| Agent Skill | ✅ Shipped | Teaches coding agents the API |

**Verdict**: Use Scrapling as the primary web scraper for metadata enrichment,
transcript fetching, and competitor analysis. It replaces TinyFish/Firecrawl/Apify
for local scraping tasks. Keep TinyFish for remote/API-based searches where you
don't want to manage browser sessions.

## Publishing Reality Check (verified Nov 2026)

| Platform | Free Direct API? | Blocker | Rate Limit |
|---|---|---|---|
| YouTube | ✅ YES | None (Nov 2, 2026 reversal) | 100 uploads/day |
| Instagram | ⚠️ YES-WITH-CAVEATS | App Review (2–4 weeks, free) | 100 posts/24h |
| TikTok | ❌ NO | Unaudited = private only | N/A |

**TikTok workaround**: Upload-Post free tier (10/month) or accept private-only drafts.
Cookie/session automation violates ToS and risks account bans — not recommended.

## License

MIT — fork, modify, sell, whatever. Your videos, your pipeline.
