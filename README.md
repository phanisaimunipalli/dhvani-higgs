# Dhvani — Real-time YouTube Dubbing

> Built at **Boson x Eigen AI Hackathon 2026**
> Stack: Higgs ASR 3 · GPT-OSS 120B · Higgs TTS 2.5 · FastAPI · Python 3.13

Paste a YouTube URL. Pick a language. Dhvani dubs the audio in real-time — with voice cloning and emotion preservation.

---

## How It Works

```
YouTube Audio
     │
     ▼
┌─────────────────┐
│  Higgs ASR 3    │  ← Cloud transcription (Eigen API)
│  (task: asr)    │
└────────┬────────┘
         │  English transcript
         ▼
┌─────────────────┐
│  GPT-OSS 120B   │  ← Translation (Eigen API)
│                 │  → Fallback: Higgs AST
└────────┬────────┘
         │  Translated text + emotion
         ▼
┌─────────────────┐
│  Higgs TTS 2.5  │  ← Voice clone + emotion-aware TTS (Boson API)
└────────┬────────┘
         │  WAV audio (base64)
         ▼
   Browser (WebSocket)
   Timestamp-synced playback
```

---

## Pipeline — 3 Concurrent Stages

```python
text_q = asyncio.Queue(maxsize=6)   # ASR → Brain
tts_q  = asyncio.Queue(maxsize=6)   # Brain → TTS

await asyncio.gather(asr_stage(), brain_stage(), tts_stage())
```

While TTS plays chunk **N** → Brain translates **N+1** → ASR transcribes **N+2**.
Bounded queues prevent memory runaway when TTS is slow.

---

## Features

- **Voice cloning** — captures speaker voice from first 5 chunks, clones it in TTS
- **Emotion detection** — pure numpy, no ML model. Maps RMS / pitch / ZCR → TTS speed
- **Hallucination filter** — ASR output validated against YouTube captions (word overlap < 30% → use caption)
- **Timestamp-aware playback** — each chunk carries `start_s / end_s`. Late = skip. Early = wait.
- **Disk cache** — chunks cached to `.cache/` per video+language. Instant replay on reconnect.
- **Transcript panel** — live original + translated text, synced to playing chunk

---

## Languages

| Code | Language |
|------|----------|
| `es` | Spanish  |
| `ja` | Japanese |
| `zh` | Chinese  |

---

## Setup

### 1. Clone

```bash
git clone https://github.com/phanisaimunipalli/dhvani-higgs.git
cd dhvani-higgs
```

### 2. Create ARM64 venv (macOS M1/M2/M3)

```bash
/opt/homebrew/bin/python3.13 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

> **Important:** Do NOT use Anaconda Python — it runs under Rosetta 2 (x86_64) and is significantly slower.

### 3. Environment variables

Create a `.env` file:

```env
BOSONAI_API_KEY=your_eigen_boson_key
```

### 4. Run

```bash
uvicorn main:app --host 0.0.0.0 --port 8011 --reload
```

Open **http://localhost:8011**

---

## API

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Frontend UI |
| `/health` | GET | Health check |
| `/api/youtube/prepare` | POST | Download + chunk YouTube audio |
| `/ws/dub/{session_id}` | WebSocket | Stream dubbed audio chunks |

### WebSocket messages (client → server)

```json
{ "type": "play",   "lang": "es", "from": 0  }
{ "type": "switch", "lang": "ja", "from": 12 }
{ "type": "stop" }
```

### WebSocket messages (server → client)

```json
{ "type": "chunk", "index": 3, "total": 42, "transcript": "...", "translation": "...", "audio_b64": "...", "start_s": 9.0, "dur_s": 3.0, "latency": 1240 }
{ "type": "done",  "lang": "es" }
{ "type": "skip",  "index": 5,  "transcript": "..." }
```

---

## Project Structure

```
dhvani-higgs/
├── main.py          # FastAPI backend — full pipeline
├── emotion.py       # Acoustic emotion detection (numpy only)
├── static/
│   └── index.html   # Frontend — YouTube IFrame API + WebSocket client
├── requirements.txt
├── .env             # BOSONAI_API_KEY (gitignored)
└── .cache/          # Per-chunk translation cache (gitignored)
```

---

## Known Gotchas

**Higgs TTS hallucination** — with voice clone + CJK languages, TTS can return 2.7MB (57s) for a 3s chunk.
Guard in place: chunks > 500KB are dropped.

**GPT-OSS rate limits** — hits 429 at burst rate. Higgs AST fallback kicks in automatically.

**OpenMP conflict** — if you see `KMP_DUPLICATE_LIB_OK` warnings, it's from numpy coexistence. Already handled in `main.py`.

---

## Changelog

### v3.1 — Higgs-native pipeline
- Replaced `faster-whisper` (local) with **Higgs ASR 3** (cloud) for transcription
- Replaced **Gemini** with **GPT-OSS 120B** as primary translation brain
- Removed `google-genai` dependency entirely
- Fallback chain: GPT-OSS → Higgs AST

### v3.0 — Shared ASR + caption hallucination filter
- Single transcript cache shared across all languages — transcribe once, reuse
- Caption-based hallucination detection (word overlap < 30% → use YouTube caption)
- Emotion detection added (`emotion.py`)

### v2.0 — 3-queue async pipeline
- ASR / Brain / TTS run concurrently via `asyncio.gather`
- Bounded queues prevent memory runaway
- Timestamp-aware audio sync — late chunks skipped, early chunks wait

### v1.0 — Initial hackathon build
- Sequential pipeline: download → chunk → ASR → translate → TTS
- Voice cloning from loudest of first 5 chunks
- Disk cache per video+language

---

## Built With

| Component | Model / Tool |
|-----------|-------------|
| ASR | Higgs ASR 3 (Eigen API) |
| Translation | GPT-OSS 120B (Eigen API) |
| TTS | Higgs TTS 2.5 (Boson API) |
| Backend | FastAPI + uvicorn |
| Frontend | Vanilla JS + YouTube IFrame API |
| Emotion | Custom numpy acoustic classifier |

---

---

## Contributors

| Name | GitHub |
|------|--------|
| Phani Sai Ram Munipalli | [@phanisaimunipalli](https://github.com/phanisaimunipalli) |
| Vinay Mokidi | [@vinaymokidi-mk](https://github.com/vinaymokidi-mk) |

---

*Dhvani (ध्वनि) — Sanskrit for "sound"*
