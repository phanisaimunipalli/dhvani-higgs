"""
Dhvani — Real-time YouTube Video Dubbing
Clean rewrite: simple sequential pipeline with full file logging.

Design:
  1. /api/youtube/prepare  → download YT audio, split into 3s chunks, store in session
  2. /ws/dub/{session_id}  → WebSocket: client sends {lang}, server streams dubbed chunks
  3. Per chunk: ASR → AST (parallel) → GPT-OSS fallback → Higgs TTS → send WAV to client

Logging: every API call timed and written to dhvani.log
"""

from __future__ import annotations
import asyncio, base64, io, json, logging, os, time, wave
from pathlib import Path

import httpx, numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from dotenv import load_dotenv
from emotion import detect_emotion

load_dotenv()

# ─── KMP fix for faster-whisper + numpy coexistence ───
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

# ─── Parakeet/Whisper local ASR (faster-whisper base.en) ───
# Loaded once at startup — no API calls, no network, flat ~1.3s per chunk
_whisper_model = None

def get_whisper():
    global _whisper_model
    if _whisper_model is None:
        from faster_whisper import WhisperModel
        L("[ASR-LOCAL] Loading faster-whisper base.en...")
        t = time.time()
        _whisper_model = WhisperModel("base.en", device="cpu", compute_type="int8")
        L(f"[ASR-LOCAL] Loaded in {time.time()-t:.1f}s")
    return _whisper_model

# ─── Gemini (Brain) ───
from google import genai as _genai
_gemini_client: "_genai.Client | None" = None

def get_gemini():
    global _gemini_client
    if _gemini_client is None:
        key = os.environ.get("GEMINI_API_KEY", "")
        if key:
            _gemini_client = _genai.Client(api_key=key)
    return _gemini_client

# ─── File logger ───
LOG_FILE = Path(__file__).parent / "dhvani.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, mode="a"),
        logging.StreamHandler(),          # also print to console
    ]
)
log = logging.getLogger("dhvani")

def L(msg: str):
    """Shorthand logger — writes to dhvani.log and console."""
    log.info(msg)

# ─── Config ───
app = FastAPI(title="Dhvani", version="3.0.0")
STATIC_DIR = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

EIGEN_BASE = "https://api-web.eigenai.com"
EIGEN_KEY  = os.environ.get("BOSONAI_API_KEY", "")

LANGUAGES = {
    "en": "English", "zh": "Chinese", "ko": "Korean",
    "ja": "Japanese", "es": "Spanish", "fr": "French",
    "de": "German",  "it": "Italian", "ru": "Russian",
}
DUB_LANGUAGES = {
    "es": {"name": "Spanish", "flag": "ES"},
    "ja": {"name": "Japanese", "flag": "JP"},
    "zh": {"name": "Chinese",  "flag": "CN"},
}

# ─── Disk cache ───
CACHE_DIR = Path(__file__).parent / ".cache"
CACHE_DIR.mkdir(exist_ok=True)

def cache_path(video_id: str, lang: str, idx: int) -> Path:
    return CACHE_DIR / f"{video_id}_{lang}_{idx}.json"

def load_chunk_cache(video_id: str, lang: str, total: int) -> dict:
    out = {}
    for i in range(total):
        p = cache_path(video_id, lang, i)
        if p.exists():
            try: out[i] = json.loads(p.read_text())
            except: pass
    if out:
        L(f"[CACHE] Loaded {len(out)}/{total} chunks from disk for {video_id}/{lang}")
    return out

def save_chunk_cache(video_id: str, lang: str, idx: int, result: dict):
    try: cache_path(video_id, lang, idx).write_text(json.dumps(result))
    except Exception as e: L(f"[CACHE] Save error chunk {idx}: {e}")

# ─── HTTP client (persistent, pooled) ───
_http: httpx.AsyncClient | None = None

async def get_http() -> httpx.AsyncClient:
    global _http
    if _http is None or _http.is_closed:
        _http = httpx.AsyncClient(
            timeout=httpx.Timeout(60.0, connect=10.0),
            limits=httpx.Limits(max_connections=10, max_keepalive_connections=5),
        )
    return _http

@app.on_event("startup")
async def startup():
    # Pre-load local ASR model in background so first request isn't slow
    loop = asyncio.get_event_loop()
    loop.run_in_executor(None, get_whisper)

@app.on_event("shutdown")
async def shutdown():
    global _http
    if _http: await _http.aclose()

# ─── Sessions ───
_sessions: dict[str, dict] = {}

# ─── Routes ───
@app.get("/")
async def index():
    return FileResponse(str(STATIC_DIR / "index.html"))

@app.get("/health")
async def health():
    return {"status": "ok", "version": "3.0.0"}


@app.post("/api/youtube/prepare")
async def youtube_prepare(request: Request):
    body = await request.json()
    url  = body.get("url", "").strip()
    if not url:
        return JSONResponse({"error": "No URL"}, status_code=400)

    import re, tempfile, subprocess
    m = re.search(r'(?:v=|youtu\.be/|shorts/)([a-zA-Z0-9_-]{11})', url)
    if not m:
        return JSONResponse({"error": "Invalid YouTube URL"}, status_code=400)
    video_id = m.group(1)

    # ── Download ──
    t0 = time.time()
    L(f"[PREPARE] Downloading video_id={video_id}")
    try:
        from pytubefix import YouTube
        yt    = YouTube(url)
        title = yt.title
        stream = yt.streams.filter(only_audio=True).order_by("abr").desc().first()
        if not stream:
            return JSONResponse({"error": "No audio stream"}, status_code=400)
        tmp_dir  = tempfile.mkdtemp(prefix="dhvani_")
        dl_path  = stream.download(output_path=tmp_dir, filename="audio")
    except Exception as e:
        return JSONResponse({"error": f"Download failed: {e}"}, status_code=400)
    L(f"[PREPARE] Download done in {time.time()-t0:.1f}s — {title}")

    # ── Convert to WAV 16kHz mono ──
    t1 = time.time()
    wav_path = os.path.join(tmp_dir, "audio.wav")
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-i", dl_path,
             "-ar", "16000", "-ac", "1", "-f", "wav", wav_path],
            capture_output=True, timeout=60,
        )
    except Exception as e:
        return JSONResponse({"error": f"ffmpeg failed: {e}"}, status_code=400)
    if not os.path.exists(wav_path):
        return JSONResponse({"error": "ffmpeg produced no output"}, status_code=400)
    L(f"[PREPARE] ffmpeg done in {time.time()-t1:.1f}s")

    # ── Read audio ──
    with wave.open(wav_path, "rb") as wf:
        frames = wf.readframes(wf.getnframes())
    data = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
    if data.ndim > 1: data = data.mean(axis=1)
    duration = len(data) / 16000

    # ── Fixed 3s chunks ──
    CHUNK_S     = 3
    chunk_samp  = 16000 * CHUNK_S
    chunks      = []
    for i in range(0, len(data), chunk_samp):
        seg = data[i:i + chunk_samp]
        if len(seg) < 1600: continue
        pcm = (np.clip(seg, -1, 1) * 32767).astype(np.int16)
        chunks.append({
            "pcm":      pcm.tobytes(),
            "start_s":  round(i / 16000, 2),
            "dur_s":    round(len(seg) / 16000, 2),
        })

    # ── Pick voice reference (loudest of first 5 chunks) ──
    voice_ref = None
    best_rms  = 0.0
    for c in chunks[:5]:
        a   = np.frombuffer(c["pcm"], dtype=np.int16).astype(np.float32) / 32768.0
        rms = float(np.sqrt(np.mean(a ** 2)))
        if rms > best_rms:
            best_rms = rms
            voice_ref = pcm_to_wav(c["pcm"], 16000)
    if best_rms < 0.01: voice_ref = None

    # ── Load disk cache for all languages ──
    lang_caches = {lang: load_chunk_cache(video_id, lang, len(chunks)) for lang in DUB_LANGUAGES}

    session_id = f"yt_{video_id}_{int(time.time())}"
    _sessions[session_id] = {
        "video_id":  video_id,
        "title":     title,
        "duration":  duration,
        "chunks":    chunks,
        "total":     len(chunks),
        "voice_ref": voice_ref,
        "cache":     lang_caches,
    }

    try:
        import shutil; shutil.rmtree(tmp_dir, ignore_errors=True)
    except: pass

    chunk_times = [{"start": c["start_s"], "dur": c["dur_s"]} for c in chunks]
    L(f"[PREPARE] Ready: {len(chunks)} chunks | {duration:.1f}s | session={session_id}")

    return {
        "session_id":  session_id,
        "video_id":    video_id,
        "title":       title,
        "duration":    round(duration, 1),
        "chunks":      len(chunks),
        "chunk_times": chunk_times,
        "languages":   DUB_LANGUAGES,
    }


# ─── WebSocket dubbing ───
@app.websocket("/ws/dub/{session_id}")
async def ws_dub(websocket: WebSocket, session_id: str):
    await websocket.accept()
    session = _sessions.get(session_id)
    if not session:
        await websocket.send_json({"type": "error", "message": "Session not found"})
        await websocket.close()
        return

    chunks    = session["chunks"]
    total     = session["total"]
    voice_ref = session["voice_ref"]
    cache     = session["cache"]
    video_id  = session["video_id"]
    active_task: asyncio.Task | None = None
    flag = {"cancelled": False}

    L(f"[WS] Connected: {session_id} ({total} chunks)")

    # ── 3-Queue Pipeline: ASR → Brain(Gemini) → TTS — all overlapping ──
    async def dub_lang(lang: str, from_idx: int):
        lang_cache = cache.get(lang, {})
        lang_name  = LANGUAGES.get(lang, "English")
        clone      = voice_ref is not None and len(voice_ref) > 3200

        # Bounded queues prevent runaway memory if TTS is slow
        text_q = asyncio.Queue(maxsize=6)   # ASR → Brain
        tts_q  = asyncio.Queue(maxsize=6)   # Brain → TTS

        L(f"[PIPELINE:{lang}] Start from {from_idx}/{total} | cached={len(lang_cache)}")

        # ── Stage 1: ASR ── runs ahead, pushes text to brain ──
        async def asr_stage():
            for i in range(from_idx, total):
                if flag["cancelled"]: break

                if i in lang_cache:
                    await text_q.put(("cached", i, None, None, None, None))
                    continue

                chunk   = chunks[i]
                pcm_raw = chunk["pcm"]
                audio   = np.frombuffer(pcm_raw, dtype=np.int16).astype(np.float32) / 32768.0
                rms     = float(np.sqrt(np.mean(audio ** 2)))

                if rms < 0.005:
                    await text_q.put(("silent", i, None, None, None, None))
                    continue

                wav_bytes = pcm_to_wav(pcm_raw, 16000)
                emo       = detect_emotion(pcm_raw, 16000)
                t_start   = time.time()

                t = time.time()
                transcript = await api_asr(wav_bytes)
                L(f"[ASR:{lang}] {i+1}/{total}: {int((time.time()-t)*1000)}ms | \"{(transcript or '')[:40]}\"")

                if not transcript or is_noise(transcript):
                    await text_q.put(("silent", i, None, None, None, None))
                else:
                    await text_q.put(("text", i, transcript, wav_bytes, emo, t_start))

            await text_q.put(None)  # sentinel

        # ── Stage 2: Brain (Gemini) ── translates, pushes to TTS ──
        async def brain_stage():
            while True:
                item = await text_q.get()
                if item is None:
                    await tts_q.put(None)
                    break

                kind, i, transcript, wav_bytes, emo, t_start = item
                chunk = chunks[i]

                if kind in ("cached", "silent"):
                    await tts_q.put((kind, i, None, None, None, chunk, t_start))
                    continue

                # Gemini Flash Lite: ~200-400ms, no rate limits
                t = time.time()
                translation = await gemini_translate(transcript, lang)
                ms_g = int((time.time() - t) * 1000)

                if translation and translation.strip() != transcript.strip():
                    L(f"[BRAIN:Gemini:{lang}] {i+1}/{total}: {ms_g}ms | \"{translation[:40]}\"")
                else:
                    # Fallback: Higgs AST
                    t = time.time()
                    translation = await api_ast(wav_bytes, lang_name)
                    L(f"[BRAIN:AST:{lang}] {i+1}/{total}: {int((time.time()-t)*1000)}ms | \"{(translation or '')[:40]}\"")

                    if not translation or translation.strip() == transcript.strip():
                        # Last resort: GPT-OSS
                        t = time.time()
                        translation = await api_translate(transcript, lang)
                        L(f"[BRAIN:GPT:{lang}] {i+1}/{total}: {int((time.time()-t)*1000)}ms | \"{(translation or '')[:40]}\"")

                if not translation or translation.strip() == transcript.strip():
                    await tts_q.put(("skip", i, transcript, None, None, chunk, t_start))
                else:
                    await tts_q.put(("speak", i, transcript, translation, emo, chunk, t_start))

        # ── Stage 3: TTS ── generates audio, sends to client ──
        async def tts_stage():
            while True:
                item = await tts_q.get()
                if item is None: break
                if flag["cancelled"]: continue

                kind, i, transcript, translation, emo, chunk, t_start = item

                if kind == "silent":
                    continue

                if kind == "cached":
                    r = lang_cache[i]
                    try:
                        await websocket.send_json({
                            "type": "chunk", "index": i, "total": total, "cached": True, **r
                        })
                    except: break
                    L(f"[TTS:{lang}] {i+1}/{total} CACHED ⚡")
                    continue

                if kind == "skip":
                    try:
                        await websocket.send_json({
                            "type": "skip", "index": i, "total": total,
                            "transcript": transcript or "",
                        })
                    except: break
                    continue

                t = time.time()
                tts_raw = await api_tts(translation, emo["speed"], voice_ref if clone else None)
                ms_tts  = int((time.time() - t) * 1000)
                L(f"[TTS:{lang}] {i+1}/{total}: {ms_tts}ms | {len(tts_raw) if tts_raw else 0}bytes | clone={clone}")

                if not tts_raw or len(tts_raw) < 1000:
                    L(f"[TTS:{lang}] {i+1}/{total} SKIP (tts empty)")
                    continue

                tts_wav  = tts_raw if clone else pcm_to_wav(tts_raw, 24000)
                tts_b64  = base64.b64encode(tts_wav).decode() if tts_wav else None
                total_ms = int((time.time() - t_start) * 1000)
                L(f"[TTS:{lang}] {i+1}/{total} DONE ✓ pipeline={total_ms}ms | \"{(transcript or '')[:30]}\" → \"{translation[:30]}\"")

                result = {
                    "transcript":  transcript,
                    "translation": translation,
                    "audio_b64":   tts_b64,
                    "emotion":     {"emoji": emo["emoji"], "label": emo["label"]},
                    "latency":     total_ms,
                    "start_s":     chunk["start_s"],
                    "dur_s":       chunk["dur_s"],
                }
                lang_cache[i] = result
                save_chunk_cache(video_id, lang, i, result)

                if flag["cancelled"]: break
                try:
                    await websocket.send_json({
                        "type": "chunk", "index": i, "total": total, "cached": False, **result
                    })
                except: break

            if not flag["cancelled"]:
                L(f"[PIPELINE:{lang}] Done — all chunks processed")
                try: await websocket.send_json({"type": "done", "lang": lang})
                except: pass

        # All 3 stages run concurrently — the actual pipeline
        await asyncio.gather(asr_stage(), brain_stage(), tts_stage())

    async def cancel():
        nonlocal active_task
        if active_task and not active_task.done():
            flag["cancelled"] = True
            active_task.cancel()
            try: await active_task
            except: pass

    try:
        await websocket.send_json({"type": "ready", "total": total})

        while True:
            data = await websocket.receive_text()
            msg  = json.loads(data)

            if msg["type"] in ("play", "switch"):
                await cancel()
                lang     = msg.get("lang", "es")
                from_idx = max(0, min(msg.get("from", 0), total - 1))
                flag["cancelled"] = False
                L(f"[WS] {msg['type'].upper()} lang={lang} from={from_idx}")
                await websocket.send_json({"type": "playing", "lang": lang, "from": from_idx})
                active_task = asyncio.create_task(dub_lang(lang, from_idx))

            elif msg["type"] == "stop":
                await cancel()
                await websocket.send_json({"type": "stopped"})
                L("[WS] Stopped by client")

    except WebSocketDisconnect:
        flag["cancelled"] = True
        if active_task: active_task.cancel()
        L(f"[WS] Disconnected: {session_id}")
    except Exception as e:
        L(f"[WS] Error: {e}")


# ─── API helpers ───

async def gemini_translate(text: str, lang: str) -> str | None:
    """Fast translation via Gemini Flash Lite — ~200ms, no rate limits."""
    import re as _re
    lang_name = LANGUAGES.get(lang, lang)
    try:
        client = get_gemini()
        if not client:
            return None
        resp = await client.aio.models.generate_content(
            model="gemini-3.1-flash-lite-preview",
            contents=f"Translate to {lang_name}. Reply with ONLY the translated text, no markdown:\n{text}",
        )
        result = (resp.text or "").strip()
        result = _re.sub(r'\*+', '', result).strip("\"'`_ \n")
        if not result or result.lower() == text.lower(): return None
        if len(result) > len(text) * 4: return None  # hallucination guard
        return result
    except Exception as e:
        L(f"[Gemini] Error: {e}")
        return None


async def api_asr(wav_bytes: bytes) -> str | None:
    """Local ASR via faster-whisper base.en — ~1.3s flat, no API calls, no spikes."""
    try:
        # Decode wav bytes → float32 numpy array
        buf = io.BytesIO(wav_bytes)
        with wave.open(buf, "rb") as wf:
            frames = wf.readframes(wf.getnframes())
            sr = wf.getframerate()
        audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0

        # Run in executor so it doesn't block the event loop
        loop = asyncio.get_event_loop()
        def _transcribe():
            model = get_whisper()
            segs, _ = model.transcribe(audio, beam_size=1, language="en", condition_on_previous_text=False)
            return " ".join(s.text for s in segs).strip()

        text = await loop.run_in_executor(None, _transcribe)
        return text or None
    except Exception as e:
        L(f"[ASR-LOCAL] Error: {e}")
        return None


async def api_ast(wav_bytes: bytes, target_lang: str) -> str | None:
    if not EIGEN_KEY: return None
    try:
        client = await get_http()
        resp = await client.post(
            f"{EIGEN_BASE}/api/v1/generate",
            headers={"Authorization": f"Bearer {EIGEN_KEY}"},
            files={"file": ("audio.wav", wav_bytes, "audio/wav")},
            data={"model": "higgs_asr_3", "task": "ast", "language": target_lang},
        )
        if resp.status_code != 200:
            L(f"[AST] HTTP {resp.status_code}")
            return None
        return resp.json().get("transcription", "").strip() or None
    except Exception as e:
        L(f"[AST] Error: {e}")
        return None


async def api_translate(text: str, lang: str) -> str | None:
    if not EIGEN_KEY: return None
    import re as _re
    tgt = LANGUAGES.get(lang, lang)
    try:
        client = await get_http()
        for attempt in range(3):
            resp = await client.post(
                f"{EIGEN_BASE}/api/v1/chat/completions",
                headers={"Authorization": f"Bearer {EIGEN_KEY}", "Content-Type": "application/json"},
                json={
                    "model": "gpt-oss-120b",
                    "messages": [
                        {"role": "system", "content": f"Translate to {tgt}. Reply with ONLY the translated text, no markdown, no quotes, no asterisks."},
                        {"role": "user", "content": text},
                    ],
                    "temperature": 0.1, "max_tokens": 300,
                },
            )
            if resp.status_code == 429:
                wait = (attempt + 1) * 2
                L(f"[GPT-OSS] 429 rate limit — retry {attempt+1}/3 in {wait}s")
                await asyncio.sleep(wait)
                continue
            if resp.status_code != 200:
                L(f"[GPT-OSS] HTTP {resp.status_code}")
                return None
            result = resp.json().get("choices", [{}])[0].get("message", {}).get("content", "")
            # Strip markdown artifacts
            result = _re.sub(r'\*+', '', result)
            result = result.strip("\"'`_ \n")
            if not result: return None
            if len(result) > len(text) * 4: return None   # hallucination guard
            return result
        return None
    except Exception as e:
        L(f"[GPT-OSS] Error: {e}")
        return None


async def api_tts(text: str, speed: float = 1.0, voice_ref_wav: bytes | None = None) -> bytes | None:
    """Higgs TTS 2.5.
    With voice_ref → multipart POST (voice clone) → returns WAV bytes.
    Without → SSE streaming → returns raw PCM16 bytes at 24kHz.
    """
    if not EIGEN_KEY: return None
    clone = voice_ref_wav and len(voice_ref_wav) > 3200
    try:
        client = await get_http()
        if clone:
            resp = await client.post(
                f"{EIGEN_BASE}/api/v1/generate",
                headers={"Authorization": f"Bearer {EIGEN_KEY}"},
                data={
                    "model": "higgs2p5", "text": text,
                    "voice_settings": json.dumps({"speed": speed}),
                    "sampling": json.dumps({"temperature": 0.85, "top_p": 0.95, "top_k": 50}),
                },
                files={"voice_reference_file": ("speaker.wav", voice_ref_wav, "audio/wav")},
            )
            if resp.status_code != 200:
                L(f"[TTS-clone] HTTP {resp.status_code}")
                return None
            return resp.content if len(resp.content) > 1000 else None
        else:
            # SSE streaming — collect base64 PCM16 chunks
            pcm = bytearray()
            async with client.stream(
                "POST", f"{EIGEN_BASE}/api/v1/generate",
                headers={"Authorization": f"Bearer {EIGEN_KEY}", "Content-Type": "application/json"},
                json={"model": "higgs2p5", "text": text, "stream": True,
                      "voice_settings": {"speed": speed},
                      "sampling": {"temperature": 0.85, "top_p": 0.95, "top_k": 50}},
            ) as resp:
                buf = ""
                async for chunk in resp.aiter_text():
                    buf += chunk
                    while "\n" in buf:
                        line, buf = buf.split("\n", 1)
                        line = line.strip()
                        if not line.startswith("data: "): continue
                        data = json.loads(line[6:])
                        if isinstance(data.get("data"), str) and len(data["data"]) > 50:
                            pcm.extend(base64.b64decode(data["data"]))
                        if data.get("type") == "done": break
            return bytes(pcm) if len(pcm) > 1000 else None
    except Exception as e:
        L(f"[TTS] Error: {e}")
        return None


# ─── Utilities ───

def pcm_to_wav(pcm: bytes, sr: int) -> bytes | None:
    if len(pcm) < 3200: return None
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(sr)
        wf.writeframes(pcm)
    return buf.getvalue()

_NOISE = {"thank you","thanks","thanks for watching","bye","goodbye","the end",
          "subscribe","like and subscribe","silence","...",".",""," "}

def is_noise(text: str) -> bool:
    if not text: return True
    c = text.strip().lower().rstrip(".!?,")
    return not c or c in _NOISE or (len(c.split()) > 2 and len(set(c.split())) == 1)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
