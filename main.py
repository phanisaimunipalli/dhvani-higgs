"""
Dhvani — Real-time YouTube Video Dubbing
Paste a YouTube URL → Select language → Dubbed audio overlays on video
100% Higgs Audio powered on Eigen AI.

Pipeline: ASR → Translate (GPT-OSS) → Streaming TTS (Higgs 2.5)
Streaming TTS sends PCM chunks as they generate → sub-second first audio.
"""

from __future__ import annotations
import asyncio
import base64
import io
import json
import os
import time
import wave
from pathlib import Path

import httpx
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from dotenv import load_dotenv
from emotion import detect_emotion

load_dotenv()

app = FastAPI(title="Dhvani", version="2.0.0")

STATIC_DIR = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

EIGEN_BASE = "https://api-web.eigenai.com"
EIGEN_KEY = os.environ.get("BOSONAI_API_KEY", "")

LANGUAGES = {
    "en": "English", "zh": "Chinese", "ko": "Korean",
    "ja": "Japanese", "es": "Spanish", "fr": "French",
    "de": "German", "it": "Italian", "ru": "Russian",
}

DUB_LANGUAGES = {
    "es": {"name": "Spanish", "flag": "ES"},
    "ja": {"name": "Japanese", "flag": "JP"},
    "zh": {"name": "Chinese", "flag": "CN"},
}

# ─── Persistent HTTP client ───
_http: httpx.AsyncClient | None = None

async def get_http() -> httpx.AsyncClient:
    global _http
    if _http is None or _http.is_closed:
        _http = httpx.AsyncClient(
            timeout=httpx.Timeout(30.0, connect=10.0),
            limits=httpx.Limits(max_connections=20, max_keepalive_connections=10),
        )
    return _http


@app.on_event("shutdown")
async def shutdown():
    global _http
    if _http:
        await _http.aclose()


@app.get("/")
async def index():
    return FileResponse(str(STATIC_DIR / "index.html"))


@app.get("/health")
async def health():
    return {"status": "Dhvani online", "version": "2.0.0"}


# ─── Session state ───
_sessions: dict[str, dict] = {}


@app.post("/api/youtube/prepare")
async def youtube_prepare(request: Request):
    body = await request.json()
    url = body.get("url", "").strip()
    if not url:
        return JSONResponse({"error": "No URL provided"}, status_code=400)

    import re, tempfile, subprocess

    vid_match = re.search(r'(?:v=|youtu\.be/|shorts/)([a-zA-Z0-9_-]{11})', url)
    if not vid_match:
        return JSONResponse({"error": "Invalid YouTube URL"}, status_code=400)
    video_id = vid_match.group(1)

    try:
        from pytubefix import YouTube
        yt = YouTube(url)
        title = yt.title
        print(f"[YT] Downloading: {title} ({yt.length}s)")
        audio_stream = yt.streams.filter(only_audio=True).order_by("abr").desc().first()
        if not audio_stream:
            return JSONResponse({"error": "No audio stream found"}, status_code=400)
        tmp_dir = tempfile.mkdtemp(prefix="dhvani_yt_")
        dl_path = audio_stream.download(output_path=tmp_dir, filename="audio")
    except Exception as e:
        return JSONResponse({"error": f"Download failed: {e}"}, status_code=400)

    wav_path = os.path.join(tmp_dir, "audio.wav")
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-i", dl_path, "-ar", "16000", "-ac", "1", "-f", "wav", wav_path],
            capture_output=True, timeout=60,
        )
    except Exception as e:
        return JSONResponse({"error": f"Conversion failed: {e}"}, status_code=400)

    if not os.path.exists(wav_path):
        return JSONResponse({"error": "Conversion produced no output"}, status_code=400)

    try:
        with wave.open(wav_path, "rb") as wf:
            frames = wf.readframes(wf.getnframes())
            data = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
    except Exception as e:
        return JSONResponse({"error": f"Could not decode: {e}"}, status_code=400)

    if data.ndim > 1:
        data = data.mean(axis=1)

    CHUNK_S = 3
    chunk_samples = 16000 * CHUNK_S
    chunks = []
    for i in range(0, len(data), chunk_samples):
        seg = data[i:i + chunk_samples]
        if len(seg) < 1600:
            continue
        pcm = (np.clip(seg, -1, 1) * 32767).astype(np.int16)
        start_s = round(i / 16000, 2)
        dur_s = round(len(seg) / 16000, 2)
        chunks.append({"pcm": pcm.tobytes(), "start_s": start_s, "duration_s": dur_s})

    voice_ref = _pick_voice_ref(chunks)

    duration = len(data) / 16000
    session_id = f"yt_{video_id}_{int(time.time())}"

    _sessions[session_id] = {
        "chunks": chunks,
        "video_id": video_id,
        "title": title,
        "duration": duration,
        "voice_ref": voice_ref,
        "total": len(chunks),
        "cache": {lang: {} for lang in DUB_LANGUAGES},
    }

    try:
        import shutil
        shutil.rmtree(tmp_dir, ignore_errors=True)
    except Exception:
        pass

    chunk_times = [{"start": c["start_s"], "dur": c["duration_s"]} for c in chunks]
    print(f"[YT] Ready: {title} | {duration:.1f}s, {len(chunks)} chunks ({CHUNK_S}s each)")

    return {
        "session_id": session_id,
        "video_id": video_id,
        "title": title,
        "duration": round(duration, 1),
        "chunks": len(chunks),
        "chunk_times": chunk_times,
        "languages": DUB_LANGUAGES,
    }


@app.websocket("/ws/dub/{session_id}")
async def ws_dub(websocket: WebSocket, session_id: str):
    """
    Streaming dubbing pipeline:
    ASR → GPT-OSS Translate → Streaming TTS (SSE, PCM chunks)
    Sends audio as WAV base64 per chunk for client playback.
    """
    await websocket.accept()
    session = _sessions.get(session_id)
    if not session:
        await websocket.send_json({"type": "error", "message": "Session not found"})
        await websocket.close()
        return

    chunks = session["chunks"]
    total = session["total"]
    voice_ref = session["voice_ref"]
    cache = session["cache"]
    active_task: asyncio.Task | None = None
    cancel_flag = {"cancelled": False}

    async def stream_lang(lang: str, from_idx: int, flag: dict):
        """
        Parallel pipeline: process N chunks simultaneously, send to client in order.
        With 4 workers: 4x faster than sequential processing.
        """
        lang_cache = cache.get(lang, {})
        lang_name = LANGUAGES.get(lang, "English")
        clone_active = voice_ref is not None and len(voice_ref) > 3200
        PARALLEL = 4  # concurrent chunk workers
        sem = asyncio.Semaphore(PARALLEL)
        # results[i] = dict (chunk result) | "skip" | "silent" | None (not done yet)
        results: dict[int, object] = {}

        async def process_one(i: int):
            if flag["cancelled"]:
                results[i] = "silent"
                return

            chunk_info = chunks[i]
            pcm_bytes = chunk_info["pcm"]
            chunk_start_s = chunk_info["start_s"]
            chunk_dur_s = chunk_info["duration_s"]
            audio = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0
            rms = float(np.sqrt(np.mean(audio ** 2)))
            if rms < 0.005:
                results[i] = "silent"
                return

            async with sem:
                if flag["cancelled"]:
                    results[i] = "silent"
                    return

                t0 = time.time()
                emo = detect_emotion(pcm_bytes, 16000)
                wav_bytes = _pcm_to_wav(pcm_bytes, 16000)
                if not wav_bytes:
                    results[i] = "silent"
                    return

                # ASR + AST in parallel
                t_asr = time.time()
                transcript, ast_translation = await asyncio.gather(
                    _higgs_asr(wav_bytes),
                    _higgs_asr_translate(wav_bytes, lang_name),
                )
                ms_asr = int((time.time() - t_asr) * 1000)
                print(f"[Dub] [{lang}] {i+1}/{total} ASR+AST: {ms_asr}ms | \"{(transcript or '')[:35]}\" -> \"{(ast_translation or '')[:35]}\"")

                if not transcript or _is_noise(transcript):
                    results[i] = "silent"
                    return

                translation = ast_translation
                if not translation or translation == transcript or _is_noise(translation):
                    t_fb = time.time()
                    translation = await _translate(transcript, lang)
                    print(f"[Dub] [{lang}] {i+1}/{total} GPT-OSS fallback: {int((time.time()-t_fb)*1000)}ms")

                if not translation or translation == transcript:
                    print(f"[Dub] [{lang}] {i+1}/{total} SKIP no translation")
                    results[i] = {"_skip": True, "transcript": transcript,
                                  "start_s": chunk_start_s, "duration_s": chunk_dur_s}
                    return

                # TTS
                t_tts = time.time()
                tts_pcm = await _higgs_tts_stream(translation, emo["speed"], voice_ref)
                ms_tts = int((time.time() - t_tts) * 1000)
                print(f"[Dub] [{lang}] {i+1}/{total} TTS: {ms_tts}ms | clone={clone_active} | TOTAL: {int((time.time()-t0)*1000)}ms")

                if tts_pcm and len(tts_pcm) > 1000:
                    tts_wav = tts_pcm if clone_active else _pcm_to_wav_24k(tts_pcm)
                    tts_b64 = base64.b64encode(tts_wav).decode() if tts_wav else None
                else:
                    tts_b64 = None

                results[i] = {
                    "transcript": transcript,
                    "translation": translation,
                    "audio_b64": tts_b64,
                    "emotion": {"emoji": emo["emoji"], "label": emo["label"]},
                    "latency": int((time.time() - t0) * 1000),
                    "start_s": chunk_start_s,
                    "duration_s": chunk_dur_s,
                }
                lang_cache[i] = results[i]

        # Pre-fill cache hits so process_one doesn't need to touch them
        for i in range(from_idx, total):
            if i in lang_cache:
                results[i] = lang_cache[i]

        # Launch all worker tasks in parallel
        tasks = [
            asyncio.create_task(process_one(i))
            for i in range(from_idx, total)
            if i not in lang_cache
        ]
        print(f"[Dub] [{lang}] launched {len(tasks)} parallel tasks (PARALLEL={PARALLEL})")

        # Send results to client in order as they complete
        next_i = from_idx
        while next_i < total and not flag["cancelled"]:
            if next_i not in results:
                await asyncio.sleep(0.05)
                continue

            r = results[next_i]
            try:
                if r == "silent":
                    pass  # skip silently
                elif isinstance(r, dict) and r.get("_skip"):
                    await websocket.send_json({
                        "type": "skip", "index": next_i, "total": total,
                        "transcript": r["transcript"],
                    })
                elif isinstance(r, dict):
                    await websocket.send_json({
                        "type": "chunk", "index": next_i, "total": total,
                        "cached": next_i in lang_cache and r is lang_cache.get(next_i), **r,
                    })
            except Exception:
                flag["cancelled"] = True
                break

            next_i += 1

        # Cancel any still-running tasks on cancel/done
        for t in tasks:
            if not t.done():
                t.cancel()

        if not flag["cancelled"]:
            try:
                await websocket.send_json({"type": "done", "lang": lang})
            except Exception:
                pass

    async def cancel_active():
        nonlocal active_task
        if active_task and not active_task.done():
            cancel_flag["cancelled"] = True
            active_task.cancel()
            try:
                await active_task
            except (asyncio.CancelledError, Exception):
                pass

    try:
        await websocket.send_json({"type": "ready", "total": total})

        while True:
            data = await websocket.receive_text()
            msg = json.loads(data)

            if msg["type"] in ("play", "switch"):
                await cancel_active()
                lang = msg.get("lang", "es")
                from_idx = max(0, min(msg.get("from", 0), total - 1))
                cancel_flag["cancelled"] = False
                print(f"[Dub] {msg['type']} {lang} from chunk {from_idx}")
                await websocket.send_json({"type": "playing", "lang": lang, "from": from_idx})
                active_task = asyncio.create_task(stream_lang(lang, from_idx, cancel_flag))

            elif msg["type"] == "stop":
                await cancel_active()
                await websocket.send_json({"type": "stopped"})

    except WebSocketDisconnect:
        cancel_flag["cancelled"] = True
        if active_task:
            active_task.cancel()
        print(f"[WS] Disconnected: {session_id}")
    except Exception as e:
        print(f"[WS Error] {e}")


# ─── Helpers ───

def _vad_chunk(data: np.ndarray, sr: int = 16000) -> list[dict]:
    """
    Split audio into chunks at natural silence/pause boundaries.
    Energy-based VAD — no extra dependencies needed.
    Returns list of {"pcm": bytes, "start_s": float, "duration_s": float}
    """
    MIN_CHUNK_S = 1.5
    MAX_CHUNK_S = 8.0
    MIN_SILENCE_S = 0.3
    FRAME_MS = 30

    frame_len = int(sr * FRAME_MS / 1000)  # 480 samples at 16kHz
    n_frames = len(data) // frame_len

    if n_frames < 10:
        pcm = (np.clip(data, -1, 1) * 32767).astype(np.int16).tobytes()
        return [{"pcm": pcm, "start_s": 0.0, "duration_s": len(data) / sr}]

    # Energy per frame
    energies = np.array([
        float(np.sqrt(np.mean(data[i * frame_len:(i + 1) * frame_len] ** 2)))
        for i in range(n_frames)
    ])

    # Adaptive threshold: median * 0.3 or fixed floor
    thresh = max(0.008, float(np.median(energies)) * 0.3)
    is_silence = energies < thresh

    min_sil_frames = int(MIN_SILENCE_S * 1000 / FRAME_MS)
    min_chunk_frames = int(MIN_CHUNK_S * 1000 / FRAME_MS)
    max_chunk_frames = int(MAX_CHUNK_S * 1000 / FRAME_MS)

    # Find split points at silence gaps
    splits = [0]
    sil_start = None

    for i in range(n_frames):
        if is_silence[i]:
            if sil_start is None:
                sil_start = i
        else:
            if sil_start is not None:
                gap = i - sil_start
                since_split = sil_start - splits[-1]
                if gap >= min_sil_frames and since_split >= min_chunk_frames:
                    splits.append(sil_start + gap // 2)
                sil_start = None

        # Force split at max length
        if i - splits[-1] >= max_chunk_frames:
            if sil_start is not None:
                splits.append(i)
                sil_start = None
            else:
                splits.append(i)

    splits.append(n_frames)

    # Deduplicate and sort
    splits = sorted(set(splits))

    # Build chunks
    chunks = []
    for idx in range(len(splits) - 1):
        sf, ef = splits[idx], splits[idx + 1]
        if ef - sf < 3:
            continue
        s_sample = sf * frame_len
        e_sample = min(ef * frame_len, len(data))
        seg = data[s_sample:e_sample]
        if len(seg) < 1600:
            continue
        pcm = (np.clip(seg, -1, 1) * 32767).astype(np.int16).tobytes()
        chunks.append({
            "pcm": pcm,
            "start_s": round(s_sample / sr, 2),
            "duration_s": round(len(seg) / sr, 2),
        })

    if not chunks:
        # Fallback: single chunk
        pcm = (np.clip(data, -1, 1) * 32767).astype(np.int16).tobytes()
        chunks = [{"pcm": pcm, "start_s": 0.0, "duration_s": len(data) / sr}]

    return chunks


def _pick_voice_ref(chunks: list[dict]) -> bytes | None:
    best_wav, best_rms = None, 0.0
    for c in chunks[:5]:
        pcm = c["pcm"]
        audio = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
        rms = float(np.sqrt(np.mean(audio ** 2)))
        if rms > best_rms:
            best_rms = rms
            best_wav = _pcm_to_wav(pcm, 16000)
    return best_wav if best_rms > 0.01 else None


async def _higgs_asr(wav_bytes: bytes) -> str | None:
    if not EIGEN_KEY:
        return None
    try:
        client = await get_http()
        resp = await client.post(
            f"{EIGEN_BASE}/api/v1/generate",
            headers={"Authorization": f"Bearer {EIGEN_KEY}"},
            files={"file": ("audio.wav", wav_bytes, "audio/wav")},
            data={"model": "higgs_asr_3", "task": "asr"},
        )
        if resp.status_code != 200:
            return None
        return resp.json().get("transcription", "").strip() or None
    except Exception as e:
        print(f"[ASR Error] {e}")
        return None


async def _higgs_asr_translate(wav_bytes: bytes, target_lang: str) -> str | None:
    """AST mode — direct speech-to-text translation via Higgs ASR3."""
    if not EIGEN_KEY:
        return None
    try:
        client = await get_http()
        resp = await client.post(
            f"{EIGEN_BASE}/api/v1/generate",
            headers={"Authorization": f"Bearer {EIGEN_KEY}"},
            files={"file": ("audio.wav", wav_bytes, "audio/wav")},
            data={"model": "higgs_asr_3", "task": "ast", "language": target_lang},
        )
        if resp.status_code != 200:
            print(f"[AST] HTTP {resp.status_code}")
            return None
        result = resp.json().get("transcription", "").strip()
        if result:
            print(f"[AST] {target_lang}: {result[:60]}")
        return result or None
    except Exception as e:
        print(f"[AST Error] {e}")
        return None


async def _translate(text: str, target_lang: str) -> str | None:
    """Translate English text to target language using GPT-OSS 120B."""
    if not EIGEN_KEY:
        return None
    tgt = LANGUAGES.get(target_lang, target_lang)
    # Very strict prompt to prevent hallucination
    messages = [
        {"role": "system", "content": f"You are a translator. Translate the following English text to {tgt}. Reply with ONLY the translated text, nothing else. Do not add explanations, quotes, or extra content."},
        {"role": "user", "content": text},
    ]
    try:
        client = await get_http()
        resp = await client.post(
            f"{EIGEN_BASE}/api/v1/chat/completions",
            headers={"Authorization": f"Bearer {EIGEN_KEY}", "Content-Type": "application/json"},
            json={"model": "gpt-oss-120b", "messages": messages, "temperature": 0.1, "max_tokens": 300},
        )
        if resp.status_code == 429:
            await asyncio.sleep(0.5)
            resp = await client.post(
                f"{EIGEN_BASE}/api/v1/chat/completions",
                headers={"Authorization": f"Bearer {EIGEN_KEY}", "Content-Type": "application/json"},
                json={"model": "gpt-oss-120b", "messages": messages, "temperature": 0.1, "max_tokens": 300},
            )
        if resp.status_code != 200:
            return None
        result = resp.json().get("choices", [{}])[0].get("message", {}).get("content", "").strip()
        # Strip quotes that models sometimes add
        result = result.strip("\"'`")
        # If result is way longer than input, it's likely hallucinating
        if len(result) > len(text) * 4:
            return None
        return result if result else None
    except Exception as e:
        print(f"[Translate Error] {e}")
        return None


QWEN3_LANG = {
    "es": "Spanish", "ja": "Japanese", "zh": "Chinese",
    "fr": "French", "de": "German", "it": "Italian",
    "ko": "Korean", "ru": "Russian",
}

async def _qwen3_tts_stream(text: str, speed: float = 1.0, lang: str = "es") -> tuple[bytes | None, int]:
    """
    Qwen3 TTS via WebSocket streaming. Returns (pcm_bytes, sample_rate).
    Binary PCM chunks streamed directly — no base64 overhead.
    """
    if not EIGEN_KEY:
        return None, 24000
    try:
        import websockets
        uri = "wss://api-web.eigenai.com/api/v1/generate/ws"
        language = QWEN3_LANG.get(lang, "Auto")

        pcm_data = bytearray()
        sample_rate = 24000

        async with websockets.connect(uri, ping_interval=None) as ws:
            # 1. Authenticate
            await ws.send(json.dumps({"token": EIGEN_KEY, "model": "qwen3-tts"}))
            auth_raw = await ws.recv()
            auth = json.loads(auth_raw) if isinstance(auth_raw, str) else {}
            if auth.get("status") not in ("authenticated", "ok", None):
                print(f"[Qwen3] Auth unexpected: {auth_raw}")

            # 2. Send TTS request
            await ws.send(json.dumps({
                "text": text,
                "voice": "Vivian",
                "language": language,
                "voice_settings": {"speed": speed},
            }))

            # 3. Stream binary PCM chunks
            async for message in ws:
                if isinstance(message, bytes):
                    pcm_data.extend(message)
                else:
                    data = json.loads(message)
                    if data.get("sample_rate"):
                        sample_rate = int(data["sample_rate"])
                    elif data.get("type") == "complete":
                        break

        return (bytes(pcm_data) if len(pcm_data) > 1000 else None), sample_rate
    except Exception as e:
        print(f"[Qwen3 TTS Error] {e}")
        return None, 24000


async def _higgs_tts_stream(text: str, speed: float = 1.0,
                            voice_ref_wav: bytes | None = None) -> bytes | None:
    """
    Higgs 2.5 TTS.
    - With voice_ref: multipart POST with voice cloning (returns WAV bytes directly).
    - Without voice_ref: SSE streaming, collect PCM16 chunks.
    """
    if not EIGEN_KEY:
        return None
    clone = voice_ref_wav and len(voice_ref_wav) > 3200
    try:
        client = await get_http()
        if clone:
            resp = await client.post(
                f"{EIGEN_BASE}/api/v1/generate",
                headers={"Authorization": f"Bearer {EIGEN_KEY}"},
                data={
                    "model": "higgs2p5",
                    "text": text,
                    "voice_settings": json.dumps({"speed": speed}),
                    "sampling": json.dumps({"temperature": 0.85, "top_p": 0.95, "top_k": 50}),
                },
                files={"voice_reference_file": ("speaker.wav", voice_ref_wav, "audio/wav")},
            )
            if resp.status_code != 200:
                print(f"[Higgs TTS clone] HTTP {resp.status_code}")
                return None
            # Response is raw WAV audio bytes — return as-is (already has WAV header)
            return resp.content if len(resp.content) > 1000 else None
        else:
            # SSE streaming — collect base64 PCM16 chunks
            pcm_data = bytearray()
            async with client.stream("POST", f"{EIGEN_BASE}/api/v1/generate",
                headers={"Authorization": f"Bearer {EIGEN_KEY}", "Content-Type": "application/json"},
                json={"model": "higgs2p5", "text": text, "stream": True,
                      "voice_settings": {"speed": speed},
                      "sampling": {"temperature": 0.85, "top_p": 0.95, "top_k": 50}},
            ) as resp:
                buffer = ""
                async for chunk in resp.aiter_text():
                    buffer += chunk
                    while "\n" in buffer:
                        line, buffer = buffer.split("\n", 1)
                        line = line.strip()
                        if not line or not line.startswith("data: "):
                            continue
                        data = json.loads(line[6:])
                        if isinstance(data.get("data"), str) and len(data["data"]) > 50:
                            pcm_data.extend(base64.b64decode(data["data"]))
                        if data.get("type") == "done":
                            break
            return bytes(pcm_data) if len(pcm_data) > 1000 else None
    except Exception as e:
        print(f"[Higgs TTS Error] {e}")
        return None


def _pcm_to_wav(pcm: bytes, sr: int) -> bytes | None:
    """Convert PCM16 to WAV at given sample rate."""
    if len(pcm) < 3200:
        return None
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm)
    return buf.getvalue()


def _pcm_to_wav_24k(pcm: bytes) -> bytes | None:
    """Convert PCM16 to WAV at 24kHz (TTS output format)."""
    return _pcm_to_wav(pcm, 24000)


_NOISE = {"thank you", "thanks", "thanks for watching", "bye", "goodbye", "the end",
          "subscribe", "like and subscribe", "thanks for listening", "subtitles", "silence", "...", ".", ""}

def _is_noise(text: str) -> bool:
    if not text:
        return True
    c = text.strip().lower().rstrip(".!?,")
    return not c or c in _NOISE or (len(c.split()) > 2 and len(set(c.split())) == 1)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
