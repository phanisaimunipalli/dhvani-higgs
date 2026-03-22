"""
Dhvani — Emotion Detection from Audio
Analyzes raw PCM audio to detect speaker emotion via acoustic features.
Maps to Higgs TTS emotion + speed parameters for emotion-preserving translation.

Thresholds calibrated for YouTube/processed audio (louder than raw mic input).
"""

import numpy as np


def detect_emotion(pcm_bytes: bytes, sample_rate: int = 16000) -> dict:
    """
    Analyze raw PCM int16 audio and return emotion classification.

    Returns:
        {
            "emotion": str,       # neutral|excited|angry|sad|whisper|happy|fearful|surprised
            "speed": float,       # 0.6 - 1.5 TTS speed
            "confidence": float,  # 0 - 1
            "features": dict,     # raw acoustic features for debugging
            "emoji": str,         # visual indicator
            "label": str,         # human-readable label
        }
    """
    audio = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0

    if len(audio) < 1600:
        return _default()

    # === FEATURE EXTRACTION ===

    # 1. RMS Energy (volume/loudness)
    rms = float(np.sqrt(np.mean(audio ** 2)))

    # 2. Peak amplitude
    peak = float(np.max(np.abs(audio)))

    # 3. Zero Crossing Rate (voice quality — breathy, harsh, etc.)
    zcr = float(np.mean(np.abs(np.diff(np.sign(audio)))) / 2)

    # 4. Pitch estimation via autocorrelation
    pitch_hz = _estimate_pitch(audio, sample_rate)

    # 5. Speaking rate proxy: energy variance over time (more variation = more animated)
    frame_len = sample_rate // 10  # 100ms frames
    if len(audio) > frame_len * 3:
        frames = [audio[i:i+frame_len] for i in range(0, len(audio) - frame_len, frame_len)]
        frame_energies = [float(np.sqrt(np.mean(f ** 2))) for f in frames]
        energy_var = float(np.std(frame_energies))
        energy_range = max(frame_energies) - min(frame_energies) if frame_energies else 0
    else:
        energy_var = 0.0
        energy_range = 0.0

    # 6. Spectral centroid (brightness of voice)
    fft = np.abs(np.fft.rfft(audio))
    freqs = np.fft.rfftfreq(len(audio), 1.0 / sample_rate)
    spectral_centroid = float(np.sum(freqs * fft) / (np.sum(fft) + 1e-10))

    # 7. Dynamic range (difference between loud and quiet parts)
    dynamic_range = peak / (rms + 1e-10)

    features = {
        "rms": round(rms, 4),
        "peak": round(peak, 4),
        "zcr": round(zcr, 4),
        "pitch_hz": round(pitch_hz, 1),
        "energy_var": round(energy_var, 4),
        "energy_range": round(energy_range, 4),
        "spectral_centroid": round(spectral_centroid, 1),
        "dynamic_range": round(dynamic_range, 2),
    }

    # === EMOTION CLASSIFICATION ===
    emotion, speed, confidence, emoji, label = _classify(
        rms, peak, zcr, pitch_hz, energy_var, energy_range, spectral_centroid, dynamic_range
    )

    return {
        "emotion": emotion,
        "speed": speed,
        "confidence": confidence,
        "features": features,
        "emoji": emoji,
        "label": label,
    }


def _estimate_pitch(audio: np.ndarray, sr: int) -> float:
    """Estimate fundamental frequency via autocorrelation."""
    # Only analyze a segment (first 0.5s) for speed
    seg = audio[:min(len(audio), sr // 2)]
    if len(seg) < 200:
        return 0.0

    # Autocorrelation
    corr = np.correlate(seg, seg, mode='full')
    corr = corr[len(corr) // 2:]

    # Find first peak after initial decay (skip lag 0)
    # Human voice: 85Hz (male low) to 300Hz (female high)
    min_lag = sr // 400  # 400 Hz max
    max_lag = sr // 70   # 70 Hz min

    if max_lag >= len(corr):
        max_lag = len(corr) - 1
    if min_lag >= max_lag:
        return 0.0

    segment = corr[min_lag:max_lag]
    if len(segment) == 0:
        return 0.0

    peak_idx = int(np.argmax(segment)) + min_lag

    if peak_idx > 0 and corr[peak_idx] > 0.1 * corr[0]:
        return float(sr / peak_idx)
    return 0.0


def _classify(rms, peak, zcr, pitch, e_var, e_range, centroid, dyn_range):
    """
    Rule-based emotion classification from acoustic features.
    Returns (emotion, speed, confidence, emoji, label).

    Thresholds raised for YouTube/mastered audio where RMS is typically 0.08-0.25.
    Normal conversational YouTube speech: RMS ~0.08-0.18, pitch 100-200.
    Only flag emotions for genuinely extreme values.
    """

    # WHISPER: very low energy, low peak
    if rms < 0.02 and peak < 0.1:
        return "whisper", 0.85, 0.8, "&#x1F92B;", "Whisper"

    # SHOUTING: extremely loud — well above normal YouTube levels
    if rms > 0.40:
        return "excited", 1.3, 0.85, "&#x1F525;", "Excited"

    # EXCITED: very loud + high pitch (both must be extreme)
    if rms > 0.30 and pitch > 220:
        return "excited", 1.25, 0.75, "&#x1F525;", "Excited"

    # ANGRY: loud + harsh voice quality (high ZCR) + lower pitch
    if rms > 0.25 and zcr > 0.15:
        if pitch < 200:
            return "angry", 1.15, 0.7, "&#x1F620;", "Angry"
        else:
            return "excited", 1.2, 0.7, "&#x1F525;", "Excited"

    # SURPRISED: sudden loud peak relative to average (extreme dynamic range)
    if peak > 0.7 and dyn_range > 5:
        return "surprised", 1.2, 0.65, "&#x1F632;", "Surprised"

    # HAPPY: notably energetic + higher pitch (well above conversational)
    if rms > 0.20 and pitch > 220:
        return "happy", 1.1, 0.65, "&#x1F60A;", "Happy"

    # SAD: low energy + low pitch (monotone quiet voice)
    if rms < 0.04 and pitch > 0 and pitch < 140:
        return "sad", 0.8, 0.6, "&#x1F622;", "Sad"

    # FEARFUL: high pitch but not loud (nervous voice)
    if pitch > 250 and rms < 0.10:
        return "fearful", 1.15, 0.55, "&#x1F628;", "Fearful"

    # DEFAULT: normal conversational voice — this is what most YouTube speech should be
    return "neutral", 1.0, 0.7, "&#x1F399;", "Speaking"


def _default():
    return {
        "emotion": "neutral",
        "speed": 1.0,
        "confidence": 0.0,
        "features": {},
        "emoji": "&#x1F399;",
        "label": "Speaking",
    }
