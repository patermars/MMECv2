import librosa
import soundfile as sf
import pyloudnorm as pyln
import numpy as np
from pathlib import Path

TARGET_SR   = 16000
TARGET_LUFS = -23.0


def preprocess_audio(input_path: str, output_path: str) -> dict:
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    audio, sr = librosa.load(input_path, sr=TARGET_SR, mono=True)
    audio, _  = librosa.effects.trim(audio, top_db=20)

    meter    = pyln.Meter(TARGET_SR)
    loudness = meter.integrated_loudness(audio)

    if not np.isneginf(loudness):
        audio = pyln.normalize.loudness(audio, loudness, TARGET_LUFS)

    audio = np.clip(audio, -1.0, 1.0)
    sf.write(output_path, audio, TARGET_SR)

    snr = estimate_snr(audio)
    return {
        "duration_s":    len(audio) / TARGET_SR,
        "snr_db":        snr,
        "quality_flag":  "low" if snr < 10 else "ok",
        "original_lufs": loudness,
        "output_path":   output_path,
    }


def estimate_snr(audio: np.ndarray) -> float:
    rms    = librosa.feature.rms(y=audio, frame_length=2048)[0]
    noise  = np.percentile(rms, 10)
    signal = np.percentile(rms, 90)
    return 20 * np.log10(signal / noise) if noise > 1e-8 else 40.0
