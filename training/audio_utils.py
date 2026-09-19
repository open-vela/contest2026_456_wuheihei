from __future__ import annotations

import wave
from pathlib import Path

import numpy as np


TARGET_SAMPLE_RATE = 16000


def read_wav_mono(path: str | Path, target_sample_rate: int = TARGET_SAMPLE_RATE) -> np.ndarray:
    """Read a PCM WAV file and return mono int16 samples at target_sample_rate."""
    path = Path(path)
    with wave.open(str(path), "rb") as wf:
        channels = wf.getnchannels()
        sample_width = wf.getsampwidth()
        sample_rate = wf.getframerate()
        frames = wf.getnframes()
        raw = wf.readframes(frames)

    if sample_width != 2:
        raise ValueError(f"{path}: only 16-bit PCM WAV is supported, got sample width {sample_width}")

    pcm = np.frombuffer(raw, dtype="<i2")
    if channels > 1:
        pcm = pcm.reshape(-1, channels).mean(axis=1).astype(np.int16)
    if sample_rate != target_sample_rate:
        pcm = resample_linear(pcm, sample_rate, target_sample_rate)
    return pcm.astype(np.int16, copy=False)


def write_wav_mono(path: str | Path, pcm: np.ndarray, sample_rate: int = TARGET_SAMPLE_RATE) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pcm = np.asarray(pcm)
    pcm = np.clip(pcm, -32768, 32767).astype("<i2")
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm.tobytes())


def resample_linear(pcm: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    if src_rate == dst_rate:
        return pcm.astype(np.int16, copy=False)
    if pcm.size == 0:
        return pcm.astype(np.int16)
    duration = pcm.size / float(src_rate)
    dst_len = max(1, int(round(duration * dst_rate)))
    src_x = np.linspace(0.0, duration, num=pcm.size, endpoint=False)
    dst_x = np.linspace(0.0, duration, num=dst_len, endpoint=False)
    out = np.interp(dst_x, src_x, pcm.astype(np.float32))
    return np.clip(out, -32768, 32767).astype(np.int16)


def fixed_window(pcm: np.ndarray, num_samples: int) -> np.ndarray:
    out = np.zeros(num_samples, dtype=np.int16)
    n = min(num_samples, int(pcm.size))
    if n:
        out[:n] = pcm[:n]
    return out
