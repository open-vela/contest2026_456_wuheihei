from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from audio_utils import read_wav_mono


FEATURE_DIM = 40
FRAME_LEN = 400
HOP_LEN = 160


def _stats(values: np.ndarray) -> list[float]:
    if values.size == 0:
        return [0.0] * 8
    half = values.size // 2
    first = values[:half] if half else values
    second = values[half:] if half else values
    return [
        float(np.mean(values)),
        float(np.std(values)),
        float(np.max(values)),
        float(np.min(values)),
        float(np.percentile(values, 25)),
        float(np.percentile(values, 50)),
        float(np.percentile(values, 75)),
        float(np.mean(second) - np.mean(first)),
    ]


def extract_features_from_pcm(pcm: np.ndarray, sample_rate: int = 16000) -> np.ndarray:
    """Extract 40 embedded-friendly time-domain features from 1 s 16 kHz PCM."""
    if sample_rate != 16000:
        raise ValueError("extract_features_from_pcm expects 16 kHz PCM")
    pcm = np.asarray(pcm, dtype=np.int16)
    if pcm.size < FRAME_LEN:
        padded = np.zeros(FRAME_LEN, dtype=np.int16)
        padded[: pcm.size] = pcm
        pcm = padded

    frame_features: list[list[float]] = []
    prev_energy = 0.0
    x = pcm.astype(np.float32) / 32768.0
    for start in range(0, x.size - FRAME_LEN + 1, HOP_LEN):
        frame = x[start : start + FRAME_LEN]
        energy = float(np.mean(frame * frame))
        signs = np.signbit(frame)
        zcr = float(np.count_nonzero(signs[1:] != signs[:-1]) / (FRAME_LEN - 1))
        mean_abs = float(np.mean(np.abs(frame)))
        max_abs = float(np.max(np.abs(frame)))
        delta_energy = abs(energy - prev_energy) if frame_features else 0.0
        frame_features.append([energy, zcr, mean_abs, max_abs, delta_energy])
        prev_energy = energy

    by_metric = np.asarray(frame_features, dtype=np.float32).T
    features: list[float] = []
    for metric in by_metric:
        features.extend(_stats(metric))
    return np.asarray(features, dtype=np.float32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wav", required=True)
    args = parser.parse_args()
    pcm = read_wav_mono(Path(args.wav))
    features = extract_features_from_pcm(pcm)
    print(" ".join(f"{v:.8f}" for v in features))


if __name__ == "__main__":
    main()
