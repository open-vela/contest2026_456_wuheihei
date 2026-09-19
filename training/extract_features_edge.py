from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import numpy as np

from audio_utils import read_wav_mono


FEATURE_DIM = 64
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


def extract_features_edge_from_pcm(pcm: np.ndarray, sample_rate: int = 16000) -> np.ndarray:
    """Extract 64 low-cost time/spectral-proxy features for MCU inference."""
    if sample_rate != 16000:
        raise ValueError("extract_features_edge_from_pcm expects 16 kHz PCM")
    pcm = np.asarray(pcm, dtype=np.int16)
    if pcm.size < FRAME_LEN:
        padded = np.zeros(FRAME_LEN, dtype=np.int16)
        padded[: pcm.size] = pcm
        pcm = padded

    frame_features: list[list[float]] = []
    previous_energy = 0.0
    x = pcm.astype(np.float32) / 32768.0
    for start in range(0, x.size - FRAME_LEN + 1, HOP_LEN):
        frame = x[start : start + FRAME_LEN]
        energy = float(np.mean(frame * frame))
        signs = np.signbit(frame)
        zcr = float(np.count_nonzero(signs[1:] != signs[:-1]) / (FRAME_LEN - 1))
        absolute = np.abs(frame)
        mean_abs = float(np.mean(absolute))
        max_abs = float(np.max(absolute))
        delta_energy = abs(energy - previous_energy) if frame_features else 0.0
        differences = frame[1:] - frame[:-1]
        difference_energy = float(np.mean(differences * differences))
        high_frequency_ratio = difference_energy / (4.0 * energy + 1e-12)
        crest_factor = max_abs / (float(np.sqrt(energy)) + 1e-12)
        lag_numerator = float(np.sum(frame[1:] * frame[:-1]))
        lag_denominator = float(
            np.sqrt(np.sum(frame[1:] * frame[1:]) * np.sum(frame[:-1] * frame[:-1]))
        )
        lag_one_correlation = lag_numerator / (lag_denominator + 1e-12)
        frame_features.append(
            [
                energy,
                zcr,
                mean_abs,
                max_abs,
                delta_energy,
                high_frequency_ratio,
                crest_factor,
                lag_one_correlation,
            ]
        )
        previous_energy = energy

    by_metric = np.asarray(frame_features, dtype=np.float32).T
    features: list[float] = []
    for metric in by_metric:
        features.extend(_stats(metric))
    return np.asarray(features, dtype=np.float32)


def build_cache(labels_path: Path, output_path: Path) -> None:
    files: list[str] = []
    labels: list[int] = []
    sources: list[str] = []
    folds: list[int] = []
    features: list[np.ndarray] = []
    with labels_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    for index, row in enumerate(rows, start=1):
        filepath = row["filepath"].replace("\\", "/")
        source = row["source_file"].replace("\\", "/")
        match = re.match(r"([1-5])-", Path(source).name)
        if match is None:
            raise ValueError(f"cannot determine ESC-50 fold: {source}")
        files.append(filepath)
        labels.append(int(row["label_id"]))
        sources.append(source)
        folds.append(int(match.group(1)))
        features.append(extract_features_edge_from_pcm(read_wav_mono(filepath)))
        if index % 250 == 0 or index == len(rows):
            print(f"extracted edge features: {index}/{len(rows)}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        X=np.stack(features).astype(np.float32),
        Y=np.asarray(labels, dtype=np.int64),
        FILES=np.asarray(files),
        SOURCES=np.asarray(sources),
        FOLDS=np.asarray(folds, dtype=np.int64),
    )
    print(f"saved {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wav")
    parser.add_argument("--labels")
    parser.add_argument("--out", default="results/esc50_edge64_features.npz")
    args = parser.parse_args()
    if args.wav:
        values = extract_features_edge_from_pcm(read_wav_mono(args.wav))
        print(" ".join(f"{value:.8f}" for value in values))
    elif args.labels:
        build_cache(Path(args.labels), Path(args.out))
    else:
        parser.error("provide --wav or --labels")


if __name__ == "__main__":
    main()
