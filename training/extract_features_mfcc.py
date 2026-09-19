from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import numpy as np

from audio_utils import read_wav_mono
from extract_features import extract_features_from_pcm
from keyword_detect import extract_mfcc


def extract_features_mfcc_from_pcm(pcm: np.ndarray, include_delta: bool = False) -> np.ndarray:
    time_features = extract_features_from_pcm(pcm)
    mfcc = extract_mfcc(pcm).astype(np.float32)
    spectral = [mfcc.mean(axis=0), mfcc.std(axis=0)]
    if include_delta:
        delta = np.diff(mfcc, axis=0)
        if len(delta) == 0:
            delta = np.zeros((1, mfcc.shape[1]), dtype=np.float32)
        spectral.extend([delta.mean(axis=0), delta.std(axis=0)])
    return np.concatenate([time_features, *spectral]).astype(np.float32)


def extract_features_for_model(pcm: np.ndarray, model: dict[str, np.ndarray]) -> np.ndarray:
    """Select the compatible feature front-end from the model input width."""
    feature_dim = int(model["W1"].shape[0])
    if feature_dim == 40:
        return extract_features_from_pcm(pcm)
    if feature_dim == 66:
        return extract_features_mfcc_from_pcm(pcm, include_delta=False)
    if feature_dim == 92:
        return extract_features_mfcc_from_pcm(pcm, include_delta=True)
    raise ValueError(f"unsupported model feature dimension: {feature_dim}")


def build_cache(labels_path: Path, output_path: Path, include_delta: bool) -> None:
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
        pcm = read_wav_mono(filepath)
        features.append(extract_features_mfcc_from_pcm(pcm, include_delta=include_delta))
        if index % 250 == 0 or index == len(rows):
            print(f"extracted MFCC features: {index}/{len(rows)}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        X=np.stack(features).astype(np.float32),
        Y=np.asarray(labels, dtype=np.int64),
        FILES=np.asarray(files),
        SOURCES=np.asarray(sources),
        FOLDS=np.asarray(folds, dtype=np.int64),
        INCLUDE_DELTA=np.asarray(include_delta),
    )
    print(f"saved {output_path}; feature_dim={features[0].shape[0]}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", default="data/processed_esc50_real/labels.csv")
    parser.add_argument("--out", default="results/esc50_mfcc66_features.npz")
    parser.add_argument("--include-delta", action="store_true")
    args = parser.parse_args()
    build_cache(Path(args.labels), Path(args.out), args.include_delta)


if __name__ == "__main__":
    main()
