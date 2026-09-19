from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from audio_utils import read_wav_mono
from extract_features import extract_features_from_pcm
from extract_features_mfcc import extract_features_mfcc_from_pcm


def shift_with_zeros(pcm: np.ndarray, shift: int) -> np.ndarray:
    output = np.zeros_like(pcm)
    if shift > 0:
        output[shift:] = pcm[:-shift]
    elif shift < 0:
        output[:shift] = pcm[-shift:]
    else:
        output[:] = pcm
    return output


def stretch_to_length(pcm: np.ndarray, rate: float) -> np.ndarray:
    source_length = len(pcm)
    stretched_length = max(1, int(round(source_length / rate)))
    source_positions = np.linspace(0.0, 1.0, source_length, endpoint=False)
    target_positions = np.linspace(0.0, 1.0, stretched_length, endpoint=False)
    stretched = np.interp(target_positions, source_positions, pcm.astype(np.float32))
    output = np.zeros(source_length, dtype=np.float32)
    copy_length = min(source_length, stretched_length)
    output[:copy_length] = stretched[:copy_length]
    return output


def add_noise(signal: np.ndarray, snr_db: float, rng: np.random.Generator) -> np.ndarray:
    signal = signal.astype(np.float32)
    power = max(float(np.mean(signal * signal)), 1.0)
    noise_power = power / (10.0 ** (snr_db / 10.0))
    return signal + rng.normal(0.0, np.sqrt(noise_power), len(signal)).astype(np.float32)


def mix_background(signal: np.ndarray, background: np.ndarray, snr_db: float) -> np.ndarray:
    signal = signal.astype(np.float32)
    background = background.astype(np.float32)
    signal_power = max(float(np.mean(signal * signal)), 1.0)
    background_power = max(float(np.mean(background * background)), 1.0)
    target_background_power = signal_power / (10.0 ** (snr_db / 10.0))
    scale = np.sqrt(target_background_power / background_power)
    return signal + background * scale


def augment(
    pcm: np.ndarray,
    label: int,
    background: np.ndarray,
    rng: np.random.Generator,
    profile: str,
) -> np.ndarray:
    value = pcm.astype(np.float32)
    if profile == "current":
        if rng.random() < 0.65:
            value = stretch_to_length(value, float(rng.uniform(0.9, 1.1)))
        if rng.random() < 0.75:
            value = shift_with_zeros(value, int(rng.integers(-2400, 2401)))
        value *= float(rng.uniform(0.55, 1.45))
        if label != 0 and rng.random() < 0.85:
            value = mix_background(value, background, float(rng.uniform(6.0, 24.0)))
        elif rng.random() < 0.55:
            value = add_noise(value, float(rng.uniform(10.0, 30.0)), rng)
    elif profile == "competition":
        # Previous competition recipe: Gaussian noise, time shift, gain and
        # time stretch. Zero padding replaces circular wrap-around so that an
        # event cannot reappear at the opposite edge of the one-second window.
        if rng.random() < 0.30:
            value = stretch_to_length(value, float(rng.uniform(0.8, 1.2)))
        if rng.random() < 0.50:
            value = shift_with_zeros(value, int(rng.integers(-3200, 3201)))
        if rng.random() < 0.50:
            value *= float(rng.uniform(0.5, 1.5))
        if rng.random() < 0.70:
            value = add_noise(value, float(rng.uniform(5.0, 25.0)), rng)
    elif profile == "hybrid":
        # Keep the previous competition transforms, but split the corruption
        # path between evaluation-aligned Gaussian noise and real same-fold
        # background audio. Same-fold sampling prevents test-fold leakage.
        if rng.random() < 0.35:
            value = stretch_to_length(value, float(rng.uniform(0.85, 1.15)))
        if rng.random() < 0.50:
            value = shift_with_zeros(value, int(rng.integers(-3200, 3201)))
        if rng.random() < 0.50:
            value *= float(rng.uniform(0.5, 1.5))

        corruption = rng.random()
        if label != 0:
            if corruption < 0.65:
                value = add_noise(value, float(rng.uniform(3.0, 25.0)), rng)
            elif corruption < 0.95:
                value = mix_background(value, background, float(rng.uniform(3.0, 20.0)))
        else:
            if corruption < 0.35:
                value = add_noise(value, float(rng.uniform(5.0, 25.0)), rng)
            elif corruption < 0.60:
                value = mix_background(value, background, float(rng.uniform(5.0, 20.0)))
    else:
        raise ValueError(f"unknown augmentation profile: {profile}")
    return np.clip(value, -32768.0, 32767.0).astype(np.int16)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-cache", default="results/esc50_real_features.npz")
    parser.add_argument("--copies", type=int, default=2)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--feature-mode", choices=("time40", "mfcc92"), default="time40")
    parser.add_argument(
        "--profile",
        choices=("current", "competition", "hybrid"),
        default="hybrid",
        help="current v10, previous-competition recipe, or their leakage-safe hybrid",
    )
    parser.add_argument("--out", default="results/esc50_robust_augmented_features.npz")
    args = parser.parse_args()

    base = np.load(args.base_cache, allow_pickle=True)
    files = [Path(str(value)) for value in base["FILES"].tolist()]
    labels = base["Y"].astype(np.int64)
    folds = base["FOLDS"].astype(np.int64)
    pcm_values = [read_wav_mono(path) for path in files]
    background_by_fold = {
        fold: np.where((folds == fold) & (labels == 0))[0] for fold in range(1, 6)
    }
    all_copies: list[np.ndarray] = []
    for copy_index in range(args.copies):
        features: list[np.ndarray] = []
        rng = np.random.default_rng(args.seed + copy_index)
        for index, (pcm, label, fold) in enumerate(zip(pcm_values, labels.tolist(), folds.tolist()), start=1):
            background_index = int(rng.choice(background_by_fold[int(fold)]))
            augmented = augment(
                pcm,
                int(label),
                pcm_values[background_index],
                rng,
                args.profile,
            )
            if args.feature_mode == "mfcc92":
                features.append(extract_features_mfcc_from_pcm(augmented, include_delta=True))
            else:
                features.append(extract_features_from_pcm(augmented))
            if index % 250 == 0 or index == len(files):
                print(f"copy {copy_index + 1}/{args.copies}: {index}/{len(files)}")
        all_copies.append(np.stack(features).astype(np.float32))

    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        X_AUG=np.stack(all_copies),
        Y=labels,
        FILES=base["FILES"],
        SOURCES=base["SOURCES"],
        FOLDS=folds,
        DESCRIPTION=np.asarray(
            "train-only gain, zero-padded shift, stretch, Gaussian noise, same-fold real background mixing"
        ),
        AUGMENTATION_PROFILE=np.asarray(args.profile),
        AUGMENTATION_SEED=np.asarray(args.seed),
    )
    print(f"saved {output}")


if __name__ == "__main__":
    main()
