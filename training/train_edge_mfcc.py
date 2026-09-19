from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from audio_utils import read_wav_mono
from extract_features_mfcc import extract_features_mfcc_from_pcm
from sweep_edge_models import forward, metrics, recording_predictions, train


LABEL_NAMES = np.asarray(["background", "cough", "glass_break", "baby_cry", "dog_bark"])


def augmented_training_data(
    clean_x: np.ndarray,
    y: np.ndarray,
    augmented: np.ndarray,
    indices: np.ndarray,
    fraction: float,
    event_fraction: float | None,
    background_fraction: float | None,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    train_x = clean_x[indices]
    train_y = y[indices]
    augmented_x = np.concatenate([copy[indices] for copy in augmented], axis=0)
    augmented_y = np.tile(train_y, augmented.shape[0])
    event_fraction = fraction if event_fraction is None else event_fraction
    background_fraction = fraction if background_fraction is None else background_fraction
    for name, value in (
        ("event_fraction", event_fraction),
        ("background_fraction", background_fraction),
    ):
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be in [0, 1], received {value}")

    rng = np.random.default_rng(seed)
    subsets = []
    for class_id in range(len(LABEL_NAMES)):
        candidates = np.where(augmented_y == class_id)[0]
        class_fraction = background_fraction if class_id == 0 else event_fraction
        use_count = min(len(candidates), int(round(len(candidates) * class_fraction)))
        if use_count:
            subsets.append(rng.choice(candidates, size=use_count, replace=False))
    if subsets:
        subset = np.concatenate(subsets)
        rng.shuffle(subset)
        train_x = np.concatenate([train_x, augmented_x[subset]], axis=0)
        train_y = np.concatenate([train_y, augmented_y[subset]], axis=0)
    return train_x, train_y


def quantize_dequantize(weights: list[dict[str, np.ndarray]]) -> list[dict[str, np.ndarray]]:
    result = []
    for layer in weights:
        quantized_layer = {}
        for name, values in layer.items():
            maximum = max(float(np.max(np.abs(values))), 1e-10)
            scale = maximum / 127.0
            quantized = np.clip(np.round(values / scale), -127, 127).astype(np.int8)
            quantized_layer[name] = (quantized.astype(np.float32) * scale).astype(np.float32)
        result.append(quantized_layer)
    return result


def save_model(
    path: Path,
    weights: list[dict[str, np.ndarray]],
    mean: np.ndarray,
    std: np.ndarray,
    extra: dict[str, np.ndarray],
) -> None:
    if len(weights) != 2:
        raise ValueError("deployment exporter currently expects one hidden layer")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        W1=weights[0]["W"].astype(np.float32),
        B1=weights[0]["B"].astype(np.float32),
        W2=weights[1]["W"].astype(np.float32),
        B2=weights[1]["B"].astype(np.float32),
        FEATURE_MEAN=mean.astype(np.float32),
        FEATURE_STD=std.astype(np.float32),
        LABEL_NAMES=LABEL_NAMES,
        **extra,
    )


def noisy_feature_cache(
    clean_cache: np.lib.npyio.NpzFile,
    snr_db: int,
    output_path: Path,
    seed: int,
) -> np.ndarray:
    if output_path.exists():
        data = np.load(output_path, allow_pickle=True)
        if data["X"].shape == clean_cache["X"].shape:
            print(f"using noisy feature cache: {output_path}")
            return data["X"].astype(np.float32)
    rng = np.random.default_rng(seed + snr_db)
    features = []
    files = [Path(str(value)) for value in clean_cache["FILES"].tolist()]
    for index, path in enumerate(files, start=1):
        pcm = read_wav_mono(path).astype(np.float32)
        signal_power = max(float(np.mean(pcm * pcm)), 1.0)
        noise_power = signal_power / (10.0 ** (snr_db / 10.0))
        noisy = pcm + rng.normal(0.0, np.sqrt(noise_power), len(pcm)).astype(np.float32)
        noisy_pcm = np.clip(noisy, -32768.0, 32767.0).astype(np.int16)
        features.append(extract_features_mfcc_from_pcm(noisy_pcm, include_delta=True))
        if index % 250 == 0 or index == len(files):
            print(f"SNR {snr_db} dB features: {index}/{len(files)}")
    x = np.stack(features).astype(np.float32)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, X=x, SNR_DB=np.asarray(snr_db), FILES=clean_cache["FILES"])
    return x


def format_summary(result: dict[str, object]) -> str:
    lines = [
        "Edge MFCC TinyMLP cross-validation and robustness report",
        "Architecture: 92 -> 64 -> 5",
        "Split: official ESC-50 folds, grouped by source recording",
        "",
    ]
    for noise_name, values in result["recording_level"].items():
        lines.append(
            f"{noise_name}: float overall={100*values['float']['overall']:.2f}% "
            f"macro={100*values['float']['macro']:.2f}%; "
            f"int8 overall={100*values['int8']['overall']:.2f}% "
            f"macro={100*values['int8']['macro']:.2f}%"
        )
        for label, accuracy in values["float"]["per_class"].items():
            lines.append(f"  {label}: {100*accuracy:.2f}%")
        lines.append("")
    lines.extend(
        [
            f"Float32 parameter and normalization bytes: {result['size']['float32_bytes']}",
            f"INT8 weight-only bytes (estimated): {result['size']['int8_bytes']}",
            f"INT8 reduction: {result['size']['int8_reduction_percent']:.2f}%",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clean-cache", default="results/esc50_mfcc92_features.npz")
    parser.add_argument("--aug-cache", default="results/esc50_mfcc92_robust_augmented_features.npz")
    parser.add_argument("--models-dir", default="models/edge_mfcc_cv")
    parser.add_argument("--final-model", default="models/tiny_mlp_mfcc92_robust_simulator.npz")
    parser.add_argument("--report-dir", default="results/edge_mfcc_final")
    parser.add_argument("--noise-cache-dir",
                        help="reuse noisy feature caches from another report directory")
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--class-weight-power", type=float, default=0.7)
    parser.add_argument("--aug-fraction", type=float, default=0.5)
    parser.add_argument(
        "--event-aug-fraction",
        type=float,
        help="fraction of cached augmented copies used for event classes; defaults to --aug-fraction",
    )
    parser.add_argument(
        "--background-aug-fraction",
        type=float,
        help="fraction used for background; defaults to --aug-fraction",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    clean_cache = np.load(args.clean_cache, allow_pickle=True)
    clean_x = clean_cache["X"].astype(np.float32)
    y = clean_cache["Y"].astype(np.int64)
    folds = clean_cache["FOLDS"].astype(np.int64)
    sources = clean_cache["SOURCES"]
    augmented_cache = np.load(args.aug_cache, allow_pickle=True)
    augmented = augmented_cache["X_AUG"].astype(np.float32)
    augmentation_profile = (
        str(augmented_cache["AUGMENTATION_PROFILE"].item())
        if "AUGMENTATION_PROFILE" in augmented_cache.files
        else "legacy-unspecified"
    )
    if clean_x.shape[1] != 92 or augmented.shape[1:] != clean_x.shape:
        raise ValueError("expected clean and augmented MFCC92 feature caches")

    report_dir = Path(args.report_dir)
    models_dir = Path(args.models_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    noise_cache_dir = Path(args.noise_cache_dir) if args.noise_cache_dir else report_dir
    feature_sets = {"clean": clean_x}
    for snr_db in (20, 10, 5):
        feature_sets[f"snr_{snr_db}db"] = noisy_feature_cache(
            clean_cache,
            snr_db,
            noise_cache_dir / f"features_snr_{snr_db}db.npz",
            args.seed + 5000,
        )

    predictions: dict[str, dict[str, list[np.ndarray]]] = {
        name: {"truth": [], "float": [], "int8": []} for name in feature_sets
    }
    fold_results = []
    for test_fold in range(1, 6):
        train_indices = np.where(folds != test_fold)[0]
        test_indices = np.where(folds == test_fold)[0]
        train_x, train_y = augmented_training_data(
            clean_x,
            y,
            augmented,
            train_indices,
            args.aug_fraction,
            args.event_aug_fraction,
            args.background_aug_fraction,
            args.seed + 1000 + test_fold,
        )
        mean = train_x.mean(axis=0).astype(np.float32)
        std = train_x.std(axis=0).astype(np.float32)
        std[std < 1e-6] = 1.0
        weights = train(
            (train_x - mean) / std,
            train_y,
            [args.hidden_dim],
            args.epochs,
            args.batch_size,
            args.lr,
            args.weight_decay,
            args.seed + test_fold,
            args.class_weight_power,
        )
        int8_weights = quantize_dequantize(weights)
        save_model(
            models_dir / f"fold_{test_fold}.npz",
            weights,
            mean,
            std,
            {"TEST_FOLD": np.asarray(test_fold)},
        )
        fold_summary = {"fold": test_fold}
        test_sources = sources[test_indices]
        for name, feature_values in feature_sets.items():
            normalized = (feature_values[test_indices] - mean) / std
            float_probabilities, _, _ = forward(normalized, weights)
            int8_probabilities, _, _ = forward(normalized, int8_weights)
            truth, float_prediction = recording_predictions(
                float_probabilities, y[test_indices], test_sources
            )
            _, int8_prediction = recording_predictions(
                int8_probabilities, y[test_indices], test_sources
            )
            predictions[name]["truth"].append(truth)
            predictions[name]["float"].append(float_prediction)
            predictions[name]["int8"].append(int8_prediction)
            if name == "clean":
                fold_summary["clean_float"] = metrics(truth, float_prediction)
        fold_results.append(fold_summary)
        print(f"finished fold {test_fold}")

    recording_level = {}
    for name, values in predictions.items():
        truth = np.concatenate(values["truth"])
        recording_level[name] = {
            "float": metrics(truth, np.concatenate(values["float"])),
            "int8": metrics(truth, np.concatenate(values["int8"])),
        }

    train_indices = np.arange(len(y))
    final_x, final_y = augmented_training_data(
        clean_x,
        y,
        augmented,
        train_indices,
        args.aug_fraction,
        args.event_aug_fraction,
        args.background_aug_fraction,
        args.seed + 2000,
    )
    final_mean = final_x.mean(axis=0).astype(np.float32)
    final_std = final_x.std(axis=0).astype(np.float32)
    final_std[final_std < 1e-6] = 1.0
    final_weights = train(
        (final_x - final_mean) / final_std,
        final_y,
        [args.hidden_dim],
        args.epochs,
        args.batch_size,
        args.lr,
        args.weight_decay,
        args.seed + 100,
        args.class_weight_power,
    )
    save_model(
        Path(args.final_model),
        final_weights,
        final_mean,
        final_std,
        {
            "FEATURE_TYPE": np.asarray("time40+mfcc_mean_std_delta_mean_std"),
            "TRAINING_WINDOWS_CLEAN": np.asarray(len(clean_x)),
            "TRAINING_WINDOWS_TOTAL": np.asarray(len(final_x)),
            "TRAINING_RECORDINGS": np.asarray(len(set(sources.tolist()))),
            "CLASS_WEIGHT_POWER": np.asarray(args.class_weight_power),
            "AUGMENTATION_FRACTION": np.asarray(args.aug_fraction),
            "EVENT_AUGMENTATION_FRACTION": np.asarray(
                args.aug_fraction if args.event_aug_fraction is None else args.event_aug_fraction
            ),
            "BACKGROUND_AUGMENTATION_FRACTION": np.asarray(
                args.aug_fraction
                if args.background_aug_fraction is None
                else args.background_aug_fraction
            ),
        },
    )

    parameter_count = (92 + 1) * args.hidden_dim + (args.hidden_dim + 1) * len(LABEL_NAMES)
    float32_bytes = (parameter_count + 92 * 2) * 4
    int8_bytes = parameter_count + 4 * 4 + 92 * 2 * 4
    result = {
        "configuration": {
            "architecture": f"92-{args.hidden_dim}-5",
            "epochs": args.epochs,
            "class_weight_power": args.class_weight_power,
            "augmentation_fraction": args.aug_fraction,
            "augmentation_profile": augmentation_profile,
            "augmentation_copies": int(augmented.shape[0]),
            "event_augmentation_fraction": (
                args.aug_fraction if args.event_aug_fraction is None else args.event_aug_fraction
            ),
            "background_augmentation_fraction": (
                args.aug_fraction
                if args.background_aug_fraction is None
                else args.background_aug_fraction
            ),
            "feature_type": "40 time features + MFCC mean/std + MFCC delta mean/std",
        },
        "folds": fold_results,
        "recording_level": recording_level,
        "size": {
            "parameters": parameter_count,
            "float32_bytes": float32_bytes,
            "int8_bytes": int8_bytes,
            "int8_reduction_percent": 100.0 * (1.0 - int8_bytes / float32_bytes),
        },
        "final_model": args.final_model,
    }
    (report_dir / "metrics.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    summary = format_summary(result)
    (report_dir / "metrics.txt").write_text(summary, encoding="utf-8")
    print("\n" + summary)
    print(f"saved final model: {args.final_model}")


if __name__ == "__main__":
    main()
