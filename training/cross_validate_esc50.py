from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

from audio_utils import read_wav_mono
from extract_features import extract_features_from_pcm


LABEL_NAMES = np.array(["background", "cough", "glass_break", "baby_cry", "dog_bark"])


def load_rows(labels_path: Path) -> tuple[list[Path], np.ndarray, list[str], np.ndarray]:
    files: list[Path] = []
    labels: list[int] = []
    sources: list[str] = []
    folds: list[int] = []
    with labels_path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            wav_path = Path(row["filepath"].replace("\\", "/"))
            source = row.get("source_file", str(wav_path)).replace("\\", "/")
            match = re.match(r"([1-5])-", Path(source).name)
            if match is None:
                raise ValueError(f"cannot determine ESC-50 fold from source file: {source}")
            files.append(wav_path)
            labels.append(int(row["label_id"]))
            sources.append(source)
            folds.append(int(match.group(1)))
    return files, np.asarray(labels, dtype=np.int64), sources, np.asarray(folds, dtype=np.int64)


def load_or_extract_features(
    files: list[Path], y: np.ndarray, sources: list[str], folds: np.ndarray, cache_path: Path
) -> np.ndarray:
    if cache_path.exists():
        cached = np.load(cache_path, allow_pickle=True)
        cached_files = [str(item) for item in cached["FILES"].tolist()]
        if cached_files == [str(item) for item in files]:
            print(f"using feature cache: {cache_path}")
            return cached["X"].astype(np.float32)
        print("feature cache does not match labels; rebuilding it")

    features: list[np.ndarray] = []
    for index, wav_path in enumerate(files, start=1):
        features.append(extract_features_from_pcm(read_wav_mono(wav_path)))
        if index % 250 == 0 or index == len(files):
            print(f"extracted features: {index}/{len(files)}")
    x = np.stack(features).astype(np.float32)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        X=x,
        Y=y,
        FILES=np.asarray([str(item) for item in files]),
        SOURCES=np.asarray(sources),
        FOLDS=folds,
    )
    return x


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits.astype(np.float64) - np.max(logits, axis=1, keepdims=True)
    exp = np.exp(shifted)
    return (exp / np.sum(exp, axis=1, keepdims=True)).astype(np.float32)


def predict(x_norm: np.ndarray, weights: dict[str, np.ndarray]) -> np.ndarray:
    hidden = np.maximum(0.0, x_norm @ weights["W1"] + weights["B1"])
    return softmax(hidden @ weights["W2"] + weights["B2"])


def balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> float:
    values = []
    for cls in range(num_classes):
        mask = y_true == cls
        if np.any(mask):
            values.append(float(np.mean(y_pred[mask] == cls)))
    return float(np.mean(values)) if values else 0.0


def train_adam(
    x_norm: np.ndarray,
    y: np.ndarray,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    seed: int,
    num_classes: int,
    title: str,
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    feature_dim = x_norm.shape[1]
    hidden_dim = 32
    weights = {
        "W1": rng.normal(0.0, np.sqrt(2.0 / feature_dim), (feature_dim, hidden_dim)).astype(np.float32),
        "B1": np.zeros(hidden_dim, dtype=np.float32),
        "W2": rng.normal(0.0, np.sqrt(2.0 / hidden_dim), (hidden_dim, num_classes)).astype(np.float32),
        "B2": np.zeros(num_classes, dtype=np.float32),
    }
    first_moment = {key: np.zeros_like(value) for key, value in weights.items()}
    second_moment = {key: np.zeros_like(value) for key, value in weights.items()}
    counts = np.bincount(y, minlength=num_classes).astype(np.float32)
    class_weights = len(y) / (num_classes * np.maximum(counts, 1.0))
    beta1, beta2, epsilon = 0.9, 0.999, 1e-8
    step = 0

    for epoch in range(1, epochs + 1):
        order = rng.permutation(len(y))
        epoch_loss = 0.0
        epoch_weight = 0.0
        for start in range(0, len(order), batch_size):
            idx = order[start : start + batch_size]
            xb = x_norm[idx]
            yb = y[idx]
            sample_weights = class_weights[yb]
            denominator = float(np.sum(sample_weights))

            hidden_pre = xb @ weights["W1"] + weights["B1"]
            hidden = np.maximum(0.0, hidden_pre)
            probabilities = softmax(hidden @ weights["W2"] + weights["B2"])
            epoch_loss += float(np.sum(-np.log(probabilities[np.arange(len(yb)), yb] + 1e-9) * sample_weights))
            epoch_weight += denominator

            grad_logits = probabilities.copy()
            grad_logits[np.arange(len(yb)), yb] -= 1.0
            grad_logits *= (sample_weights / max(denominator, 1e-12))[:, None]
            gradients = {
                "W2": hidden.T @ grad_logits + weight_decay * weights["W2"],
                "B2": np.sum(grad_logits, axis=0),
            }
            grad_hidden = grad_logits @ weights["W2"].T
            grad_hidden[hidden_pre <= 0.0] = 0.0
            gradients["W1"] = xb.T @ grad_hidden + weight_decay * weights["W1"]
            gradients["B1"] = np.sum(grad_hidden, axis=0)

            step += 1
            for key in weights:
                gradient = gradients[key].astype(np.float32)
                first_moment[key] = beta1 * first_moment[key] + (1.0 - beta1) * gradient
                second_moment[key] = beta2 * second_moment[key] + (1.0 - beta2) * gradient * gradient
                corrected_first = first_moment[key] / (1.0 - beta1**step)
                corrected_second = second_moment[key] / (1.0 - beta2**step)
                weights[key] -= lr * corrected_first / (np.sqrt(corrected_second) + epsilon)

        if epoch == 1 or epoch % 25 == 0 or epoch == epochs:
            train_pred = predict(x_norm, weights).argmax(axis=1)
            train_accuracy = float(np.mean(train_pred == y))
            train_macro = balanced_accuracy(y, train_pred, num_classes)
            print(
                f"{title} epoch {epoch:03d}/{epochs} "
                f"loss={epoch_loss / max(epoch_weight, 1e-12):.4f} "
                f"accuracy={train_accuracy:.3f} balanced={train_macro:.3f}"
            )
    return weights


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> np.ndarray:
    matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    for truth, prediction in zip(y_true.tolist(), y_pred.tolist()):
        matrix[int(truth), int(prediction)] += 1
    return matrix


def metrics_from_confusion(matrix: np.ndarray, label_names: np.ndarray) -> dict[str, object]:
    total = int(np.sum(matrix))
    correct = int(np.trace(matrix))
    per_class: dict[str, dict[str, float | int]] = {}
    recalls = []
    for cls, name in enumerate(label_names.tolist()):
        support = int(np.sum(matrix[cls]))
        predicted = int(np.sum(matrix[:, cls]))
        true_positive = int(matrix[cls, cls])
        recall = true_positive / support if support else 0.0
        precision = true_positive / predicted if predicted else 0.0
        f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
        recalls.append(recall)
        per_class[str(name)] = {
            "support": support,
            "correct": true_positive,
            "accuracy_recall": recall,
            "precision": precision,
            "f1": f1,
        }
    return {
        "overall_accuracy": correct / total if total else 0.0,
        "macro_accuracy": float(np.mean(recalls)) if recalls else 0.0,
        "correct": correct,
        "total": total,
        "per_class": per_class,
        "confusion_matrix": matrix.tolist(),
    }


def recording_predictions(
    probabilities: np.ndarray, y: np.ndarray, sources: list[str]
) -> tuple[np.ndarray, np.ndarray]:
    grouped_probabilities: dict[str, list[np.ndarray]] = defaultdict(list)
    grouped_labels: dict[str, int] = {}
    for probability, label, source in zip(probabilities, y.tolist(), sources):
        grouped_probabilities[source].append(probability)
        if source in grouped_labels and grouped_labels[source] != int(label):
            raise ValueError(f"source file has conflicting labels: {source}")
        grouped_labels[source] = int(label)
    ordered_sources = sorted(grouped_probabilities)
    truth = np.asarray([grouped_labels[source] for source in ordered_sources], dtype=np.int64)
    prediction = np.asarray(
        [np.mean(grouped_probabilities[source], axis=0).argmax() for source in ordered_sources],
        dtype=np.int64,
    )
    return truth, prediction


def save_model(
    path: Path,
    weights: dict[str, np.ndarray],
    mean: np.ndarray,
    std: np.ndarray,
    label_names: np.ndarray,
    extra: dict[str, np.ndarray] | None = None,
) -> None:
    payload: dict[str, np.ndarray] = {
        "W1": weights["W1"].astype(np.float32),
        "B1": weights["B1"].astype(np.float32),
        "W2": weights["W2"].astype(np.float32),
        "B2": weights["B2"].astype(np.float32),
        "FEATURE_MEAN": mean.astype(np.float32),
        "FEATURE_STD": std.astype(np.float32),
        "LABEL_NAMES": label_names,
    }
    if extra:
        payload.update(extra)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **payload)


def percent(value: float) -> str:
    return f"{100.0 * value:.2f}%"


def report_text(result: dict[str, object], label_names: np.ndarray) -> str:
    lines = [
        "ESC-50 five-fold cross-validation",
        "Split unit: original 5-second recording (no 1-second-window leakage)",
        "",
    ]
    folds = result["folds"]
    assert isinstance(folds, list)
    for fold in folds:
        assert isinstance(fold, dict)
        recording = fold["recording_level"]
        assert isinstance(recording, dict)
        lines.append(
            f"Fold {fold['fold']}: recording accuracy {percent(float(recording['overall_accuracy']))}, "
            f"macro {percent(float(recording['macro_accuracy']))}"
        )
    for level_key, level_title in (("recording_level", "Recording-level aggregate"), ("window_level", "Window-level aggregate")):
        metrics = result[level_key]
        assert isinstance(metrics, dict)
        lines.extend(
            [
                "",
                level_title,
                f"Overall accuracy: {percent(float(metrics['overall_accuracy']))} ({metrics['correct']}/{metrics['total']})",
                f"Macro accuracy: {percent(float(metrics['macro_accuracy']))}",
            ]
        )
        per_class = metrics["per_class"]
        assert isinstance(per_class, dict)
        for name in label_names.tolist():
            item = per_class[str(name)]
            lines.append(
                f"  {name}: {percent(float(item['accuracy_recall']))} "
                f"({item['correct']}/{item['support']}), precision {percent(float(item['precision']))}, "
                f"F1 {percent(float(item['f1']))}"
            )
        lines.append("Confusion matrix (rows=true, columns=predicted):")
        for row in metrics["confusion_matrix"]:
            lines.append("  " + " ".join(f"{int(value):4d}" for value in row))
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Train TinyMLP with official ESC-50 five-fold cross-validation")
    parser.add_argument("--labels", default="data/processed_esc50_real/labels.csv")
    parser.add_argument("--feature-cache", default="results/esc50_real_features.npz")
    parser.add_argument("--output-dir", default="results/esc50_cv")
    parser.add_argument("--models-dir", default="models/esc50_cv")
    parser.add_argument("--final-model", default="models/tiny_mlp_5class_esc50_real.npz")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    labels_path = Path(args.labels)
    files, y, sources, folds = load_rows(labels_path)
    if len(files) == 0:
        raise RuntimeError("no training clips found")
    num_classes = int(np.max(y)) + 1
    label_names = LABEL_NAMES[:num_classes]
    x = load_or_extract_features(files, y, sources, folds, Path(args.feature_cache))

    output_dir = Path(args.output_dir)
    models_dir = Path(args.models_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    all_window_truth: list[np.ndarray] = []
    all_window_prediction: list[np.ndarray] = []
    all_recording_truth: list[np.ndarray] = []
    all_recording_prediction: list[np.ndarray] = []
    fold_results: list[dict[str, object]] = []

    for test_fold in range(1, 6):
        train_idx = np.where(folds != test_fold)[0]
        test_idx = np.where(folds == test_fold)[0]
        mean = x[train_idx].mean(axis=0).astype(np.float32)
        std = x[train_idx].std(axis=0).astype(np.float32)
        std[std < 1e-6] = 1.0
        x_train = (x[train_idx] - mean) / std
        x_test = (x[test_idx] - mean) / std
        weights = train_adam(
            x_train,
            y[train_idx],
            args.epochs,
            args.batch_size,
            args.lr,
            args.weight_decay,
            args.seed + test_fold,
            num_classes,
            f"fold {test_fold}",
        )
        test_probabilities = predict(x_test, weights)
        window_prediction = test_probabilities.argmax(axis=1)
        window_matrix = confusion_matrix(y[test_idx], window_prediction, num_classes)
        test_sources = [sources[index] for index in test_idx.tolist()]
        recording_truth, recording_prediction = recording_predictions(
            test_probabilities, y[test_idx], test_sources
        )
        recording_matrix = confusion_matrix(recording_truth, recording_prediction, num_classes)
        window_metrics = metrics_from_confusion(window_matrix, label_names)
        recording_metrics = metrics_from_confusion(recording_matrix, label_names)
        fold_result: dict[str, object] = {
            "fold": test_fold,
            "train_windows": int(len(train_idx)),
            "test_windows": int(len(test_idx)),
            "test_recordings": int(len(recording_truth)),
            "window_level": window_metrics,
            "recording_level": recording_metrics,
        }
        fold_results.append(fold_result)
        all_window_truth.append(y[test_idx])
        all_window_prediction.append(window_prediction)
        all_recording_truth.append(recording_truth)
        all_recording_prediction.append(recording_prediction)
        save_model(
            models_dir / f"fold_{test_fold}.npz",
            weights,
            mean,
            std,
            label_names,
            {
                "TEST_FOLD": np.asarray(test_fold),
                "TEST_FILES": np.asarray([str(files[index]) for index in test_idx.tolist()]),
                "TEST_LABELS": y[test_idx],
            },
        )
        print(
            f"fold {test_fold} finished: recording={percent(float(recording_metrics['overall_accuracy']))} "
            f"macro={percent(float(recording_metrics['macro_accuracy']))}"
        )

    aggregate_window = metrics_from_confusion(
        confusion_matrix(np.concatenate(all_window_truth), np.concatenate(all_window_prediction), num_classes),
        label_names,
    )
    aggregate_recording = metrics_from_confusion(
        confusion_matrix(
            np.concatenate(all_recording_truth), np.concatenate(all_recording_prediction), num_classes
        ),
        label_names,
    )
    result: dict[str, object] = {
        "dataset": "ESC-50 selected with import_esc50.py",
        "architecture": "40-feature TinyMLP (40-32-5)",
        "split": "official ESC-50 folds grouped by source recording",
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "weight_decay": args.weight_decay,
        "folds": fold_results,
        "recording_level": aggregate_recording,
        "window_level": aggregate_window,
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    text = report_text(result, label_names)
    (output_dir / "metrics.txt").write_text(text, encoding="utf-8")
    print("\n" + text)

    final_mean = x.mean(axis=0).astype(np.float32)
    final_std = x.std(axis=0).astype(np.float32)
    final_std[final_std < 1e-6] = 1.0
    final_weights = train_adam(
        (x - final_mean) / final_std,
        y,
        args.epochs,
        args.batch_size,
        args.lr,
        args.weight_decay,
        args.seed + 100,
        num_classes,
        "final",
    )
    final_model = Path(args.final_model)
    save_model(
        final_model,
        final_weights,
        final_mean,
        final_std,
        label_names,
        {
            "TRAINING_DATASET": np.asarray("ESC-50 real subset"),
            "TRAINING_WINDOWS": np.asarray(len(y)),
            "TRAINING_RECORDINGS": np.asarray(len(set(sources))),
            "CV_METRICS_FILE": np.asarray(str(output_dir / "metrics.json")),
        },
    )
    print(f"saved final all-data model: {final_model}")


if __name__ == "__main__":
    main()
