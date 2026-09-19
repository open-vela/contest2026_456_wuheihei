from __future__ import annotations

import argparse
import itertools
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np


LABEL_NAMES = ["background", "cough", "glass_break", "baby_cry", "dog_bark"]


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits.astype(np.float64) - np.max(logits, axis=1, keepdims=True)
    exp = np.exp(shifted)
    return (exp / np.sum(exp, axis=1, keepdims=True)).astype(np.float32)


def forward(x: np.ndarray, weights: list[dict[str, np.ndarray]]) -> tuple[np.ndarray, list[np.ndarray], list[np.ndarray]]:
    activations = [x]
    preactivations: list[np.ndarray] = []
    value = x
    for layer_index, layer in enumerate(weights):
        pre = value @ layer["W"] + layer["B"]
        preactivations.append(pre)
        value = np.maximum(0.0, pre) if layer_index < len(weights) - 1 else pre
        activations.append(value)
    return softmax(value), activations, preactivations


def train(
    x: np.ndarray,
    y: np.ndarray,
    hidden_dims: list[int],
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    seed: int,
    class_weight_power: float,
) -> list[dict[str, np.ndarray]]:
    rng = np.random.default_rng(seed)
    dimensions = [x.shape[1], *hidden_dims, int(np.max(y)) + 1]
    weights: list[dict[str, np.ndarray]] = []
    for fan_in, fan_out in zip(dimensions[:-1], dimensions[1:]):
        weights.append(
            {
                "W": rng.normal(0.0, np.sqrt(2.0 / fan_in), (fan_in, fan_out)).astype(np.float32),
                "B": np.zeros(fan_out, dtype=np.float32),
            }
        )
    first = [{key: np.zeros_like(value) for key, value in layer.items()} for layer in weights]
    second = [{key: np.zeros_like(value) for key, value in layer.items()} for layer in weights]
    class_counts = np.bincount(y, minlength=dimensions[-1]).astype(np.float32)
    class_weights = (len(y) / (dimensions[-1] * np.maximum(class_counts, 1.0))) ** class_weight_power
    beta1, beta2, epsilon = 0.9, 0.999, 1e-8
    step = 0

    for _ in range(epochs):
        order = rng.permutation(len(y))
        for start in range(0, len(order), batch_size):
            indices = order[start : start + batch_size]
            xb, yb = x[indices], y[indices]
            probabilities, activations, preactivations = forward(xb, weights)
            sample_weights = class_weights[yb]
            gradient = probabilities.copy()
            gradient[np.arange(len(yb)), yb] -= 1.0
            gradient *= (sample_weights / np.sum(sample_weights))[:, None]

            gradients: list[dict[str, np.ndarray]] = [dict() for _ in weights]
            for layer_index in range(len(weights) - 1, -1, -1):
                gradients[layer_index]["W"] = (
                    activations[layer_index].T @ gradient + weight_decay * weights[layer_index]["W"]
                )
                gradients[layer_index]["B"] = np.sum(gradient, axis=0)
                if layer_index > 0:
                    gradient = gradient @ weights[layer_index]["W"].T
                    gradient[preactivations[layer_index - 1] <= 0.0] = 0.0

            step += 1
            for layer_index, layer in enumerate(weights):
                for key in layer:
                    grad = gradients[layer_index][key].astype(np.float32)
                    first[layer_index][key] = beta1 * first[layer_index][key] + (1.0 - beta1) * grad
                    second[layer_index][key] = beta2 * second[layer_index][key] + (1.0 - beta2) * grad * grad
                    corrected_first = first[layer_index][key] / (1.0 - beta1**step)
                    corrected_second = second[layer_index][key] / (1.0 - beta2**step)
                    layer[key] -= lr * corrected_first / (np.sqrt(corrected_second) + epsilon)
    return weights


def recording_predictions(
    probabilities: np.ndarray, labels: np.ndarray, sources: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    grouped: dict[str, list[np.ndarray]] = defaultdict(list)
    truth: dict[str, int] = {}
    for probability, label, source_value in zip(probabilities, labels.tolist(), sources.tolist()):
        source = str(source_value)
        grouped[source].append(probability)
        truth[source] = int(label)
    names = sorted(grouped)
    y_true = np.asarray([truth[name] for name in names], dtype=np.int64)
    y_pred = np.asarray([np.mean(grouped[name], axis=0).argmax() for name in names], dtype=np.int64)
    return y_true, y_pred


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, object]:
    matrix = np.zeros((len(LABEL_NAMES), len(LABEL_NAMES)), dtype=np.int64)
    for truth, prediction in zip(y_true.tolist(), y_pred.tolist()):
        matrix[int(truth), int(prediction)] += 1
    per_class = {}
    recalls = []
    for label_id, name in enumerate(LABEL_NAMES):
        support = int(np.sum(matrix[label_id]))
        correct = int(matrix[label_id, label_id])
        recall = correct / support if support else 0.0
        recalls.append(recall)
        per_class[name] = recall
    return {
        "overall": float(np.trace(matrix) / np.sum(matrix)),
        "macro": float(np.mean(recalls)),
        "per_class": per_class,
        "confusion_matrix": matrix.tolist(),
    }


def parse_architecture(value: str) -> list[int]:
    result = [int(item) for item in value.split("x") if item]
    if not result or any(item <= 0 for item in result):
        raise argparse.ArgumentTypeError(f"invalid hidden dimensions: {value}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", default="results/esc50_real_features.npz")
    parser.add_argument("--aug-cache")
    parser.add_argument("--aug-fraction", type=float, default=1.0,
                        help="fraction of all cached augmented copies used in each training fold")
    parser.add_argument("--architectures", nargs="+", type=parse_architecture,
                        default=[[32], [64], [96], [64, 32], [96, 48], [128, 64]])
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--class-weight-powers", nargs="+", type=float, default=[1.0])
    parser.add_argument("--out", default="results/edge_model_sweep.json")
    args = parser.parse_args()

    cache = np.load(args.cache, allow_pickle=True)
    x = cache["X"].astype(np.float32)
    y = cache["Y"].astype(np.int64)
    folds = cache["FOLDS"].astype(np.int64)
    sources = cache["SOURCES"]
    augmented = None
    if args.aug_cache:
        augmented_data = np.load(args.aug_cache, allow_pickle=True)
        augmented = augmented_data["X_AUG"].astype(np.float32)
        if augmented.shape[1:] != x.shape:
            raise ValueError(f"augmented cache shape {augmented.shape} does not match base {x.shape}")
        print(f"using {augmented.shape[0]} robust augmented copies per training window")
    results = []

    for hidden_dims, class_weight_power in itertools.product(
        args.architectures, args.class_weight_powers
    ):
        started = time.monotonic()
        fold_scores = []
        all_truth, all_prediction = [], []
        feature_dim = int(x.shape[1])
        architecture = f"{feature_dim}-" + "-".join(str(value) for value in hidden_dims) + "-5"
        parameter_count = sum(
            (fan_in + 1) * fan_out
            for fan_in, fan_out in zip([feature_dim, *hidden_dims], [*hidden_dims, 5])
        )
        print(
            f"testing {architecture} ({parameter_count} parameters), "
            f"class_weight_power={class_weight_power:.2f}"
        )
        for test_fold in range(1, 6):
            train_indices = np.where(folds != test_fold)[0]
            test_indices = np.where(folds == test_fold)[0]
            train_x = x[train_indices]
            train_y = y[train_indices]
            if augmented is not None:
                augmented_x = np.concatenate([copy[train_indices] for copy in augmented], axis=0)
                augmented_y = np.tile(train_y, augmented.shape[0])
                use_count = min(
                    len(augmented_x),
                    max(0, int(round(len(augmented_x) * args.aug_fraction))),
                )
                if use_count:
                    subset_rng = np.random.default_rng(args.seed + 1000 + test_fold)
                    subset = subset_rng.choice(len(augmented_x), size=use_count, replace=False)
                    train_x = np.concatenate([train_x, augmented_x[subset]], axis=0)
                    train_y = np.concatenate([train_y, augmented_y[subset]], axis=0)
            mean = train_x.mean(axis=0)
            std = train_x.std(axis=0)
            std[std < 1e-6] = 1.0
            weights = train(
                (train_x - mean) / std,
                train_y,
                hidden_dims,
                args.epochs,
                args.batch_size,
                args.lr,
                args.weight_decay,
                args.seed + test_fold,
                class_weight_power,
            )
            probabilities, _, _ = forward((x[test_indices] - mean) / std, weights)
            truth, prediction = recording_predictions(probabilities, y[test_indices], sources[test_indices])
            score = metrics(truth, prediction)
            fold_scores.append(score)
            all_truth.append(truth)
            all_prediction.append(prediction)
            print(f"  fold {test_fold}: overall={score['overall']:.4f} macro={score['macro']:.4f}")
        aggregate = metrics(np.concatenate(all_truth), np.concatenate(all_prediction))
        result = {
            "architecture": architecture,
            "hidden_dims": hidden_dims,
            "parameters": parameter_count,
            "float32_parameter_bytes": parameter_count * 4,
            "int8_weight_bytes_approx": parameter_count,
            "class_weight_power": class_weight_power,
            "folds": fold_scores,
            "aggregate": aggregate,
            "seconds": time.monotonic() - started,
        }
        results.append(result)
        print(
            f"  aggregate: overall={aggregate['overall']:.4f} macro={aggregate['macro']:.4f} "
            f"time={result['seconds']:.1f}s"
        )

    payload = {
        "epochs": args.epochs,
        "learning_rate": args.lr,
        "weight_decay": args.weight_decay,
        "augmentation_copies": 0 if augmented is None else int(augmented.shape[0]),
        "augmentation_fraction": args.aug_fraction,
        "results": results,
    }
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"saved {output}")


if __name__ == "__main__":
    main()
