#!/usr/bin/env python3
"""Evaluate the repository's TinyMLP on an external labelled WAV dataset."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from audio_utils import fixed_window, read_wav_mono
from extract_features_mfcc import extract_features_for_model
from model_np import label_names, load_model, predict_features


def summarize(cm: np.ndarray, names: list[str]) -> dict[str, object]:
    total = int(cm.sum())
    correct = int(np.trace(cm))
    classes: dict[str, dict[str, float | int]] = {}
    recalls: list[float] = []
    for idx, name in enumerate(names):
        support = int(cm[idx].sum())
        predicted = int(cm[:, idx].sum())
        tp = int(cm[idx, idx])
        accuracy = tp / support if support else 0.0
        precision = tp / predicted if predicted else 0.0
        f1 = (
            2.0 * precision * accuracy / (precision + accuracy)
            if precision + accuracy
            else 0.0
        )
        recalls.append(accuracy)
        classes[name] = {
            "support": support,
            "correct": tp,
            "accuracy": accuracy,
            "precision": precision,
            "f1": f1,
        }
    return {
        "total": total,
        "correct": correct,
        "overall_accuracy": correct / total if total else 0.0,
        "macro_accuracy": float(np.mean(recalls)) if recalls else 0.0,
        "classes": classes,
        "confusion_matrix": cm.tolist(),
    }


def print_summary(title: str, result: dict[str, object]) -> None:
    print(title)
    print(
        f"overall_accuracy={result['overall_accuracy']:.4f} "
        f"macro_accuracy={result['macro_accuracy']:.4f} "
        f"correct={result['correct']}/{result['total']}"
    )
    print("class support correct accuracy precision f1")
    for name, metrics in result["classes"].items():
        print(
            f"{name} {metrics['support']} {metrics['correct']} "
            f"{metrics['accuracy']:.4f} {metrics['precision']:.4f} "
            f"{metrics['f1']:.4f}"
        )
    print("confusion_matrix rows=true cols=predicted")
    for row in result["confusion_matrix"]:
        print(" ".join(str(value) for value in row))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--output", default="results/real_dataset_evaluation.json")
    args = parser.parse_args()

    model = load_model(args.model)
    names = label_names(model)
    label_to_id = {name: idx for idx, name in enumerate(names)}
    rows = list(csv.DictReader(Path(args.labels).open(encoding="utf-8")))

    window_cm = np.zeros((len(names), len(names)), dtype=np.int64)
    recording_probs: dict[str, list[np.ndarray]] = defaultdict(list)
    recording_truth: dict[str, int] = {}

    for index, row in enumerate(rows, 1):
        truth = label_to_id[row["label_name"]]
        wav_path = Path(row["filepath"])
        pcm = fixed_window(read_wav_mono(wav_path), 16000)
        features = extract_features_for_model(pcm, model)
        probs = predict_features(features, model)
        pred = int(np.argmax(probs))
        window_cm[truth, pred] += 1

        source = row.get("source_file") or str(wav_path)
        recording_probs[source].append(probs)
        recording_truth[source] = truth
        if index % 250 == 0:
            print(f"evaluated {index}/{len(rows)} windows", flush=True)

    recording_cm = np.zeros_like(window_cm)
    for source, probabilities in recording_probs.items():
        truth = recording_truth[source]
        pred = int(np.argmax(np.mean(probabilities, axis=0)))
        recording_cm[truth, pred] += 1

    result = {
        "model": str(Path(args.model)),
        "labels": str(Path(args.labels)),
        "label_order": names,
        "recording_level": summarize(recording_cm, names),
        "window_level": summarize(window_cm, names),
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print_summary("RECORDING_LEVEL", result["recording_level"])
    print()
    print_summary("WINDOW_LEVEL", result["window_level"])
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
