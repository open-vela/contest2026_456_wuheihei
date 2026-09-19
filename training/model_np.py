from __future__ import annotations

from pathlib import Path

import numpy as np


def softmax(logits: np.ndarray) -> np.ndarray:
    logits = logits.astype(np.float64)
    logits -= np.max(logits, axis=-1, keepdims=True)
    exp = np.exp(logits)
    return (exp / np.sum(exp, axis=-1, keepdims=True)).astype(np.float32)


def load_model(path: str | Path) -> dict[str, np.ndarray]:
    data = np.load(path, allow_pickle=True)
    return {k: data[k] for k in data.files}


def predict_features(features: np.ndarray, model: dict[str, np.ndarray]) -> np.ndarray:
    x = (features.astype(np.float32) - model["FEATURE_MEAN"]) / (model["FEATURE_STD"] + 1e-6)
    h = np.maximum(0.0, x @ model["W1"] + model["B1"])
    logits = h @ model["W2"] + model["B2"]
    return softmax(logits)


def label_names(model: dict[str, np.ndarray]) -> list[str]:
    return [str(x) for x in model["LABEL_NAMES"].tolist()]
