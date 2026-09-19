from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from audio_utils import fixed_window, read_wav_mono
from extract_features_mfcc import extract_features_for_model
from model_np import label_names, load_model, predict_features


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/tiny_mlp.npz")
    args = parser.parse_args()
    model = load_model(args.model)
    names = label_names(model)
    files = [Path(str(x).replace("\\", "/")) for x in model.get("TEST_FILES", np.array([])).tolist()]
    labels = model.get("TEST_LABELS", np.array([], dtype=np.int64)).astype(int)
    if not files:
        raise RuntimeError("model does not contain TEST_FILES; retrain with train.py")

    preds = []
    for wav in files:
        pcm = fixed_window(read_wav_mono(wav), 16000)
        probs = predict_features(extract_features_for_model(pcm, model), model)
        preds.append(int(probs.argmax()))
    preds = np.asarray(preds)
    cm = np.zeros((len(names), len(names)), dtype=int)
    for truth, pred in zip(labels, preds):
        cm[truth, pred] += 1
    accuracy = float(np.mean(preds == labels))
    lines = [f"accuracy: {accuracy:.4f}", "class precision recall f1"]
    for i, name in enumerate(names):
        tp = cm[i, i]
        precision = tp / max(1, cm[:, i].sum())
        recall = tp / max(1, cm[i, :].sum())
        f1 = 2 * precision * recall / max(1e-12, precision + recall)
        lines.append(f"{name}: precision={precision:.4f} recall={recall:.4f} f1={f1:.4f}")
    lines.append("confusion matrix:")
    lines.extend(" ".join(str(v) for v in row) for row in cm)
    Path("results").mkdir(exist_ok=True)
    Path("results/evaluation.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    try:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(5, 4))
        ax.imshow(cm, cmap="Blues")
        ax.set_xticks(range(len(names)), names, rotation=30, ha="right")
        ax.set_yticks(range(len(names)), names)
        for r in range(len(names)):
            for c in range(len(names)):
                ax.text(c, r, str(cm[r, c]), ha="center", va="center")
        ax.set_xlabel("predicted")
        ax.set_ylabel("true")
        fig.tight_layout()
        fig.savefig("results/confusion_matrix.png")
    except Exception as exc:
        print(f"warning: could not save confusion_matrix.png: {exc}")


if __name__ == "__main__":
    main()
