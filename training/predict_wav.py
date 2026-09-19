from __future__ import annotations

import argparse
from pathlib import Path

from audio_utils import fixed_window, read_wav_mono
from extract_features_mfcc import extract_features_for_model
from model_np import label_names, load_model, predict_features


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wav", required=True)
    parser.add_argument("--model", default="models/tiny_mlp.npz")
    parser.add_argument("--threshold", type=float, default=0.80)
    args = parser.parse_args()

    model = load_model(args.model)
    names = label_names(model)
    pcm = fixed_window(read_wav_mono(args.wav), 16000)
    features = extract_features_for_model(pcm, model)
    probs = predict_features(features, model)
    pred = int(probs.argmax())
    confidence = float(probs[pred])

    print(f"Input: {args.wav}")
    print(f"Prediction: {names[pred]}")
    print(f"Confidence: {confidence:.3f}")
    print("Probabilities:")
    for name, prob in zip(names, probs):
        print(f"{name}: {float(prob):.3f}")
    if pred != 0 and confidence > args.threshold:
        print("[ALERT] Local alarm triggered")


if __name__ == "__main__":
    main()
