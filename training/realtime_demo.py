from __future__ import annotations

import argparse
import time

import numpy as np

from audio_utils import fixed_window, read_wav_mono
from extract_features_mfcc import extract_features_for_model
from model_np import label_names, load_model, predict_features


def report(pcm: np.ndarray, model: dict[str, np.ndarray], names: list[str], threshold: float) -> None:
    window = fixed_window(pcm, 16000)
    probs = predict_features(extract_features_for_model(window, model), model)
    pred = int(probs.argmax())
    confidence = float(probs[pred])
    print(f"{names[pred]} {confidence:.3f}")
    if pred != 0 and confidence > threshold:
        print("[ALERT] Local alarm triggered")


def run_wav_loop(path: str, model: dict[str, np.ndarray], names: list[str], threshold: float) -> None:
    pcm = read_wav_mono(path)
    step = 8000
    pos = 0
    while True:
        if pos + 16000 > len(pcm):
            window = np.concatenate([pcm[pos:], pcm[: max(0, pos + 16000 - len(pcm))]])
            pos = (pos + step) % max(1, len(pcm))
        else:
            window = pcm[pos : pos + 16000]
            pos += step
        report(window, model, names, threshold)
        time.sleep(0.5)


def run_microphone(model: dict[str, np.ndarray], names: list[str], threshold: float) -> None:
    import sounddevice as sd

    sr = 16000
    step = 8000
    buffer = np.zeros(sr, dtype=np.float32)

    def callback(indata, frames, time_info, status):
        nonlocal buffer
        if status:
            print(status)
        mono = indata[:, 0]
        buffer = np.concatenate([buffer, mono])[-sr:]

    with sd.InputStream(channels=1, samplerate=sr, blocksize=step, callback=callback):
        print("listening... Ctrl+C to stop")
        while True:
            pcm = np.clip(buffer * 32768.0, -32768, 32767).astype(np.int16)
            report(pcm, model, names, threshold)
            time.sleep(0.5)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/tiny_mlp.npz")
    parser.add_argument("--input_wav")
    parser.add_argument("--threshold", type=float, default=0.80)
    args = parser.parse_args()
    model = load_model(args.model)
    names = label_names(model)
    if args.input_wav:
        run_wav_loop(args.input_wav, model, names, args.threshold)
    else:
        run_microphone(model, names, args.threshold)


if __name__ == "__main__":
    main()
