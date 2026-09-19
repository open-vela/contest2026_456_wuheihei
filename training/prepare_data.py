from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from audio_utils import read_wav_mono, write_wav_mono


LABELS = {
    "background": 0,
    "cough": 1,
    "glass_break": 2,
    "baby_cry": 3,
    "dog_bark": 4,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert raw WAV files into 1 s 16 kHz mono clips.")
    parser.add_argument("--raw_dir", default="data/raw")
    parser.add_argument("--output_dir", default="data/processed")
    parser.add_argument("--sample_rate", type=int, default=16000)
    parser.add_argument("--window_sec", type=float, default=1.0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.output_dir)
    window_samples = int(round(args.sample_rate * args.window_sec))
    rows: list[dict[str, str | int]] = []

    for label_name, label_id in LABELS.items():
        src_dir = raw_dir / label_name
        dst_dir = out_dir / label_name
        dst_dir.mkdir(parents=True, exist_ok=True)
        if not src_dir.exists():
            print(f"warning: missing source directory {src_dir}")
            continue
        for wav_path in sorted(src_dir.glob("*.wav")):
            pcm = read_wav_mono(wav_path, args.sample_rate)
            segment_count = max(1, int(np.ceil(pcm.size / window_samples)))
            for idx in range(segment_count):
                start = idx * window_samples
                segment = np.zeros(window_samples, dtype=np.int16)
                chunk = pcm[start : start + window_samples]
                segment[: chunk.size] = chunk
                out_name = f"{wav_path.stem}_{idx:03d}.wav"
                out_path = dst_dir / out_name
                if out_path.exists() and not args.overwrite:
                    pass
                else:
                    write_wav_mono(out_path, segment, args.sample_rate)
                rows.append({
                    "filepath": str(out_path.as_posix()),
                    "label_id": label_id,
                    "label_name": label_name,
                    "source_file": str(wav_path.as_posix()),
                })

    labels_csv = out_dir / "labels.csv"
    with labels_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["filepath", "label_id", "label_name", "source_file"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} clips")
    print(f"labels: {labels_csv}")


if __name__ == "__main__":
    main()
