from __future__ import annotations

import argparse
import csv
import random
import shutil
from pathlib import Path


TARGETS = {
    "glass_breaking": "glass_break",
    "coughing": "cough",
    "dog": "dog_bark",
    "crying_baby": "baby_cry",
}
DEFAULT_ESC50_DIR = Path("data/datasets/ESC-50")
DEFAULT_OUTPUT_DIR = Path("data/raw_esc50")


def copy_file(src: Path, dst: Path, overwrite: bool) -> bool:
    if dst.exists() and not overwrite:
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Import cough/glass/dog/baby/background WAVs from ESC-50.")
    parser.add_argument("--esc50_dir", default=str(DEFAULT_ESC50_DIR))
    parser.add_argument("--output_dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--background_per_class", type=int, default=5)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    esc50_dir = Path(args.esc50_dir)
    audio_dir = esc50_dir / "audio"
    meta_csv = esc50_dir / "meta" / "esc50.csv"
    out_dir = Path(args.output_dir)
    rng = random.Random(args.seed)

    if not meta_csv.exists():
        raise FileNotFoundError(f"ESC-50 metadata not found: {meta_csv}")

    rows: list[dict[str, str]] = []
    with meta_csv.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    grouped: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        grouped.setdefault(row["category"], []).append(row)

    counts = {"glass_break": 0, "cough": 0, "dog_bark": 0, "baby_cry": 0, "background": 0}
    for category, label in TARGETS.items():
        for row in grouped.get(category, []):
            src = audio_dir / row["filename"]
            dst = out_dir / label / row["filename"]
            if copy_file(src, dst, args.overwrite):
                counts[label] += 1

    background_rows: list[dict[str, str]] = []
    for category, items in grouped.items():
        if category in TARGETS:
            continue
        sample_count = min(args.background_per_class, len(items))
        background_rows.extend(rng.sample(items, sample_count))

    for row in background_rows:
        src = audio_dir / row["filename"]
        dst = out_dir / "background" / row["filename"]
        if copy_file(src, dst, args.overwrite):
            counts["background"] += 1

    for label in ["glass_break", "cough", "dog_bark", "baby_cry", "background"]:
        print(f"{label}: {counts[label]} files")


if __name__ == "__main__":
    main()
