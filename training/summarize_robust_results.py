"""Compare robustness reports and print the submission-table view."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


CONDITIONS = (("clean", "Clean"), ("snr_20db", "20 dB"), ("snr_10db", "10 dB"), ("snr_5db", "5 dB"))


def load(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def percent(value: float) -> str:
    return f"{100.0 * value:.2f}%"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", required=True)
    parser.add_argument("--baseline")
    args = parser.parse_args()

    result = load(args.metrics)
    baseline = load(args.baseline) if args.baseline else None
    print("| 条件 | Float 总体准确率 | Float 宏平均召回率 | INT8 总体准确率 | INT8 宏平均召回率 |")
    print("|---|---:|---:|---:|---:|")
    for key, label in CONDITIONS:
        values = result["recording_level"][key]
        print(
            f"| {label} | {percent(values['float']['overall'])} | "
            f"{percent(values['float']['macro'])} | {percent(values['int8']['overall'])} | "
            f"{percent(values['int8']['macro'])} |"
        )

    if baseline is not None:
        print("\n| 条件 | Float 总体变化 | Float 宏召回变化 | INT8 总体变化 | INT8 宏召回变化 |")
        print("|---|---:|---:|---:|---:|")
        for key, label in CONDITIONS:
            current = result["recording_level"][key]
            previous = baseline["recording_level"][key]
            deltas = (
                current["float"]["overall"] - previous["float"]["overall"],
                current["float"]["macro"] - previous["float"]["macro"],
                current["int8"]["overall"] - previous["int8"]["overall"],
                current["int8"]["macro"] - previous["int8"]["macro"],
            )
            print("| " + label + " | " + " | ".join(f"{100.0 * value:+.2f} pp" for value in deltas) + " |")

    print("\n5 dB 各类别召回率（Float / INT8）")
    for name in ("background", "cough", "glass_break", "baby_cry", "dog_bark"):
        values = result["recording_level"]["snr_5db"]
        print(f"- {name}: {percent(values['float']['per_class'][name])} / {percent(values['int8']['per_class'][name])}")


if __name__ == "__main__":
    main()
