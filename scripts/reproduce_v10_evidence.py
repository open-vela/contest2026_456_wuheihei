#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import subprocess
import sys
import wave
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


LABELS = ["background", "cough", "glass_break", "baby_cry", "dog_bark"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def describe(path: Path, root: Path | None = None) -> dict[str, object]:
    stat = path.stat()
    return {
        "path": str(path.relative_to(root) if root else path),
        "size_bytes": stat.st_size,
        "modified_local": datetime.fromtimestamp(stat.st_mtime).astimezone().isoformat(),
        "sha256": sha256(path),
    }


def run_logged(command: list[str], cwd: Path, log_path: Path) -> None:
    with log_path.open("a", encoding="utf-8") as log:
        log.write("$ " + " ".join(command) + "\n")
        process = subprocess.Popen(
            command,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            sys.stdout.write(line)
            log.write(line)
        return_code = process.wait()
        log.write(f"[exit_code={return_code}]\n")
    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, command)


def per_class_details(matrix_values: list[list[int]]) -> dict[str, dict[str, float | int]]:
    matrix = np.asarray(matrix_values, dtype=np.int64)
    details: dict[str, dict[str, float | int]] = {}
    for class_id, label in enumerate(LABELS):
        true_positive = int(matrix[class_id, class_id])
        support = int(matrix[class_id].sum())
        predicted = int(matrix[:, class_id].sum())
        recall = true_positive / support if support else 0.0
        precision = true_positive / predicted if predicted else 0.0
        f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
        details[label] = {
            "support": support,
            "correct": true_positive,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
    return details


def expected_label(path: Path) -> str:
    name = path.name.lower()
    for label in ("background", "cough", "baby_cry", "dog_bark", "glass_break"):
        if label in name:
            return label
    raise ValueError(f"cannot infer expected class from {path.name}")


def wav_stats(path: Path) -> dict[str, float | int]:
    with wave.open(str(path), "rb") as handle:
        channels = handle.getnchannels()
        sample_rate = handle.getframerate()
        sample_width = handle.getsampwidth()
        frames = handle.getnframes()
        raw = handle.readframes(frames)
    if channels != 1 or sample_width != 2:
        raise ValueError(f"expected 16-bit mono WAV: {path}")
    pcm = np.frombuffer(raw, dtype="<i2").astype(np.float64)
    return {
        "duration_seconds": frames / sample_rate,
        "sample_rate": sample_rate,
        "rms": float(np.sqrt(np.mean(pcm * pcm))) if len(pcm) else 0.0,
        "peak": int(np.max(np.abs(pcm))) if len(pcm) else 0,
    }


def run_simulator_suite(
    simulator_binary: Path,
    wavs: list[Path],
    output_tsv: Path,
) -> dict[str, object]:
    with output_tsv.open("w", encoding="utf-8") as output:
        subprocess.run(
            [str(simulator_binary), *map(str, wavs)],
            check=True,
            text=True,
            stdout=output,
        )
    rows = list(csv.DictReader(output_tsv.open(encoding="utf-8"), delimiter="\t"))
    results = []
    for row in rows:
        expected = expected_label(Path(row["file"]))
        energy = float(row["energy"])
        valid_event_audio = expected == "background" or energy >= 2.0e-5
        results.append(
            {
                "file": row["file"],
                "expected": expected,
                "predicted": row["predicted"],
                "correct": expected == row["predicted"],
                "valid_event_audio": valid_event_audio,
                "confidence": float(row["confidence"]),
                "energy": energy,
                "windows": int(row["windows"]),
                "alert": row["alert"] == "yes",
                "wav": wav_stats(Path(row["file"])),
            }
        )
    valid_results = [item for item in results if item["valid_event_audio"]]
    return {
        "correct": sum(int(item["correct"]) for item in results),
        "total": len(results),
        "valid_audio_correct": sum(int(item["correct"]) for item in valid_results),
        "valid_audio_total": len(valid_results),
        "results": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Freeze the current v10 artifacts and reproduce metrics without modifying them."
    )
    parser.add_argument("--python-root", type=Path, required=True)
    parser.add_argument("--openvela-root", type=Path, required=True)
    parser.add_argument("--phone-test-root", type=Path, required=True)
    parser.add_argument("--harness", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    python_root = args.python_root.resolve()
    openvela_root = args.openvela_root.resolve()
    app_root = openvela_root / "apps/audiodetect"
    phone_test_root = args.phone_test_root.resolve()
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "execution.log"

    app_files = sorted(
        path
        for path in app_root.iterdir()
        if path.is_file()
        and path.suffix in {".c", ".h"}
        or path.name in {"Kconfig", "Makefile", "CMakeLists.txt", "Make.defs", "README.md"}
    )
    python_files = [
        python_root / "audio_utils.py",
        python_root / "extract_features_mfcc.py",
        python_root / "sweep_edge_models.py",
        python_root / "train_edge_mfcc.py",
        python_root / "results/esc50_mfcc92_features.npz",
        python_root / "results/esc50_mfcc92_robust_augmented_features.npz",
        python_root / "results/edge_mfcc_final/features_snr_20db.npz",
        python_root / "results/edge_mfcc_final/features_snr_10db.npz",
        python_root / "results/edge_mfcc_final/features_snr_5db.npz",
        python_root / "models/tiny_mlp_mfcc92_edge_robust.npz",
    ]
    board_files = [
        openvela_root / "nuttx/.config",
        openvela_root / "nuttx/defconfig",
        openvela_root / "nuttx/nuttx.bin",
        openvela_root
        / "vendor/allwinnertech/boards/r528/r528s3-gemini-s1/src/etc/init.d/rcS.nsh",
    ]
    required = python_files + board_files + app_files + [args.harness.resolve()]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("missing required files:\n" + "\n".join(missing))

    clean_cache = np.load(python_root / "results/esc50_mfcc92_features.npz", allow_pickle=True)
    labels = clean_cache["Y"].astype(int)
    sources = [str(value) for value in clean_cache["SOURCES"].tolist()]
    folds = clean_cache["FOLDS"].astype(int)
    source_to_label: dict[str, int] = {}
    for source, label in zip(sources, labels.tolist()):
        if source in source_to_label and source_to_label[source] != label:
            raise ValueError(f"source has conflicting labels: {source}")
        source_to_label[source] = label

    model = np.load(python_root / "models/tiny_mlp_mfcc92_edge_robust.npz", allow_pickle=True)
    parameter_count = int(
        model["W1"].size + model["B1"].size + model["W2"].size + model["B2"].size
    )
    config_path = openvela_root / "nuttx/.config"
    config_prefixes = (
        "CONFIG_ARCH_CHIP_CUSTOM_NAME=",
        "CONFIG_ARCH_BOARD_CUSTOM_DIR=",
        "CONFIG_ARCH_BOARD_R528S3_GEMINI_S1=",
        "CONFIG_R528_AUDIO=",
        "CONFIG_AUDIO=",
        "CONFIG_AUDIO_FORMAT_PCM=",
        "CONFIG_INPUT_TOUCHSCREEN=",
        "CONFIG_LCD=",
        "CONFIG_GRAPHICS_LVGL=",
        "CONFIG_AE_AUDIODETECT=",
        "CONFIG_AE_AUDIODETECT_UI=",
        "CONFIG_AE_AUDIODETECT_UI_AUTOSTART=",
    )
    selected_config = [
        line.strip()
        for line in config_path.read_text(encoding="utf-8", errors="replace").splitlines()
        if line.startswith(config_prefixes)
    ]

    image_path = openvela_root / "nuttx/nuttx.bin"
    manifest = {
        "schema": "openvela-audio-sentinel-v10-freeze-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "definition": (
            "v10 is the latest server image by modification time and the source/configuration "
            "used by that image; this evidence run does not modify deployment sources."
        ),
        "team": {"name": "呜嘿嘿", "members": ["李炳霖", "吴安琪"]},
        "official_repository": "https://github.com/open-vela/contest2026_456_wuheihei",
        "environment": {
            "host": platform.node(),
            "platform": platform.platform(),
            "python": sys.version,
            "working_directory": os.getcwd(),
        },
        "boundaries": {
            "python_training": str(python_root),
            "c_simulator": "host-compiled current OpenVela C feature/int8/three-second pipeline",
            "real_openvela_board": str(app_root),
            "accuracy_claim_location": "Python five-fold evaluation; C simulator is a smoke/parity test only",
        },
        "dataset": {
            "feature_windows": int(len(labels)),
            "recordings": int(len(source_to_label)),
            "window_class_counts": {
                LABELS[key]: int(value) for key, value in sorted(Counter(labels.tolist()).items())
            },
            "recording_class_counts": {
                LABELS[key]: int(value)
                for key, value in sorted(Counter(source_to_label.values()).items())
            },
            "fold_recording_counts": {
                str(fold): len(
                    {
                        source
                        for source, source_fold in zip(sources, folds.tolist())
                        if source_fold == fold
                    }
                )
                for fold in sorted(set(folds.tolist()))
            },
        },
        "model": {
            "architecture": f"{model['W1'].shape[0]}-{model['W1'].shape[1]}-{model['W2'].shape[1]}",
            "parameters": parameter_count,
            "feature_type": str(model["FEATURE_TYPE"].item()),
            "training_windows_clean": int(model["TRAINING_WINDOWS_CLEAN"].item()),
            "training_windows_total": int(model["TRAINING_WINDOWS_TOTAL"].item()),
            "training_recordings": int(model["TRAINING_RECORDINGS"].item()),
        },
        "board": {
            "image": describe(image_path),
            "selected_config": selected_config,
        },
        "files": {
            "python_training_and_inputs": [describe(path, python_root) for path in python_files],
            "openvela_app": [describe(path, app_root) for path in app_files],
            "board_build": [describe(path, openvela_root) for path in board_files],
            "evidence_harness": describe(args.harness.resolve()),
        },
    }
    (output_dir / "v10_freeze_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    training_dir = output_dir / "python_cv"
    command = [
        sys.executable,
        "train_edge_mfcc.py",
        "--clean-cache",
        "results/esc50_mfcc92_features.npz",
        "--aug-cache",
        "results/esc50_mfcc92_robust_augmented_features.npz",
        "--models-dir",
        str(training_dir / "fold_models"),
        "--final-model",
        str(training_dir / "tiny_mlp_mfcc92_edge_robust_reproduced.npz"),
        "--report-dir",
        str(training_dir),
        "--noise-cache-dir",
        "results/edge_mfcc_final",
        "--epochs",
        "120",
        "--hidden-dim",
        "64",
        "--class-weight-power",
        "0.7",
        "--aug-fraction",
        "0.5",
        "--batch-size",
        "64",
        "--lr",
        "0.003",
        "--weight-decay",
        "0.0001",
        "--seed",
        "42",
    ]
    run_logged(command, python_root, log_path)

    metrics_path = training_dir / "metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    extended = {
        "method": {
            "unit": "source recording",
            "split": "official ESC-50 folds; all windows from one source stay in one fold",
            "folds": 5,
            "warning": (
                "These are simulator/offline evaluation results, not measured real-board accuracy. "
                "Overall accuracy is influenced by the larger background class; macro and per-class metrics "
                "must be reported alongside it."
            ),
        },
        "fold_clean_float_accuracy": [
            {
                "fold": int(fold["fold"]),
                "overall": float(fold["clean_float"]["overall"]),
                "macro_recall": float(fold["clean_float"]["macro"]),
            }
            for fold in metrics["folds"]
        ],
        "conditions": {},
    }
    for condition, values in metrics["recording_level"].items():
        extended["conditions"][condition] = {}
        for precision_mode in ("float", "int8"):
            item = values[precision_mode]
            extended["conditions"][condition][precision_mode] = {
                "overall_accuracy": item["overall"],
                "macro_recall": item["macro"],
                "per_class": per_class_details(item["confusion_matrix"]),
                "confusion_matrix": item["confusion_matrix"],
            }
    (training_dir / "metrics_extended.json").write_text(
        json.dumps(extended, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    simulator_dir = output_dir / "c_simulator"
    simulator_dir.mkdir(parents=True, exist_ok=True)
    simulator_binary = simulator_dir / "audiodetect_v10_sim"
    compile_command = [
        "gcc",
        "-std=c11",
        "-O2",
        "-Wall",
        "-Wextra",
        "-I",
        str(app_root),
        str(args.harness.resolve()),
        str(app_root / "audio_classifier.c"),
        str(app_root / "wav_reader.c"),
        str(app_root / "feature_extractor.c"),
        str(app_root / "keyword_detect.c"),
        str(app_root / "model_infer_int8.c"),
        "-lm",
        "-o",
        str(simulator_binary),
    ]
    run_logged(compile_command, output_dir, log_path)

    phone_wavs = sorted(phone_test_root.glob("*_phone_test.wav"))
    one_second_wavs = sorted((phone_test_root / "wav_samples").glob("*.wav"))
    if len(phone_wavs) != 5 or len(one_second_wavs) != 15:
        raise RuntimeError(
            f"expected five phone tracks and fifteen one-second files; "
            f"found {len(phone_wavs)} and {len(one_second_wavs)}"
        )
    phone_suite = run_simulator_suite(
        simulator_binary, phone_wavs, simulator_dir / "phone_tracks.tsv"
    )
    one_second_suite = run_simulator_suite(
        simulator_binary, one_second_wavs, simulator_dir / "one_second_samples.tsv"
    )
    simulator_report = {
        "purpose": (
            "Functional smoke test of the current board-equivalent C path: WAV reader, 92-feature "
            "extractor, INT8 TinyMLP, quiet gate, three one-second windows, and 0.80 event threshold."
        ),
        "accuracy_claim": (
            "Do not use this five-file convenience suite as the model accuracy claim; use the strict "
            "five-fold results in python_cv/metrics_extended.json."
        ),
        "interpretation": (
            "The long phone tracks are 15 seconds, while the frozen board classifier intentionally reads "
            "at most the first three seconds. The dog-bark phone track begins with a silent source clip, so "
            "that convenience track is rejected by the quiet gate. The one-second suite exposes the same "
            "silent dog_bark_01.wav input. Results excluding invalid silent event-labelled input are reported "
            "only as smoke-test diagnostics, not as accuracy."
        ),
        "phone_tracks_first_three_seconds": phone_suite,
        "one_second_samples": one_second_suite,
        "binary": describe(simulator_binary),
    }
    (simulator_dir / "smoke_test.json").write_text(
        json.dumps(simulator_report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    evidence = {
        "freeze_manifest_sha256": sha256(output_dir / "v10_freeze_manifest.json"),
        "reproduced_metrics_sha256": sha256(metrics_path),
        "extended_metrics_sha256": sha256(training_dir / "metrics_extended.json"),
        "reproduced_model_sha256": sha256(
            training_dir / "tiny_mlp_mfcc92_edge_robust_reproduced.npz"
        ),
        "c_simulator_smoke_sha256": sha256(simulator_dir / "smoke_test.json"),
        "execution_log_sha256": sha256(log_path),
    }
    (output_dir / "evidence_checksums.json").write_text(
        json.dumps(evidence, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(evidence, indent=2))


if __name__ == "__main__":
    main()
