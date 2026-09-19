from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np


def fmt_float(v: float) -> str:
    return f"{float(v):.9e}f"


def c_array(name: str, arr: np.ndarray, dims: str = "") -> str:
    arr = arr.astype(np.float32)
    if arr.ndim == 2:
        rows = []
        for row in arr:
            rows.append("    { " + ", ".join(fmt_float(v) for v in row) + " }")
        return f"static const float {name}{dims} = {{\n" + ",\n".join(rows) + "\n};\n"
    values = ", ".join(fmt_float(v) for v in arr.ravel())
    return f"static const float {name}{dims} = {{ {values} }};\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/tiny_mlp_5class_augmented.npz")
    parser.add_argument("--out", default="embedded/audiodetect/model_weights.h")
    parser.add_argument("--openvela_out", default="../openvela_app/audiodetect/model_weights.h")
    args = parser.parse_args()

    data = np.load(args.model, allow_pickle=True)
    names = [str(x) for x in data["LABEL_NAMES"].tolist()]
    num_classes = len(names)
    feature_dim = int(data["W1"].shape[0])
    hidden_dim = int(data["W1"].shape[1])
    text = f"""#ifndef MODEL_WEIGHTS_H
#define MODEL_WEIGHTS_H

#define FEATURE_DIM {feature_dim}
#define HIDDEN_DIM {hidden_dim}
#define NUM_CLASSES {num_classes}

"""
    text += c_array("W1", data["W1"], "[FEATURE_DIM][HIDDEN_DIM]")
    text += c_array("B1", data["B1"], "[HIDDEN_DIM]")
    text += c_array("W2", data["W2"], "[HIDDEN_DIM][NUM_CLASSES]")
    text += c_array("B2", data["B2"], "[NUM_CLASSES]")
    text += c_array("FEATURE_MEAN", data["FEATURE_MEAN"], "[FEATURE_DIM]")
    text += c_array("FEATURE_STD", data["FEATURE_STD"], "[FEATURE_DIM]")
    labels = ", ".join(f'"{n}"' for n in names)
    text += f"static const char *LABEL_NAMES[NUM_CLASSES] __attribute__((unused)) = {{ {labels} }};\n\n"
    text += "#endif\n"

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    openvela_out = Path(args.openvela_out)
    openvela_out.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(out, openvela_out)
    # 同步到 Linux C 便携版与 OpenVela 部署目录，避免多副本权重分叉
    for extra in ("../embedded/audiodetect/model_weights.h",
                  "../../../audio-event-openvela/openvela_app/audiodetect/model_weights.h"):
        dst = Path(extra)
        if dst.resolve() == out.resolve():
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(out, dst)
        print(f"Copied to {dst}")
    print(f"wrote {out}")
    print(f"copied {openvela_out}")


if __name__ == "__main__":
    main()
