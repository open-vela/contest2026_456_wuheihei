#!/usr/bin/env python3
"""Export the project model to every C weight header used by the repository.

One model, three headers, produced in a single run so the OpenVela board weights
and the host-simulator weights can never diverge:

  app/audiodetect/model_weights.h            float32 weights
  app/audiodetect/model_weights_int8.h       weight-only INT8, board guard
  simulator/generated/model_weights_int8.h   weight-only INT8, simulator guard

INT8 here is weight-only: weights are stored as int8 with a per-tensor float
scale and dequantized back to float32 at run time. Activations stay float32, so
the benefit is weight storage, not integer-only MAC throughput.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]


def fmt_float(v: float) -> str:
    return f"{float(v):.9e}f"


def c_float_array(name: str, arr: np.ndarray, dims: str = "") -> str:
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim == 2:
        rows = ["    { " + ", ".join(fmt_float(v) for v in row) + " }" for row in arr]
        return f"static const float {name}{dims} = {{\n" + ",\n".join(rows) + "\n};\n"
    values = ", ".join(fmt_float(v) for v in arr.ravel())
    return f"static const float {name}{dims} = {{ {values} }};\n"


def c_int8_array(name: str, arr: np.ndarray, dims: str = "") -> str:
    arr = np.asarray(arr)
    if arr.ndim == 2:
        rows = ["    { " + ", ".join(f"{int(v):d}" for v in row) + " }" for row in arr]
        return f"static const int8_t {name}{dims} = {{\n" + ",\n".join(rows) + "\n};\n"
    values = ", ".join(f"{int(v):d}" for v in arr.ravel())
    return f"static const int8_t {name}{dims} = {{ {values} }};\n"


def c_scalar(name: str, value: float) -> str:
    return f"static const float {name} = {{ {float(value):.9e}f }};\n"


def quantize_symmetric(values: np.ndarray) -> tuple[np.ndarray, float]:
    """Symmetric per-tensor int8 quantization: float = int8 * scale."""
    maximum = max(float(np.max(np.abs(values))), 1e-10)
    scale = maximum / 127.0
    quantized = np.clip(np.round(values / scale), -127.0, 127.0).astype(np.int8)
    return quantized, scale


def label_block(names: list[str], unused_attribute: bool) -> str:
    labels = ", ".join(f'"{n}"' for n in names)
    attribute = " __attribute__((unused))" if unused_attribute else ""
    return f"static const char *LABEL_NAMES[NUM_CLASSES]{attribute} = {{ {labels} }};\n"


def float_header(model: np.lib.npyio.NpzFile, names: list[str]) -> str:
    feature_dim = int(model["W1"].shape[0])
    hidden_dim = int(model["W1"].shape[1])
    num_classes = len(names)
    text = (
        "#ifndef MODEL_WEIGHTS_H\n"
        "#define MODEL_WEIGHTS_H\n"
        "\n"
        f"#define FEATURE_DIM {feature_dim}\n"
        f"#define HIDDEN_DIM {hidden_dim}\n"
        f"#define NUM_CLASSES {num_classes}\n"
        "\n"
    )
    text += c_float_array("W1", model["W1"], "[FEATURE_DIM][HIDDEN_DIM]")
    text += c_float_array("B1", model["B1"], "[HIDDEN_DIM]")
    text += c_float_array("W2", model["W2"], "[HIDDEN_DIM][NUM_CLASSES]")
    text += c_float_array("B2", model["B2"], "[NUM_CLASSES]")
    text += c_float_array("FEATURE_MEAN", model["FEATURE_MEAN"], "[FEATURE_DIM]")
    text += c_float_array("FEATURE_STD", model["FEATURE_STD"], "[FEATURE_DIM]")
    text += label_block(names, unused_attribute=True)
    text += "\n#endif\n"
    return text


def int8_header(
    model: np.lib.npyio.NpzFile,
    names: list[str],
    guard: str,
    unused_attribute: bool = True,
) -> str:
    feature_dim = int(model["W1"].shape[0])
    hidden_dim = int(model["W1"].shape[1])
    num_classes = len(names)
    text = (
        f"#ifndef {guard}\n"
        f"#define {guard}\n"
        "\n"
        "#include <stdint.h>\n"
        "\n"
        f"#define FEATURE_DIM {feature_dim}\n"
        f"#define HIDDEN_DIM {hidden_dim}\n"
        f"#define NUM_CLASSES {num_classes}\n"
        "\n"
    )
    for name, dims in (
        ("W1", "[FEATURE_DIM][HIDDEN_DIM]"),
        ("B1", "[HIDDEN_DIM]"),
        ("W2", "[HIDDEN_DIM][NUM_CLASSES]"),
        ("B2", "[NUM_CLASSES]"),
    ):
        quantized, scale = quantize_symmetric(model[name])
        text += c_int8_array(f"{name}_INT8", quantized, dims)
        text += c_scalar(f"{name}_SCALE", scale)
    text += c_float_array("FEATURE_MEAN", model["FEATURE_MEAN"], "[FEATURE_DIM]")
    text += c_float_array("FEATURE_STD", model["FEATURE_STD"], "[FEATURE_DIM]")
    text += label_block(names, unused_attribute=unused_attribute)
    text += "\n#endif\n"
    return text


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="model/tiny_mlp_mfcc92_robust_simulator.npz")
    parser.add_argument("--float-out", default="app/audiodetect/model_weights.h")
    parser.add_argument("--int8-out", default="app/audiodetect/model_weights_int8.h")
    parser.add_argument("--simulator-int8-out", default="simulator/generated/model_weights_int8.h")
    args = parser.parse_args()

    model_path = Path(args.model)
    if not model_path.is_absolute():
        model_path = REPO_ROOT / model_path
    model = np.load(model_path, allow_pickle=True)
    names = [str(x) for x in model["LABEL_NAMES"].tolist()]

    outputs = {
        args.float_out: float_header(model, names),
        args.int8_out: int8_header(model, names, "MODEL_WEIGHTS_INT8_H"),
        args.simulator_int8_out: int8_header(
            model, names, "SIMULATOR_MODEL_WEIGHTS_INT8_H", unused_attribute=False
        ),
    }
    for relative, text in outputs.items():
        out = Path(relative)
        if not out.is_absolute():
            out = REPO_ROOT / out
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")
        print(f"wrote {out.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
