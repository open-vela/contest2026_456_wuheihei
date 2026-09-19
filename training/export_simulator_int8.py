"""Export a weight-only INT8 model for the host C simulator only.

This exporter deliberately has no OpenVela or board output path.  It keeps the
frozen board headers untouched while allowing the Python robustness experiment
to be exercised by the same C feature and inference implementation on a host.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def quantize(values: np.ndarray) -> tuple[np.ndarray, float]:
    maximum = max(float(np.max(np.abs(values))), 1e-10)
    scale = maximum / 127.0
    result = np.clip(np.round(values / scale), -127, 127).astype(np.int8)
    return result, scale


def int8_array(name: str, values: np.ndarray, dimensions: str) -> str:
    if values.ndim == 2:
        rows = ["    { " + ", ".join(str(int(value)) for value in row) + " }" for row in values]
        return f"static const int8_t {name}{dimensions} = {{\n" + ",\n".join(rows) + "\n};\n"
    body = ", ".join(str(int(value)) for value in values.ravel())
    return f"static const int8_t {name}{dimensions} = {{ {body} }};\n"


def float_array(name: str, values: np.ndarray, dimensions: str) -> str:
    body = ", ".join(f"{float(value):.9e}f" for value in values.ravel())
    return f"static const float {name}{dimensions} = {{ {body} }};\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", default="simulator/generated/model_weights_int8.h")
    parser.add_argument("--metadata-out", default="simulator/generated/model_metadata.json")
    args = parser.parse_args()

    model = np.load(args.model, allow_pickle=True)
    labels = [str(value) for value in model["LABEL_NAMES"].tolist()]
    feature_dim, hidden_dim = model["W1"].shape
    num_classes = model["W2"].shape[1]
    quantized = {}
    scales = {}
    for name in ("W1", "B1", "W2", "B2"):
        quantized[name], scales[name] = quantize(model[name])

    text = f"""#ifndef SIMULATOR_MODEL_WEIGHTS_INT8_H
#define SIMULATOR_MODEL_WEIGHTS_INT8_H

#include <stdint.h>

#define FEATURE_DIM {feature_dim}
#define HIDDEN_DIM {hidden_dim}
#define NUM_CLASSES {num_classes}

"""
    text += int8_array("W1_INT8", quantized["W1"], "[FEATURE_DIM][HIDDEN_DIM]")
    text += float_array("W1_SCALE", np.asarray([scales["W1"]]), "")
    text += int8_array("B1_INT8", quantized["B1"], "[HIDDEN_DIM]")
    text += float_array("B1_SCALE", np.asarray([scales["B1"]]), "")
    text += int8_array("W2_INT8", quantized["W2"], "[HIDDEN_DIM][NUM_CLASSES]")
    text += float_array("W2_SCALE", np.asarray([scales["W2"]]), "")
    text += int8_array("B2_INT8", quantized["B2"], "[NUM_CLASSES]")
    text += float_array("B2_SCALE", np.asarray([scales["B2"]]), "")
    text += float_array("FEATURE_MEAN", model["FEATURE_MEAN"], "[FEATURE_DIM]")
    text += float_array("FEATURE_STD", model["FEATURE_STD"], "[FEATURE_DIM]")
    text += "static const char *LABEL_NAMES[NUM_CLASSES] = { "
    text += ", ".join(json.dumps(label) for label in labels)
    text += " };\n\n#endif\n"

    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text, encoding="utf-8")

    parameter_count = sum(model[name].size for name in ("W1", "B1", "W2", "B2"))
    metadata = {
        "source_model": str(Path(args.model)),
        "architecture": f"{feature_dim}-{hidden_dim}-{num_classes}",
        "parameters": int(parameter_count),
        "quantization": "symmetric per-tensor weight-only INT8; float32 activations",
        "scope": "host C simulator only; OpenVela board headers are intentionally untouched",
    }
    metadata_output = Path(args.metadata_out)
    metadata_output.parent.mkdir(parents=True, exist_ok=True)
    metadata_output.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {output}")
    print(f"wrote {metadata_output}")


if __name__ == "__main__":
    main()
