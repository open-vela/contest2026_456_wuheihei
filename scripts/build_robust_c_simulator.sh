#!/usr/bin/env sh
set -eu

script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
repo_dir=$(CDPATH= cd -- "$script_dir/.." && pwd)
app_dir=${APP_DIR:-$repo_dir/app/audiodetect}
out_dir=${OUT_DIR:-$repo_dir/out/robust-simulator}

if [ ! -f "$repo_dir/simulator/generated/model_weights_int8.h" ]; then
  echo "missing simulator/generated/model_weights_int8.h; run export_simulator_int8.py first" >&2
  exit 2
fi

mkdir -p "$out_dir"
cc -std=c11 -O2 -Wall -Wextra \
  -I "$repo_dir/simulator" -I "$app_dir" \
  "$repo_dir/scripts/board_simulator_harness.c" \
  "$app_dir/audio_classifier.c" \
  "$app_dir/wav_reader.c" \
  "$app_dir/feature_extractor.c" \
  "$app_dir/keyword_detect.c" \
  "$repo_dir/simulator/model_infer_int8.c" \
  -lm -o "$out_dir/audiodetect_robust_sim"

echo "Built simulator-only binary: $out_dir/audiodetect_robust_sim"
echo "OpenVela board sources and model headers were not modified."
