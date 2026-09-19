#!/usr/bin/env sh
set -eu

repo_dir=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
app_dir=$repo_dir/app/audiodetect
out_dir=$repo_dir/out

mkdir -p "$out_dir"
cc -std=c11 -O2 -Wall -Wextra -I "$app_dir" \
  "$repo_dir/scripts/board_simulator_harness.c" \
  "$app_dir/audio_classifier.c" \
  "$app_dir/wav_reader.c" \
  "$app_dir/feature_extractor.c" \
  "$app_dir/keyword_detect.c" \
  "$app_dir/model_infer_int8.c" \
  -lm -o "$out_dir/audiodetect_v10_sim"

echo "Built: $out_dir/audiodetect_v10_sim"

