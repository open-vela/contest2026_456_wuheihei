#!/usr/bin/env sh
set -eu

repo_dir=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$repo_dir"

required_files="
README.md
contest2026_456_wuheihei.xml
app/audiodetect/audiodetect_main.c
app/audiodetect/audiodetect_ui.c
app/audiodetect/audio_classifier.c
app/audiodetect/model_weights.h
app/audiodetect/model_weights_int8.h
simulator/model_infer_int8.c
simulator/generated/model_weights_int8.h
training/export_to_c.py
training/train_edge_mfcc.py
training/extract_features_mfcc.py
model/tiny_mlp_mfcc92_robust_simulator.npz
board/r528s3-gemini-s1/configs/nsh/defconfig
board/r528s3-gemini-s1/src/etc/init.d/rcS.nsh
results/python_cv/metrics.json
"

for file in $required_files; do
  test -s "$file" || { echo "missing or empty: $file" >&2; exit 1; }
done

if grep -R "your-github-login\|hello_app\|hello_quickapp" \
  README.md contest2026_456_wuheihei.xml app board training model results 2>/dev/null; then
  echo "placeholder content found" >&2
  exit 1
fi

expected=c96333aaae66d324b8b8cec48ffb0ec981ac4e975d3cf9a7d4a2fdba4307c97e
if command -v sha256sum >/dev/null 2>&1; then
  actual=$(sha256sum model/tiny_mlp_mfcc92_robust_simulator.npz | awk '{print $1}')
else
  actual=$(shasum -a 256 model/tiny_mlp_mfcc92_robust_simulator.npz | awk '{print $1}')
fi
test "$actual" = "$expected" || {
  echo "model checksum mismatch: $actual" >&2
  exit 1
}

python3 -m json.tool results/python_cv/metrics.json >/dev/null

# The three generated weight headers must describe the same network.
for header in app/audiodetect/model_weights.h \
              app/audiodetect/model_weights_int8.h \
              simulator/generated/model_weights_int8.h; do
  test "$(grep -c '#define FEATURE_DIM 92' "$header")" = "1" || {
    echo "unexpected FEATURE_DIM in $header" >&2; exit 1; }
  test "$(grep -c '#define HIDDEN_DIM 64' "$header")" = "1" || {
    echo "unexpected HIDDEN_DIM in $header" >&2; exit 1; }
  test "$(grep -c '#define NUM_CLASSES 5' "$header")" = "1" || {
    echo "unexpected NUM_CLASSES in $header" >&2; exit 1; }
done

echo "Submission source tree is complete."
echo "Model SHA256: $actual"
