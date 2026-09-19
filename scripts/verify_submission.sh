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
app/audiodetect/model_weights_int8.h
training/train_edge_mfcc.py
training/extract_features_mfcc.py
model/tiny_mlp_mfcc92_edge_robust.npz
board/r528s3-gemini-s1/configs/nsh/defconfig
board/r528s3-gemini-s1/src/etc/init.d/rcS.nsh
results/python_cv/metrics_extended.json
results/c_simulator/smoke_test.json
"

for file in $required_files; do
  test -s "$file" || { echo "missing or empty: $file" >&2; exit 1; }
done

if grep -R "your-github-login\|hello_app\|hello_quickapp" \
  README.md contest2026_456_wuheihei.xml app board training model results 2>/dev/null; then
  echo "placeholder content found" >&2
  exit 1
fi

expected=5382765b0aeb0c0aa9f96f7f78b26728acfcd20efd35ac7b045637889b50df62
if command -v sha256sum >/dev/null 2>&1; then
  actual=$(sha256sum model/tiny_mlp_mfcc92_edge_robust.npz | awk '{print $1}')
else
  actual=$(shasum -a 256 model/tiny_mlp_mfcc92_edge_robust.npz | awk '{print $1}')
fi
test "$actual" = "$expected" || {
  echo "model checksum mismatch: $actual" >&2
  exit 1
}

python3 -m json.tool results/python_cv/metrics_extended.json >/dev/null
python3 -m json.tool results/c_simulator/smoke_test.json >/dev/null

echo "Submission source tree is complete."
echo "Model SHA256: $actual"
