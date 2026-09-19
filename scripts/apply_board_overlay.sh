#!/usr/bin/env sh
set -eu

if [ "$#" -ne 1 ]; then
  echo "usage: $0 OPENVELA_WORKSPACE_ROOT" >&2
  exit 2
fi

workspace=$1
team_dir=$workspace/contest2026_456_wuheihei
board_dir=$workspace/vendor/allwinnertech/boards/r528/r528s3-gemini-s1

test -d "$workspace/nuttx"
test -d "$board_dir/configs/nsh"
test -f "$team_dir/board/r528s3-gemini-s1/configs/nsh/defconfig"
test -f "$team_dir/board/r528s3-gemini-s1/src/etc/init.d/rcS.nsh"

cp "$team_dir/board/r528s3-gemini-s1/configs/nsh/defconfig" \
  "$board_dir/configs/nsh/defconfig"
cp "$team_dir/board/r528s3-gemini-s1/src/etc/init.d/rcS.nsh" \
  "$board_dir/src/etc/init.d/rcS.nsh"

echo "Applied Audio Sentinel v10 board overlay to: $board_dir"

