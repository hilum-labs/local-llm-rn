#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PKG_DIR="$(dirname "$SCRIPT_DIR")"
TMP_DIR="$(mktemp -d "${TMPDIR:-/tmp}/local-llm-rn-vendoring.XXXXXX")"
STAGE_DIR="$TMP_DIR/pkg"

cleanup() {
  rm -rf "$TMP_DIR"
}
trap cleanup EXIT

if ! command -v rsync >/dev/null 2>&1; then
  echo "ERROR: rsync is required for vendoring verification"
  exit 1
fi

echo "Staging local-llm-rn into $STAGE_DIR"
mkdir -p "$STAGE_DIR"
rsync -a \
  --exclude '.git' \
  --exclude 'node_modules' \
  --exclude 'android/.cxx' \
  --exclude 'android/build' \
  --exclude 'android/.gradle' \
  --exclude 'cpp' \
  "$PKG_DIR/" "$STAGE_DIR/"

assert_exists() {
  local path="$1"
  if [ ! -e "$path" ]; then
    echo "ERROR: missing required path: $path"
    exit 1
  fi
}

assert_not_exists() {
  local path="$1"
  if [ -e "$path" ]; then
    echo "ERROR: path should not exist: $path"
    exit 1
  fi
}

assert_file_contains() {
  local path="$1"
  local pattern="$2"
  if ! grep -F -q "$pattern" "$path"; then
    echo "ERROR: expected pattern '$pattern' in $path"
    exit 1
  fi
}

echo "Verifying prepare.sh layout"
(cd "$STAGE_DIR" && bash scripts/prepare.sh)

if [ ! -L "$STAGE_DIR/cpp" ]; then
  echo "ERROR: prepare.sh did not create cpp symlink"
  exit 1
fi

CPP_TARGET="$(readlink "$STAGE_DIR/cpp")"
if [ "$CPP_TARGET" != "vendor/hilum-local-llm-engine" ] && [ "$CPP_TARGET" != "$STAGE_DIR/vendor/hilum-local-llm-engine" ]; then
  echo "ERROR: unexpected cpp symlink target: $CPP_TARGET"
  exit 1
fi

assert_exists "$STAGE_DIR/cpp/hilum/hilum_llm.h"
assert_exists "$STAGE_DIR/cpp/tools/mtmd/mtmd.h"

echo "Verifying prepack.sh layout"
rm -f "$STAGE_DIR/cpp"
(cd "$STAGE_DIR" && bash scripts/prepack.sh)

if [ -L "$STAGE_DIR/cpp" ]; then
  echo "ERROR: prepack.sh left cpp as symlink"
  exit 1
fi

assert_exists "$STAGE_DIR/cpp/CMakeLists.txt"
assert_exists "$STAGE_DIR/cpp/LICENSE"
assert_exists "$STAGE_DIR/cpp/src"
assert_exists "$STAGE_DIR/cpp/include"
assert_exists "$STAGE_DIR/cpp/cmake"
assert_exists "$STAGE_DIR/cpp/licenses"
assert_exists "$STAGE_DIR/cpp/ggml"
assert_exists "$STAGE_DIR/cpp/common"
assert_exists "$STAGE_DIR/cpp/common/build-info.cpp"
assert_exists "$STAGE_DIR/cpp/hilum"
assert_exists "$STAGE_DIR/cpp/mtmd"
assert_exists "$STAGE_DIR/cpp/vendor/cpp-httplib"
assert_exists "$STAGE_DIR/cpp/vendor/nlohmann"
assert_exists "$STAGE_DIR/cpp/vendor/stb"
assert_exists "$STAGE_DIR/cpp/vendor/miniaudio"
assert_exists "$STAGE_DIR/cpp/mtmd/mtmd.h"
assert_exists "$STAGE_DIR/cpp/mtmd/mtmd-helper.h"
assert_file_contains "$STAGE_DIR/cpp/hilum/hilum_llm.cpp" "#include \"mtmd.h\""
assert_file_contains "$STAGE_DIR/cpp/hilum/hilum_llm.cpp" "#include \"mtmd-helper.h\""

assert_not_exists "$STAGE_DIR/cpp/ggml/src/ggml-cuda"
assert_not_exists "$STAGE_DIR/cpp/ggml/src/ggml-opencl"
assert_not_exists "$STAGE_DIR/cpp/ggml/src/ggml-hip"
assert_not_exists "$STAGE_DIR/cpp/mtmd/mtmd-cli.cpp"

echo "Vendoring verification passed"
