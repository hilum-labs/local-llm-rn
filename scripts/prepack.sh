#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PKG_DIR="$(dirname "$SCRIPT_DIR")"
ENGINE_DIR="$PKG_DIR/vendor/hilum-local-llm-engine"
ENGINE_REPO_URL="https://github.com/hilum-labs/hilum-local-llm-engine.git"

if [ ! -d "$ENGINE_DIR/src" ]; then
  if [ -d "$PKG_DIR/.git" ] || [ -f "$PKG_DIR/.git" ]; then
    echo "Engine source missing, attempting submodule initialization..."
    git -C "$PKG_DIR" submodule update --init --recursive
  fi
fi

if [ ! -d "$ENGINE_DIR/src" ]; then
  echo "Engine source still missing, cloning engine repository..."
  mkdir -p "$(dirname "$ENGINE_DIR")"
  rm -rf "$ENGINE_DIR"
  git clone --depth 1 "$ENGINE_REPO_URL" "$ENGINE_DIR"
fi

if [ ! -d "$ENGINE_DIR/src" ]; then
  echo "ERROR: Engine source not found at $ENGINE_DIR"
  echo "Run: git submodule update --init --recursive or clone $ENGINE_REPO_URL"
  exit 1
fi

# Remove symlink or stale copy (if symlink, remove it only — don't follow into vendor)
if [ -L "$PKG_DIR/cpp" ]; then
  rm "$PKG_DIR/cpp"
else
  rm -rf "$PKG_DIR/cpp"
fi
mkdir -p "$PKG_DIR/cpp"

echo "Copying engine source (~20 MB)..."

# Core inference engine
cp -r "$ENGINE_DIR/src"     "$PKG_DIR/cpp/src"
cp -r "$ENGINE_DIR/include" "$PKG_DIR/cpp/include"
cp -r "$ENGINE_DIR/cmake"   "$PKG_DIR/cpp/cmake"
cp -r "$ENGINE_DIR/licenses" "$PKG_DIR/cpp/licenses"

# Tensor library + Metal shaders
cp -r "$ENGINE_DIR/ggml"    "$PKG_DIR/cpp/ggml"

# Grammar support, chat templates, json-schema-to-grammar
cp -r "$ENGINE_DIR/common"  "$PKG_DIR/cpp/common"

# Vision / multimodal
cp -r "$ENGINE_DIR/tools/mtmd" "$PKG_DIR/cpp/mtmd"

# libhilum shared C API
cp -r "$ENGINE_DIR/hilum" "$PKG_DIR/cpp/hilum"

# Third-party headers (nlohmann/json, stb_image, miniaudio)
mkdir -p "$PKG_DIR/cpp/vendor"
cp -r "$ENGINE_DIR/vendor/cpp-httplib" "$PKG_DIR/cpp/vendor/cpp-httplib"
cp -r "$ENGINE_DIR/vendor/nlohmann" "$PKG_DIR/cpp/vendor/nlohmann"
cp -r "$ENGINE_DIR/vendor/stb"      "$PKG_DIR/cpp/vendor/stb"
cp -r "$ENGINE_DIR/vendor/miniaudio" "$PKG_DIR/cpp/vendor/miniaudio"

# Top-level CMakeLists.txt (needed by ggml's internal references)
cp "$ENGINE_DIR/CMakeLists.txt" "$PKG_DIR/cpp/CMakeLists.txt"
cp "$ENGINE_DIR/LICENSE" "$PKG_DIR/cpp/LICENSE"

# Generate build-info.cpp (normally produced by CMake)
cat > "$PKG_DIR/cpp/common/build-info.cpp" <<'BUILDINFO'
int LLAMA_BUILD_NUMBER = 0;
char const *LLAMA_COMMIT = "unknown";
char const *LLAMA_COMPILER = "Apple Clang";
char const *LLAMA_BUILD_TARGET = "ios";
BUILDINFO

# Strip backends that neither iOS nor Android need.
# Keep:
# - ggml-blas   for Apple Accelerate
# - ggml-metal  for iOS/macOS
# - ggml-vulkan for Android GPU
for backend in \
  ggml-cuda \
  ggml-sycl \
  ggml-cann \
  ggml-hip \
  ggml-kompute \
  ggml-opencl \
  ggml-webgpu \
  ggml-hexagon \
  ggml-musa \
  ggml-zdnn \
  ggml-zendnn \
  ggml-rpc \
  ggml-virtgpu
do
  rm -rf "$PKG_DIR/cpp/ggml/src/$backend"
done

# Strip non-build multimodal assets and tooling.
rm -rf "$PKG_DIR/cpp/mtmd/legacy-models"
rm -f "$PKG_DIR/cpp/mtmd/README.md"
rm -f "$PKG_DIR/cpp/mtmd/requirements.txt"
rm -f "$PKG_DIR/cpp/mtmd/tests.sh"
rm -f "$PKG_DIR/cpp/mtmd/test-1.jpeg"
rm -f "$PKG_DIR/cpp/mtmd/test-2.mp3"
rm -f "$PKG_DIR/cpp/mtmd/mtmd-cli.cpp"




# Strip Android build artifacts (npm pack may include them despite .npmignore)
rm -rf "$PKG_DIR/android/.cxx" "$PKG_DIR/android/build" "$PKG_DIR/android/.gradle"

FINAL_SIZE=$(du -sh "$PKG_DIR/cpp" | cut -f1)
FILE_COUNT=$(find "$PKG_DIR/cpp" -type f | wc -l | tr -d ' ')
echo "Packed cpp/ -- ${FINAL_SIZE}, ${FILE_COUNT} files"
