#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PKG_DIR="$(dirname "$SCRIPT_DIR")"
ENGINE_DIR="$PKG_DIR/vendor/hilum-local-llm-engine"

# In a dev checkout, prefer the symlink so the native build sees the full engine tree.
# Skip if cpp/ is a real directory with engine (prepack output) — don't replace before npm pack.
if [ -d "$PKG_DIR/cpp/hilum" ] && [ ! -L "$PKG_DIR/cpp" ]; then
  exit 0
fi

if [ -d "$ENGINE_DIR" ]; then
  if [ -L "$PKG_DIR/cpp" ]; then
    TARGET="$(readlink "$PKG_DIR/cpp")"
    if [ "$TARGET" = "vendor/hilum-local-llm-engine" ] || [ "$TARGET" = "$ENGINE_DIR" ]; then
      exit 0
    fi
  fi

  rm -rf "$PKG_DIR/cpp"
  ln -sf "vendor/hilum-local-llm-engine" "$PKG_DIR/cpp"
  echo "Symlinked cpp/ -> vendor/hilum-local-llm-engine"
fi
