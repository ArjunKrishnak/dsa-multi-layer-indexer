#!/bin/bash
# Clone SGLang v0.5.16 and apply the MLI overlay (pure Python, no kernel rebuild).
#   scripts/setup/prepare_sglang.sh /path/to/sglang-mli
set -euo pipefail
DEST="${1:?usage: prepare_sglang.sh <dest-dir>}"
PATCH="$(cd "$(dirname "$0")/../.." && pwd)/sglang_overlay/mli-sglang-v0.5.16.patch"
git clone --depth 1 --branch v0.5.16 https://github.com/sgl-project/sglang.git "$DEST"
git -C "$DEST" apply --check "$PATCH"
git -C "$DEST" apply "$PATCH"
echo "MLI overlay applied to $DEST"
