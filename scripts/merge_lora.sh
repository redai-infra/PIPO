#!/usr/bin/env bash
set -euo pipefail

PIPO_DIR="$(dirname "$0")"
export PYTHONPATH="${PIPO_DIR}:${PIPO_DIR}/third_party/ms-swift${PYTHONPATH:+:${PYTHONPATH}}"

adapters="$1"
COMPRESSOR_TYPE="${2:-mlp}" HF_HUB_OFFLINE=1 swift export \
  --adapters "$adapters" \
  --merge_lora true \
  --use_hf true \
  --model_type qwen3_5_mtp \
  --external_plugins "${PIPO_DIR}/pipo/trainer/swift_plugin.py"
