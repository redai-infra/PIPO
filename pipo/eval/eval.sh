#!/usr/bin/env bash
set -euo pipefail

if [ $# -lt 1 ]; then
  echo "Usage: $0 <exp_dir_name>" >&2
  exit 1
fi

EXP_DIR="$1"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

if [ ! -d "${EXP_DIR}" ]; then
  echo "[eval] error: ${EXP_DIR} is not a directory" >&2
  exit 1
fi

echo "=== [eval] exp_dir = ${EXP_DIR} ==="

echo "--- [eval] stage 1: rule-based ---"
python pipo/eval/eval_1_rule.py "${EXP_DIR}"

echo "--- [eval] stage 2: livecodebench ---"
bash pipo/eval/eval_2_lcb.sh "${EXP_DIR}"

echo "--- [eval] stage 3: export stats + excel ---"
python pipo/eval/eval_3_export_to_excel.py "${EXP_DIR}"

echo "=== [eval] done: ${EXP_DIR}/stats.xlsx ==="
