#!/usr/bin/env bash
# Stage 3: LiveCodeBench in-place evaluation.
#
# 1. Convert {exp_dir}/livecodebench-results.jsonl into the lcb_runner format
#    -> {exp_dir}/livecodebench-results_converted.json
# 2. Run lcb_runner.runner.custom_evaluator from inside ./third_party/LiveCodeBench
#    -> *_codegeneration_output_eval_all.json (contains graded_list per qid)
# 3. Merge graded_list back into the jsonl as per-sample accuracies + pass@k.
set -euo pipefail

if [ $# -lt 1 ]; then
  echo "Usage: $0 <exp_dir_name>" >&2
  exit 1
fi

EXP_DIR="$1"
JSONL="${EXP_DIR%/}/livecodebench-results.jsonl"

if [ ! -f "${JSONL}" ]; then
  echo "[eval_3] skip livecodebench: ${JSONL} not found"
  exit 0
fi

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "${REPO_ROOT}"

CONVERTED="${JSONL%.jsonl}_converted.json"
EVAL_ALL="${JSONL%.jsonl}_converted_codegeneration_output_eval_all.json"

echo "[eval_2] step 1/3: convert jsonl -> lcb input"
python -c "from pipo.eval.eval_utils import convert_jsonl_to_lcb_input; from pathlib import Path; convert_jsonl_to_lcb_input(Path('${JSONL}'))"

echo "[eval_2] step 2/3: run lcb custom_evaluator"
( cd ./third_party/LiveCodeBench \
  && python -m lcb_runner.runner.custom_evaluator \
       --custom_output_file "../../${CONVERTED}" \
       --release_version release_v6 \
       --start_date 2025-02-01 \
       --num_process_evaluate 32 \
       --trust_remote_code )

echo "[eval_2] step 3/3: merge graded_list back into jsonl"
python -c "from pipo.eval.eval_utils import merge_lcb_eval_into_jsonl; from pathlib import Path; merge_lcb_eval_into_jsonl(Path('${JSONL}'), Path('${EVAL_ALL}'))"

echo "[eval_2] done."
