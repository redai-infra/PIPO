"""Stage 1: rule-based re-evaluation (aime2025, gpqa_diamond, lb2).

For each ``{dataset}-results.jsonl`` found under ``exp_dir``, re-run the
rule-based [`evaluator`](pipo/eval/evaluator.py) on every completion and
write the per-sample ``accuracies`` / ``extracted_answers`` / ``pass@k`` back
into the same file (in place, deduplicated, sorted by ``micro_index``).
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.append(os.getcwd())


from pipo.eval import evaluator
from pipo.eval.eval_utils import (
    RULE_DATASETS,
    load_jsonl_dedup,
    normalize_record,
    write_jsonl,
)


def reeval_record(r: dict) -> dict:
    dataset = r["src_item"]["dataset"]
    ev = evaluator.evaluator_map[dataset]
    accs: list[float] = []
    extracted: list[str] = []
    for text in r["completion_texts"]:
        try:
            ok, ex = ev(text, r["src_item"]["final_answer"])
            accs.append(1.0 if ok else 0.0)
            extracted.append(ex)
        except Exception as e:
            accs.append(0.0)
            extracted.append(f"[eval-error] {type(e).__name__}: {e}")
    r["accuracies"] = accs
    r["extracted_answers"] = extracted
    r["pass@k"] = max(accs) if accs else 0.0
    return normalize_record(r)


def process_dataset(exp_dir: Path, dataset: str) -> None:
    path = exp_dir / f"{dataset}-results.jsonl"
    if not path.exists():
        print(f"[eval_1] skip {dataset}: {path} not found")
        return
    records = load_jsonl_dedup(path)
    if not records:
        print(f"[eval_1] skip {dataset}: empty jsonl")
        return
    for r in records:
        reeval_record(r)
    write_jsonl(path, records)
    n_correct = sum(int(r["pass@k"] > 0) for r in records)
    print(
        f"[eval_1] {dataset}: {len(records)} questions, "
        f"pass@k>0 in {n_correct} -> wrote {path}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Rule-based eval (aime/gpqa/lb2)")
    parser.add_argument("exp_dir_name", type=str)
    parser.add_argument(
        "--datasets",
        type=str,
        default=",".join(RULE_DATASETS),
        help="comma-separated subset of rule-based datasets to eval",
    )
    args = parser.parse_args()

    exp_dir = Path(args.exp_dir_name)
    if not exp_dir.is_dir():
        print(f"[eval_1] error: {exp_dir} is not a directory")
        sys.exit(1)

    for ds in args.datasets.split(","):
        ds = ds.strip()
        if not ds:
            continue
        if ds not in evaluator.evaluator_map:
            print(f"[eval_1] skip {ds}: no rule-based evaluator registered")
            continue
        process_dataset(exp_dir, ds)


if __name__ == "__main__":
    main()
