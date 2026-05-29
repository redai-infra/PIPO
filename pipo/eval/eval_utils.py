import json
from pathlib import Path

import numpy as np


# ---- benchmark groups ----
RULE_DATASETS = ("aime2025", "gpqa_diamond", "lb2")
# IFBENCH_DATASET = "ifbench"  # OOD dataset, we do not test it
LCB_DATASET = "livecodebench"

# Benchmarks listed in the unified excel (in column order on the "overall" sheet).
EXCEL_BENCHMARKS = ("aime2025", "gpqa_diamond", LCB_DATASET, "lb2")

# Context-length buckets shared across benchmarks.
LENGTH_LIMITS = {
    "2K": 2048,
    "4K": 4096,
    "8K": 8192,
    "16K": 16384,
    "32K": 32768,
}
LENGTH_ORDER = list(LENGTH_LIMITS.keys())


def safe_mean(xs) -> float:
    return float(np.mean(xs)) if len(xs) else 0.0


# ---- jsonl io ----

def load_jsonl_dedup(path: Path) -> list[dict]:
    """Load a results jsonl, dedup by micro_index (last wins), sort by micro_index.

    Also normalizes legacy records so the rest of the pipeline sees the unified
    schema (see :func:`normalize_record`).
    """
    seen: dict[int, dict] = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            mi = r["src_item"]["micro_index"]
            seen[mi] = r
    records = sorted(seen.values(), key=lambda r: r["src_item"]["micro_index"])
    return [normalize_record(r) for r in records]


def write_jsonl(path: Path, records: list[dict]) -> None:
    """Overwrite jsonl atomically with ``records`` in order."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    tmp.replace(path)


# ---- record helpers ----

def normalize_record(r: dict) -> dict:
    """Force a record to follow the unified key set.

    Also coerces legacy values (accuracies=[None,...], pass@k=[None,...],
    extracted_answers=[]) to the new ``None`` sentinel so downstream checks
    can simply test ``r["accuracies"] is None``.
    """
    r.setdefault("extracted_answers", None)
    r.setdefault("accuracies", None)
    r.setdefault("pass@k", None)

    if r.get("n_pads") is None:
        r["n_pads"] = [0] * len(r.get("n_tokens", []))

    accs = r.get("accuracies")
    if isinstance(accs, list) and (len(accs) == 0 or any(a is None for a in accs)):
        r["accuracies"] = None
    pk = r.get("pass@k")
    if isinstance(pk, list):
        r["pass@k"] = None
    ea = r.get("extracted_answers")
    if isinstance(ea, list) and len(ea) == 0:
        r["extracted_answers"] = None
    return r


# ---- stats ----

def compute_statistics(records: list[dict]) -> dict:
    """Compute the unified statistics dict from a list of unified records."""
    if not records:
        raise ValueError("empty records")

    all_n_tokens = [t for r in records for t in r["n_tokens"]]
    all_n_pads = [p for r in records for p in r["n_pads"]]

    multi: dict = {}
    for limit_name, limit_tokens in LENGTH_LIMITS.items():
        sample_pass: list[float] = []
        question_pass: list[float] = []
        finished_num = 0

        for r in records:
            accs = r["accuracies"] or []
            n_toks = r["n_tokens"]
            fins = r["finished"]
            q_pass = False
            for acc, n_tok, fin in zip(accs, n_toks, fins):
                within = n_tok <= limit_tokens
                eff_acc = acc if within else 0.0
                sample_pass.append(eff_acc)
                if within and fin:
                    finished_num += 1
                if eff_acc > 0:
                    q_pass = True
            question_pass.append(1.0 if q_pass else 0.0)

        multi[limit_name] = {
            "finished_num": finished_num,
            "pass@1": safe_mean(sample_pass),
            "pass@k": safe_mean(question_pass),
        }

    return {
        "question_num": len(records),
        "avg_token_length-all": safe_mean(all_n_tokens),
        "avg_pad_length-all": safe_mean(all_n_pads),
        "multi_length_stats": multi,
    }


def write_statistics_json(exp_dir: Path, dataset: str, stats: dict) -> Path:
    out = exp_dir / f"{dataset}-statistics.json"
    out.write_text(json.dumps(stats, indent=2, ensure_ascii=False))
    return out


# ---- lcb conversion (jsonl <-> lcb evaluator io) ----

import re

_PY_CODE_RE = re.compile(r"```python\n(.*?)```", re.DOTALL)


def extract_python_code(text: str) -> str:
    matches = _PY_CODE_RE.findall(text or "")
    return matches[-1] if matches else ""


def convert_jsonl_to_lcb_input(jsonl_path: Path) -> Path:
    """Build ``{stem}_converted.json`` (lcb custom_evaluator input)."""
    records = load_jsonl_dedup(jsonl_path)
    items = []
    for r in records:
        items.append({
            "question_id": r["src_item"]["metadata"]["question_id"],
            "code_list": [extract_python_code(c) for c in r["completion_texts"]],
        })
    items.sort(key=lambda x: x["question_id"])
    out = jsonl_path.parent / f"{jsonl_path.stem}_converted.json"
    out.write_text(json.dumps(items, ensure_ascii=False, indent=2))
    return out


def merge_lcb_eval_into_jsonl(jsonl_path: Path, eval_all_path: Path) -> None:
    """Write per-sample ``accuracies`` + ``pass@k`` back into the jsonl in place.

    Maps records by ``src_item.metadata.question_id`` <-> lcb_eval ``question_id``.
    """
    eval_data = json.loads(eval_all_path.read_text())
    qid2graded: dict = {}
    for e in eval_data:
        qid2graded[str(e["question_id"])] = e["graded_list"]

    records = load_jsonl_dedup(jsonl_path)
    n_missing = 0
    for r in records:
        qid = str(r["src_item"]["metadata"]["question_id"])
        graded = qid2graded.get(qid)
        if graded is None:
            n_missing += 1
            r["accuracies"] = [0.0] * len(r["completion_texts"])
            r["pass@k"] = 0.0
            continue
        accs = [1.0 if g else 0.0 for g in graded]
        r["accuracies"] = accs
        r["pass@k"] = max(accs) if accs else 0.0
        r["extracted_answers"] = None
        normalize_record(r)
    write_jsonl(jsonl_path, records)
    if n_missing:
        print(f"[lcb-merge] warning: {n_missing} records missing in eval_all")
    print(f"[lcb-merge] updated {len(records)} records in {jsonl_path}")
