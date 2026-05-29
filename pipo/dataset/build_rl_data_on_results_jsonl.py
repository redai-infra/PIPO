from __future__ import annotations

import argparse
import glob
import json
import logging
import os
import sys
import tqdm
from pathlib import Path
from typing import Any, Callable

os.environ["HF_HUB_OFFLINE"] = "1"

sys.path.append(os.getcwd())


def _merge_no_proxy_localhosts() -> None:
    for key in ("NO_PROXY", "no_proxy"):
        cur = os.environ.get(key, "")
        parts = [p.strip() for p in cur.split(",") if p.strip()]
        for h in ("localhost", "127.0.0.1", "::1"):
            if h not in parts:
                parts.append(h)
        if parts:
            os.environ[key] = ",".join(parts)


_merge_no_proxy_localhosts()

logger = logging.getLogger(__name__)

# Hard-coded known file names -------------------------------------------------
CODEFORCES_STATS_GLOB = "codeforces_rl-exec-statistics-*.json"


def infer_dataset_name(jsonl_path: Path) -> str:
    """Infer dataset name from ``<dataset>-results.jsonl``."""
    stem = jsonl_path.stem  # e.g. "dapo_math_rl-results"
    if stem.endswith("-results"):
        return stem[: -len("-results")]
    raise ValueError(
        f"Cannot infer dataset name from {jsonl_path.name!r}; "
        f"expected pattern <dataset>-results.jsonl"
    )


def user_content_from_src_item(src_item: dict[str, Any]) -> str:
    p = src_item.get("prompt")
    if isinstance(p, str):
        return p
    if isinstance(p, list) and p and isinstance(p[0], dict):
        return str(p[0].get("value", ""))
    raise ValueError(f"Unsupported prompt layout: {type(p)!r}")


def load_jsonl_by_micro_index(path: Path) -> dict[int, dict[str, Any]]:
    """Load JSONL records indexed by micro_index."""
    by_mi: dict[int, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            si = record.get("src_item", {}) or {}
            try:
                mi = int(si.get("micro_index", -1))
            except (TypeError, ValueError):
                continue
            if mi < 0:
                continue
            by_mi[mi] = record
    return by_mi


# ---------------------------------------------------------------------------
# Codeforces stats loading
# ---------------------------------------------------------------------------

def acc_map_from_codeforces_stats(stats: dict[str, Any]) -> dict[int, list[float]]:
    mis = stats.get("micro_indices_sorted")
    acc_block = stats.get("all_pass1_sorted")
    if not isinstance(mis, list) or not isinstance(acc_block, list):
        raise KeyError(
            "Codeforces stats must include micro_indices_sorted and all_pass1_sorted "
            "(re-run pipo/eval/eval_openr1_codeforces.py)."
        )
    if len(mis) != len(acc_block):
        raise ValueError(
            f"stats length mismatch: micro_indices_sorted={len(mis)} "
            f"all_pass1_sorted={len(acc_block)}"
        )
    return {int(mi): [float(x) for x in row] for mi, row in zip(mis, acc_block)}


def resolve_codeforces_stats_path(jsonl_path: Path) -> Path:
    """Auto-discover the newest ``codeforces_rl-exec-statistics-*.json`` next to *jsonl_path*."""
    parent = jsonl_path.parent
    pat = str(parent / CODEFORCES_STATS_GLOB)
    matches = glob.glob(pat)
    if not matches:
        raise FileNotFoundError(
            f"No {CODEFORCES_STATS_GLOB} found under {parent}"
        )
    return Path(max(matches, key=lambda p: Path(p).stat().st_mtime))


# ---------------------------------------------------------------------------
# Per-dataset builders
# ---------------------------------------------------------------------------

def compute_average_accuracy(record: dict[str, Any]) -> float | None:
    """Compute average accuracy across all completions for a record."""
    accuracies = record.get("accuracies")
    if not isinstance(accuracies, list) or len(accuracies) == 0:
        return None
    try:
        return sum(float(a) for a in accuracies) / len(accuracies)
    except (TypeError, ValueError):
        return None


def build_rows_dapo_math(
    jsonl_path: Path,
    acc_threshold: float,
) -> list[dict[str, Any]]:
    """Build RL rows for dapo_math (accuracies come from JSONL directly)."""
    source = "dapo_math"
    by_mi = load_jsonl_by_micro_index(jsonl_path)
    rows: list[dict[str, Any]] = []
    n_bad = 0
    n_below_threshold = 0

    for micro_index in tqdm.tqdm(sorted(by_mi), desc=source):
        record = by_mi[micro_index]
        si = record.get("src_item", {}) or {}

        avg_acc = compute_average_accuracy(record)
        if avg_acc is None:
            n_bad += 1
            continue

        if avg_acc < acc_threshold:
            n_below_threshold += 1
            continue

        try:
            user_prompt = user_content_from_src_item(si)
        except ValueError:
            n_bad += 1
            continue

        rows.append({
            "messages": [
                {"role": "user", "content": user_prompt},
            ],
            "source": source,
            "source_index": micro_index,
        })

    if n_bad:
        logger.warning("%s: skipped %d problems (bad/missing fields)", source, n_bad)
    logger.info(
        "%s: %d RL rows from %d questions (%d below threshold %.2f)",
        source,
        len(rows),
        len(by_mi),
        n_below_threshold,
        acc_threshold,
    )
    return rows


def build_rows_codeforces(
    jsonl_path: Path,
    stats: dict[str, Any],
    acc_threshold: float,
) -> list[dict[str, Any]]:
    """Build RL rows for codeforces (accuracies come from exec stats file)."""
    source = "codeforces"
    acc_by_mi = acc_map_from_codeforces_stats(stats)
    by_mi = load_jsonl_by_micro_index(jsonl_path)
    rows: list[dict[str, Any]] = []
    n_missing = 0
    n_mismatch = 0
    n_bad = 0
    n_below_threshold = 0

    for micro_index in tqdm.tqdm(sorted(by_mi), desc=source):
        record = by_mi[micro_index]
        si = record.get("src_item", {}) or {}
        acc_list = acc_by_mi.get(micro_index)
        if acc_list is None:
            n_missing += 1
            continue

        completions = record.get("completion_texts") or []
        if not isinstance(completions, list):
            completions = []
        if len(acc_list) != len(completions):
            n_mismatch += 1
            continue

        if not acc_list:
            n_bad += 1
            continue

        avg_acc = sum(acc_list) / len(acc_list)
        if avg_acc < acc_threshold:
            n_below_threshold += 1
            continue

        try:
            user_prompt = user_content_from_src_item(si)
        except ValueError:
            n_bad += 1
            continue

        rows.append({
            "messages": [
                {"role": "user", "content": user_prompt},
            ],
            "source": source,
            "source_index": micro_index,
        })

    if n_missing:
        logger.warning("%s: %d problems missing from exec stats", source, n_missing)
    if n_mismatch:
        logger.warning("%s: %d problems skipped (field length mismatch)", source, n_mismatch)
    if n_bad:
        logger.warning("%s: skipped %d problems (bad/missing fields)", source, n_bad)
    logger.info(
        "%s: %d RL rows from %d questions (%d below threshold %.2f)",
        source,
        len(rows),
        len(by_mi),
        n_below_threshold,
        acc_threshold,
    )
    return rows


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

RlBuilder = Callable[..., list[dict[str, Any]]]

RL_SOURCE_REGISTRY: dict[str, dict[str, Any]] = {
    "dapo_math_rl": {
        "builder": build_rows_dapo_math,
        "needs_stats": False,
    },
    "codeforces_rl": {
        "builder": build_rows_codeforces,
        "needs_stats": True,
    },
}


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def write_rl_jsonl(rows: list[dict[str, Any]], path: Path) -> int:
    """Write one RL record per line."""
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    return n


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build RL JSONL from result JSONL files by average accuracy threshold.",
    )
    parser.add_argument(
        "jsonl_paths",
        type=Path,
        nargs="+",
        help=(
            "One or more *_rl-results.jsonl files. The dataset name is inferred "
            "from the filename."
        ),
    )
    parser.add_argument(
        "--acc_threshold",
        type=float,
        default=0.5,
        help="Minimum average accuracy for a question to be included (default: 0.5)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output JSONL path (default: data/rl_{acc_threshold}.jsonl)",
    )
    args = parser.parse_args()

    if args.output is None:
        # Format threshold nicely, e.g. 0.5 -> "0.5", 0.75 -> "0.75"
        threshold_str = f"{args.acc_threshold:g}"
        args.output = Path(f"data/rl_{threshold_str}.jsonl")

    logging.basicConfig(
        format="%(asctime)s — %(levelname)s — %(message)s",
        level=logging.INFO,
    )

    rows: list[dict[str, Any]] = []

    for jsonl_path in args.jsonl_paths:
        jsonl_path = jsonl_path.resolve()
        if not jsonl_path.is_file():
            logger.warning("Skipping — file not found: %s", jsonl_path)
            continue

        dataset_name = infer_dataset_name(jsonl_path)
        spec = RL_SOURCE_REGISTRY.get(dataset_name)
        if spec is None:
            logger.warning(
                "Skipping — unknown dataset %r (inferred from %s). "
                "Known: %s",
                dataset_name,
                jsonl_path.name,
                ", ".join(RL_SOURCE_REGISTRY),
            )
            continue

        logger.info(
            "Processing %s  [dataset=%s, threshold=%.2f]",
            jsonl_path, dataset_name, args.acc_threshold,
        )
        builder: RlBuilder = spec["builder"]

        if spec["needs_stats"]:
            stats_path = resolve_codeforces_stats_path(jsonl_path)
            cf_stats = json.loads(stats_path.read_text(encoding="utf-8"))
            logger.info("Loaded codeforces stats from %s", stats_path)
            part = builder(jsonl_path, cf_stats, args.acc_threshold)
        else:
            part = builder(jsonl_path, args.acc_threshold)

        rows.extend(part)

    if not rows:
        raise RuntimeError("No rows built — check input paths.")

    logger.info("Total RL training samples: %d", len(rows))

    n_written = write_rl_jsonl(rows, args.output)
    logger.info("Written RL JSONL: %s (%d lines)", args.output, n_written)


if __name__ == "__main__":
    main()
