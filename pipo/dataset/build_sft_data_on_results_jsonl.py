from __future__ import annotations

import argparse
import glob
import json
import logging
import os
import re
import sys
import tqdm
from collections import Counter
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
CODEFORCES_JSONL_NAME = "codeforces-results.jsonl"
CODEFORCES_STATS_GLOB = "codeforces-exec-statistics-*.json"

# Dataset name is inferred from the JSONL file name: <dataset>-results.jsonl
KNOWN_DATASETS = {"dapo_math", "codeforces"}


def infer_dataset_name(jsonl_path: Path) -> str:
    """Infer dataset name from ``<dataset>-results.jsonl``."""
    stem = jsonl_path.stem  # e.g. "dapo_math-results"
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


def messages_from_chat(
    user_prompt: str,
    completion: str,
) -> dict[str, Any]:
    """Build messages dict for ms-swift SFT format."""
    return {
        "messages": [
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": completion},
        ]
    }


def has_repetition(
    text: str,
    min_line_repeat: int = 3,
    ngram_size: int = 10,
    ngram_threshold: float = 0.35,
) -> bool:
    """Detect repetitive patterns in generated text."""
    if not text or len(text) < 100:
        return False

    lines = text.split("\n")
    line_counts = Counter(line.strip() for line in lines if line.strip())
    for line, count in line_counts.items():
        if len(line) > 5 and count >= min_line_repeat:
            return True

    clean_text = re.sub(r"\s+", " ", text).strip()
    if len(clean_text) < ngram_size * 2:
        return False

    ngrams = [clean_text[i : i + ngram_size] for i in range(len(clean_text) - ngram_size + 1)]
    if not ngrams:
        return False

    unique_ngrams = len(set(ngrams))
    total_ngrams = len(ngrams)
    if total_ngrams > 0 and unique_ngrams / total_ngrams < ngram_threshold:
        return True

    words = clean_text.split()
    if len(words) >= 6:
        for i in range(len(words) - 5):
            window = words[i : i + 6]
            word_counts = Counter(window)
            if any(c >= 4 for c in word_counts.values()):
                return True

    return False


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
    """Auto-discover the newest ``codeforces-exec-statistics-*.json`` next to *jsonl_path*."""
    parent = jsonl_path.parent
    pat = str(parent / CODEFORCES_STATS_GLOB)
    matches = glob.glob(pat)
    if not matches:
        raise FileNotFoundError(
            f"No {CODEFORCES_STATS_GLOB} found under {parent}"
        )
    return Path(max(matches, key=lambda p: Path(p).stat().st_mtime))


# ---------------------------------------------------------------------------
# Per-dataset builders  (shortest-correct or all-correct strategy)
# ---------------------------------------------------------------------------

def _pick_shortest_correct(
    completions: list[str],
    accs: list[float],
    finished: list[bool],
    n_tokens: list[int],
) -> tuple[str, int] | None:
    """Return (completion, token_count) of the shortest correct & finished completion, or None."""
    best: tuple[str, int] | None = None
    for comp, acc, fin, ntok in zip(completions, accs, finished, n_tokens):
        if float(acc) <= 0:
            continue
        if not fin:
            continue
        # if has_repetition(comp):
        #     continue
        if best is None or ntok < best[1]:
            best = (comp, ntok)
    return best


def _pick_all_correct(
    completions: list[str],
    accs: list[float],
    finished: list[bool],
    n_tokens: list[int],
) -> list[tuple[str, int]]:
    """Return list of (completion, token_count) for all correct & finished completions."""
    results: list[tuple[str, int]] = []
    for comp, acc, fin, ntok in zip(completions, accs, finished, n_tokens):
        if float(acc) <= 0:
            continue
        if not fin:
            continue
        # if has_repetition(comp):
        #     continue
        results.append((comp, ntok))
    return results


def build_rows_dapo_math(jsonl_path: Path, length_mode: str = "shortest") -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    by_mi = load_jsonl_by_micro_index(jsonl_path)
    n_bad = 0
    n_no_correct = 0
    for micro_index in tqdm.tqdm(sorted(by_mi), desc="dapo_math"):
        record = by_mi[micro_index]
        si = record.get("src_item", {}) or {}
        user_prompt = user_content_from_src_item(si)
        completions = record.get("completion_texts") or []
        if not isinstance(completions, list):
            completions = []
        accs = record.get("accuracies")
        finished = record.get("finished")
        n_tokens = record.get("n_tokens")
        if (
            not isinstance(accs, list)
            or len(accs) != len(completions)
            or not isinstance(finished, list)
            or len(finished) != len(completions)
            or not isinstance(n_tokens, list)
            or len(n_tokens) != len(completions)
        ):
            n_bad += 1
            continue

        if length_mode == "shortest":
            pick = _pick_shortest_correct(completions, accs, finished, n_tokens)
            if pick is None:
                n_no_correct += 1
                continue
            rows.append(messages_from_chat(user_prompt, pick[0]))
        else:  # all
            picks = _pick_all_correct(completions, accs, finished, n_tokens)
            if not picks:
                n_no_correct += 1
                continue
            for pick, _ in picks:
                rows.append(messages_from_chat(user_prompt, pick))

    if n_bad:
        logger.warning("dapo_math: skipped %d problems (bad/missing field lists)", n_bad)
    logger.info(
        "dapo_math: %d SFT rows from %d questions (%d had no correct completion)",
        len(rows),
        len(by_mi),
        n_no_correct,
    )
    return rows


def build_rows_codeforces(
    jsonl_path: Path,
    stats: dict[str, Any],
    length_mode: str = "shortest",
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    acc_by_mi = acc_map_from_codeforces_stats(stats)
    by_mi = load_jsonl_by_micro_index(jsonl_path)
    n_missing = 0
    n_mismatch = 0
    n_no_correct = 0
    for micro_index in tqdm.tqdm(sorted(by_mi), desc="codeforces"):
        record = by_mi[micro_index]
        si = record.get("src_item", {}) or {}
        acc_list = acc_by_mi.get(micro_index)
        if acc_list is None:
            n_missing += 1
            continue
        user_prompt = user_content_from_src_item(si)
        completions = record.get("completion_texts") or []
        finished = record.get("finished")
        n_tokens = record.get("n_tokens")
        if not isinstance(completions, list):
            completions = []
        if (
            not isinstance(finished, list)
            or len(finished) != len(completions)
            or not isinstance(n_tokens, list)
            or len(n_tokens) != len(completions)
        ):
            n_mismatch += 1
            continue
        if len(acc_list) != len(completions):
            n_mismatch += 1
            continue

        if length_mode == "shortest":
            pick = _pick_shortest_correct(completions, acc_list, finished, n_tokens)
            if pick is None:
                n_no_correct += 1
                continue
            rows.append(messages_from_chat(user_prompt, pick[0]))
        else:  # all
            picks = _pick_all_correct(completions, acc_list, finished, n_tokens)
            if not picks:
                n_no_correct += 1
                continue
            for pick, _ in picks:
                rows.append(messages_from_chat(user_prompt, pick))

    if n_missing:
        logger.warning(
            "codeforces: %d problems missing from exec stats", n_missing
        )
    if n_mismatch:
        logger.warning(
            "codeforces: %d problems skipped (field length mismatch)", n_mismatch
        )
    logger.info(
        "codeforces: %d SFT rows from %d questions (%d had no correct completion)",
        len(rows),
        len(by_mi),
        n_no_correct,
    )
    return rows


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

SftBuilder = Callable[..., list[dict[str, Any]]]

SFT_SOURCE_REGISTRY: dict[str, dict[str, Any]] = {
    "dapo_math": {
        "builder": build_rows_dapo_math,
        "needs_stats": False,
    },
    "codeforces": {
        "builder": build_rows_codeforces,
        "needs_stats": True,
    },
}


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def write_swift_sft_jsonl(rows: list[dict[str, Any]], path: Path) -> int:
    """Write one ``{"messages": ...}`` per line."""
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            msgs = r.get("messages")
            if not msgs:
                continue
            f.write(json.dumps({"messages": msgs}, ensure_ascii=False) + "\n")
            n += 1
    return n


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build SFT JSONL from result JSONL files."
        ),
    )
    parser.add_argument(
        "jsonl_paths",
        type=Path,
        nargs="+",
        help=(
            "One or more *_results.jsonl files. The dataset name is inferred "
            "from the filename (e.g. dapo_math_results.jsonl -> dapo_math)."
        ),
    )
    parser.add_argument(
        "--length",
        type=str,
        choices=["shortest", "all"],
        default="shortest",
        help="Selection mode: 'shortest' for shortest correct completion per question, 'all' for all correct completions (default: shortest)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output JSONL path for ms-swift SFT (default: data/sft_{length}.jsonl)",
    )
    args = parser.parse_args()

    # Set default output path based on length mode if not provided
    if args.output is None:
        args.output = Path(f"data/sft_{args.length}.jsonl")

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
        spec = SFT_SOURCE_REGISTRY.get(dataset_name)
        if spec is None:
            logger.warning(
                "Skipping — unknown dataset %r (inferred from %s). "
                "Known: %s",
                dataset_name,
                jsonl_path.name,
                ", ".join(SFT_SOURCE_REGISTRY),
            )
            continue

        logger.info("Processing %s  [dataset=%s, mode=%s]", jsonl_path, dataset_name, args.length)
        builder: SftBuilder = spec["builder"]

        if spec["needs_stats"]:
            stats_path = resolve_codeforces_stats_path(jsonl_path)
            cf_stats = json.loads(stats_path.read_text(encoding="utf-8"))
            logger.info("Loaded codeforces stats from %s", stats_path)
            part = builder(jsonl_path, cf_stats, args.length)
        else:
            part = builder(jsonl_path, args.length)

        rows.extend(part)

    if not rows:
        raise RuntimeError("No rows built — check input paths.")

    logger.info("Total training samples: %d", len(rows))

    n_written = write_swift_sft_jsonl(rows, args.output)
    logger.info("Written swift SFT JSONL: %s (%d lines)", args.output, n_written)


if __name__ == "__main__":
    main()
