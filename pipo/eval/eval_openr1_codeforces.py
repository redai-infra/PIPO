#!/usr/bin/env python3
"""Execute open-r1/Codeforces model outputs against official (and optional generated) tests.

Joins each jsonl record to the Hugging Face row at the same ``micro_index`` (same split as
``preprocess_datasets.py``: ``open-r1/codeforces``, ``verifiable-prompts``, ``train[:90%]``).
Results are deduped by ``micro_index`` (last line wins) and processed in sorted index order.

Requires either local ``piston_compile.sh`` / ``piston_run.sh`` (default) or a Piston HTTP
API (``--executor piston``) with the ``codeforces`` 1.0.0 package installed.

Use ``--workers N`` for parallel questions. On Linux, workers use ``fork`` so the HF dataset
stays memory-mapped once (copy-on-write). On Windows / spawn, each worker reloads the split.

The statistics JSON includes ``micro_indices_sorted``, ``all_passk_sorted`` (pass@k per
problem), and ``all_pass1_sorted`` (per-completion 0/1 accuracies, parallel to
``micro_indices_sorted``).

Reference: https://huggingface.co/datasets/open-r1/codeforces#verifying-problems
"""

from __future__ import annotations

import argparse
import json
import multiprocessing
import os
import re
import shutil
import subprocess
import sys
import tempfile
import math
import tqdm
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.compute as pc
import pyarrow.parquet as pq
import requests
from datasets import load_dataset

sys.path.append(os.getcwd())

from pipo.utils import safe_mean

_LOCAL_NO_PROXY_HOSTS = ("localhost", "127.0.0.1", "::1")


def ensure_no_proxy_localhosts() -> None:
    """Append loopback hosts so ``requests`` bypasses a global HTTP proxy for local services."""
    for key in ("NO_PROXY", "no_proxy"):
        cur = os.environ.get(key, "")
        parts = [p.strip() for p in cur.split(",") if p.strip()]
        for h in _LOCAL_NO_PROXY_HOSTS:
            if h not in parts:
                parts.append(h)
        if parts:
            os.environ[key] = ",".join(parts)


def language_to_piston(language: str) -> tuple[str, str, str]:
    """Return (piston_language, main_filename, PISTON_LANGUAGE for compile/run scripts)."""
    if language == "cpp":
        return "cf_c++17", "main.cpp", "c++17"
    if language == "python":
        return "cf_python3", "main.py", "python3"
    raise ValueError(f"Unsupported problem language: {language!r} (expected cpp or python)")


_CPP_BLOCKS = re.compile(r"```(?:cpp|c\+\+|cxx)\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)
_PY_BLOCKS = re.compile(r"```(?:python|py)\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)
_GENERIC_BLOCKS = re.compile(r"```[a-z0-9+]*\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)


def extract_solution_code(completion_text: str, language: str) -> str:
    """Take the last fenced code block; models often emit several ```cpp``` drafts."""
    if language == "cpp":
        blocks = _CPP_BLOCKS.findall(completion_text)
        if not blocks:
            blocks = _GENERIC_BLOCKS.findall(completion_text)
    elif language == "python":
        blocks = _PY_BLOCKS.findall(completion_text)
        if not blocks:
            blocks = _GENERIC_BLOCKS.findall(completion_text)
    else:
        blocks = _GENERIC_BLOCKS.findall(completion_text)
    if not blocks:
        return ""
    return blocks[-1].strip()


def load_generated_tests(root: Path, problem: dict[str, Any]) -> list[dict[str, str]]:
    """Load extra tests from HF layout: ``test_cases_{contest_id}.parquet``."""
    cid = problem.get("contest_id")
    pid = problem.get("id")
    if cid is None or pid is None:
        return []
    fp = root / f"test_cases_{cid}.parquet"
    if not fp.is_file():
        return []
    table = pq.read_table(fp, columns=["problem_id", "input", "output"])
    mask = pc.equal(table["problem_id"], pid)
    filtered = table.filter(mask)
    if filtered.num_rows == 0:
        return []
    inputs = filtered.column("input").to_pylist()
    outputs = filtered.column("output").to_pylist()
    return [{"input": str(i), "output": str(o)} for i, o in zip(inputs, outputs)]


def build_grader_config(problem: dict[str, Any]) -> str:
    lines = [
        f"TIME_LIMIT={problem['time_limit']}",
        f"MEMORY_LIMIT={problem['memory_limit']}",
        f"INPUT_MODE={problem['input_mode']}",
    ]
    return "\n".join(lines) + "\n"


def run_tests_local(
    source_code: str,
    problem: dict[str, Any],
    tests: list[dict[str, str]],
    compile_sh: Path,
    run_sh: Path,
) -> tuple[bool, str]:
    """Run all tests in a temp dir using repo piston scripts. Returns (all_passed, detail)."""
    if not tests:
        return False, "no tests"
    piston_lang = language_to_piston(problem["language"])[2]
    main_name = language_to_piston(problem["language"])[1]
    checker = problem.get("generated_checker") or ""

    with tempfile.TemporaryDirectory(prefix="cf_eval_") as tmp:
        wd = Path(tmp)
        shutil.copy2(compile_sh, wd / "piston_compile.sh")
        shutil.copy2(run_sh, wd / "piston_run.sh")
        for name in ("piston_compile.sh", "piston_run.sh"):
            (wd / name).chmod(0o755)

        (wd / main_name).write_text(source_code, encoding="utf-8")
        (wd / "grader_config").write_text(build_grader_config(problem), encoding="utf-8")
        if checker:
            (wd / "checker.py").write_text(checker, encoding="utf-8")

        env = os.environ.copy()
        env["PISTON_LANGUAGE"] = piston_lang

        if piston_lang != "python3":
            cr = subprocess.run(
                ["bash", str(wd / "piston_compile.sh"), main_name],
                cwd=str(wd),
                env=env,
                capture_output=True,
                text=True,
                timeout=120,
            )
            if cr.returncode != 0 or not (wd / "a.out").exists():
                return False, f"compile_failed rc={cr.returncode} stderr={cr.stderr[:500]}"

        for ti, case in enumerate(tests):
            (wd / "input.txt").write_text(case["input"], encoding="utf-8")
            (wd / "correct_output.txt").write_text(case["output"], encoding="utf-8")
            run_cmd = ["bash", str(wd / "piston_run.sh")]
            if piston_lang == "python3":
                run_cmd.append(main_name)
            run_timeout = max(30.0, float(problem["time_limit"]) * 2 + 5.0)
            rr = subprocess.run(
                run_cmd,
                cwd=str(wd),
                env=env,
                capture_output=True,
                text=True,
                timeout=math.ceil(run_timeout),
            )
            out_first = (rr.stdout or "").strip().splitlines()
            score = out_first[0].strip() if out_first else ""
            if rr.returncode != 0 or score != "1":
                return False, f"test[{ti}] score={score!r} rc={rr.returncode} stderr={(rr.stderr or '')[:200]}"
        return True, "ok"


def run_tests_piston(
    source_code: str,
    problem: dict[str, Any],
    tests: list[dict[str, str]],
    piston_execute_url: str,
    session: requests.Session,
    timeout: float,
) -> tuple[bool, str]:
    piston_language, main_name, _ = language_to_piston(problem["language"])
    checker = problem.get("generated_checker") or ""
    extra_checker = (
        [{"name": "checker.py", "content": checker}] if checker else []
    )
    grader = build_grader_config(problem)

    for ti, case in enumerate(tests):
        payload = {
            "language": piston_language,
            "version": "*",
            "files": [
                {"name": main_name, "content": source_code},
                {"name": "input.txt", "content": case["input"]},
                {"name": "correct_output.txt", "content": case["output"]},
                *extra_checker,
                {"name": "grader_config", "content": grader},
            ],
        }
        try:
            r = session.post(piston_execute_url, json=payload, timeout=timeout)
            r.raise_for_status()
            data = r.json()
        except Exception as e:
            return False, f"test[{ti}] http_error {e}"

        comp = data.get("compile") or {}
        run = data.get("run") or {}
        if comp.get("code") not in (0, None):
            return False, f"test[{ti}] compile_code={comp.get('code')}"
        if run.get("code") not in (0, None):
            return False, f"test[{ti}] run_code={run.get('code')}"
        stdout = (run.get("stdout") or "").strip().split()
        if not stdout or stdout[0] != "1":
            return False, f"test[{ti}] stdout={run.get('stdout')!r}"
    return True, "ok"


def _as_bool_list(x: Any, n: int) -> list[bool]:
    if isinstance(x, list):
        out = [bool(v) for v in x[:n]]
        if len(out) < n:
            out.extend([True] * (n - len(out)))
        return out
    return [True] * n


def _collect_token_lengths(n_tokens_val: Any) -> list[float]:
    if isinstance(n_tokens_val, list):
        out: list[float] = []
        for v in n_tokens_val:
            try:
                out.append(float(v))
            except Exception:
                continue
        return out
    try:
        return [float(n_tokens_val)]
    except Exception:
        return []


def _detect_dataset_stem(path: Path) -> str:
    """Return 'codeforces' or 'codeforces_rl' from the results filename."""
    name = path.name
    if "codeforces_rl" in name:
        return "codeforces_rl"
    if "codeforces" in name:
        return "codeforces"
    return "codeforces"


def load_jsonl_by_micro_index(path: Path) -> dict[int, dict[str, Any]]:
    """Parse jsonl; keep one record per ``micro_index`` (last line wins)."""
    by_mi: dict[int, dict[str, Any]] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            si = record.get("src_item", {}) or {}
            try:
                mi = int(si.get("micro_index", -1))
            except (TypeError, ValueError):
                continue
            if mi < 0:
                continue
            by_mi[mi] = record
    return by_mi


def row_to_problem(row: Any) -> dict[str, Any]:
    """HF ``datasets`` rows may use numpy types; normalize to plain Python for file I/O / piston."""

    def conv(v: Any) -> Any:
        if isinstance(v, np.ndarray):
            return v.tolist()
        if isinstance(v, np.generic):
            return v.item()
        if isinstance(v, dict):
            return {k: conv(x) for k, x in v.items()}
        if isinstance(v, list):
            return [conv(x) for x in v]
        return v

    return {k: conv(v) for k, v in dict(row).items()}


# Populated in the parent before forking workers (Linux ``fork``); children share COW pages.
_MP_SHARED: dict[str, Any] = {}

# Lazy per-process Session for piston (do not share across forks from parent).
_worker_piston_session: requests.Session | None = None


def _worker_piston_session_get() -> requests.Session:
    global _worker_piston_session
    if _worker_piston_session is None:
        ensure_no_proxy_localhosts()
        _worker_piston_session = requests.Session()
    return _worker_piston_session


def eval_one_question(
    micro_index: int,
    record: dict[str, Any],
    cf_ds: Any,
    n_hf: int,
    compile_sh: Path,
    run_sh: Path,
    executor: str,
    piston_execute_url: str,
    piston_timeout: float,
    gen_root: Path | None,
) -> dict[str, Any]:
    """Evaluate one jsonl record. Returns a result dict (including ``skipped``)."""
    out: dict[str, Any] = {
        "micro_index": micro_index,
        "skipped": True,
        "skip_reason": "",
        "finished_num": 0,
        "accuracies": [],
        "pass1": 0.0,
        "passk": 0.0,
        "n_tokens_lengths": [],
    }

    si = record.get("src_item", {}) or {}
    ds_name = si.get("dataset")
    if ds_name not in ("codeforces", "codeforces_rl"):
        out["skip_reason"] = f"dataset={ds_name!r}"
        return out

    if micro_index < 0 or micro_index >= n_hf:
        out["skip_reason"] = f"oob_index>={n_hf}"
        return out

    problem = row_to_problem(cf_ds[micro_index])
    official = problem.get("official_tests") or []
    tests: list[dict[str, str]] = [
        {"input": str(t["input"]), "output": str(t["output"])} for t in official
    ]
    if gen_root and gen_root.is_dir():
        tests = tests + load_generated_tests(gen_root, problem)

    comp_list = record.get("completion_texts")
    if not isinstance(comp_list, list):
        comp_list = []
    n_comp = len(comp_list)
    finished = _as_bool_list(record.get("finished"), n_comp)

    out["skipped"] = False
    out["finished_num"] = int(sum(bool(x) for x in finished))

    accuracies: list[float] = []
    session = _worker_piston_session_get() if executor == "piston" else None
    for j in range(n_comp):
        text = comp_list[j]
        try:
            code = extract_solution_code(text, problem["language"])
            if not code.strip():
                accuracies.append(0.0)
                continue
            if executor == "local":
                ok, _detail = run_tests_local(code, problem, tests, compile_sh, run_sh)
            else:
                assert session is not None
                ok, _detail = run_tests_piston(
                    code,
                    problem,
                    tests,
                    piston_execute_url,
                    session,
                    piston_timeout,
                )
            accuracies.append(1.0 if ok else 0.0)
        except Exception:
            accuracies.append(0.0)

    if not accuracies:
        out["pass1"] = 0.0
        out["passk"] = 0.0
    else:
        out["pass1"] = float(accuracies[0])
        out["passk"] = float(np.max(accuracies))
    out["accuracies"] = accuracies
    out["n_tokens_lengths"] = _collect_token_lengths(record.get("n_tokens"))
    return out


def _mp_eval_job(job: tuple[int, dict[str, Any]]) -> dict[str, Any]:
    """Pool worker: read config from ``_MP_SHARED`` (filled before ``fork`` or in spawn init)."""
    micro_index, record = job
    s = _MP_SHARED
    return eval_one_question(
        micro_index,
        record,
        s["cf_ds"],
        s["n_hf"],
        s["compile_sh"],
        s["run_sh"],
        s["executor"],
        s["piston_execute_url"],
        float(s["piston_timeout"]),
        s["gen_root"],
    )


def _mp_init_worker_spawn(cfg: dict[str, Any]) -> None:
    """Spawn workers: each process loads HF data (higher RAM; no shared COW)."""
    global _MP_SHARED
    ds = load_dataset(cfg["hf_dataset"], name=cfg["hf_config"], split=cfg["hf_split"])
    gr = cfg.get("gen_root") or ""
    _MP_SHARED = {
        "cf_ds": ds,
        "n_hf": len(ds),
        "compile_sh": Path(cfg["compile_sh"]),
        "run_sh": Path(cfg["run_sh"]),
        "executor": cfg["executor"],
        "piston_execute_url": cfg["piston_execute_url"],
        "piston_timeout": cfg["piston_timeout"],
        "gen_root": Path(gr) if gr else None,
    }


def _pool_context_and_spawn_reload(workers: int) -> tuple[multiprocessing.context.BaseContext, bool]:
    """Return (context, needs_spawn_dataset_reload). Fork avoids reloading HF in each worker."""
    if workers <= 1:
        raise ValueError("workers must be > 1")
    if sys.platform == "win32":
        return multiprocessing.get_context("spawn"), True
    try:
        return multiprocessing.get_context("fork"), False
    except ValueError:
        return multiprocessing.get_context(), True


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Codeforces jsonl via test execution")
    parser.add_argument(
        "results_jsonl",
        type=str,
        default="",
    )
    parser.add_argument(
        "--hf-dataset",
        type=str,
        default="open-r1/codeforces",
        help="HF dataset id (must match the split used when generating results)",
    )
    parser.add_argument(
        "--hf-config",
        type=str,
        default="verifiable-prompts",
        help="HF config / subset name= (matches preprocess open-r1/codeforces)",
    )
    parser.add_argument(
        "--hf-split",
        type=str,
        default="train[:90%]",
        help="HF split (matches preprocess sft parquet: train[:90%%])",
    )
    parser.add_argument(
        "--executor",
        choices=("local", "piston"),
        default="local",
        help="local: bash piston_compile/run in a temp dir; piston: HTTP API",
    )
    parser.add_argument(
        "--piston-url",
        type=str,
        default=os.environ.get("PISTON_URL", "http://127.0.0.1:2000/api/v2"),
        help="Base URL for Piston v2 (used when executor=piston); POST .../execute",
    )
    parser.add_argument(
        "--piston-timeout",
        type=float,
        default=120.0,
        help="Per-request timeout (piston mode)",
    )
    parser.add_argument(
        "--generated-tests-dir",
        type=str,
        default=None,
        help="Optional dir with test_cases_*.parquet (HF download layout)",
    )
    parser.add_argument(
        "--max-questions",
        type=int,
        default=0,
        help="If >0, stop after this many unique questions (debug)",
    )
    parser.add_argument(
        "--out-stats",
        type=str,
        default=None,
        help="Output statistics JSON path (default: alongside jsonl, codeforces-exec-*.json)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, (os.cpu_count() or 4) - 1),
        help="Parallel worker processes (1 = sequential). On Linux, uses fork so the HF "
        "dataset stays mapped once in memory (copy-on-write).",
    )
    args = parser.parse_args()

    results_path = Path(args.results_jsonl)
    if not results_path.is_file():
        raise FileNotFoundError(results_path)

    gen_root = Path(args.generated_tests_dir).resolve() if args.generated_tests_dir else None

    print(
        f"Loading HF {args.hf_dataset!r} (config={args.hf_config!r}, split={args.hf_split!r}) ...",
        flush=True,
    )
    cf_ds = load_dataset(
        args.hf_dataset,
        name=args.hf_config,
        split=args.hf_split,
    )
    n_hf = len(cf_ds)

    results_by_mi = load_jsonl_by_micro_index(results_path)
    if not results_by_mi:
        raise RuntimeError(f"No usable records in {results_path}")

    eval_dir = Path(__file__).resolve().parent
    compile_sh = eval_dir / "piston_compile.sh"
    run_sh = eval_dir / "piston_run.sh"
    if args.executor == "local" and (not compile_sh.is_file() or not run_sh.is_file()):
        raise FileNotFoundError(f"Missing {compile_sh} or {run_sh}")

    piston_execute_url = args.piston_url.rstrip("/") + "/execute"

    sorted_mis = sorted(results_by_mi)
    if args.max_questions > 0:
        sorted_mis = sorted_mis[: args.max_questions]

    # Validate dataset labels before spawning work.
    dataset: str | None = None
    for micro_index in sorted_mis:
        record = results_by_mi[micro_index]
        si = record.get("src_item", {}) or {}
        ds = si.get("dataset")
        if dataset is None:
            dataset = ds
        if ds is not None and dataset != ds:
            raise ValueError(f"Mixed datasets: {dataset} vs {ds}")
        if dataset not in ("codeforces", "codeforces_rl"):
            raise ValueError(f"Expected dataset codeforces or codeforces_rl, got {dataset!r}")

    jobs: list[tuple[int, dict[str, Any]]] = [
        (mi, results_by_mi[mi]) for mi in sorted_mis
    ]

    workers = max(1, int(args.workers))
    results: list[dict[str, Any]] = []

    if workers == 1:
        for job in tqdm.tqdm(jobs):
            results.append(
                eval_one_question(
                    job[0],
                    job[1],
                    cf_ds,
                    n_hf,
                    compile_sh,
                    run_sh,
                    args.executor,
                    piston_execute_url,
                    args.piston_timeout,
                    gen_root,
                )
            )
    else:
        ctx, spawn_reload = _pool_context_and_spawn_reload(workers)
        chunksize = max(1, len(jobs) // (workers * 8) or 1)
        if spawn_reload:
            spawn_cfg: dict[str, Any] = {
                "hf_dataset": args.hf_dataset,
                "hf_config": args.hf_config,
                "hf_split": args.hf_split,
                "compile_sh": str(compile_sh),
                "run_sh": str(run_sh),
                "executor": args.executor,
                "piston_execute_url": piston_execute_url,
                "piston_timeout": args.piston_timeout,
                "gen_root": str(gen_root) if gen_root else "",
            }
            pool_kw: dict[str, Any] = {
                "processes": workers,
                "initializer": _mp_init_worker_spawn,
                "initargs": (spawn_cfg,),
            }
        else:
            _MP_SHARED.clear()
            _MP_SHARED.update(
                {
                    "cf_ds": cf_ds,
                    "n_hf": n_hf,
                    "compile_sh": compile_sh,
                    "run_sh": run_sh,
                    "executor": args.executor,
                    "piston_execute_url": piston_execute_url,
                    "piston_timeout": args.piston_timeout,
                    "gen_root": gen_root,
                }
            )
            pool_kw = {"processes": workers}
        with ctx.Pool(**pool_kw) as pool:
            for r in tqdm.tqdm(
                pool.imap_unordered(_mp_eval_job, jobs, chunksize=chunksize),
                total=len(jobs),
            ):
                results.append(r)

    by_mi: dict[int, dict[str, Any]] = {}
    for r in results:
        if r.get("skipped"):
            if r.get("skip_reason"):
                print(
                    f"[skip] micro_index={r['micro_index']} {r['skip_reason']}",
                    flush=True,
                )
            continue
        mi = int(r["micro_index"])
        by_mi[mi] = r

    if not by_mi:
        raise RuntimeError("No records processed (all skipped or empty jobs).")

    micro_indices_sorted = sorted(by_mi)
    all_passk_sorted: list[float] = [float(by_mi[mi]["passk"]) for mi in micro_indices_sorted]
    all_pass1_sorted: list[list[float]] = [
        [float(x) for x in by_mi[mi]["accuracies"]] for mi in micro_indices_sorted
    ]

    question_num = len(by_mi)
    finished_num = sum(int(by_mi[mi]["finished_num"]) for mi in micro_indices_sorted)
    pass1_per_question = [float(by_mi[mi]["pass1"]) for mi in micro_indices_sorted]
    passk_per_question = [float(by_mi[mi]["passk"]) for mi in micro_indices_sorted]
    all_acc: list[float] = []
    all_n: list[float] = []
    for mi in micro_indices_sorted:
        all_acc.extend(by_mi[mi]["accuracies"])
        all_n.extend(by_mi[mi]["n_tokens_lengths"])

    avg_acc = safe_mean(all_acc)
    pass1_mean = safe_mean(pass1_per_question)
    passk_mean = safe_mean(passk_per_question)
    avg_token_length_all = safe_mean(all_n)

    statistics = {
        "question_num": question_num,
        "finished_num": finished_num,
        "avg_accuracy": avg_acc,
        "pass@k": passk_mean,
        "pass@1": pass1_mean,
        "avg_token_length-all": avg_token_length_all,
        "micro_indices_sorted": micro_indices_sorted,
        "all_passk_sorted": all_passk_sorted,
        "all_pass1_sorted": all_pass1_sorted,
    }

    if args.out_stats:
        out_path = Path(args.out_stats)
    else:
        dataset_stem = _detect_dataset_stem(results_path)
        out_path = (
            results_path.parent
            / f"{dataset_stem}-exec-statistics-{pass1_mean:.4f}-{passk_mean:.4f}-{int(avg_token_length_all)}.json"
        )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(statistics, indent=2), encoding="utf-8")
    print(f"Wrote: {out_path}", flush=True)


if __name__ == "__main__":
    main()
