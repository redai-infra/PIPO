import argparse
import asyncio
import fcntl
import json
import os
import signal
import subprocess
import time
from pathlib import Path

from tqdm import tqdm
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
# os.environ.setdefault("RAYON_NUM_THREADS", "1")
# os.environ.setdefault("OMP_NUM_THREADS", "1")
# os.environ.setdefault("MKL_NUM_THREADS", "1")
# os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
# os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

from transformers import AutoTokenizer
import sglang as sgl

from pipo.eval.benchmark_loader import load_datasets, list_datasets
from pipo.utils import get_timestamp


def _is_stop_finish(finish_reason) -> bool:
    """Return True when the generation ended on a stop token (not truncated)."""
    if finish_reason is None:
        return False
    if isinstance(finish_reason, dict):
        return finish_reason.get("type") == "stop"
    return finish_reason.to_json().get("type") == "stop"


def load_checkpoint_done(log_dir: Path, datasets: list[str]) -> set[tuple[str, int]]:
    """Load (dataset, micro_index) of already-processed questions."""
    done = set()
    for dataset in datasets:
        path = log_dir / f"{dataset}-results.jsonl"
        if not path.exists():
            continue
        with open(path) as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    r = json.loads(line)
                    si = r.get("src_item", {})
                    done.add((si.get("dataset"), si.get("micro_index", -1)))
                except json.JSONDecodeError:
                    continue
    return done


def make_record(qidx: int, src_item: dict, completion_texts: list[str], meta_infos: list[dict]) -> dict:
    """Build the unified result record. Evaluation (accuracies / extracted_answers /
    pass@k) is performed post hoc by pipo/eval/eval.sh.
    """
    finished = [_is_stop_finish(m.get("finish_reason")) for m in meta_infos]
    n_tokens = [m.get("completion_tokens", 0) for m in meta_infos]
    n_pads = [m.get("n_pad", 0) for m in meta_infos]
    return {
        "question_idx": qidx,
        "src_item": src_item,
        "completion_texts": completion_texts,
        "extracted_answers": None,
        "accuracies": None,
        "pass@k": None,
        "n_tokens": n_tokens,
        "n_pads": n_pads,
        "finished": finished,
        "meta_infos": meta_infos,
    }


async def run_streaming(
    engine: sgl.Engine,
    tokenizer,
    src_items: list[dict],
    num_samples: int,
    sampling_params: dict,
    exp_dir: Path,
    max_concurrent: int = 0,
) -> float:
    """Generate and append each question to {dataset}-results.jsonl (no eval)."""
    max_concurrent = min(len(src_items), 512)
    semaphore = asyncio.Semaphore(max_concurrent)
    print(f"[run_streaming] num_samples={num_samples}, max_concurrent={max_concurrent}, total_questions={len(src_items)}", flush=True)

    async def generate_one(idx: int, src_item: dict):
        async with semaphore:
            messages = [{"role": "user", "content": src_item["prompt"]}]
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=True,
            )

            sample_rets = []
            for _ in range(num_samples):
                try:
                    ret = await engine.async_generate(prompt=prompt, sampling_params=sampling_params)
                    sample_rets.append(ret)
                except Exception as e:
                    print(f"  [SKIP] generate failed: {type(e).__name__}: {e}", flush=True)
                    return None

            return idx, sample_rets

    tasks = [asyncio.create_task(generate_one(i, item)) for i, item in enumerate(src_items)]
    start_time = time.time()

    for coro in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Generating", unit="q", ncols=80, leave=True):
        try:
            res = await coro
            if res is None:
                continue
            qidx, sample_rets = res
            src_item = src_items[qidx]
            dataset = src_item["dataset"]

            completion_texts = []
            meta_infos = []
            for ret in sample_rets:
                completion_texts.append(f'<think>\n{ret.get("text", "").removeprefix('<think>\n')}')
                meta = dict(ret.get("meta_info", {}))
                output_ids = ret.get("output_ids") or []
                meta["n_pad"] = int(sum(int(tok == tokenizer.pad_token_id) for tok in output_ids))
                fr = meta.get("finish_reason")
                if fr is not None and not isinstance(fr, dict):
                    meta["finish_reason"] = fr.to_json()
                meta_infos.append(meta)

            record = make_record(qidx, src_item, completion_texts, meta_infos)

            results_path = exp_dir / f"{dataset}-results.jsonl"
            def _write_with_lock():
                with open(results_path, "a") as f:
                    fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                    try:
                        f.write(json.dumps(record, ensure_ascii=False) + "\n")
                        f.flush()
                    finally:
                        fcntl.flock(f.fileno(), fcntl.LOCK_UN)
            await asyncio.to_thread(_write_with_lock)

        except KeyboardInterrupt:
            print("[Interrupt] KeyboardInterrupt", flush=True)
            return time.time() - start_time
        except Exception:
            import traceback
            traceback.print_exc()

    return time.time() - start_time


def get_args():
    parser = argparse.ArgumentParser(description="In-process sglang.Engine evaluation")

    # Engine / model args
    parser.add_argument("--model_path", type=str, default="Qwen/Qwen3.5-4B")
    parser.add_argument("--tp_size", type=int, default=1)
    parser.add_argument("--dp_size", type=int, default=8)
    parser.add_argument("--mem_fraction_static", type=float, default=0.8)
    parser.add_argument("--context_length", type=int, default=262144)

    parser.add_argument("--enable_pipo", action="store_true", help="Enable PIPO")
    parser.add_argument("--pipo_conf_threshold", type=float, default=0.95)
    parser.add_argument("--enable_eagle", action="store_true", help="Enable NEXTN (EAGLE-2) speculative decoding")

    parser.add_argument("--disable_cuda_graph", action="store_true")
    parser.add_argument("--debug_cuda", action="store_true")
    parser.add_argument("--log_info", action="store_true", help="Enable info-level logging in sglang.Engine")

    # Data args
    parser.add_argument("--datasets", type=str, default="aime2025,gpqa_diamond,livecodebench,lb2",
                        help="Comma-separated dataset names")
    parser.add_argument("--list_datasets", action="store_true",
                        help="List available datasets and exit")

    # Eval args
    parser.add_argument("--output_dir", type=str, default="")
    parser.add_argument("--num_samples", type=int, default=4,
                        help="Number of samples per question")
    parser.add_argument("--max_concurrent", type=int, default=512)
    parser.add_argument("--max_generated_tokens", type=int, default=32768)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--presence_penalty", type=float, default=1.5)
    parser.add_argument("--log_suffix", type=str, default="")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--remaining_ratio_start", type=float, default=0)
    parser.add_argument("--remaining_ratio_end", type=float, default=1)
    parser.add_argument("--start_index", type=int, default=None)
    parser.add_argument("--end_index", type=int, default=None)
    parser.add_argument("--skip_eval", action="store_true",
                        help="Do not invoke pipo/eval/eval.sh at the end.")

    args = parser.parse_args()

    if args.debug:
        args.output_dir = f"./temp/{get_timestamp()}"
        if "outputs" in args.model_path:
            args.enable_pipo = True
            print("Warning: Using trained checkpoint, --enable_pipo converted to True")

    elif args.output_dir == "":
        if "outputs" in args.model_path:
            args.output_dir = f"./{args.model_path}/eval"
            args.enable_pipo = True
            # aligning max_generated_slots to max_generated_tokens
            args.max_generated_tokens = args.max_generated_tokens * 2
            print("Warning: Using trained checkpoint, --enable_pipo converted to True")

        elif args.model_path.startswith('Qwen/'):
            output_dir = f"./outputs/{args.model_path.split('/')[1]}"
            if args.enable_eagle:
                output_dir += "/eagle"
            else:
                output_dir += "/regular"
            args.output_dir = output_dir

    return args


def main():
    args = get_args()

    if args.list_datasets:
        print("Available datasets:")
        for name in list_datasets():
            print(f"  - {name}")
        return

    datasets = args.datasets.split(",")

    # Load data directly from HuggingFace
    print(f"Loading datasets: {datasets}")
    src_items, loaded = load_datasets(datasets)
    if not src_items:
        print("No data loaded.")
        return

    # Resume from checkpoint
    exp_name = f"{args.num_samples}_{args.temperature}_{args.top_p}_{args.top_k}_{args.presence_penalty}_{args.max_generated_tokens}"
    if args.enable_pipo:
        exp_name += f"_{args.pipo_conf_threshold}"

    if args.log_suffix:
        exp_name += f"_{args.log_suffix}"
    exp_dir = Path(args.output_dir) / exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)

    if args.start_index is not None or args.end_index is not None:
        print(f"Resuming: slicing questions by index {args.start_index}:{args.end_index}")
        src_items = src_items[args.start_index:args.end_index]

    done_keys = load_checkpoint_done(exp_dir, loaded)
    if done_keys:
        n_before = len(src_items)
        src_items = [x for x in src_items if (x["dataset"], x["micro_index"]) not in done_keys]
        n_skip = n_before - len(src_items)
        if n_skip:
            print(f"Resuming: skipping {n_skip} already-processed ({len(src_items)} remaining)")

    # Slice by ratio (only used when there is still work to do)
    if src_items:
        n_items = len(src_items)
        remaining_start_index = int(args.remaining_ratio_start * n_items)
        remaining_end_index = int(args.remaining_ratio_end * n_items)
        src_items = src_items[remaining_start_index:remaining_end_index]
        print(f"Evaluating {len(src_items)} questions (indices {remaining_start_index}:{remaining_end_index})")

    if src_items:
        # CUDA debug
        if args.debug_cuda:
            os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
            os.environ["TORCH_USE_CUDA_DSA"] = "1"

        # Build engine
        engine_kwargs = dict(
            model_path=args.model_path,
            tp_size=args.tp_size,
            dp_size=args.dp_size,
            mem_fraction_static=args.mem_fraction_static,
            context_length=args.context_length,
            reasoning_parser="qwen3",
        )
        if args.log_info:
            engine_kwargs["log_level"] = "info"

        if args.enable_pipo:
            engine_kwargs["enable_pipo"] = True
            engine_kwargs["disable_radix_cache"] = True
            os.environ["PIPO_CONF_THRESHOLD"] = str(args.pipo_conf_threshold)

        elif args.enable_eagle:
            os.environ["SGLANG_ENABLE_SPEC_V2"] = "1"
            engine_kwargs["speculative_algorithm"] = "NEXTN"
            engine_kwargs["speculative_num_steps"] = 3
            engine_kwargs["speculative_eagle_topk"] = 1
            engine_kwargs["speculative_num_draft_tokens"] = 4
            engine_kwargs["mamba_scheduler_strategy"] = "extra_buffer"

        if args.disable_cuda_graph:
            engine_kwargs["disable_cuda_graph"] = True

        print(f"Launching sglang.Engine: {engine_kwargs}", flush=True)
        engine = sgl.Engine(**engine_kwargs)
        # Promote SIGTERM (e.g. `kill <pid>`) to KeyboardInterrupt so it
        # follows the same orderly-shutdown path as Ctrl+C.  Without this,
        # SIGTERM kills the main process abruptly and every spawned sglang
        # scheduler / detokenizer / tokenizer_manager subprocess is
        # re-parented to PID 1; if PID 1 is a non-reaping stub like
        # `sleep infinity` (typical in dev containers) they accumulate as
        # <defunct> zombies forever, slowly eroding the per-host pthread /
        # VMA budget.
        def _sigterm_as_interrupt(signum, frame):
            raise KeyboardInterrupt
        signal.signal(signal.SIGTERM, _sigterm_as_interrupt)
        try:
            tokenizer = AutoTokenizer.from_pretrained(args.model_path)

            print("Starting generation...", flush=True)
            time_taken = engine.loop.run_until_complete(
                run_streaming(
                    engine, tokenizer, src_items, args.num_samples,
                    {
                        "temperature": args.temperature,
                        "top_p": args.top_p,
                        "top_k": args.top_k,
                        "presence_penalty": args.presence_penalty,
                        "max_new_tokens": args.max_generated_tokens,
                    },
                    exp_dir, args.max_concurrent
                )
            )
            print(f"\nGeneration time: {time_taken:.1f}s")
        except KeyboardInterrupt:
            print(
                "\n[sglang_eval] KeyboardInterrupt — cancelling pending requests "
                "and shutting down sglang.Engine before exit.",
                flush=True,
            )
            # Cancel pending asyncio tasks so engine.shutdown() does not hang
            # waiting on in-flight async_generate() coroutines.
            try:
                loop = engine.loop
                pending = [t for t in asyncio.all_tasks(loop) if not t.done()]
                for t in pending:
                    t.cancel()
                if pending:
                    loop.run_until_complete(
                        asyncio.gather(*pending, return_exceptions=True)
                    )
            except Exception as e:
                print(
                    f"[sglang_eval] pending task cancellation raised: "
                    f"{type(e).__name__}: {e}",
                    flush=True,
                )
            raise
        finally:
            # Always tear down the engine, even on KeyboardInterrupt / errors,
            # so that scheduler / detokenizer / tokenizer_manager subprocesses
            # exit cleanly and get reaped by their actual parent (this Python
            # process) rather than re-parented to PID 1 and becoming zombies.
            try:
                engine.shutdown()
            except Exception as e:
                print(
                    f"[sglang_eval] engine.shutdown() raised: "
                    f"{type(e).__name__}: {e}",
                    flush=True,
                )
    else:
        print("All questions already processed (no generation needed).")
        return

    print(f"\nResults written to {exp_dir}")

    # Post-hoc evaluation pipeline (rule-based + ifbench + lcb + excel).
    if args.skip_eval:
        print("[run_streaming] --skip_eval set; not invoking eval.sh")
        return
    repo_root = Path(__file__).resolve().parent
    eval_script = repo_root / "pipo" / "eval" / "eval.sh"
    print(f"\n[run_streaming] invoking {eval_script} {exp_dir}", flush=True)
    proc = subprocess.run(["bash", str(eval_script), str(exp_dir)], cwd=str(repo_root))
    if proc.returncode != 0:
        print(f"[run_streaming] eval.sh exited with code {proc.returncode}")


if __name__ == "__main__":
    main()
