"""
Benchmark dataset loader - HuggingFace and local parquet loading.

Loads and processes benchmark datasets from HuggingFace or local parquet files.
"""
import json
import random
from datetime import datetime
from pathlib import Path
from typing import Callable

import pandas as pd
from datasets import load_dataset as hf_load_dataset
import sys
import os
sys.path.append(os.getcwd())
from pipo.constants import PROMPT

DATA_DIR = Path(os.getcwd()) / "data"


def _format_mcq_question(question_text: str, choices: list[str], correct_idx: int = 0, seed: int = 0) -> tuple[str, str]:
    """Build MCQ prompt with shuffled choices. Returns (formatted_question, correct_letter)."""
    rng = random.Random(seed)
    perm = rng.sample(range(4), 4)
    shuffled = [choices[i] for i in perm]
    new_correct_pos = perm.index(correct_idx)
    correct_letter = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"[new_correct_pos]

    lines = [question_text.strip(), ""]
    for letter, choice in zip("ABCDEFGHIJKLMNOPQRSTUVWXYZ", shuffled):
        lines.append(f"{letter}) {choice}")
    formatted = "\n".join(lines)
    return formatted, correct_letter


# =============================================================================
# Dataset Processors
# =============================================================================

def _process_livecodebench(version_tag: str = "release_v6", start_date: str = "2025-02-01T00:00:00") -> list[dict]:
    """LiveCodeBench - coding problems."""
    ds = hf_load_dataset("livecodebench/code_generation_lite", version_tag=version_tag, trust_remote_code=True)
    filter_date = datetime.strptime(start_date, "%Y-%m-%dT%H:%M:%S")
    
    data = []
    for example in ds["test"]:
        contest_date = datetime.strptime(example["contest_date"], "%Y-%m-%dT%H:%M:%S")
        if contest_date < filter_date:
            continue
        prompt = PROMPT.get_lcb_prompt(example["question_content"], example["starter_code"])
        data.append({
            "prompt": prompt,
            "final_answer": '',
            "metadata": {
                'question_title': example["question_title"],
                'question_id': example["question_id"],
                'contest_id': example["contest_id"],
                'contest_date': example["contest_date"],
                'difficulty': example["difficulty"],
            }
        })
    return sorted(data, key=lambda x: x['metadata']['question_id'])


def _process_dapo_math(split: str = "sft") -> list[dict]:
    """DAPO-Math-17k-Processed dataset."""
    split_spec = "train[:90%]" if split == "sft" else "train[90%:]"
    raw_data = hf_load_dataset("open-r1/DAPO-Math-17k-Processed", split=split_spec)
    
    data = []
    for idx, example in enumerate(raw_data):
        prompt = PROMPT.MATH_QUERY_TEMPLATE.format(Question=example["prompt"])
        final_answer = example["reward_model"]["ground_truth"]
        data.append({
            "prompt": prompt,
            "final_answer": final_answer,
            "metadata": {"split": split, "index": idx}
        })
    return data


def _process_aime2025() -> list[dict]:
    """AIME 2025 math problems."""
    raw_data = hf_load_dataset("math-ai/aime25", split="test")
    
    data = []
    for idx, example in enumerate(raw_data):
        prompt = PROMPT.MATH_QUERY_TEMPLATE.format(Question=example["problem"])
        data.append({
            "prompt": prompt,
            "final_answer": example["answer"],
            "metadata": {"id": example["id"]}
        })
    return data


def _process_gpqa() -> list[dict]:
    """GPQA Diamond MCQ dataset."""
    raw_data = hf_load_dataset("Idavidrein/gpqa", "gpqa_diamond", split="train")
    
    candidate_keys = ["Correct Answer", "Incorrect Answer 1", "Incorrect Answer 2", "Incorrect Answer 3"]
    
    data = []
    for idx, example in enumerate(raw_data):
        choices = [example[k] for k in candidate_keys]
        question_with_choices, correct_letter = _format_mcq_question(
            example["Question"], choices, correct_idx=0, seed=idx
        )
        prompt = PROMPT.MCQ_QUERY_TEMPLATE.format(Question=question_with_choices)
        data.append({
            "prompt": prompt,
            "final_answer": correct_letter,
            "metadata": {"index": idx}
        })
    return data


def _process_codeforces(split: str = "sft") -> list[dict]:
    """Codeforces coding problems."""
    split_spec = "train[:90%]" if split == "sft" else "train[90%:]"
    raw_data = hf_load_dataset("open-r1/codeforces", "verifiable-prompts", split=split_spec)
    
    data = []
    for idx, example in enumerate(raw_data):
        data.append({
            "prompt": example["prompt"],
            "final_answer": "",
            "metadata": {
                "split": split,
                "index": idx,
                "raw_example": json.dumps(example, default=str),
            }
        })
    return data


def _process_lb2() -> list[dict]:
    """LongBench-v2 long-context evaluation."""
    raw_data = hf_load_dataset("zai-org/LongBench-v2", split="train")
    choice_keys = ["choice_A", "choice_B", "choice_C", "choice_D"]
    
    data = []
    for idx, example in enumerate(raw_data):
        if example['length'] != 'short':
            continue
        choices = [example[k] for k in choice_keys]
        correct_idx = ord(example["answer"]) - ord("A")
        formatted_question, correct_letter = _format_mcq_question(
            example["question"], choices, correct_idx=correct_idx, seed=idx
        )
        question_with_context = f"Read the following text and answer the question below.\n{example['context']}\n{formatted_question}"
        prompt = PROMPT.MCQ_QUERY_TEMPLATE.format(Question=question_with_context)
        
        data.append({
            "prompt": prompt,
            "final_answer": correct_letter,
            "metadata": {
                'question_id': example["_id"],
                "length": example["length"],
                "difficulty": example["difficulty"],
                "index": idx,
            }
        })
    return data


# =============================================================================
# Registry
# =============================================================================

_DATASET_PROCESSORS: dict[str, Callable] = {
    "livecodebench": _process_livecodebench,
    "dapo_math": lambda: _process_dapo_math("sft"),
    "dapo_math_sft": lambda: _process_dapo_math("sft"),
    "dapo_math_rl": lambda: _process_dapo_math("rl"),
    "aime2025": _process_aime2025,
    "gpqa": _process_gpqa,
    "gpqa_diamond": _process_gpqa,
    "codeforces": lambda: _process_codeforces("sft"),
    "codeforces_sft": lambda: _process_codeforces("sft"),
    "codeforces_rl": lambda: _process_codeforces("rl"),
    "lb2": _process_lb2,
    "longbench_v2": _process_lb2,
}


# =============================================================================
# Public API
# =============================================================================

def load_dataset(dataset_name: str) -> list[dict]:
    """Load and process a dataset directly from HuggingFace.
    
    Returns list of items with keys: prompt, final_answer, metadata
    """
    if dataset_name not in _DATASET_PROCESSORS:
        available = ", ".join(sorted(_DATASET_PROCESSORS.keys()))
        raise ValueError(f"Unknown dataset: {dataset_name}. Available: {available}")
    
    return _DATASET_PROCESSORS[dataset_name]()


def load_datasets(dataset_names: list[str]) -> tuple[list[dict], list[str]]:
    """Load multiple datasets and prepare src_items for evaluation.
    
    Returns: (src_items, loaded_dataset_names)
    src_items format: {"dataset": str, "prompt": str, "final_answer": str, "micro_index": int, "metadata": dict}
    """
    src_items = []
    loaded = []
    
    for name in dataset_names:
        data = load_dataset(name)
        for i, item in enumerate(data):
            src_items.append({
                "dataset": name,
                "prompt": item["prompt"],
                "final_answer": item["final_answer"],
                "micro_index": i,
                "metadata": item.get("metadata", {})
            })
        loaded.append(name)
        print(f"Loaded {len(data)} items from {name}")
    
    return src_items, loaded


def list_datasets() -> list[str]:
    """Return list of available dataset names."""
    return sorted(_DATASET_PROCESSORS.keys())
