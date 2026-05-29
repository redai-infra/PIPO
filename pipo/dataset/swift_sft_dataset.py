"""Datasets for PIPO: pre-training (packed) and SFT (no packing, no loss on prompt).
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pyarrow.parquet as pq
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast

logger = logging.getLogger(__name__)


class SFTDataset(Dataset):
    """SFT dataset: one sample per row, no packing, labels = -100 on prompt (user input).

    Expects parquet with columns: ``text``, ``prompt_length``. Optional ``total_length`` (ignored).
    Loss is only on the assistant response tokens.
    """

    def __init__(
        self,
        parquet_path: str,
        tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
        max_length: int = 2048,
        truncation_side: str = "right",
    ):
        table = pq.read_table(parquet_path)
        self.texts = table.column("text").to_pylist()
        self.prompt_lengths = table.column("prompt_length").to_pylist()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.truncation_side = truncation_side  # "right" = keep prompt, truncate response end
        logger.info("Loaded %d SFT samples from %s (max_length=%d)", len(self.texts), parquet_path, max_length)

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        text = self.texts[idx]
        prompt_len = int(self.prompt_lengths[idx])
        encoded = self.tokenizer(
            text,
            add_special_tokens=False,
            return_attention_mask=False,
            truncation=True,
            max_length=self.max_length,
            truncation_side=self.truncation_side,
        )
        ids = encoded["input_ids"]
        input_ids = torch.tensor(ids, dtype=torch.long)
        labels = input_ids.clone()
        # No supervision on prompt (user input)
        n_prompt = min(prompt_len, len(ids))
        labels[:n_prompt] = -100
        return {"input_ids": input_ids, "labels": labels}
