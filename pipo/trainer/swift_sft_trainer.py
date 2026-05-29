from __future__ import annotations

import os
from typing import Any, Optional

import torch
import torch.nn.functional as F
from swift.trainers.seq2seq_trainer import Seq2SeqTrainer
from swift.rlhf_trainers.utils import aggressive_empty_cache


N_BOT_TOKENS = 2


class PIPOSeq2SeqTrainer(Seq2SeqTrainer):
    """Seq2SeqTrainer with PIPO-specific loss logic (random PAD insertion, per-component logging)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        args = self.args
        # ``max_pad_ratio`` is the UPPER BOUND of the per-step pad-ratio
        # sampling distribution. Each forward draws
        # ``pad_ratio ~ Uniform[0, max_pad_ratio]`` so the model sees the full
        # density spectrum it might encounter at inference time. Legacy users
        # of ``PIPO_PAD_RATIO`` (fixed value) are auto-migrated: that
        # env-var, if set, becomes the new ``max_pad_ratio`` upper bound (we
        # log a deprecation note on the rank-0 logger via Trainer.log).
        if not hasattr(args, "max_pad_ratio"):
            legacy = os.environ.get("PIPO_PAD_RATIO")
            if legacy is not None:
                args.max_pad_ratio = float(legacy)
            else:
                args.max_pad_ratio = float(
                    os.environ.get("PIPO_MAX_PAD_RATIO", "0")
                )
        if not hasattr(args, "mtp_loss_weight"):
            args.mtp_loss_weight = float(os.environ.get("MTP_LOSS_WEIGHT", "1.0"))

    # ------------------------------------------------------------------
    # compute_loss — main entry point
    # ------------------------------------------------------------------

    def compute_loss(
        self,
        model: torch.nn.Module,
        inputs: dict,
        return_outputs: bool = False,
        num_items_in_batch: Optional[int] = None,
        **kwargs: Any,
    ):
        # Strip swift-specific keys that the model doesn't accept.
        inputs = dict(inputs)
        for k in (
            "compute_loss_func",
            "loss_scale",
            "text_position_ids",
            "channel",
            "logits_to_keep",
        ):
            inputs.pop(k, None)

        # --- Mask begin-of-thinking tokens (<think>\n) from supervision ---
        # Find the last -100 in labels and set the following N_BOT_TOKENS labels
        # to -100 as well, so that <think>\n are not supervised.
        labels: torch.Tensor = inputs["labels"]  # [1, T]
        mask_positions = (labels[0] == -100).nonzero(as_tuple=True)[0]
        if mask_positions.numel() > 0:
            last_neg = mask_positions[-1].item()
            end = min(last_neg + 1 + N_BOT_TOKENS, labels.shape[1])
            if end > last_neg + 1:
                labels = labels.clone()
                labels[0, last_neg + 1 : end] = -100
                inputs["labels"] = labels

        max_pad_ratio = float(getattr(self.args, "max_pad_ratio", 0))
        assert max_pad_ratio >= 0, (
            f"max_pad_ratio must be >= 0; got {max_pad_ratio}"
        )
        if max_pad_ratio > 0:
            pad_ratio = float(torch.rand(()).item()) * max_pad_ratio
        else:
            pad_ratio = 0.0

        # --- Random PAD insertion (pad_ratio > 0) -------------------------
        input_ids: torch.Tensor = inputs["input_ids"]  # [1, T]
        labels = inputs["labels"]  # [1, T] (possibly updated above)

        # Build padded input_ids and labels with random PAD insertion.
        new_input_ids, new_labels, n_pads = self._build_random_padded_inputs(
            input_ids, labels, pad_ratio
        )

        # Forward with modified inputs.
        inputs["input_ids"] = new_input_ids
        inputs["labels"] = new_labels
        inputs.pop("attention_mask", None)  # Let the model rebuild attention mask if needed.
        outputs = model(**inputs)
        loss = outputs.loss
        self._log_component_losses(outputs, prefix="")
        if (
            outputs.loss is not None
            and self.state.global_step % max(self.args.logging_steps, 1) == 0
        ):
            self.log({
                "pipo/pad_ratio_step": round(pad_ratio, 4),
                "pipo/n_pads_step": float(n_pads),
            })

        return (loss, outputs) if return_outputs else loss

    def _build_random_padded_inputs(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
        pad_ratio: float,
    ) -> tuple[torch.Tensor, torch.Tensor, int]:
        """Build new input_ids and labels with random PAD insertion at token2 positions.

        **Prompt even-length guarantee**: The prompt part (contiguous leading
        tokens where label == -100) must have even length so that pair boundaries
        align with the prompt/generation boundary.  If the prompt is odd-length,
        a PAD token (label=-100) is appended at the end of the prompt before
        pairing.

        **Total even-length guarantee**: The total sequence length is also padded
        to be even.
        
        All PAD tokens in the resulting sequence will be at odd indices (the second
        token of a pair).

        For each token pair ``(t_{2p}, t_{2p+1})`` in the **generation** part
        (both tokens have label != -100), the pair is eligible for PAD insertion.
        Among eligible pairs, ``pad_ratio`` fraction are randomly selected and
        split: ``{t_{2p}, t_{2p+1}}`` → ``{t_{2p}, PAD}, {t_{2p+1}, PAD}``.
        Labels at PAD positions are ``-100``.

        All operations are vectorized (no Python per-token for-loops).

        Args:
            input_ids: ``[1, T]`` original token ids.
            labels: ``[1, T]`` original labels.
            pad_ratio: fraction of eligible pairs to split (0.0–1.0).

        Returns:
            ``(new_input_ids, new_labels, n_pads)`` where
            new shapes are ``[1, T']`` with ``T' >= T``.
        """
        ids = input_ids[0]   # [T]
        labs = labels[0]     # [T]
        device = ids.device
        pad_id = self.tokenizer.pad_token_id

        T = ids.shape[0]

        pad_id_t = torch.tensor([pad_id], dtype=ids.dtype, device=device)
        neg_label = torch.tensor([-100], dtype=labs.dtype, device=device)

        # --- Step 1: Ensure prompt has even length ---
        non_prompt = (labs != -100).nonzero(as_tuple=True)[0]
        if non_prompt.numel() > 0:
            prompt_len = non_prompt[0].item()
        else:
            prompt_len = T

        if prompt_len % 2 != 0:
            # Insert a PAD token at position prompt_len (end of prompt).
            ids = torch.cat([ids[:prompt_len], pad_id_t, ids[prompt_len:]], dim=0)
            labs = torch.cat([labs[:prompt_len], neg_label, labs[prompt_len:]], dim=0)
            T += 1

        # --- Step 2: Ensure total length is even ---
        if T % 2 != 0:
            ids = torch.cat([ids, pad_id_t], dim=0)
            labs = torch.cat([labs, neg_label], dim=0)
            T += 1

        # --- Step 3: Pair up and identify eligible generation pairs ---
        n_pairs = T // 2

        # Eligible pair: token2 (odd-indexed element) has label != -100.
        # Vectorized: check labs[1::2] != -100
        eligible_mask = labs[1::2] != -100  # [n_pairs]

        n_eligible = eligible_mask.sum().item()
        if n_eligible == 0:
            return ids.unsqueeze(0), labs.unsqueeze(0), 0

        # --- Step 4: Randomly select pairs to split ---
        n_to_split = max(1, int(round(n_eligible * pad_ratio)))
        eligible_indices = eligible_mask.nonzero(as_tuple=True)[0]
        perm = torch.randperm(n_eligible, device=device)[:n_to_split]
        split_indices = eligible_indices[perm]

        split_mask = torch.zeros(n_pairs, dtype=torch.bool, device=device)
        split_mask[split_indices] = True

        # --- Step 5: Build the new sequence (vectorized) ---
        # Reshape ids/labs into pairs: [n_pairs, 2]
        ids_pairs = ids.view(n_pairs, 2)
        labs_pairs = labs.view(n_pairs, 2)

        # For non-split pairs: output 2 tokens  (t1, t2)
        # For split pairs:     output 4 tokens  (t1, PAD, t2, PAD)
        
        out_ids = torch.full((n_pairs, 4), pad_id, dtype=ids.dtype, device=device)
        out_labs = torch.full((n_pairs, 4), -100, dtype=labs.dtype, device=device)

        not_split = ~split_mask

        # Non-split rows: place t1 at col0, t2 at col1
        out_ids[not_split, 0] = ids_pairs[not_split, 0]
        out_ids[not_split, 1] = ids_pairs[not_split, 1]
        out_labs[not_split, 0] = labs_pairs[not_split, 0]
        out_labs[not_split, 1] = labs_pairs[not_split, 1]

        # Split rows: col0=t1, col1=PAD, col2=t2, col3=PAD
        out_ids[split_mask, 0] = ids_pairs[split_mask, 0]
        out_ids[split_mask, 2] = ids_pairs[split_mask, 1]
        out_labs[split_mask, 0] = labs_pairs[split_mask, 0]
        out_labs[split_mask, 2] = labs_pairs[split_mask, 1]

        valid = torch.zeros(n_pairs, 4, dtype=torch.bool, device=device)
        valid[:, 0] = True
        valid[:, 1] = True
        valid[split_mask, 2] = True
        valid[split_mask, 3] = True

        new_ids = out_ids[valid]   # [T']
        new_labs = out_labs[valid]  # [T']

        return new_ids.unsqueeze(0), new_labs.unsqueeze(0), int(n_to_split)

    def _log_component_losses(self, outputs, prefix: str = "") -> None:
        if outputs.loss is None:
            return
        if self.state.global_step % max(self.args.logging_steps, 1) != 0:
            return

        logs = {}
        if getattr(outputs, "backbone_loss", None) is not None:
            logs[f"{prefix}backbone_loss"] = outputs.backbone_loss.detach().item()
        if getattr(outputs, "mtp_loss", None) is not None:
            logs[f"{prefix}mtp_loss"] = outputs.mtp_loss.detach().item()
        if getattr(outputs, "conf_loss", None) is not None:
            # SFT-stage ConfidenceHead BCE (target = p_s_mtp(label_token2),
            # i.e. EAGLE per-pair accept rate under the deterministic-teacher
            # assumption). Surfaced unweighted so users can read it
            # independently of ``SFT_CONF_LOSS_WEIGHT``.
            logs[f"{prefix}conf_loss"] = outputs.conf_loss.detach().item()
            for attr, name in (
                ("conf_mean_pred", "conf/mean_pred"),
                ("conf_mean_target", "conf/mean_target"),
                ("conf_commit_rate_0_5", "conf/commit_rate_0.5"),
                ("conf_commit_rate_0_7", "conf/commit_rate_0.7"),
                ("conf_commit_rate_0_8", "conf/commit_rate_0.8"),
                ("conf_commit_rate_0_9", "conf/commit_rate_0.9"),
                ("conf_commit_rate_0_95", "conf/commit_rate_0.95"),
                ("conf_num_valid", "conf/num_valid"),
            ):
                value = getattr(outputs, attr, None)
                if value is not None:
                    logs[f"{prefix}{name}"] = float(value)
        if logs:
            self.log(logs)

    def training_step(self, model, inputs, *args, **kwargs):
        with self.template.forward_context(self.model, inputs):
            out = super().training_step(model, inputs, *args, **kwargs)
        # aggressive_empty_cache()
        return out
