from __future__ import annotations

import os
import torch
from torch import nn
from typing import TYPE_CHECKING

from .modeling_qwen3_5 import Qwen3_5RMSNorm

if TYPE_CHECKING:
    from .configuration_qwen3_5 import Qwen3_5TextConfig


# ---------------------------------------------------------------------------
# Confidence head: per-position commit/skip predictor for MTP token2
# ---------------------------------------------------------------------------

class ConfidenceHead(nn.Module):
    """Predict a per-position confidence ``c ∈ [0, 1]`` for MTP ``token2``.

    Used at inference (SGLang Phase 2) to gate ``token2`` commit.
    The supervision target is the **EAGLE per-pair accept rate** of ``token2``
    under the verify model (vanilla teacher):

      * sampled-KL OPD path → per-token accept rate
        ``α(y) = min(1, p_t(y) / p_s(y))`` evaluated at the sampled label.
      * full-vocab / topk JSD OPD path → expected accept rate at this position
        ``Σ_y min(p_s(y), p_t(y)) = 1 − TV(p_s, p_t)`` (over the candidate
        subset under topk mode).

    See :meth:`pipo.trainer.swift_gkd_trainer.PIPOGKDTrainer._accumulate_conf_chunk`
    for the construction. ``sigmoid(ConfHead)`` is consumed at serving time
    as the per-pair accept probability — high → commit ``token2``, low →
    replace with PAD.

    Architecture (intentionally tiny — ~``H² + H/2`` params, ~6.6M for 4B):
        ``RMSNorm(2H) → Linear(2H, H) → SiLU → Linear(H, 1)``  → logit (scalar)

    Inputs are ``(backbone_hidden, mtp_hidden)`` both shaped ``[..., H]`` and
    ALWAYS detached upstream by the trainer — the head is a post-hoc evaluator
    and must not push gradients back into the MTP / backbone representations
    (would induce a degenerate solution where the MTP head sacrifices token2
    quality to make itself easier to predict-confidence on).

    Output is a raw logit; downstream apply ``sigmoid`` for inference threshold
    comparison or ``BCEWithLogitsLoss`` for training.
    """

    def __init__(self, config: Qwen3_5TextConfig):
        super().__init__()
        H = config.hidden_size
        self.norm = Qwen3_5RMSNorm(2 * H, eps=config.rms_norm_eps)
        self.fc1 = nn.Linear(2 * H, H, bias=False)
        self.act = nn.SiLU()
        self.fc2 = nn.Linear(H, 1, bias=False)

    def forward(
        self,
        backbone_hidden: torch.Tensor,
        mtp_hidden: torch.Tensor,
    ) -> torch.Tensor:
        """Return per-position confidence logits with the leading dims of inputs.

        Args:
            backbone_hidden: ``[..., H]`` — the compressed backbone hidden state at
                the pair containing token1 (Phase 1 output before MTP fusion).
            mtp_hidden: ``[..., H]`` — the post-norm MTP block output (Phase 2
                hidden, the same tensor that produces ``logits2`` via ``lm_head``).

        Returns:
            ``[...]`` (one fewer dim than inputs) — raw logits; apply sigmoid for
            probabilities. Caller is responsible for detaching the inputs if the
            head should not influence upstream representations (the training path
            in ``PIPOGKDTrainer`` always detaches; the SGLang inference path
            runs under ``torch.no_grad`` so detach is implicit).
        """
        x = torch.cat([backbone_hidden, mtp_hidden], dim=-1)
        x = self.norm(x)
        x = self.fc1(x)
        x = self.act(x)
        return self.fc2(x).squeeze(-1)


class PIPOCompressor(nn.Module):
    """Base class for token-pair compressors: ``[..., 2T, H] → [..., T, ...]``.

    Subclasses must implement :meth:`forward` and optionally :meth:`init_weights`.

    The ``forward()`` signature accepts ``**kwargs`` so that future compressor
    variants can receive extra inputs (e.g. token probabilities) without
    breaking the call-site.  Input tensors may be 2-D ``[num_tokens, H]``
    (SGLang packed format) or 3-D ``[B, T, H]`` (HuggingFace batched format);
    both must be supported.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

    def forward(self, embeds: torch.Tensor, **kwargs) -> torch.Tensor:
        raise NotImplementedError

    def init_weights(self):
        """Optional initialization for training. Called during model initialization."""
        pass


class LinearCompressor(PIPOCompressor):
    """Pair-compress via learned linear projection: ``Linear(2H → H, bias=True)``."""

    def __init__(self, config):
        super().__init__(config)
        hidden_size = getattr(config, 'hidden_size', config)
        self.linear = nn.Linear(hidden_size * 2, hidden_size, bias=True)

    def forward(self, embeds: torch.Tensor, **kwargs) -> torch.Tensor:
        if embeds.dim() == 2:
            # SGLang packed: [num_tokens, H]
            T2, H = embeds.shape
            return self.linear(embeds.view(T2 // 2, 2 * H))
        else:
            # HuggingFace batched: [B, T, H]
            B, T, H = embeds.shape
            return self.linear(embeds.reshape(B, T // 2, H * 2))

    def init_weights(self):
        if os.environ.get('PIPO_COMPRESSOR_RANDOM_INIT', 'false').lower() == 'true':
            return
        """Init so that f([e_i; e_{i+1}]) = e_i + e_{i+1} at training start.

        Weight W has shape [H, 2H].  Setting W = [I_H | I_H] and bias = 0 gives
        W @ [e_i; e_{i+1}] = e_i + e_{i+1}, a neutral starting point.
        """
        H = self.linear.weight.shape[0]
        nn.init.zeros_(self.linear.bias)
        with torch.no_grad():
            eye = torch.eye(H, device=self.linear.weight.device, dtype=self.linear.weight.dtype)
            self.linear.weight.copy_(torch.cat([eye, eye], dim=1))


# This class is discarded in early ablation
class GatedCompressor(PIPOCompressor):
    """Gated pair-compressor: ``α·e_i + β·e_{i+1} + δ(e_i − e_{i+1})``."""

    def __init__(self, config):
        super().__init__(config)
        hidden_size = getattr(config, 'hidden_size', config)
        self.alpha = nn.Linear(hidden_size, 1, bias=True)
        self.beta = nn.Linear(hidden_size, 1, bias=True)
        self.delta = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, embeds: torch.Tensor, **kwargs) -> torch.Tensor:
        if embeds.dim() == 2:
            # SGLang packed: [num_tokens, H] → split even/odd
            e_i = embeds[0::2, :]
            e_ip1 = embeds[1::2, :]
        else:
            # HuggingFace batched: [B, T, H]
            e_i = embeds[:, 0::2, :]
            e_ip1 = embeds[:, 1::2, :]
        a = torch.sigmoid(self.alpha(e_i))
        b = torch.sigmoid(self.beta(e_ip1))
        return a * e_i + b * e_ip1 + self.delta(e_i - e_ip1)

    def init_weights(self):
        """Init biases to 0 → sigmoid gates start at ~0.5, giving f ≈ 0.5*(e_i + e_{i+1})."""
        nn.init.zeros_(self.alpha.bias)
        nn.init.zeros_(self.beta.bias)


class MLPCompressor(PIPOCompressor):
    """MLP-based compressor with SiLU activation."""

    def __init__(self, config):
        super().__init__(config)
        hidden_size = getattr(config, 'hidden_size', config)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size * 2, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size * 2, hidden_size, bias=True),
        )

    def forward(self, embeds: torch.Tensor, **kwargs) -> torch.Tensor:
        if embeds.dim() == 2:
            # SGLang packed: [num_tokens, H]
            T2, H = embeds.shape
            return self.mlp(embeds.view(T2 // 2, 2 * H))
        else:
            # HuggingFace batched: [B, T, H]
            B, T, H = embeds.shape
            return self.mlp(embeds.reshape(B, T // 2, H * 2))

    def init_weights(self):
        if os.environ.get('PIPO_COMPRESSOR_RANDOM_INIT', 'false').lower() == 'true':
            return
        """Initialize MLP to approximate identity (sum of two halves)."""
        l1 = self.mlp[0]
        l2 = self.mlp[2]
        h = l2.weight.shape[0]  # output dim = hidden_size

        with torch.no_grad():
            # First layer: pass through both halves
            l1.weight.zero_()
            l1.bias.zero_()
            l1.weight.copy_(torch.eye(2 * h))

            # Second layer: sum the two halves
            l2.weight.zero_()
            l2.bias.zero_()
            l2.weight[:, :h] = torch.eye(h)
            l2.weight[:, h:] = torch.eye(h)


# Registry for compressor types — extend here to add new variants.
COMPRESSOR_REGISTRY: dict[str, type[PIPOCompressor]] = {
    "linear": LinearCompressor,
    "mlp": MLPCompressor,
}


def get_compressor(config, compressor_type: str = None) -> PIPOCompressor:
    """Factory function to create a compressor instance.
    
    Args:
        config: Model config with hidden_size attribute
        compressor_type: Type of compressor (linear, mlp). 
                        If None, reads from config.compressor_type or defaults to "linear".
    
    Returns:
        Instantiated compressor
    """
    if compressor_type is None:
        compressor_type = getattr(config, "compressor_type", "linear")
    
    compressor_cls = COMPRESSOR_REGISTRY.get(compressor_type)
    if compressor_cls is None:
        raise ValueError(
            f"Unknown compressor_type={compressor_type!r}. "
            f"Choose from: {sorted(COMPRESSOR_REGISTRY)}"
        )
    return compressor_cls(config)
