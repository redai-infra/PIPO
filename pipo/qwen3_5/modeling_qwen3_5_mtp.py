"""Qwen3.5 Pair-In Pari-Out (PIPO): compressed backbone + MTP.

Training path (compressed)::

    tokens [B, T]
      ↓ embed_tokens
    embeddings [B, T, H]
      ↓ view pairs → [B, T//2, 2H]
      ↓ compressor  Linear(2H → H)
    compressed [B, L, H]   (L = T//2)
      ↓ backbone (32 decoder layers, causal attention)
    hidden [B, L, H]
      ├─ lm_head → logits[i]  predicts t_{2i+2}   (backbone CE)
      └─ MTP(hidden[i], embed(t_{2i+2})) → logits'[i]  predicts t_{2i+3}  (MTP CE)

Generation path (uncompressed) — standard autoregressive decoding.

Generation path (compressed prompt forward):

- If prompt length T is even, run the backbone on pair-compressed embeddings and
  return the backbone logits for the next token ``t_T``.
- If prompt length T is odd, run the backbone on the first ``T-1`` tokens (even),
  then run MTP on the last backbone hidden state fused with the last token embedding
  to return logits for the next token ``t_T``.

This keeps HuggingFace generation compatibility by always returning *one-step*
next-token logits, while exploiting compression on the prompt when possible.

Architecture::

    Qwen3_5ForCausalPIPO
    ├── model           Qwen3_5TextModel  (32 decoder layers)
    │   └── embed_tokens   (used by both backbone and MTP via inputs_embeds)
    ├── compressor      Linear(2H → H)
    ├── lm_head
    └── mtp             MTP predictor
        ├── fc           Linear(2H → H)
        ├── pre_fc_norm_{hidden,embedding}
        ├── layers       1 full-attention decoder layer
        ├── norm
        └── rotary_emb
"""

import copy
import os
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from transformers.cache_utils import Cache, DynamicCache
from transformers.generation import GenerationMixin
from transformers.masking_utils import create_causal_mask
from transformers.modeling_outputs import ModelOutput
from transformers.utils import logging

from .configuration_qwen3_5 import Qwen3_5TextConfig
from .modeling_qwen3_5 import (
    Qwen3_5DecoderLayer,
    Qwen3_5PreTrainedModel,
    Qwen3_5RMSNorm,
    Qwen3_5TextModel,
    Qwen3_5TextRotaryEmbedding,
)
from .compressor import (
    PIPOCompressor,
    LinearCompressor,
    GatedCompressor,
    MLPCompressor,
    COMPRESSOR_REGISTRY,
    ConfidenceHead,
)

logger = logging.get_logger(__name__)

# ms-swift / HF trainers pass these for the *uncompressed* token length; the backbone
# here sees pair-compressed sequences [B, L, H] with L = T//2. Forwarding them causes
# shape mismatches and CUDA asserts inside masking / RoPE.
_COMPRESSED_BACKBONE_IGNORE_KWARGS = frozenset({
    "position_ids",
    "text_position_ids",
    "attention_mask",
    "cache_position",
    "token_type_ids",
})

# When training at long context (e.g. 64K compressed tokens), materializing the
# full ``[B, L, V]`` logits tensor for ``lm_head(hidden)`` is the dominant memory
# cost (Qwen3.5 has V=248320, so [1,64K,V] in bf16 is ~31.7 GB and the implicit
# fp32 upcast inside ``F.cross_entropy`` adds another ~63 GB — for *each* of
# backbone and MTP heads, easily 100+ GB peak).  The chunk size below caps the
# transient logits buffer to roughly ``LOGITS_CHUNK_SIZE × V`` floats per chunk.
LOGITS_CHUNK_SIZE = int(os.environ.get("PIPO_LOGITS_CHUNK", "2048"))


def _chunked_linear_cross_entropy(
    hidden_states: torch.Tensor,
    targets: torch.Tensor,
    lm_head: nn.Linear,
    ignore_index: int = -100,
    chunk_size: int = LOGITS_CHUNK_SIZE,
    return_log_p_at_label: bool = False,
):
    """Compute ``F.cross_entropy(lm_head(hidden_states), targets, ignore_index, reduction='mean')``
    in chunks along the token dimension to avoid materializing the full ``[N, V]``
    logits tensor (and its implicit fp32 copy inside ``F.cross_entropy``).

    Numerically equivalent (up to summation order) to the un-chunked path; gradients
    flow through ``hidden_states`` and ``lm_head.weight`` in the standard way (chunked
    backward accumulates ``lm_head.weight.grad`` automatically).

    Args:
        hidden_states: ``[..., H]`` features (typically ``[B, L, H]`` or ``[N, H]``).
        targets: ``[...]`` integer labels (must broadcast to the leading dims of
            ``hidden_states``); ``ignore_index`` positions are ignored in the mean.
        lm_head: language-model head ``Linear(H -> V)``.
        ignore_index: label value to skip (default ``-100``).
        chunk_size: number of tokens per chunk; transient logits buffer is
            ``chunk_size × V``.  Smaller = less memory, more Python overhead.
        return_log_p_at_label: if True, additionally return a flat
            ``[N_total]`` ``fp32`` tensor of ``log p_s(target)`` per token,
            **detached** from the autograd graph. Positions where
            ``target == ignore_index`` carry a ``0.0`` placeholder that the
            downstream consumer must mask out (``ignore_index`` would not be
            a valid vocabulary index for ``gather`` anyway). The intent is
            to let consumers like :func:`_chunked_conf_loss` reuse the
            per-token student probability **without a second lm-head
            forward** — at V≈248K the lm-head is the dominant cost in the
            chunk loop, so piggy-backing this here is a free ~2× speed-up
            on the SFT conf path. When False (default), this path is
            inactive and behaviour is identical to the original
            implementation.

    Returns:
        * ``return_log_p_at_label=False`` (default): scalar loss tensor on
          ``hidden_states.dtype``; ``0.0`` if no valid targets.
        * ``return_log_p_at_label=True``: ``(loss, log_p_at_label_flat)``
          where ``log_p_at_label_flat`` has shape ``[N_total]`` (fp32,
          detached, ``0.0`` at ignore positions).
    """
    flat_hidden = hidden_states.reshape(-1, hidden_states.shape[-1])
    flat_targets = targets.reshape(-1)
    n_tokens = flat_hidden.shape[0]

    n_valid = (flat_targets != ignore_index).sum()
    if n_valid == 0:
        # Multiply by hidden_states.sum() * 0 to keep autograd connected (so DDP
        # doesn't complain about parameters with no grad in this rank).
        zero_loss = flat_hidden.sum() * 0.0
        if return_log_p_at_label:
            return zero_loss, flat_hidden.new_zeros(n_tokens, dtype=torch.float32)
        return zero_loss

    total_loss = hidden_states.new_zeros((), dtype=torch.float32)
    if return_log_p_at_label:
        # Pre-allocate a flat fp32 tensor to collect detached per-token gold
        # log-prob across chunks. fp32 matches log_softmax's internal
        # precision; downstream consumers cast as needed.
        log_p_at_label = flat_hidden.new_zeros(n_tokens, dtype=torch.float32)

    for start in range(0, n_tokens, chunk_size):
        end = min(start + chunk_size, n_tokens)
        h_chunk = flat_hidden[start:end]
        t_chunk = flat_targets[start:end]
        logits_chunk = lm_head(h_chunk)

        if return_log_p_at_label:
            # Manual log_softmax + nll_loss decomposition — PyTorch's
            # ``F.cross_entropy`` impl is the SAME decomposition, so this is
            # numerically identical (up to summation order) to the else
            # branch. The only addition is that we keep ``log_probs`` long
            # enough to gather the per-token gold log-prob, then discard it.
            # Memory envelope is unchanged: autograd already had to retain
            # the ``log_probs`` tensor under ``F.cross_entropy`` for the
            # backward pass.
            log_probs = F.log_softmax(logits_chunk, dim=-1)               # [chunk, V]
            total_loss = total_loss + F.nll_loss(
                log_probs, t_chunk,
                ignore_index=ignore_index, reduction="sum",
            )
            with torch.no_grad():
                safe_t = t_chunk.clamp(min=0).unsqueeze(-1)               # [chunk, 1]
                log_p_chunk = log_probs.gather(-1, safe_t).squeeze(-1).float()
                log_p_chunk = log_p_chunk.masked_fill(
                    t_chunk == ignore_index, 0.0,
                )
                log_p_at_label[start:end] = log_p_chunk
        else:
            # F.cross_entropy already upcasts internally; reduction='sum' lets us
            # divide once at the end so chunk boundaries don't bias the mean.
            total_loss = total_loss + F.cross_entropy(
                logits_chunk,
                t_chunk,
                ignore_index=ignore_index,
                reduction="sum",
            )

    loss = (total_loss / n_valid.to(total_loss.dtype)).to(hidden_states.dtype)
    if return_log_p_at_label:
        return loss, log_p_at_label
    return loss


def _chunked_conf_loss(
    backbone_hidden: torch.Tensor,
    mtp_hidden: torch.Tensor,
    labels: torch.Tensor,
    confidence_head: nn.Module,
    conf_target_flat: torch.Tensor,
    detach_inputs: bool = True,
    ignore_index: int = -100,
    chunk_size: int = LOGITS_CHUNK_SIZE,
    return_metrics: bool = False,
) -> torch.Tensor | Tuple[torch.Tensor, dict[str, float]]:
    """SFT-stage ConfidenceHead BCE — target is **caller-provided**.

    Under the *deterministic teacher* assumption (i.e. the SFT label is the
    teacher's argmax, ``p_t = δ_{y_t}``), the EAGLE per-pair accept rate
    reduces to::

        AR_pos  =  Σ_y min(p_s(y), p_t(y))  =  min(p_s(y_t), 1)  =  p_s(y_t)

    so the BCE target is ``p_s_mtp(label_token2)``. This matches the
    OPD-stage conf target in the deterministic-teacher limit
    (``min(1, p_t/p_s)`` and ``Σ min(p_s, p_t)`` both collapse to the same
    quantity), so a head trained here transfers seamlessly into OPD without
    a parameter reset.

    The target tensor is provided by the caller so we **avoid a second
    lm-head forward** here: ``_chunked_linear_cross_entropy(...,
    return_log_p_at_label=True)`` already exposes ``log p_s(label)`` as a
    detached side output, so the caller computes
    ``conf_target = log_p_at_label.exp().clamp(0,1).detach()`` once and
    feeds it in. At V≈248K the lm-head is the dominant cost in any chunk
    loop, so piggy-backing the target derivation on the MTP CE path is a
    free ~2× speed-up on the SFT conf path.

    Chunking here is purely for confidence-head forward isolation (the head
    itself outputs ``[chunk]`` so its peak is small); we keep it for
    consistency with ``_chunked_linear_cross_entropy``.

    Args:
        backbone_hidden: ``[B, L-1, H]`` — pre-MTP backbone hidden at each
            pair containing token1.
        mtp_hidden: ``[B, L-1, H]`` — post-norm MTP block output (the same
            tensor that produces ``logits2`` via ``lm_head``).
        labels: ``[B, L-1]`` — token2 ground-truth ids; ``-100`` skipped.
        confidence_head: trained ``ConfidenceHead`` instance.
        conf_target_flat: ``[B*(L-1)]`` precomputed BCE target in ``[0, 1]``
            (already detached). Typically piggy-backed off the MTP CE
            forward via ``log_p_at_label.exp().clamp(0, 1).detach()``.
            Positions corresponding to ``ignore_index`` labels are
            irrelevant (masked out below).
        detach_inputs: detach ``(backbone_hidden, mtp_hidden)`` before the
            conf head — guards against the degenerate solution where the
            MTP head sacrifices token2 quality to game its own confidence.
        ignore_index: label value to skip (default ``-100``).
        chunk_size: tokens per chunk; tune via ``PIPO_LOGITS_CHUNK``.

    Returns:
        Scalar BCE loss (mean over valid positions), in the same dtype as
        ``backbone_hidden``. Returns a 0-grad ghost loss when no valid
        positions exist on this rank (preserves DDP/ZeRO grad-bucket
        alignment so the conf head's params always appear in the graph).
        If ``return_metrics=True``, also returns detached confidence
        calibration metrics for logging.
    """
    flat_back = backbone_hidden.reshape(-1, backbone_hidden.shape[-1])
    flat_mtp = mtp_hidden.reshape(-1, mtp_hidden.shape[-1])
    flat_lbl = labels.reshape(-1)
    n_tokens = flat_back.shape[0]

    assert conf_target_flat.shape == (n_tokens,), (
        f"conf_target_flat shape {tuple(conf_target_flat.shape)} must match "
        f"flattened token count {n_tokens}."
    )

    valid = (flat_lbl != ignore_index)
    n_valid = valid.sum()
    metrics: dict[str, float] = {
        "conf_mean_pred": 0.0,
        "conf_mean_target": 0.0,
        "conf_commit_rate_0.5": 0.0,
        "conf_commit_rate_0.7": 0.0,
        "conf_commit_rate_0.8": 0.0,
        "conf_commit_rate_0.9": 0.0,
        "conf_commit_rate_0.95": 0.0,
        "conf_num_valid": 0.0,
    }
    if n_valid == 0:
        # Zero-grad ghost: keep both hidden tensors AND confidence_head's
        # params in the autograd graph so DDP's all-reduce buckets are
        # aligned across ranks (mirrors the OPD trainer's _finalise_conf).
        ghost = (flat_back.sum() + flat_mtp.sum()) * 0.0
        dummy_b = flat_back[:1].detach() * 0.0
        dummy_m = flat_mtp[:1].detach() * 0.0
        ghost = ghost + (confidence_head(dummy_b, dummy_m).sum() * 0.0)
        loss = ghost.to(backbone_hidden.dtype)
        return (loss, metrics) if return_metrics else loss

    total_loss = backbone_hidden.new_zeros((), dtype=torch.float32)
    pred_sum = 0.0
    target_sum = 0.0
    commit_counts = {0.5: 0, 0.7: 0, 0.8: 0, 0.9: 0, 0.95: 0}
    valid_count = 0
    for start in range(0, n_tokens, chunk_size):
        end = min(start + chunk_size, n_tokens)
        b_chunk = flat_back[start:end]
        m_chunk = flat_mtp[start:end]
        l_chunk = flat_lbl[start:end]
        chunk_mask = (l_chunk != ignore_index)
        if not chunk_mask.any():
            continue

        # ── Conf head forward (no lm_head — target already provided) ──
        if detach_inputs:
            conf_logit = confidence_head(b_chunk.detach(), m_chunk.detach())
        else:
            conf_logit = confidence_head(b_chunk, m_chunk)               # [chunk]

        cl = conf_logit[chunk_mask]
        ct = conf_target_flat[start:end][chunk_mask].to(cl.dtype)
        total_loss = total_loss + F.binary_cross_entropy_with_logits(
            cl, ct, reduction="sum",
        )

        if return_metrics:
            with torch.no_grad():
                pred = torch.sigmoid(cl.float())
                target = ct.float()
                pred_sum += float(pred.sum().item())
                target_sum += float(target.sum().item())
                valid_count += int(pred.numel())
                for thr in commit_counts:
                    commit_counts[thr] += int((pred >= thr).sum().item())

    loss = (total_loss / n_valid.to(total_loss.dtype)).to(backbone_hidden.dtype)
    if return_metrics:
        denom = max(valid_count, 1)
        metrics = {
            "conf_mean_pred": pred_sum / denom,
            "conf_mean_target": target_sum / denom,
            "conf_commit_rate_0.5": commit_counts[0.5] / denom,
            "conf_commit_rate_0.7": commit_counts[0.7] / denom,
            "conf_commit_rate_0.8": commit_counts[0.8] / denom,
            "conf_commit_rate_0.9": commit_counts[0.9] / denom,
            "conf_commit_rate_0.95": commit_counts[0.95] / denom,
            "conf_num_valid": float(valid_count),
        }
        return loss, metrics
    return loss


# ---------------------------------------------------------------------------
# Output dataclass
# ---------------------------------------------------------------------------

@dataclass
class Qwen3_5MTPCausalLMOutput(ModelOutput):
    """Output type for :class:`Qwen3_5ForCausalPIPO`.

    Attributes:
        loss: Combined (weighted) backbone + MTP loss (+ conf BCE if active).
        backbone_loss: Backbone-only CE loss (for logging).
        mtp_loss: MTP-only CE loss before weighting (for logging).
        conf_loss: SFT-stage ConfidenceHead BCE before weighting, or
            ``None`` when the head is disabled / its loss weight is 0.
        logits: Backbone next-token logits ``[B, T, V]``.
        mtp_logits: List of *k* tensors, one per MTP head.  Head *i*
            has shape ``[B, T-1-i, V]`` and predicts token ``t+2+i``.
        backbone_hidden_states: Post-norm backbone hidden states
            ``[B, T, H]`` (always returned — useful for downstream heads).
        past_key_values: Backbone KV cache for incremental decoding.
        sampled_token1: First-stage sampled token ids.
        sampled_token2: Second-stage sampled token ids from MTP logits.
        sampled_tokens: Stacked ``[sampled_token1, sampled_token2]``.
    """
    loss: Optional[torch.FloatTensor] = None
    backbone_loss: Optional[torch.FloatTensor] = None
    mtp_loss: Optional[torch.FloatTensor] = None
    conf_loss: Optional[torch.FloatTensor] = None
    conf_mean_pred: Optional[float] = None
    conf_mean_target: Optional[float] = None
    conf_commit_rate_0_5: Optional[float] = None
    conf_commit_rate_0_7: Optional[float] = None
    conf_commit_rate_0_8: Optional[float] = None
    conf_commit_rate_0_9: Optional[float] = None
    conf_commit_rate_0_95: Optional[float] = None
    conf_num_valid: Optional[float] = None
    logits: Optional[torch.FloatTensor] = None
    mtp_logits: Optional[list[torch.FloatTensor]] = None
    backbone_hidden_states: Optional[torch.FloatTensor] = None
    past_key_values: Optional[Cache] = None
    sampled_token1: Optional[torch.LongTensor] = None
    sampled_token2: Optional[torch.LongTensor] = None
    sampled_tokens: Optional[torch.LongTensor] = None


# ---------------------------------------------------------------------------
# MTP predictor (inner module — no lm_head, no backbone)
# ---------------------------------------------------------------------------

class Qwen3_5MultiTokenPredictor(nn.Module):
    """One MTP block: fuse hidden-state + next-token embedding → decoder layer → norm."""

    def __init__(self, config: Qwen3_5TextConfig):
        super().__init__()
        self.config = config
        self.num_mtp_layers = max(getattr(config, "mtp_num_hidden_layers", 1), 1)

        # self.embed_tokens: nn.Embedding = None  # caller passes inputs_embeds directly
        self.fc = nn.Linear(config.hidden_size * 2, config.hidden_size, bias=False)
        self.pre_fc_norm_hidden = Qwen3_5RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.pre_fc_norm_embedding = Qwen3_5RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        mtp_config = copy.deepcopy(config)
        mtp_config.num_hidden_layers = self.num_mtp_layers
        mtp_config.layer_types = ["full_attention"] * self.num_mtp_layers
        self._mtp_config = mtp_config

        self.layers = nn.ModuleList(
            [Qwen3_5DecoderLayer(mtp_config, layer_idx=i) for i in range(self.num_mtp_layers)]
        )

        self.norm = Qwen3_5RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen3_5TextRotaryEmbedding(config=config)

    def forward(
        self,
        # input_ids: Optional[torch.LongTensor] = None,
        hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        # if inputs_embeds is None:
        #     inputs_embeds = self.embed_tokens(input_ids)

        inputs_embeds = self.pre_fc_norm_embedding(inputs_embeds)
        hidden_states = self.pre_fc_norm_hidden(hidden_states)
        hidden_states = torch.cat([inputs_embeds, hidden_states], dim=-1)
        hidden_states = self.fc(hidden_states)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache()

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + hidden_states.shape[1], device=hidden_states.device
            )

        if position_ids is None:
            position_ids = cache_position.view(1, 1, -1).expand(3, hidden_states.shape[0], -1)
        elif position_ids.ndim == 2:
            position_ids = position_ids[None, ...].expand(3, position_ids.shape[0], -1)

        if position_ids.ndim == 3 and position_ids.shape[0] == 4:
            text_position_ids = position_ids[0]
            position_ids = position_ids[1:]
        else:
            text_position_ids = position_ids[0]

        causal_mask = create_causal_mask(
            config=self._mtp_config,
            inputs_embeds=hidden_states,
            attention_mask=attention_mask,
            cache_position=cache_position,
            past_key_values=past_key_values,
            position_ids=text_position_ids,
        )

        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        for layer in self.layers:
            hidden_states = layer(
                hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                cache_position=cache_position,
                **kwargs,
            )

        hidden_states = self.norm(hidden_states)
        return hidden_states


# ---------------------------------------------------------------------------
# Unified model: backbone + MTP + lm_head
# ---------------------------------------------------------------------------
# Note: Compressor classes are imported from .compressor module


class Qwen3_5ForCausalPIPO(Qwen3_5PreTrainedModel, GenerationMixin):
    """Backbone + *k* chained MTP heads in one ``PreTrainedModel``.

    * ``forward(input_ids)`` runs the full backbone, then each MTP head
      in sequence (each predicting one more token ahead).
    * ``from_pretrained(path)`` loads backbone **and** MTP weights in one
      call, sharing ``embed_tokens`` between both.
    """

    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}
    _keys_to_ignore_on_load_unexpected = [r"^model\.visual\..*"]
    config: Qwen3_5TextConfig

    def __init__(self, config: Qwen3_5TextConfig):
        super().__init__(config)

        # ── backbone ──
        self.model = Qwen3_5TextModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # ── input compressor: merge every 2 consecutive token embeddings ──
        # Type is selected via config.compressor_type (default "linear").
        compressor_type = getattr(config, "compressor_type", "linear")
        compressor_cls = COMPRESSOR_REGISTRY.get(compressor_type)
        if compressor_cls is None:
            raise ValueError(
                f"Unknown compressor_type={compressor_type!r}. "
                f"Choose from: {sorted(COMPRESSOR_REGISTRY)}"
            )
        self.compressor: PIPOCompressor = compressor_cls(config)

        # ── MTP head ──
        self.num_mtp_heads = max(getattr(config, "mtp_num_hidden_layers", 1), 1)
        self.mtp = Qwen3_5MultiTokenPredictor(config)

        self.mtp_loss_weight: float = getattr(config, "mtp_loss_weight", 1.0)

        self.confidence_head: ConfidenceHead = ConfidenceHead(config)

        # ── SFT-stage ConfidenceHead supervision ──
        # Current research config keeps the head instantiated and uses
        # ``SFT_CONF_LOSS_WEIGHT=1``.  The SFT target is
        # ``p_s_mtp(label_token2)``: a warm-start calibration signal before OPD
        # trains the serving-aligned sampled-token EAGLE target
        # ``min(1, P_teacher(y) / P_student(y))``.  The OPD trainer uses its
        # own conf-loss path and bypasses model.forward()'s loss calculation, so
        # setting ``SFT_CONF_LOSS_WEIGHT`` while running OPD has no effect.
        #   * ``SFT_CONF_LOSS_WEIGHT`` (float, default 1): BCE weight added to
        #     the combined SFT loss.
        #   * ``SFT_CONF_DETACH_INPUTS`` ('1'/'0', default '1'): detach
        #     ``(backbone_hidden, mtp_hidden)`` before the head — guards
        #     against the degenerate solution where MTP sacrifices token2
        #     quality to game its own confidence prediction.
        self.sft_conf_loss_weight: float = float(
            os.environ.get("SFT_CONF_LOSS_WEIGHT", "1.0")
        )
        self.sft_conf_detach_inputs: bool = (
            os.environ.get("SFT_CONF_DETACH_INPUTS", "1") == "1"
        )

        self.post_init()

    @torch.no_grad()
    def _init_weights(self, module: nn.Module):
        """Delegate to parent for standard modules.

        NOTE: PIPOCompressor is intentionally NOT initialized here.
        HuggingFace calls _init_weights on modules it considers "new" (not
        present in the original pretrained checkpoint) AFTER loading the saved
        weights — which would silently overwrite the trained compressor with
        the identity matrix.  Compressor initialization is handled explicitly
        by the caller (see swift_plugin.py :: init_compressor_weights).
        """
        super()._init_weights(module)

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def _embed_pad_and_compress(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Embed tokens and pair-compress.

        ``input_ids`` **must** have even length — callers are responsible for
        right-replication-padding to even length before calling this method
        (done at the top of :meth:`forward` for the compressed path).
        """
        B, T = input_ids.shape

        embeds = self.model.embed_tokens(input_ids)  # [B, T, H]
        compressed = self.compressor(embeds)          # [B, T//2, H]

        if attention_mask is None:
            pair_mask = None
        else:
            pair_mask = attention_mask.reshape(B, T // 2, 2).any(dim=-1).to(attention_mask.dtype)

        return compressed, pair_mask

    def _sample_from_logits(
        self,
        logits: torch.Tensor,
        temperature: float = 1e-6,
        top_k: int = 1e9,
    ) -> torch.LongTensor:
        if logits.numel() == 0:
            return torch.zeros(*logits.shape[:-1], device=logits.device, dtype=torch.long)

        # Keep ultra-low temperature deterministic by design.
        if temperature <= 1e-5:
            return logits.argmax(dim=-1)

        scores = logits.float() / max(temperature, 1e-6)
        vocab_size = scores.size(-1)
        k = max(1, min(int(top_k), vocab_size)) if top_k is not None else vocab_size

        if k < vocab_size:
            topk_scores, topk_ids = torch.topk(scores, k=k, dim=-1)
            probs = F.softmax(topk_scores, dim=-1)
            sampled_rel = torch.multinomial(probs.reshape(-1, k), 1).view(*probs.shape[:-1], 1)
            return torch.gather(topk_ids, dim=-1, index=sampled_rel).squeeze(-1)

        probs = F.softmax(scores, dim=-1)
        sampled = torch.multinomial(probs.reshape(-1, vocab_size), 1).view(*probs.shape[:-1], 1)
        return sampled.squeeze(-1)

    # ------------------------------------------------------------------
    # Unified forward for both training and inference
    # ------------------------------------------------------------------
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Qwen3_5MTPCausalLMOutput:
        use_compressed_prompt = kwargs.pop("use_compressed_prompt", True)
        temperature = float(kwargs.pop("temperature", 1e-6))
        top_k = int(kwargs.pop("top_k", 20))

        if input_ids is None and inputs_embeds is None:
            raise ValueError("Either `input_ids` or `inputs_embeds` must be provided.")
        if labels is not None and input_ids is None:
            raise ValueError("`labels` requires `input_ids`.")
        if labels is not None and use_cache is None:
            use_cache = False

        # For cache decoding with single tokens, keep the standard uncompressed path.
        can_compress = (
            use_compressed_prompt
            and inputs_embeds is None
            and input_ids is not None
            and (past_key_values is None or input_ids.shape[1] > 1)
        )

        if can_compress:
            if input_ids.shape[1] % 2 != 0:
                pad_id = 248044  # TODO: hard coded pad id
                input_ids = torch.cat([input_ids, torch.full((labels.shape[0], 1), pad_id, dtype=torch.long, device=input_ids.device)], dim=1)
                if attention_mask is not None:
                    attention_mask = torch.cat([attention_mask, attention_mask[:, -1:]], dim=1)
                if labels is not None:
                    labels = torch.cat(
                        [labels, labels.new_full((labels.shape[0], 1), -100)], dim=1
                    )

            compressed_embeds, pair_mask = self._embed_pad_and_compress(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            backbone_kw = {k: v for k, v in kwargs.items() if k not in _COMPRESSED_BACKBONE_IGNORE_KWARGS}
            backbone_out = self.model(
                inputs_embeds=compressed_embeds,
                attention_mask=pair_mask,
                past_key_values=past_key_values,
                use_cache=use_cache,
                **backbone_kw,
            )
        else:
            backbone_out = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                cache_position=cache_position,
                **kwargs,
            )

        backbone_hidden = backbone_out.last_hidden_state

        # ----- Training path: never materialize the full [B, L, V] logits.
        # On long context (L=64K, V=248320) the two lm_head outputs alone cost
        # ~63 GB in bf16 plus ~126 GB in the fp32 upcast inside F.cross_entropy.
        # Chunked linear+CE caps the transient logits buffer to chunk_size × V.
        if labels is not None:
            backbone_hidden_in = backbone_hidden[:, :-1]
            mtp_hidden_in = backbone_hidden[:, :-1]
            # Teacher-forcing: use ground truth token1 during training.
            sampled_token1 = input_ids[:, 2::2]

            mtp_out = self.mtp(
                inputs_embeds=self.model.embed_tokens(sampled_token1),
                hidden_states=mtp_hidden_in,
                attention_mask=torch.ones_like(sampled_token1, dtype=torch.bool),
                use_cache=False,
            )

            backbone_loss = _chunked_linear_cross_entropy(
                backbone_hidden_in, labels[:, 2::2], self.lm_head, ignore_index=-100,
            )
            # Conditionally piggy-back log p_s(label) on the MTP CE forward
            # so the SFT-stage ConfidenceHead BCE can reuse it (avoids a
            # SECOND lm_head forward over V≈248K — that head is the chunk
            # loop's dominant cost, so the saving is substantial). When
            # conf is OFF we keep the original ``F.cross_entropy`` fast
            # path (zero overhead).
            need_sft_conf = (
                self.confidence_head is not None and self.sft_conf_loss_weight > 0
            )
            if need_sft_conf:
                mtp_loss, mtp_log_p_at_label = _chunked_linear_cross_entropy(
                    mtp_out, labels[:, 3::2], self.lm_head, ignore_index=-100,
                    return_log_p_at_label=True,
                )
            else:
                mtp_loss = _chunked_linear_cross_entropy(
                    mtp_out, labels[:, 3::2], self.lm_head, ignore_index=-100,
                )

            loss = backbone_loss + self.mtp_loss_weight * mtp_loss

            # ── SFT-stage ConfidenceHead BCE (optional) ──
            # Activated when (a) the head is instantiated and (b) the SFT
            # conf-loss weight is positive. See ``_chunked_conf_loss`` for
            # the EAGLE-AR-under-deterministic-teacher derivation
            # (``target = p_s_mtp(label_token2)``).
            conf_loss: Optional[torch.Tensor] = None
            conf_metrics: dict[str, float] = {}
            if need_sft_conf:
                # Reuse the per-token log p_s(label) emitted by the MTP CE
                # path — already detached / fp32, just exp+clamp into [0, 1].
                conf_target_flat = (
                    mtp_log_p_at_label.exp().clamp_(0.0, 1.0)
                )
                conf_loss, conf_metrics = _chunked_conf_loss(
                    backbone_hidden=backbone_hidden_in,
                    mtp_hidden=mtp_out,
                    labels=labels[:, 3::2],
                    confidence_head=self.confidence_head,
                    conf_target_flat=conf_target_flat,
                    detach_inputs=self.sft_conf_detach_inputs,
                    return_metrics=True,
                )
                loss = loss + self.sft_conf_loss_weight * conf_loss
            elif self.confidence_head is not None:
                # Head exists but no supervision — keep its parameters in
                # the autograd graph with a 0-grad ghost forward so DDP /
                # DeepSpeed Zero2 don't complain about params with no grad
                # on this rank. Mirrors the OPD trainer's same pattern.
                dummy_b = backbone_hidden_in[:, :1].detach() * 0.0
                dummy_m = mtp_out[:, :1].detach() * 0.0
                ghost = (self.confidence_head(dummy_b, dummy_m).sum() * 0.0).to(loss.dtype)
                loss = loss + ghost

            return Qwen3_5MTPCausalLMOutput(
                loss=loss,
                backbone_loss=backbone_loss,
                mtp_loss=mtp_loss,
                conf_loss=conf_loss,
                conf_mean_pred=conf_metrics.get("conf_mean_pred"),
                conf_mean_target=conf_metrics.get("conf_mean_target"),
                conf_commit_rate_0_5=conf_metrics.get("conf_commit_rate_0.5"),
                conf_commit_rate_0_7=conf_metrics.get("conf_commit_rate_0.7"),
                conf_commit_rate_0_8=conf_metrics.get("conf_commit_rate_0.8"),
                conf_commit_rate_0_9=conf_metrics.get("conf_commit_rate_0.9"),
                conf_commit_rate_0_95=conf_metrics.get("conf_commit_rate_0.95"),
                conf_num_valid=conf_metrics.get("conf_num_valid"),
                # Skip returning full logits during training to save memory; downstream
                # (PIPOSeq2SeqTrainer) only consumes loss/backbone_loss/mtp_loss.
                logits=None,
                mtp_logits=None,
                backbone_hidden_states=backbone_hidden,
                past_key_values=backbone_out.past_key_values,
                sampled_token1=sampled_token1,
                sampled_token2=None,
                sampled_tokens=None,
            )

        # ----- Inference path: keep original behavior (materialize last-step logits only)
        logits = self.lm_head(backbone_hidden)
        logits1 = logits[:, -1:]
        mtp_hidden_in = backbone_hidden[:, -1:]
        sampled_token1 = self._sample_from_logits(
            logits1,
            temperature=temperature,
            top_k=top_k,
        )

        mtp_out = self.mtp(
            inputs_embeds=self.model.embed_tokens(sampled_token1),
            hidden_states=mtp_hidden_in,
            attention_mask=torch.ones_like(sampled_token1, dtype=torch.bool),
            use_cache=False,
        )
        logits2 = self.lm_head(mtp_out)

        sampled_token2 = self._sample_from_logits(
            logits2,
            temperature=temperature,
            top_k=top_k,
        )

        sampled_tokens = torch.stack([sampled_token1, sampled_token2], dim=-1)
        return Qwen3_5MTPCausalLMOutput(
            loss=None,
            logits=logits1,
            mtp_logits=[logits2] if logits2.shape[1] > 0 else None,
            backbone_hidden_states=backbone_hidden,
            past_key_values=backbone_out.past_key_values,
            sampled_token1=sampled_token1,
            sampled_token2=sampled_token2,
            sampled_tokens=sampled_tokens,
        )


__all__ = [
    "Qwen3_5MTPCausalLMOutput",
    "Qwen3_5MultiTokenPredictor",
    "ConfidenceHead",
    "Qwen3_5ForCausalPIPO",
    "PIPOCompressor",
    "LinearCompressor",
    "GatedCompressor",
    "MLPCompressor",
    "COMPRESSOR_REGISTRY",
]
