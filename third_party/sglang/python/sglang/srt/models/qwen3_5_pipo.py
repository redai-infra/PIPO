# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Inference-only Qwen3_5 PIPO (Pair-In Pair-Out) model.

Standalone CausalLM (not speculative draft). Flow:
  input_ids_pair -> embeddings_pair -> compressed (PIPOCompressor) -> backbone -> hidden
  -> logit1 -> sample id1 -> logit2 (MTP) -> sample id2 -> next_input_ids_pair

When --enable-pipo is set, each decode step consumes 2 tokens and produces 2 tokens.
"""

import copy
import logging
import os
import sys
from pathlib import Path
from typing import Iterable, Optional, Tuple

import torch
from torch import nn
from transformers import PretrainedConfig

from sglang.srt.distributed import get_pp_group, get_tensor_model_parallel_world_size
from sglang.srt.eplb.expert_distribution import get_global_expert_distribution_recorder
from sglang.srt.eplb.expert_location import ModelConfigForExpertLocation
from sglang.srt.layers.layernorm import GemmaRMSNorm
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE
from sglang.srt.layers.vocab_parallel_embedding import ParallelLMHead
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.models.qwen3_5_latent import Qwen3_5ForCausalLMLatent
from sglang.srt.server_args import get_global_server_args
from sglang.srt.utils import add_prefix

logger = logging.getLogger(__name__)


# Add repo root to path to import 
# FIXME: This is a temporary fix to import PIPO modules from the repo root.
_repo_root = Path(__file__).resolve().parent.parent.parent.parent.parent.parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from pipo.qwen3_5.compressor import (
    PIPOCompressor,
    LinearCompressor,
    MLPCompressor,
    ConfidenceHead,
    COMPRESSOR_REGISTRY,
)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class Qwen3_5ForCausalPIPO(nn.Module):
    """Standalone PIPO CausalLM: backbone on compressed embeddings + MTP head."""

    def __init__(
        self,
        config: PretrainedConfig,
        quant_config=None,
        prefix: str = "",
    ) -> None:
        nn.Module.__init__(self)

        self.is_multimodal = hasattr(config, "text_config")
        if self.is_multimodal:
            config = config.text_config

        # The MTP model is unquantized in the nvfp4 checkpoint.
        if quant_config and quant_config.get_name() == "modelopt_fp4":
            quant_config = None

        self.config = config
        self.tp_size = get_tensor_model_parallel_world_size()
        self.quant_config = quant_config
        self.pp_group = get_pp_group()

        # ── Input compressor ──
        # Mirrors training-side PIPOCompressor class hierarchy.
        # Selected via config.compressor_type; default "linear" for backward compat.
        compressor_type = getattr(config, "compressor_type", "linear")
        compressor_cls = COMPRESSOR_REGISTRY.get(compressor_type)
        if compressor_cls is None:
            raise ValueError(
                f"Unknown compressor_type={compressor_type!r}. "
                f"Choose from: {sorted(COMPRESSOR_REGISTRY)}"
            )
        self.compressor: PIPOCompressor = compressor_cls(config)

        # Full backbone (32 layers) - operates on compressed embeddings
        backbone_config = copy.deepcopy(config)
        self.backbone = Qwen3_5ForCausalLMLatent(
            backbone_config,
            quant_config,
            prefix=add_prefix("model", prefix),
            is_nextn=False,
        )

        # MTP block: fc + pre_fc_norm + 1 decoder layer
        # The single MTP layer must use a layer_id outside the backbone range
        # (0 .. num_hidden_layers-1) so that:
        #   a) its KV cache slot doesn't collide with any backbone layer, and
        #   b) HybridLinearAttnBackend can route it to the full-attention path.
        self._mtp_layer_id_offset = config.num_hidden_layers  # e.g. 32
        mtp_config = copy.deepcopy(config)
        mtp_config.num_hidden_layers = 1
        mtp_config.full_attention_interval = 1
        # Tell backbone builder to offset layer IDs for MTP block
        mtp_config._layer_id_offset = self._mtp_layer_id_offset
        self.mtp_fc = nn.Linear(2 * config.hidden_size, config.hidden_size, bias=False)
        RMSNorm_cls = GemmaRMSNorm
        self.mtp_pre_fc_norm_embedding = RMSNorm_cls(
            config.hidden_size, config.rms_norm_eps
        )
        self.mtp_pre_fc_norm_hidden = RMSNorm_cls(
            config.hidden_size, config.rms_norm_eps
        )
        self.mtp_block = Qwen3_5ForCausalLMLatent(
            mtp_config,
            quant_config,
            prefix=add_prefix("mtp", prefix),
            is_nextn=True,
        )
        # Total layers visible to KV cache / attention backend
        self.end_layer = self._mtp_layer_id_offset + 1  # backbone layers + 1 MTP layer

        self.confidence_head: ConfidenceHead = ConfidenceHead(config)

        if get_pp_group().is_last_rank:
            if config.tie_word_embeddings:
                self.lm_head = self.backbone.embed_tokens
            else:
                self.lm_head = ParallelLMHead(
                    config.vocab_size,
                    config.hidden_size,
                    quant_config=quant_config,
                    prefix=add_prefix("lm_head", prefix),
                )

        self.logits_processor = LogitsProcessor(config)

    @classmethod
    def get_model_config_for_expert_location(cls, config):
        text_config = getattr(config, "text_config", config)
        num_experts = getattr(text_config, "num_experts", None)
        if num_experts is None:
            return None
        return ModelConfigForExpertLocation(
            num_layers=text_config.num_hidden_layers,
            num_logical_experts=num_experts,
            num_groups=None,
        )

    def get_embed_and_head(self):
        return self.backbone.embed_tokens.weight, self.lm_head.weight

    def set_embed_and_head(self, embed, head):
        del self.backbone.embed_tokens.weight
        if not self.config.tie_word_embeddings:
            del self.lm_head.weight

        self.backbone.embed_tokens.weight = embed
        self.lm_head.weight = head
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        enable_pipo = get_global_server_args().enable_pipo
        pipo_phase = getattr(forward_batch, "pipo_phase", 1)

        # Phase 2 (decode only): MTP predicts logit2 from backbone_hidden + embed(token1)
        if enable_pipo and pipo_phase == 2:
            backbone_hidden = forward_batch.pipo_backbone_hidden
            token1 = forward_batch.pipo_sampled_token_from_phase1

            embed1 = self.backbone.embed_tokens(token1)
            input_embeds_mtp = self.mtp_pre_fc_norm_embedding(embed1)
            hidden_mtp = self.mtp_pre_fc_norm_hidden(backbone_hidden)
            fused = torch.cat([input_embeds_mtp, hidden_mtp], dim=-1)
            fused = self.mtp_fc(fused)
            with get_global_expert_distribution_recorder().disable_this_region():
                hidden_states = self.mtp_block(
                    token1, positions, forward_batch, input_embeds=fused
                )

            # ── Confidence head (post-hoc commit predictor for token2) ──
            # Reads ``backbone_hidden`` (Phase 1 output, pre-MTP fusion) and
            # ``hidden_states`` (Phase 2 MTP output, the same tensor that
            # ``logits_processor`` is about to project to logit2). Output is
            # stashed on ``forward_batch`` for ``tp_worker`` to consume after
            # token2 has been sampled — see the ``pipo_conf_threshold``
            # branch in ``forward_batch_generation``.
            if self.confidence_head is not None:
                conf_logit = self.confidence_head(backbone_hidden, hidden_states)
                forward_batch.pipo_token2_conf = torch.sigmoid(conf_logit)

            return self.logits_processor(
                token1, hidden_states, self.lm_head, forward_batch
            )

        # Phase 1 (backbone) or prefill: compress -> backbone -> logit1
        if not enable_pipo:
            # Standard flow: backbone on raw input_ids
            hidden_states = self.backbone(
                input_ids, positions, forward_batch, input_embeds=None
            )
            return self.logits_processor(
                input_ids, hidden_states, self.lm_head, forward_batch
            )

        if forward_batch.forward_mode.is_extend():
            # ----- Prefill: compress token-pair embeddings -----
            # The scheduler has already:
            #   1. Padded origin_input_ids to even length
            #   2. Set seq_lens / extend_seq_lens to compressed (L//2) granularity
            #   3. Allocated L//2 KV slots via out_cache_loc
            # So forward_batch metadata already matches compressed token counts.
            #
            # We receive the full padded input_ids via pipo_padded_input_ids
            # and the per-request padded extend lengths via pipo_padded_extend_lens.
            padded_ids = forward_batch.pipo_padded_input_ids  # [total_padded_tokens]
            padded_extend_lens = forward_batch.pipo_padded_extend_lens  # list[int]

            # Embed all padded tokens
            embeds = self.backbone.embed_tokens(padded_ids)  # [total_padded_tokens, H]

            # Compress all tokens at once since each request's length is even,
            # global pair-splitting produces the same result as per-request pair-splitting
            compressed = self.compressor(embeds)  # [comp_total, H]

            # The ForwardBatch metadata (seq_lens, extend_seq_lens, out_cache_loc,
            # positions) is already at compressed granularity — no need for
            # _make_compressed_forward_batch or post-prefill compaction.
            hidden_states = self.backbone(
                input_ids, positions, forward_batch,
                input_embeds=compressed,
            )

            return self.logits_processor(
                input_ids, hidden_states, self.lm_head, forward_batch
            )

        elif forward_batch.forward_mode.is_decode():
            # ----- Decode: each request contributes a pair (t_i, t_{i+1}) -----
            # ``input_ids_pair`` is set by the scheduler with shape [batch, 2].
            # During CUDA-graph capture this buffer is also provided (with
            # dummy data) so that the compression code path is captured.
            input_ids_pair = getattr(forward_batch, "input_ids_pair", None)
            if input_ids_pair is not None:
                ids_flat = input_ids_pair.view(-1)              # [batch * 2]
                embeds = self.backbone.embed_tokens(ids_flat)   # [batch*2, H]
                compressed = self.compressor(embeds)            # [batch, H]

                hidden_states = self.backbone(
                    input_ids, positions, forward_batch, input_embeds=compressed
                )
            else:
                # No pair available: run backbone on single-token embeddings
                # without compression (non-latent fallback).
                hidden_states = self.backbone(
                    input_ids, positions, forward_batch, input_embeds=None
                )

            forward_batch.pipo_backbone_hidden = hidden_states
            return self.logits_processor(
                input_ids, hidden_states, self.lm_head, forward_batch
            )

        else:
            hidden_states = self.backbone(
                input_ids, positions, forward_batch, input_embeds=None
            )
            return self.logits_processor(
                input_ids, hidden_states, self.lm_head, forward_batch
            )

    def load_weights(
        self, weights: Iterable[Tuple[str, torch.Tensor]], is_mtp: bool = False
    ):
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]

        # Params for MoE experts (non-fused/fused)
        num_experts = getattr(self.config, "num_experts", None)
        if num_experts is not None:
            expert_params_mapping = FusedMoE.make_expert_params_mapping(
                ckpt_gate_proj_name="gate_proj",
                ckpt_down_proj_name="down_proj",
                ckpt_up_proj_name="up_proj",
                num_experts=num_experts,
            )
        else:
            expert_params_mapping = []

        # Skip loading extra parameters for GPTQ/modelopt models.
        ignore_suffixes = (
            ".bias",
            "_bias",
            ".k_scale",
            "_k_scale",
            ".v_scale",
            "_v_scale",
            ".weight_scale",
            "_weight_scale",
            ".input_scale",
            "_input_scale",
        )

        # fused experts: experts.w13_weight / experts.w2_weight
        is_fused_expert = False
        fused_expert_params_mapping = [
            ("experts.w13_weight", "experts.gate_up_proj", 0, "w1"),
            ("experts.w2_weight", "experts.down_proj", 0, "w2"),
        ]

        def load_fused_expert_weights(
            name: str,
            params_dict: dict,
            loaded_weight: torch.Tensor,
            shard_id: str,
            num_experts: int,
        ):
            param = params_dict[name]
            weight_loader = param.weight_loader
            for expert_id in range(num_experts):
                curr_expert_weight = loaded_weight[expert_id]
                weight_loader(
                    param,
                    curr_expert_weight,
                    name,
                    shard_id,
                    expert_id,
                )
            return True

        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()

        for name, loaded_weight in weights:
            if "rotary_emb.inv_freq" in name:
                continue

            # ----- Remap checkpoint name → model parameter name -----
            # Checkpoint structure (after retraining):
            #   model.language_model.{embed_tokens,layers,norm}.* → backbone.*
            #   mtp.fc.weight → mtp_fc.weight
            #   mtp.pre_fc_norm_embedding.* → mtp_pre_fc_norm_embedding.*
            #   mtp.pre_fc_norm_hidden.* → mtp_pre_fc_norm_hidden.*
            #   mtp.layers.* → mtp_block.layers.*
            #   mtp.norm.* → mtp_block.norm.*
            #   lm_head.* → lm_head.*
            #   compressor.* → compressor.*  (direct match — no remapping)

            mapped_name = name

            # Backbone: model.language_model.X → backbone.X
            if mapped_name.startswith("model.language_model."):
                mapped_name = "backbone." + mapped_name[len("model.language_model."):]
            elif mapped_name.startswith("model."):
                # Fallback: model.X → backbone.X (in case naming varies)
                mapped_name = "backbone." + mapped_name[len("model."):]

            # MTP block weights
            if mapped_name.startswith("mtp."):
                rest = mapped_name[len("mtp."):]
                if rest.startswith("fc."):
                    mapped_name = "mtp_fc." + rest[len("fc."):]
                elif rest.startswith("pre_fc_norm_embedding."):
                    mapped_name = "mtp_pre_fc_norm_embedding." + rest[len("pre_fc_norm_embedding."):]
                elif rest.startswith("pre_fc_norm_hidden."):
                    mapped_name = "mtp_pre_fc_norm_hidden." + rest[len("pre_fc_norm_hidden."):]
                elif rest.startswith("layers.") or rest.startswith("norm."):
                    mapped_name = "mtp_block." + rest
                else:
                    mapped_name = "mtp_block." + rest

            # In the checkpoint, attention params live under ".self_attn."
            # but in the model they are direct children of the layer module.
            # E.g. mtp_block.layers.0.self_attn.q_proj → mtp_block.layers.0.q_proj
            # Same applies to backbone: backbone.layers.X.self_attn.Y → backbone.layers.X.Y
            if ".self_attn." in mapped_name:
                mapped_name = mapped_name.replace(".self_attn.", ".")

            # ----- Handle stacked / sharded parameters -----
            loaded = False
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in mapped_name:
                    continue
                if "mlp.experts" in mapped_name:
                    continue

                stacked_name = mapped_name.replace(weight_name, param_name)
                if stacked_name.endswith(ignore_suffixes) and stacked_name not in params_dict:
                    continue
                if stacked_name not in params_dict:
                    continue

                param = params_dict[stacked_name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight, shard_id)
                loaded_params.add(stacked_name)
                loaded = True
                break

            if loaded:
                continue

            # ----- Handle MoE expert weights -----
            is_expert_weight = False
            for mapping in expert_params_mapping:
                param_name, weight_name, expert_id, shard_id = mapping
                if weight_name not in mapped_name:
                    continue

                is_expert_weight = True
                name_mapped = mapped_name.replace(weight_name, param_name)

                if is_fused_expert and num_experts is not None:
                    if "experts.gate_up_proj" in mapped_name:
                        loaded_w1, loaded_w3 = loaded_weight.chunk(2, dim=-2)
                        load_fused_expert_weights(name_mapped, params_dict, loaded_w1, "w1", num_experts)
                        load_fused_expert_weights(name_mapped, params_dict, loaded_w3, "w3", num_experts)
                    else:
                        load_fused_expert_weights(name_mapped, params_dict, loaded_weight, shard_id, num_experts)
                else:
                    if name_mapped.endswith(ignore_suffixes) and name_mapped not in params_dict:
                        continue
                    if name_mapped not in params_dict:
                        break
                    param = params_dict[name_mapped]
                    weight_loader = param.weight_loader
                    weight_loader(param, loaded_weight, name_mapped, shard_id=shard_id, expert_id=expert_id)
                loaded_params.add(name_mapped)
                loaded = True
                break

            if loaded or is_expert_weight:
                continue

            # ----- Regular (non-stacked, non-expert) parameters -----
            if mapped_name.endswith(ignore_suffixes) and mapped_name not in params_dict:
                continue

            if mapped_name in params_dict:
                param = params_dict[mapped_name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
                loaded_params.add(mapped_name)
            else:
                logger.warning_once(
                    f"Parameter {name} (mapped to {mapped_name}) not found in params_dict, skip loading"
                )

        return loaded_params


EntryClass = [Qwen3_5ForCausalPIPO]
