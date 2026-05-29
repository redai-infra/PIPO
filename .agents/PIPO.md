# PIPO — Project Knowledge Base

> **Purpose of this file**: Concise background for AI agents working on this codebase across sessions.

---

## What Is PIPO?

**PIPO** is a "Pair-In / Pair-Out" LLM architecture on top of Qwen3.5.
The key idea: compress every consecutive pair of token embeddings into a single latent vector (via a learned `PIPOCompressor`), then run the backbone on this half-length sequence. A single MTP decoder layer predicts the second token of each pair, doubling effective throughput.

- Base model: **Qwen/Qwen3.5-4B** (hybrid architecture: 24 linear-attention + 8 full-attention layers across 32 total), and Qwen/Qwen3.5-9B.
- Fine-tuning strategy: **LoRA** via ms-swift + custom trainer.
- Training framework: **ms-swift** `swift sft` / `swift rlhf`, with a custom plugin/trainer.
- Serving framework: **SGLang** with `--enable-pipo` flag.
- **Compressor + ConfidenceHead are ALWAYS instantiated for PIPO** (no opt-in flag); both are saved into / loaded from every PIPO checkpoint.
- **Inference-time gating is conf-threshold ONLY**: there is no entropy-tau gate and no random-PAD gate anywhere in the serving path. There is no `mtp_no_verify` mode either — it was removed during the rename to PIPO.
- **PAD token id**: read from `tokenizer.pad_token_id`; for Qwen3.5 the default is `248044`.
  The OPD trainer asserts this value at init (`pipo/trainer/swift_gkd_trainer.py` L155-159).

---

## Architecture: `Qwen3_5ForCausalPIPO` (training-side HF model)

File: [`pipo/qwen3_5/modeling_qwen3_5_mtp.py`](pipo/qwen3_5/modeling_qwen3_5_mtp.py)

```
Qwen3_5ForCausalPIPO
├── model              Qwen3_5TextModel  (32 layers, hybrid linear+full attention)
│   └── embed_tokens   shared / tied with lm_head
├── compressor         PIPOCompressor (selected via config.compressor_type)
│   ├─ LinearCompressor:  linear = Linear(2H → H, bias=True)
│   ├─ GatedCompressor:   alpha, beta = Linear(H → 1), delta = Linear(H → H)
│   └─ MLPCompressor:     two-layer MLP with SiLU activation
├── lm_head            Linear(H → V, bias=False)    ← tied with embed_tokens, FROZEN
├── mtp                Qwen3_5MultiTokenPredictor
│   ├── pre_fc_norm_hidden     RMSNorm(H)
│   ├── pre_fc_norm_embedding  RMSNorm(H)
│   ├── fc             Linear(2H → H, bias=False)
│   ├── layers         ModuleList[1 × Qwen3_5DecoderLayer (full_attention)]
│   │   ├── self_attn.{q,k,v,o}_proj   ← LoRA rank-64 applied
│   │   ├── mlp.{gate,up,down}_proj    ← LoRA rank-64 applied
│   │   ├── input_layernorm            ← FROZEN (Qwen3.5-4B pretrained value)
│   │   ├── post_attention_layernorm   ← FROZEN (Qwen3.5-4B pretrained value)
│   │   ├── self_attn.q_norm           ← FROZEN
│   │   └── self_attn.k_norm           ← FROZEN
│   ├── norm           RMSNorm(H)     ← fully trained (modules_to_save)
│   └── rotary_emb     (no trainable params, computed from config)
└── confidence_head    ConfidenceHead   ← ALWAYS instantiated for PIPO
    ├── norm           RMSNorm(2H)
    ├── fc1            Linear(2H → H, bias=False)
    ├── act            SiLU
    └── fc2            Linear(H → 1, bias=False)
       Inputs: (backbone_hidden, mtp_hidden) → sigmoid(logit) ∈ [0,1]
       Trained in OPD against the **EAGLE per-pair accept rate** of `token2`
       under the verify model — see
       pipo/trainer/swift_gkd_trainer.py::_accumulate_conf_chunk
       (called from `_compute_chunked_jsd_loss` / `_compute_chunked_sampled_kl`).
       Replaces the entropy-tau heuristic at SGLang inference (see "SGLang
       Gating" section below).
```

**Weight tying**: `lm_head.weight == model.embed_tokens.weight` (`_tied_weights_keys`) for Qwen3.5-4B and smaller models.
`lm_head` is **not** in LoRA target_modules and not in `modules_to_save` → frozen during training.

### Compressor Classes ([`PIPOCompressor`](pipo/qwen3_5/compressor.py))

Both training and SGLang share the same class hierarchy. The base class is `PIPOCompressor(nn.Module)` with `forward(self, embeds, **kwargs)`. The `**kwargs` allows future variants to receive extra inputs (e.g. token probabilities).

| Type | `config.compressor_type` | Checkpoint keys | Formula |
|------|--------------------------|-----------------|---------|
| `LinearCompressor` | `"linear"` | `compressor.linear.{weight,bias}` | `Linear(reshape([e_i; e_{i+1}]))` |
| `MLPCompressor` | `"mlp"` | `compressor.mlp.*` | Two-layer MLP with SiLU on concatenated pair |

`forward()` handles both 2D `[num_tokens, H]` (SGLang packed) and 3D `[B, T, H]` (HuggingFace batched).
Registry: `COMPRESSOR_REGISTRY` maps type strings to classes; both training and SGLang share the same registry via `from pipo.qwen3_5.compressor import ...`.

---

## Training Data Flow (Forward Pass)

### Compressed (training) path — `use_compressed_prompt=True`
```
input_ids [B, T]
  → embed_tokens → [B, T, H]
  → pad to even T if needed (replicate last token)
  → compressor → [B, L, H]   (L = T//2)
  → Qwen3_5TextModel (backbone)
  → backbone_hidden [B, L, H]
  ├─ lm_head → logits [B, L, V]
  │   loss_1 = CE(logits[:, :-1], labels[:, 2::2])  ← backbone predicts t_{2i+2}
  └─ mtp(embed(token1), backbone_hidden[:, :-1]) → mtp_hidden [B, L-1, H]
      lm_head(mtp_hidden) → logits2 [B, L-1, V]
      loss_2 = CE(logits2, labels[:, 3::2])          ← MTP predicts t_{2i+3}
total_loss = loss_1 + mtp_loss_weight (1.0) * loss_2 + sft_conf_loss_weight * conf_bce
```

---

## SFT Configuration ([`swift_sft.sh`](swift_sft.sh))

| Parameter | Value |
|-----------|-------|
| Base model | `Qwen/Qwen3.5-4B` |
| model_type | `qwen3_5_mtp` (registered via plugin) |
| Tuner | LoRA, rank=64, alpha=128, dropout=0.05 |
| target_modules | `q_proj k_proj v_proj o_proj gate_proj up_proj down_proj` |
| modules_to_save | `confidence_head compressor mtp.fc mtp.pre_fc_norm_hidden mtp.pre_fc_norm_embedding mtp.norm mtp.layers.0.input_layernorm mtp.layers.0.post_attention_layernorm mtp.layers.0.self_attn.q_norm mtp.layers.0.self_attn.k_norm` |
| max_length | up to 64K |
| Devices | 8 × GPU (DeepSpeed ZeRO-2) |
| Task type | `pipo` → [`PIPOSeq2SeqTrainer`](pipo/trainer/swift_sft_trainer.py) (registered in `third_party/ms-swift/swift/trainers/trainer_factory.py`) |
| attn_impl | `flash_attention_2` |
| Env: `COMPRESSOR_TYPE` | `"mlp"` (default) / `"linear"` |
| Env: `PIPO_MAX_PAD_RATIO` | `"0.25"` (default). Upper bound; each step samples `pad_ratio ~ Uniform[0, max_pad_ratio]`. Pure training-time data augmentation that exposes the model to PAD-separated pairs, mirroring what the SGLang conf gate produces at serving time. NOT used during inference. |
| Env: `PIPO_LOGITS_CHUNK` | `"2048"` (default). Logits chunk size for SFT forward memory optimization. |
| Env: `MTP_LOSS_WEIGHT` | `"1.0"` (default). Weight of MTP loss in total combined loss. |
| Env: `SFT_CONF_LOSS_WEIGHT` | `"1"` (default). BCE weight for SFT-stage ConfidenceHead. |
| Env: `SFT_CONF_DETACH_INPUTS` | `"1"` (default). Detach `(backbone_hidden, mtp_hidden)` before the head. |

**LoRA scope**: `target_modules` matches by suffix, so LoRA is applied to **both** backbone layers AND `mtp.layers.0` projections simultaneously.

---

## Plugin & Trainer Chain

```
swift_sft.sh
  --external_plugins pipo/trainer/swift_plugin.py
    → register_model(model_type="qwen3_5_mtp", task_type="pipo", loader=Qwen3_5MtpLoader)
  --model_type qwen3_5_mtp
    → Qwen3_5MtpLoader.get_model() → Qwen3_5ForCausalPIPO.from_pretrained(...)
  task_type="pipo"
    → TrainerFactory → PIPOSeq2SeqTrainer (pipo/trainer/swift_sft_trainer.py)

swift_opd.sh
  --external_plugins pipo/trainer/swift_plugin.py
    → swift_plugin also hot-patches TrainerFactory.TRAINER_MAPPING['gkd']
      to point at pipo.trainer.swift_gkd_trainer.PIPOGKDTrainer
  --rlhf_type gkd
    → TrainerFactory → PIPOGKDTrainer
```

[`PIPOSeq2SeqTrainer.compute_loss`](pipo/trainer/swift_sft_trainer.py):
- `max_pad_ratio == 0` (default): standard teacher-forcing, ensures even sequence lengths (prompt and total), rebuilt attention mask, no PAD insertion.
- `max_pad_ratio > 0`: **random PAD insertion** via `_build_random_padded_inputs`:
  1. **Clean up**: Strips any existing trailing padding from `ms-swift` sequence processing.
  2. **Prompt even-length guarantee**: if prompt length (contiguous leading labels == -100) is odd, insert one PAD at the end of the prompt so pair boundaries align with the prompt/generation boundary.
  3. **Total even-length guarantee**: total length is padded to be even, ensuring cleanly paired embeddings.
  4. **Odd PAD Index Guarantee**: All PAD tokens across the sequence will reside on odd indices (as the second token of a pair).
  5. **Random generation-part splitting**: for each token pair `(t_{2p}, t_{2p+1})` in the generation part (label != -100), `pad_ratio` fraction of eligible pairs are randomly split into `{t_{2p}, PAD}, {t_{2p+1}, PAD}`. Labels at PAD positions are -100.
  6. Finally, dynamically updates the sequence's `attention_mask`. This trains the model to handle PAD-separated pairs exactly as presented at serving time.

**Thinking-template masking**: `N_BOT_TOKENS = 2` (lines 25-26 of `swift_sft_trainer.py`). The first two tokens after the last `-100` label (`<think>\n`) are masked from supervision so the model is not penalized for the thinking prefix format.

---

## Checkpoint Weight Layout

**Merged checkpoint key layout**:
- `model.language_model.*` → ~426 backbone keys
- `mtp.*` → 17 keys (fc, pre_fc_norms, layers.0.*, norm)
- `compressor.*` → NUM_KEY_COMPRESSOR keys (depends on compressor type)
- `confidence_head.*` → 4 keys (norm, fc1, fc2) — ALWAYS present for PIPO checkpoints
- `lm_head` absent (tied with `model.language_model.embed_tokens.weight`)

**LoRA adapter key layout**:
- `base_model.model.model.layers.*` → backbone LoRA (lora_A/lora_B pairs)
- `base_model.model.mtp.layers.0.*` → MTP decoder layer LoRA (lora_A/lora_B pairs)
- `base_model.model.{compressor,mtp.fc,mtp.norm,...,confidence_head}` → modules_to_save full weights

**Config persistence**: `compressor_type` is stored in `config.json` (added to `Qwen3_5TextConfig.__init__`), so SGLang can read it directly. The `Qwen3_5MtpLoader` plugin reads it from multiple sources in priority order: env > LoraConfig > additional_config.json > path-derived > config default.

---

## OPD Training Flow (On-Policy Distillation)

File: [`pipo/trainer/swift_gkd_trainer.py`](pipo/trainer/swift_gkd_trainer.py)

OPD is a three-stage pipeline: (1) **SGLang rollout** → (2) **batch encoding** → (3) **loss computation**.
The trainer is registered by the plugin when `--rlhf_type gkd` is used.

### OPD Trainer: `PIPOGKDTrainer`

Key design decisions (all fast-fail with `assert`):
- **No self-distillation**: teacher must be a separate frozen model (e.g. `Qwen/Qwen3.5-9B`).
- **SGLang-only rollout**: `rollout_backend == 'sglang'` with colocate mode.
- **Force PIPO serving**: `_prepare_sglang_engine()` overrides CLI args to set `enable_pipo=True` and `disable_radix_cache=True` regardless of user input.
- **Pad token assertion**: tokenizer `pad_token_id` must be set (Qwen3.5 = 248044).
- **Response prefix assertion**: template must have a non-empty `response_prefix` (qwen3_5 default is `"<think>\n"`).

### Response-Prefix Fix & Thinking Templates

For thinking templates (e.g. qwen3_5), SGLang's inference auto-appends `response_prefix='<think>\n'` to the prompt. The rollout `response_token_ids` therefore does **not** contain these prefix tokens — generation continues *after* them.

However, `GKDTrainer._prepare_batch_inputs` calls `replace_assistant_response_with_ids` which discards the original message content (which had `<think>\n`) and injects bare `response_token_ids` directly after `<|im_start|>assistant\n`, skipping the `response_prefix` branch. This causes a **2-token shift** between rollout-time and training-time prefixes.

**Fix**: `_encode_with_rollout_response` (L451-554) encodes prompt-only first (which auto-appends `response_prefix`), then manually concatenates the rollout `response_token_ids`. Additional parity-PAD is appended if the prompt length is odd, exactly mirroring SGLang's `tokenizer_manager.py::_tokenize_one_request` behavior.

### PAD Compaction for Teacher Forward

SGLang's PIPO serving (with `pipo_conf_threshold` gating, or with the SFT-stage random-pad augmentation visible in the dataset) inserts PAD tokens at rejected positions. The vanilla Qwen3.5 teacher has never seen PAD in mid-sequence.

**Solution**: `_teacher_logits_pad_compacted` / `_teacher_hidden_pad_compacted` (L702-856):
1. Strip PAD tokens from the input sequence (stable argsort + gather).
2. Run the teacher on the compacted clean sequence.
3. Re-map teacher outputs back to original positions via `cumsum(non_pad_mask) - 1`:
   - non-PAD position p → same token's compacted index.
   - PAD position p → previous non-PAD's compacted index (teacher conditioned on prefix-without-PAD).

This is used for **both** full-logits and chunked hidden-states paths (the latter avoids materialising `[B, T, V]` teacher logits at long context).

### KL-Divergence Modes

Controlled by `OPD_KL_MODE` (default `"sampled"`) and `beta` (default `1.0`):

| Mode | Condition | Description |
|------|-----------|-------------|
| `sampled` | `beta == 1.0` | Monte-Carlo reverse-KL on on-policy sampled tokens only: `l^sample = log p_s(y) − log p_t(y)`. Per-token unbiased estimator of `D_KL(p_s ‖ p_t)`. |
| `topk` | `beta == 1.0` + `OPD_KL_MODE=topk` | Restrict comparison to the per-position **union** of student top-k, teacher top-k, and the sampled-label index; `log_softmax` is renormalised over that union. `OPD_TOPK_SOURCE` is accepted for backward-compat but **ignored** in this path. Conf-head BCE target on this path uses the SAMPLED-mode ratio formula at the rolled-out label (NOT `Σ min(p_s, p_t)`). |
| `full` | `beta == 1.0` + `OPD_KL_MODE=full` | Full-vocab reverse-KL via chunked JSD. |
| JSD | `beta != 1.0` | Symmetric beta-JSD between student and teacher. |

**Chunked lm-head**: Both `_compute_chunked_jsd_loss` and `_compute_chunked_sampled_kl` chunk along the token dimension with size `PIPO_OPD_CHUNK_SIZE` (default 2048) to cap peak memory at `O(chunk_size × V)` instead of `O(T × V)`.

### Student Forward in OPD

`_student_compressed_logits` (L590-697) runs the student in **compressed mode** (the exact inference path):
- `embed_pad_compress(input_ids) → compressed_embeds` → backbone → `backbone_hidden`
- Teacher-force `token1 = input_ids[:, 2::2]` → MTP head → `mtp_hidden`
- Returns `(backbone_logits, mtp_logits, T)` or `(backbone_hidden, mtp_hidden, T, lm_head)` for chunked projection.

Why compressed mode for OPD:
1. **Parameter coverage**: every trainable parameter (compressor + MTP norms/fc + LoRA on backbone & MTP) receives gradient. Uncompressed mode would only train backbone LoRA.
2. **Distribution match**: backbone sees compressed-pair latents at training time, identical to inference.
3. **PAD-skip semantics**: pair `(t_p, PAD)` compressed to one latent → backbone predicts the next non-PAD token, exactly mirroring runtime behavior.

### OPD Loss Composition

Current fixed research config: `MTP_LOSS_WEIGHT=1`, `OPD_CONF_LOSS_WEIGHT=1`, `OPD_KL_MODE=sampled`, `beta=1.0`.

```
total_loss = loss_backbone + MTP_LOSS_WEIGHT * loss_mtp + OPD_CONF_LOSS_WEIGHT * loss_conf
```

- `loss_backbone`: reverse-KL / JSD between student backbone logits and teacher logits at even positions `[2, 4, ..., T-2]`.
- `loss_mtp`: same divergence between student MTP logits and teacher logits at odd positions `[3, 5, ..., T-1]`.
- `loss_conf`: BCE on `ConfidenceHead` predicting the EAGLE accept rate of `token2`.

Per-sample chunking (`B>1` → `_compute_loss_micro_chunks`): when the batch contains multiple samples with `response_token_ids`, each sample is encoded in an isolated `B=1` chunk to guarantee zero cross-sample padding. Sub-losses are summed and divided by N (sample-mean). **No in-loop `backward()`** — DeepSpeed ZeRO-2 would desynchronize.

---

## SGLang Serving Architecture

File: [`third_party/sglang/python/sglang/srt/models/qwen3_5_pipo.py`](third_party/sglang/python/sglang/srt/models/qwen3_5_pipo.py)

### PIPO-Exclusive Operations (SGLang Code Path Trace)

Below traces **every** PIPO-specific code divergence from standard SGLang, in execution order.
All cross-batch metadata fields use the `pipo_*` prefix.

#### Startup / Initialization

| Step | File | Location | What |
|------|------|----------|------|
| S1 | `server_args.py` | `ServerArgs` field | `--enable-pipo` CLI flag → `enable_pipo: bool` |
| S2 | `scheduler.py` | `Scheduler.__init__()` | Force-disable overlap scheduling when `enable_pipo` |
| S3 | `model_runner.py` | `ModelRunner` init | MTP `layer_id=32` added to `full_attention_layer_ids` (only architecture allowed is `Qwen3_5ForCausalPIPO`) |
| S4 | `cuda_graph_runner.py` | `DecodeInputBuffers.create()` | Allocate `input_ids_pair: [max_bs, 2]` GPU buffer |
| S5 | `cuda_graph_runner.py` | `capture_one_batch_size()` | Set dummy `input_ids_pair` on ForwardBatch → force compression path during capture |
| S6 | `cuda_graph_runner.py` | `capture_one_batch_size()` | Save `pipo_backbone_hidden` tensor reference after capture |
| S7 | `cuda_graph_runner.py` | `capture()` | Store per-graph-key `pipo_backbone_hidden_buffers` dict |

#### Prefill (Extend Mode)

| Step | File | Location | What |
|------|------|----------|------|
| P0 | `tokenizer_manager.py` | `_tokenize_one_request()` | Pad `input_ids` to even length if odd (append PAD token) |
| P1 | `tokenizer_manager.py` | `_batch_tokenize_and_process()` | Same padding for batch tokenization |
| P2 | `schedule_policy.py` | `_add_dllm_req()`, `add_chunked_req()`, `add_one_req_ignore_eos()`, `add_one_req()` | Ensure `trunc_len` / `extend_input_len` is **even** for PIPO. Since original sequence length is even (from P0-P1), and truncation length is even, remaining tokens = even - even = even. No padding needed in `prepare_for_extend()` |
| P3 | `schedule_batch.py` | `prepare_for_extend()` | Convert `prefix_indices` (compressed) to uncompressed offset (`*2`), compute uncompressed extend length. Both pre_len and extend_len are guaranteed even by schedule_policy. Save full padded `input_ids` in `pipo_padded_input_ids`; halve `seq_lens`, `orig_seq_lens`, `extend_lens` to compressed granularity |
| P4 | `schedule_batch.py` | `ScheduleBatch.prepare_for_idle()` | Set `input_ids_pair = None` |
| P5 | `schedule_batch.py` | `ModelWorkerBatch` | Pass `input_ids_pair=None`, `pipo_phase=1`, `pipo_padded_input_ids`, `pipo_padded_extend_lens` |
| P6 | `forward_batch_info.py` | `ForwardBatch.init_new()` | Copy PIPO fields; `compute_position()` uses compressed `extend_prefix_lens` / `extend_seq_lens` → correct positions `[0, 1, ..., L//2-1]` |
| P7 | `qwen3_5_pipo.py` | `forward()` — extend branch | Embed full padded tokens via `pipo_padded_input_ids`, compress per-request, feed compressed embeddings directly to backbone. ForwardBatch metadata is already at compressed granularity |
| P8 | `tp_worker.py` | `forward_batch_generation()` | Prefill: sample token1, then pair it with PAD as `[token1, PAD]` |
| P9 | `scheduler_output_processor_mixin.py` | `process_batch_result_prefill()` | `next_token_id` is `[t1, PAD]` → `req.output_ids.extend(next_token_id)` |

**Key Invariant**: For PIPO, all sequence lengths are guaranteed even throughout the pipeline:
- Tokenization: padded to even if needed (P0-P1)
- Chunked prefill: `trunc_len` adjusted to even in schedule_policy (P2)
- Therefore: `extend_len = total_len - prefix_len` = even - even = even

#### Decode (Two-Phase)

| Step | File | Location | What |
|------|------|----------|------|
| D1 | `schedule_batch.py` | `prepare_for_decode()` | Build `input_ids_pair [batch, 2]` from last 2 tokens per request (O(1) tail access); set `input_ids = input_ids_pair[:, 0]`; `token_per_req = 1` (compressed KV) |
| D2 | `schedule_batch.py` | `get_model_worker_batch()` | Pass `input_ids_pair` to `ModelWorkerBatch` |
| D3 | `forward_batch_info.py` | `ForwardBatch.init_new()` | Copy `input_ids_pair` to ForwardBatch |
| D4 | `cuda_graph_runner.py` | `can_run()` | Return `False` if `pipo_phase == 2` (Phase 2 must not use Phase 1's captured graph) |
| D5 | `cuda_graph_runner.py` | `populate_from_forward_batch()` | Copy `forward_batch.input_ids_pair` → pre-allocated GPU buffer |
| D6 | `cuda_graph_runner.py` | `replay()` | After graph replay, re-attach `pipo_backbone_hidden` buffer to `forward_batch` |
| **--- Phase 1 (backbone, CUDA Graph) ---** | | | |
| D7 | `qwen3_5_pipo.py` | `forward()` — decode branch | `input_ids_pair.view(-1)` → `embed_tokens` → `compressor()` → `backbone()` → save `pipo_backbone_hidden` → `logits_processor` → return logits1 |
| D8 | `tp_worker.py` | `forward_batch_generation()` | `token1 = self.model_runner.sample(logits_output, forward_batch)` ← **GPU→CPU sync point** |
| **--- Phase 2 (MTP, no CUDA Graph) ---** | | | |
| D9 | `tp_worker.py` | (L543-544) | Set `forward_batch.pipo_phase = 2`, `forward_batch.pipo_sampled_token_from_phase1 = token1` |
| D10 | `model_runner.py` | `_forward_raw()` | `can_run()` returns False → falls through to `forward_decode()` |
| D11 | `model_runner.py` | `forward_decode()` | `attn_backend.init_forward_metadata(forward_batch)` ← redundant CPU work (same metadata as Phase 1) |
| D12 | `qwen3_5_pipo.py` | `forward()` — Phase 2 branch | `embed(token1)` → `mtp_pre_fc_norm_embedding` → `mtp_pre_fc_norm_hidden(backbone_hidden)` → `cat` → `mtp_fc` → `mtp_block` (1 decoder layer) → `confidence_head(backbone_hidden, mtp_hidden) → sigmoid → forward_batch.pipo_token2_conf` → `logits_processor` → return logits2 |
| D13 | `tp_worker.py` | (L535-561) | `token2 = sample(logits2)`, then if `PIPO_CONF_THRESHOLD >= 0` replace token2 with PAD where `pipo_token2_conf < threshold`; `torch.stack([token1, token2], dim=1)` → `batch_result.next_token_ids` |

**Phase-2 fast paths** (`tp_worker.py` L537-541):
- `PIPO_CONF_THRESHOLD >= 1.0` → always PAD (model never trusted enough); short-circuits to `[token1, PAD]` and avoids the MTP forward, halving per-step decode latency.

**Gating (single mechanism, no fallback)** (`tp_worker.py`):
1. **Confidence head** (`PIPO_CONF_THRESHOLD >= 0`): uses `sigmoid(conf_logit)` from Phase 2 forward; PAD-replace token2 where conf < threshold. The confidence head is always loaded with the model, so there is no "head missing" branch to worry about.
2. **No gating** (`PIPO_CONF_THRESHOLD < 0`, default): commit token2 as-is.

There is no entropy-tau gate and no random rollout-PAD gate in the SGLang code path — both heuristics were removed during the PIPO rename. Training-time PAD augmentation (`PIPO_MAX_PAD_RATIO`) is a SFT-stage data augmentation only; it never runs at inference.

#### Post-Processing

| Step | File | Location | What |
|------|------|----------|------|
| R1 | `scheduler_output_processor_mixin.py` | `process_batch_result_decode()` | Skip logprob collection when `enable_pipo` (phase1+phase2 merge not implemented) |
| R2 | `scheduler_output_processor_mixin.py` | (L418-420) | `next_token_id` is `[t1, t2]` list → `req.output_ids.extend(next_token_id)` |
| R3 | `schedule_batch.py` | `filter_batch()` | Filter `input_ids_pair` by `keep_indices` when requests finish; gracefully clears stale `input_ids_pair` if size mismatches |
| R4 | `schedule_batch.py` | `merge_batch()` | When merging decode batch (has `input_ids_pair`) with extend batch (no `input_ids_pair`), clears `input_ids_pair` — `prepare_for_decode` rebuilds it before next forward |

### Model: `Qwen3_5ForCausalPIPO` (SGLang side)

```
Qwen3_5ForCausalPIPO
├── compressor              PIPOCompressor (from COMPRESSOR_REGISTRY[config.compressor_type])
├── backbone                Qwen3_5ForCausalLMLatent (32 layers, from qwen3_5_latent.py)
│   └── embed_tokens
├── mtp_fc                  Linear(2H → H, bias=False)
├── mtp_pre_fc_norm_embedding   GemmaRMSNorm(H)
├── mtp_pre_fc_norm_hidden      GemmaRMSNorm(H)
├── mtp_block               Qwen3_5ForCausalLMLatent (1 layer, layer_id=32)
│   └── self_attn flattened (q_proj, k_proj, v_proj → qkv_proj)
├── lm_head                 tied to backbone.embed_tokens
├── logits_processor        LogitsProcessor
└── confidence_head         ConfidenceHead (always present)
```

**MTP layer ID**: `_mtp_layer_id_offset = config.num_hidden_layers` (e.g. 32). This ensures:
- KV cache slot doesn't collide with any backbone layer.
- `HybridLinearAttnBackend` routes it to the full-attention path (`full_attention_layer_ids`).

### Weight Name Mapping (checkpoint → model param)

File: [`third_party/sglang/python/sglang/srt/models/qwen3_5_pipo.py::load_weights`](third_party/sglang/python/sglang/srt/models/qwen3_5_pipo.py:301-484)

| Checkpoint key | Model parameter | Notes |
|----------------|-----------------|-------|
| `model.language_model.*` | `backbone.*` | Strip prefix |
| `model.*` | `backbone.*` | Fallback strip |
| `mtp.fc.*` | `mtp_fc.*` | Flatten from `mtp.` |
| `mtp.pre_fc_norm_*.*` | `mtp_pre_fc_norm_*.*` | Flatten from `mtp.` |
| `mtp.layers.*` / `mtp.norm.*` | `mtp_block.layers.*` / `mtp_block.norm.*` | Remap to block |
| `*.self_attn.*` | `*.*` (remove `.self_attn.`) | SGLang flattens self_attn into direct children (e.g. `q_proj` → `qkv_proj` via stacked_params_mapping) |
| `compressor.*` | `compressor.*` | **Direct match** — no remapping |
| `confidence_head.*` | `confidence_head.*` | **Direct match** — no remapping |

**Stacked params**: SGLang shards `q_proj/k_proj/v_proj` into `qkv_proj` and `gate_proj/up_proj` into `gate_up_proj` via `stacked_params_mapping`. MoE expert weights use `FusedMoE.make_expert_params_mapping`.

### Forward Flow

**Prefill (extend mode)**:
```
Scheduler: fill_ids → compute uncompressed extend (prefix_len*2 → uncompressed offset)
         → pad extend to even → halve metadata (seq_lens//2, extend_lens//2; prefix_lens stays compressed)
         → allocate L//2 KV slots → pass pipo_padded_input_ids (full padded tokens)
Model:     pipo_padded_input_ids → embed_tokens → compressor(embeds per request)
         → backbone (compressed embeds + compressed ForwardBatch metadata)
         → hidden → logits_processor → logit1
         → sample token1 → pair as [token1, PAD]
```

**Decode (two-phase)**:
```
Phase 1:
  input_ids_pair [batch, 2] → embed → compressor → compressed [batch, H]
  → backbone(compressed) → hidden → logits → sample token1
Phase 2:
  embed(token1) → pre_fc_norm_embedding
  backbone_hidden → pre_fc_norm_hidden
  cat → mtp_fc → mtp_block → logits → sample token2
  confidence_head(backbone_hidden, mtp_hidden) → sigmoid → forward_batch.pipo_token2_conf
  if PIPO_CONF_THRESHOLD >= 0 and conf < threshold: token2 := PAD
Output: [token1, token2]
```

---

## Confidence Head (Post-hoc MTP token2 commit predictor)

### Module
* `ConfidenceHead` is defined ONCE in [`pipo/qwen3_5/compressor.py`](pipo/qwen3_5/compressor.py) and imported by both the training model and the SGLang model, guaranteeing structural parity. Architecture: `RMSNorm(2H) → Linear(2H, H) → SiLU → Linear(H, 1)`. Param count `~H² + 3H` (≈ 6.6M for 4B). Inputs are `(backbone_hidden, mtp_hidden)`, shaped `[B, L-1, H]` at training, `[batch, H]` at SGLang Phase-2 decode.

### Activation flow
1. **SFT-stage training** ([`pipo/qwen3_5/modeling_qwen3_5_mtp.py::Qwen3_5ForCausalPIPO.forward`](pipo/qwen3_5/modeling_qwen3_5_mtp.py)): the head is ALWAYS instantiated; when `SFT_CONF_LOSS_WEIGHT > 0`, the training-branch forward calls `_chunked_conf_loss(...)` to add a BCE term to the combined loss. Target derivation under the *deterministic teacher* assumption (SFT label = teacher's argmax, `p_t = δ_{y_t}`):
       AR_pos  =  Σ_y min(p_s(y), p_t(y))  =  min(p_s(y_t), 1)  =  p_s(y_t)
   so `target = p_s_mtp(label_token2)` — i.e. the student MTP head's probability on the gold token. This collapses to the SAME quantity as both OPD targets in the deterministic-teacher limit, which guarantees the SFT-trained head transfers seamlessly into OPD without a parameter reset. Implementation: chunked along the token dim with peak memory `chunk_size × V` (same envelope as `_chunked_linear_cross_entropy`); the lm-head re-forward used to derive the target runs under `no_grad`.
   When `SFT_CONF_LOSS_WEIGHT == 0`, a 0-grad ghost forward keeps the head's params in the autograd graph (DDP / ZeRO grad bucket alignment).
2. **OPD-stage training (fused into the OPD chunk loop, no extra lm-head forward)**: current config sets `OPD_CONF_LOSS_WEIGHT=1`, so `_compute_loss_single` passes the `confidence_head` and `backbone_hidden` into the same chunked function that's already computing `loss_mtp` (`_compute_chunked_sampled_kl`). Inside each chunk, the conf target is the **EAGLE sampled-token accept probability** of the actually rolled-out `token2` under the verify model, derived from the SAME log-prob tensors the OPD loss is using:
   - sampled-KL path → `α(y) = min(1, p_t(y)/p_s(y)) = exp(-per_pos_loss.clamp(min=0))` where `per_pos_loss = log p_s(y) − log p_t(y)` (full-vocab `log_softmax`).
   - **topk (union)** path → SAME sampled-token ratio formula as above, but the log-probs are read from the **union-renormalised** `log_softmax` at the label's slot inside `union_idx = cat(s_top, t_top, label)`. The label is appended to the union so the lookup is always valid; if it ALSO appears in `s_top`/`t_top` (common case), `argmax` on the equality mask picks the FIRST (non-duplicate) occurrence so the live log-prob (not the masked `-inf` duplicate) is gathered.
   - full-vocab JSD ablation path → `AR_pos = Σ_y min(p_s, p_t) = 1 − TV`.
   All forms are `.detach()`-ed before BCE so the conf head can never leak gradient back into the student logits via the target. BCE is accumulated alongside the OPD loss, so the lm-head is forwarded **once per chunk** instead of twice.
3. **SGLang Phase-2 forward**: the model unconditionally stashes `forward_batch.pipo_token2_conf = sigmoid(conf_logit)` for `tp_worker` to consume after token2 sampling.
4. **`tp_worker.forward_batch_generation`**: gating is conf-only (`PIPO_CONF_THRESHOLD >= 0` ⇒ PAD-replace where conf < threshold; otherwise commit). There is no entropy-tau fallback path.

### Training env vars
| Stage | Env | Default | Purpose |
|-------|-----|---------|---------|
| SFT ([`swift_sft.sh`](swift_sft.sh)) | `SFT_CONF_LOSS_WEIGHT` | `1` | Float weight for SFT-stage BCE conf loss; consumed inside `Qwen3_5ForCausalPIPO.forward` (NOT the OPD trainer). |
| SFT ([`swift_sft.sh`](swift_sft.sh)) | `SFT_CONF_DETACH_INPUTS` | `1` | Detach `(backbone_hidden, mtp_hidden)` before the head — guards against the degenerate solution where MTP sacrifices token2 quality to game the head. |
| OPD ([`swift_opd.sh`](swift_opd.sh)) | `OPD_CONF_LOSS_WEIGHT` | `1` | Float weight for BCE conf loss in the total OPD loss. |
| OPD ([`swift_opd.sh`](swift_opd.sh)) | `OPD_CONF_DETACH_INPUTS` | `1` | Same role as `SFT_CONF_DETACH_INPUTS` but for the OPD-stage path. |

### Inference env / CLI args
| CLI / env | Default | Purpose |
|-----------|---------|---------|
| `--enable-pipo` (SGLang CLI) / `engine_kwargs["enable_pipo"]=True` ([`sglang_eval.py`](sglang_eval.py)) | False | Switch on the PIPO inference path (pair compression + 2-phase decode). |
| `--pipo_conf_threshold X` ([`sglang_eval.py`](sglang_eval.py) CLI flag) / `PIPO_CONF_THRESHOLD=X` (env, read by SGLang's `tp_worker`) | `-1` | Commit threshold ∈ [0,1] on `sigmoid(conf_logit)`. Negative disables gating (always commit). `1.0` = always PAD (skips Phase 2 entirely). `0.0` = always commit. |

### Diagnostic metrics emitted (training)
* **OPD stage** ([`pipo/trainer/swift_gkd_trainer.py`](pipo/trainer/swift_gkd_trainer.py)): `opd_conf_loss`, `opd_conf_mean_pred`, `opd_conf_mean_target`, `opd_conf_num_valid`. Healthy regime: `mean_target ∈ [0.7, 0.9]` and `mean_pred` tracks it with a small lag.
* **SFT stage** ([`pipo/trainer/swift_sft_trainer.py`](pipo/trainer/swift_sft_trainer.py)): `conf_loss` is logged (alongside `backbone_loss` / `mtp_loss`). Decreasing `conf_loss` over training is the primary signal.

### Key invariants
* Compressor + ConfidenceHead are ALWAYS instantiated for PIPO checkpoints (HF model unconditionally adds them in `__init__`; SGLang model does the same).
* Targets are always `.detach()`-ed before BCE — head is a post-hoc evaluator.
* `OPD_CONF_DETACH_INPUTS=1` (default) detaches inputs to the head — forbids gradient leakage back into the MTP/backbone representations.
* SGLang `load_weights` uses prefix match; `confidence_head.*` keys flow through with no rename (same param names on both sides).
* **Ghost forward for ZeRO-2 alignment**: when all valid positions in a batch are masked out, the chunk loop never calls `confidence_head`. Returning a 0-grad scalar without the head in the graph causes DDP/ZeRO-2 to hang at the ALLREDUCE for the conf-head bucket. `_finalise_conf` (L1027-1093) detects `n==0` and runs a tiny detached ghost forward through the head (scaled by 0) so every parameter gets a zero gradient, keeping the bucket aligned across ranks.

---

## Evaluation Entrypoint

File: [`sglang_eval.py`](sglang_eval.py) — in-process `sglang.Engine` evaluation (no separate HTTP server).

| Arg | Default | Notes |
|-----|---------|-------|
| `--model_path` | `Qwen/Qwen3.5-4B` | Checkpoint path |
| `--tp_size` | 1 | Tensor parallelism |
| `--dp_size` | 8 | Data parallelism |
| `--enable_pipo` | False | Switch on the PIPO inference path |
| `--pipo_conf_threshold` | -1.0 | Confidence head commit threshold (exported to env as `PIPO_CONF_THRESHOLD`) |
| `--enable_eagle` | False | (alternative path) NEXTN EAGLE-2 speculative decoding |
| `--disable_cuda_graph` | False | Disable CUDA graph for debugging |
| `--datasets` | `aime2025,gpqa_diamond,livecodebench,lb2` | Comma-separated dataset names (see [`pipo/eval/benchmark_loader.py`](pipo/eval/benchmark_loader.py)) |
| `--num_samples` | 4 | Samples per question |
| `--max_generated_tokens` | 32768 | Max new tokens |
| `--presence_penalty` | 1.5 | Presence penalty for generation |

**Experiment directory naming**: `exp_name` is built from sampling params; when PIPO is enabled, the conf threshold is appended.

**PIPO engine initialization** (L287-290):
- `enable_pipo=True`
- `disable_radix_cache=True` (forced, required for correctness)
- `os.environ["PIPO_CONF_THRESHOLD"] = str(args.pipo_conf_threshold)`

**Resume support**: Loads already-processed `(dataset, micro_index)` pairs from existing `{dataset}-results.jsonl` files and skips them. Supports `--start_index` / `--end_index` and `--remaining_ratio_start` / `--remaining_ratio_end` for partial evaluation.

**Post-hoc evaluation** ([`pipo/eval/eval.sh`](pipo/eval/eval.sh) — invoked automatically unless `--skip_eval`):
1. `pipo/eval/eval_1_rule.py` — rule-based scoring for AIME / GPQA / LongBench.
2. `pipo/eval/eval_2_lcb.sh` — LiveCodeBench in-place evaluation (uses `third_party/LiveCodeBench/lcb_runner`).
3. `pipo/eval/eval_3_export_to_excel.py` — aggregate stats and write `stats.xlsx`.

Execution-based datasets (`codeforces`, `livecodebench`, `ifbench`) skip rule-based evaluation; only text and metadata are saved. Other datasets use `evaluator.evaluator_map[dataset]` for answer extraction and accuracy scoring. Results are written atomically with file locking (`fcntl.LOCK_EX`).

---

## Environment Variables Complete Reference

### SFT-stage ([`swift_sft.sh`](swift_sft.sh))
| Variable | Default | Range | Purpose |
|----------|---------|-------|---------|
| `COMPRESSOR_TYPE` | `mlp` | `linear` / `mlp` | Compressor architecture |
| `PIPO_MAX_PAD_RATIO` | `0.25` | `0.0`–`1.0` | Upper bound of per-step PAD-ratio sampling (SFT-stage data augmentation only; not used at inference) |
| `PIPO_LOGITS_CHUNK` | `2048` | int > 0 | Logits chunk size for SFT forward |
| `MTP_LOSS_WEIGHT` | `1.0` | float | Weight of MTP loss in total loss |
| `SFT_CONF_LOSS_WEIGHT` | `1` | float ≥ 0 | BCE weight for SFT-stage confidence head |
| `SFT_CONF_DETACH_INPUTS` | `1` | `0` / `1` | Detach inputs before confidence head |

### OPD-stage ([`swift_opd.sh`](swift_opd.sh))
| Variable | Default | Range | Purpose |
|----------|---------|-------|---------|
| `COMPRESSOR_TYPE` | `mlp` | `linear` / `mlp` | Compressor architecture |
| `OPD_KL_MODE` | `topk` (script default) / `sampled` (trainer default) | `sampled` / `topk` / `full` | KL divergence approximation mode |
| `OPD_TOPK_SOURCE` | `teacher` | `teacher` / `student` | **Deprecated / ignored.** Accepted for launch-script backward-compat. `OPD_KL_MODE=topk` always uses the union of student top-k, teacher top-k, and the sampled label index. |
| `OPD_CONF_LOSS_WEIGHT` | `1` | float ≥ 0 | BCE weight for OPD-stage confidence head |
| `OPD_CONF_DETACH_INPUTS` | `1` | `0` / `1` | Detach inputs before confidence head |
| `PIPO_OPD_CHUNK_SIZE` | `4096` (script) / `1024` (trainer default) | int > 0 | LM-head chunk size for OPD loss computation |
| `PIPO_EMPTY_CACHE_STEPS` | `1` | int ≥ 0 | GPU memory cleanup frequency (0 = disable). Default 1 = every step. |
| `PIPO_DEBUG_WEIGHT_SYNC` | `0` | `0` / `1` | Enable compressor weight-sync parity checks on first OPD weight sync |
| `PIPO_CONF_THRESHOLD` | `-1` | float | Confidence threshold for SGLang rollout (`<0` = disabled). Currently we rely on the random-PAD SFT augmentation to expose the model to PAD distribution; rollout-time gating is OFF by default. |
| `SGLANG_TP_SIZE` | `1` | int > 0 | SGLang tensor parallelism size |
| `SGLANG_MEM_FRACTION_STATIC` | `0.3` | float | SGLang static memory fraction |
| `SGLANG_CONTEXT_LENGTH` | (empty) | int | SGLang context length override |
| `GKD_LOGITS_TOPK` | `32` | int > 0 | Top-k candidate count when `OPD_KL_MODE=topk` |
| `PRESENCE_PENALTY` | `1.5` | float | Presence penalty for rollout generation |
| `SLEEP_LEVEL` | `0` | `0` / `1` | SGLang sleep level (`0` = keep weights GPU-resident; `1` = release KV cache between rollouts) |
| `ATTN_IMPL` | `flash_attention_2` | `flash_attention_2` / `sdpa` | Attention implementation. Use `sdpa` to avoid upstream Qwen3.5 FA2 illegal-memory-access bug at batch_size=1 with mRoPE. |
| `PER_DEVICE_TRAIN_BATCH_SIZE` | `1` | int > 0 | Per-device train batch size |
| `GRADIENT_ACCUMULATION_STEPS` | `4` | int > 0 | Gradient accumulation steps |
| `TRAIN_DATALOADER_SHUFFLE` | `true` | `true` / `false` | Shuffle training dataloader |

### SGLang serving (read by `tp_worker`)
| Variable | Default | Purpose |
|----------|---------|---------|
| `PIPO_CONF_THRESHOLD` | `-1` | Sole gating knob at inference. `<0` = always commit token2; `[0, 1)` = PAD-replace where `sigmoid(conf_logit) < threshold`; `>= 1.0` = always PAD (Phase 2 is skipped entirely for latency). |

---

## Environment

```bash
conda activate pipo           # training/inference/SGLang environment
export PYTHONPATH="/path/to/PIPO:/path/to/PIPO/third_party/ms-swift:$PYTHONPATH"
```

Qwen3.5-4B checkpoint cache: `~/.cache/huggingface/hub/models--Qwen--Qwen3.5-4B/`

---

## Key File Index

| File | Purpose |
|------|---------|
| [`pipo/qwen3_5/modeling_qwen3_5_mtp.py`](pipo/qwen3_5/modeling_qwen3_5_mtp.py) | Training HF model: `Qwen3_5ForCausalPIPO`, MTP predictor, chunked CE / chunked conf loss |
| [`pipo/qwen3_5/configuration_qwen3_5.py`](pipo/qwen3_5/configuration_qwen3_5.py) | Config class with `compressor_type`, `mtp_num_hidden_layers` |
| [`pipo/qwen3_5/compressor.py`](pipo/qwen3_5/compressor.py) | `PIPOCompressor` hierarchy + `ConfidenceHead` (shared by training and SGLang) |
| [`pipo/trainer/swift_plugin.py`](pipo/trainer/swift_plugin.py) | ms-swift plugin: model registration + loader + GKD trainer re-route |
| [`pipo/trainer/swift_sft_trainer.py`](pipo/trainer/swift_sft_trainer.py) | SFT trainer (`PIPOSeq2SeqTrainer`): compute_loss, random PAD insertion, per-component logging |
| [`pipo/trainer/swift_gkd_trainer.py`](pipo/trainer/swift_gkd_trainer.py) | OPD trainer (`PIPOGKDTrainer`): SGLang rollout, compressed student forward, PAD-compacted teacher forward, chunked JSD/sampled-KL, confidence head |
| [`swift_sft.sh`](swift_sft.sh) | SFT training launch script |
| [`swift_opd.sh`](swift_opd.sh) | OPD training launch script (registers PIPO plugin, forces SGLang `enable_pipo` + `disable_radix_cache`) |
| [`sglang_eval.py`](sglang_eval.py) | In-process `sglang.Engine` evaluation, optionally invokes the `pipo/eval/eval.sh` pipeline |
| [`pipo/eval/eval.sh`](pipo/eval/eval.sh) | 3-stage eval pipeline: rule-based → LCB → excel export |
| [`pipo/eval/benchmark_loader.py`](pipo/eval/benchmark_loader.py) | Dataset loaders for AIME / GPQA / LiveCodeBench / LongBench / IFBench |
| [`pipo/eval/evaluator.py`](pipo/eval/evaluator.py) | Per-dataset rule-based evaluators |
| [`pipo/eval/eval_utils.py`](pipo/eval/eval_utils.py) | Shared utilities: `RULE_DATASETS`, `EXCEL_BENCHMARKS`, jsonl helpers, LCB conversion |
| [`third_party/sglang/python/sglang/srt/models/qwen3_5_pipo.py`](third_party/sglang/python/sglang/srt/models/qwen3_5_pipo.py) | SGLang inference model with compressor + two-phase decode |
| [`third_party/sglang/python/sglang/srt/models/qwen3_5_latent.py`](third_party/sglang/python/sglang/srt/models/qwen3_5_latent.py) | SGLang backbone (shared by main model and MTP block); "latent" refers to the compressed pair latent space PIPO operates on |
| [`third_party/sglang/python/sglang/srt/managers/tp_worker.py`](third_party/sglang/python/sglang/srt/managers/tp_worker.py) | Two-phase sampling logic (L472-601) + `PIPO_CONF_THRESHOLD` gating |
| [`third_party/sglang/python/sglang/srt/managers/schedule_batch.py`](third_party/sglang/python/sglang/srt/managers/schedule_batch.py) | `input_ids_pair` + `pipo_padded_input_ids` preparation for prefill / decode |
| [`third_party/sglang/python/sglang/srt/model_executor/forward_batch_info.py`](third_party/sglang/python/sglang/srt/model_executor/forward_batch_info.py) | `ForwardBatch.pipo_*` fields |
| [`third_party/sglang/python/sglang/srt/model_executor/cuda_graph_runner.py`](third_party/sglang/python/sglang/srt/model_executor/cuda_graph_runner.py) | CUDA Graph capture/replay with PIPO support (`pipo_backbone_hidden` buffer cache) |
| [`third_party/sglang/python/sglang/srt/model_executor/model_runner.py`](third_party/sglang/python/sglang/srt/model_executor/model_runner.py) | Registers MTP layer id as full-attention for `Qwen3_5ForCausalPIPO` |
| [`third_party/ms-swift/swift/trainers/trainer_factory.py`](third_party/ms-swift/swift/trainers/trainer_factory.py) | Maps `task_type=pipo` → `pipo.trainer.swift_sft_trainer.PIPOSeq2SeqTrainer` |
