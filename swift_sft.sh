#!/usr/bin/env bash
set -euo pipefail

PROJ_DIR="$(cd "$(dirname "$0")" && pwd)"
export PYTHONPATH="${PROJ_DIR}:${PROJ_DIR}/third_party/ms-swift${PYTHONPATH:+:${PYTHONPATH}}"
export HF_HUB_OFFLINE=1
# Cap per-process intra-op thread pools so 8 DDP ranks on a many-core host
# don't exhaust pthread resources / per-process VMA limit (vm.max_map_count)
# during tokenizer/NCCL init. Without these, EAGAIN bubbles up as either
# Rust tokenizers Rayon panic or NCCL ProcessGroup "Resource temporarily
# unavailable".
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
# export RAYON_NUM_THREADS="${RAYON_NUM_THREADS:-1}"
# export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
# export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
# export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
# export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

compressor_type="${1:-mlp}"
dataset_name="${2:-sft_all}"
max_length="${3:-65535}"
max_pad_ratio="${4:-0.25}"
model_name="${5:-Qwen3.5-4B}"
tuner_type="${6:-lora}"
num_train_epochs="${7:-2}"
resume_from_checkpoint="${8:-}"
gradient_accumulation_steps="${9:-16}"
log_suffix="${10:-}"

export PIPO_LOGITS_CHUNK="${PIPO_LOGITS_CHUNK:-2048}"

# ── SFT-stage ConfidenceHead supervision ───────────────────────────────
# Current research config keeps the confidence head ON and all train-side
# loss weights at 1.  SFT target: p_s_mtp(label_token2) ∈ [0, 1], used as a
# warm-start calibration signal before OPD trains the sampled-token EAGLE target.
export SFT_CONF_LOSS_WEIGHT="${SFT_CONF_LOSS_WEIGHT:-1}"
export SFT_CONF_DETACH_INPUTS="${SFT_CONF_DETACH_INPUTS:-1}"
export MTP_LOSS_WEIGHT="${MTP_LOSS_WEIGHT:-1}"

# ── Tuner-specific flags ───────────────────────────────────────────────
# Two branches:
#   * lora — LoRA-only flags (--lora_rank / --lora_alpha / --target_modules
#            / --modules_to_save) + LR 1e-4 (standard LoRA regime).
#   * full — ALL params trainable (backbone + compressor + MTP block +
#            lm_head/embed_tokens via _tied_weights_keys + confidence_head).
#            LR drops to 1e-5 since every weight is updated directly.
# LR can still be overridden by exporting LEARNING_RATE before invocation.
tuner_args=(--tuner_type "${tuner_type}")
if [[ "${tuner_type}" == "lora" ]]; then
  tuner_args+=(
    --lora_rank 64
    --lora_alpha 128
    --target_modules q_proj k_proj v_proj o_proj gate_proj up_proj down_proj
    --modules_to_save confidence_head compressor mtp.fc mtp.pre_fc_norm_hidden mtp.pre_fc_norm_embedding mtp.norm mtp.layers.0.input_layernorm mtp.layers.0.post_attention_layernorm mtp.layers.0.self_attn.q_norm mtp.layers.0.self_attn.k_norm
    --learning_rate "${LEARNING_RATE:-1e-4}"
  )
elif [[ "${tuner_type}" == "full" ]]; then
  tuner_args+=(
    --learning_rate "${LEARNING_RATE:-1e-5}"
  )
else
  echo "[swift_sft.sh] unknown tuner_type=${tuner_type}, expected 'lora' or 'full'" >&2
  exit 1
fi

if [[ "${tuner_type}" == "lora" ]]; then
  output_name="sft_${compressor_type}_${dataset_name}_${max_length}_${max_pad_ratio}_${num_train_epochs}epochs${log_suffix:+_${log_suffix}}"
else
  output_name="sft_${compressor_type}_${dataset_name}_${max_length}_${max_pad_ratio}_${num_train_epochs}epochs${log_suffix:+_${log_suffix}}"
fi

COMPRESSOR_TYPE="${compressor_type}" \
PIPO_MAX_PAD_RATIO="${max_pad_ratio}" \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
NPROC_PER_NODE=8 \
swift sft \
  --use_hf true \
  --model "Qwen/${model_name}" \
  --model_type qwen3_5_mtp \
  --external_plugins "${PROJ_DIR}/pipo/trainer/swift_plugin.py" \
  --cached_dataset "${PROJ_DIR}/data/${dataset_name}.jsonl.cache/train" \
  --save_steps 100 \
  --logging_steps 1 \
  --template qwen3_5 \
  --max_length ${max_length} \
  --truncation_strategy right \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps "${gradient_accumulation_steps}" \
  --warmup_ratio 0.05 \
  --num_train_epochs ${num_train_epochs} \
  --output_dir "${PROJ_DIR}/outputs/${model_name}/${output_name}" \
  --attn_impl flash_attention_2 \
  "${tuner_args[@]}" \
  --add_version false \
  --load_from_cache_file false \
  --deepspeed zero2 \
  --gradient_checkpointing true \
  --gradient_checkpointing_kwargs '{"use_reentrant": false}' \
  --group_by_length true \
  --resume_from_checkpoint "${resume_from_checkpoint}"
