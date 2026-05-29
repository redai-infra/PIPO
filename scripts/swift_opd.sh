#!/usr/bin/env bash
# Notes
#   * `--rlhf_type gkd` triggers TrainerFactory; the plugin re-routes it to
#     PIPOGKDTrainer, which forces enable_pipo=True / disable_radix_cache=True
#     on the SGLang engine regardless of CLI args.
#   * `--rollout_backend sglang` triggers the SGLang-colocate path inside
#     RolloutTrainerMixin._prepare_vllm.
#   * `--vllm_enable_lora false` keeps LoRA weights merged into the base before
#     each weight sync (LoRA hot-swap on SGLang is out of scope).
#   * `--sleep_level 1` releases the SGLang KV cache between rollouts — required
#     to fit the trainer's optimizer states alongside the rollout engine.
#
# Current research config (kept fixed unless explicitly overridden):
#   * Compressor + Confidence head are ALWAYS instantiated for PIPO.
#   * All train-side loss weights are 1: MTP_LOSS_WEIGHT=1 and OPD_CONF_LOSS_WEIGHT=1.
#   * OPD uses sampled-token reverse KL: OPD_KL_MODE=sampled, beta=1.0, per-token
#     loss = log P_student(y) - log P_teacher(y).
#   * Inference-time gating ONLY uses confidence head threshold
#     (PIPO_CONF_THRESHOLD); entropy-tau and random-PAD gates have been removed.
#
# Confidence head env vars (post-hoc commit predictor for MTP token2):
#   OPD_CONF_LOSS_WEIGHT     float weight for the BCE conf loss                (default: 1)
#   OPD_CONF_DETACH_INPUTS   1 to detach (back, mtp) hiddens from conf head    (default: 1)
#                            target is sampled-token EAGLE accept prob:
#                            min(1, P_teacher(y) / P_student(y))
#   PIPO_CONF_THRESHOLD      inference threshold ∈ [0,1] on sigmoid(conf)      (default: -1
#                            (consumed by SGLang tp_worker; <0 disables gating). = disabled)
# ============================================================================
set -euo pipefail

PROJ_DIR="$(cd "$(dirname "$0")" && pwd)"
export PYTHONPATH="${PROJ_DIR}:${PROJ_DIR}/third_party/ms-swift${PYTHONPATH:+:${PYTHONPATH}}"
export HF_HUB_OFFLINE=1
# Cap per-process intra-op thread pools so 8 DDP ranks + colocated SGLang
# scheduler subprocesses on a many-core host don't exhaust pthread resources
# / per-process VMA limit (vm.max_map_count) during tokenizer + NCCL init.
# Without these, EAGAIN bubbles up as either Rust tokenizers Rayon panic or
# NCCL ProcessGroup "Resource temporarily unavailable".
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
# export RAYON_NUM_THREADS="${RAYON_NUM_THREADS:-1}"
# export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
# export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
# export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
# export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

student_ckpt="${1:-}"
dataset="${2:-data/rl_0.5.jsonl}"
teacher_model="${3:-Qwen/Qwen3.5-9B}"
output_suffix="${4:-}"
num_train_epochs="${5:-1}"

pipo_conf_threshold="${PIPO_CONF_THRESHOLD:-0.95}"

if [[ -z "${student_ckpt}" || -z "${teacher_model}" || -z "${dataset}" ]]; then
    echo "Usage: $0 <student_ckpt> <dataset> <teacher_model> [output_suffix] [num_train_epochs]" >&2
    exit 1
fi

base_model="$(basename "$(dirname "$(dirname "$student_ckpt")")")"
sft_exp_name="$(basename "$(dirname "$student_ckpt")")"
dataset_basename="${dataset#data/}"; dataset_basename="${dataset_basename%.jsonl}"
out_dir="${PROJ_DIR}/outputs/${base_model}/${sft_exp_name}_opd_${dataset_basename}_${pipo_conf_threshold}_${output_suffix:+_${output_suffix}}"

# SGLang-side tunables
sglang_tp_size="${SGLANG_TP_SIZE:-1}"
sglang_mem_fraction_static="${SGLANG_MEM_FRACTION_STATIC:-0.35}"
opd_kl_mode="${OPD_KL_MODE:-topk}"
gkd_logits_topk="${GKD_LOGITS_TOPK:-32}"
gkd_logits_topk_args=()
if [[ -n "${gkd_logits_topk}" ]]; then
    gkd_logits_topk_args=(--gkd_logits_topk "${gkd_logits_topk}")
fi
opd_topk_source="${OPD_TOPK_SOURCE:-teacher}"
presence_penalty="${PRESENCE_PENALTY:-1.5}"
opd_max_length="${OPD_MAX_LENGTH:-66559}"  # 65535 + 1024 (prompt)
opd_max_completion_length="${OPD_MAX_COMPLETION_LENGTH:-65535}"
sglang_context_length="${SGLANG_CONTEXT_LENGTH:-}"
sglang_context_args=()
if [[ -n "${sglang_context_length}" ]]; then
    sglang_context_args=(--sglang_context_length "${sglang_context_length}")
fi
# setting sleep_level to 1 or 2 will NOT save memory due to enable_memory_saver BUG
sleep_level="${SLEEP_LEVEL:-0}"
attn_impl="${ATTN_IMPL:-flash_attention_2}"
ddp_timeout="${DDP_TIMEOUT:-7200}"

# Confidence head tunables (training side — see header for description).
opd_conf_loss_weight="${OPD_CONF_LOSS_WEIGHT:-1}"
opd_conf_detach_inputs="${OPD_CONF_DETACH_INPUTS:-1}"

train_dataloader_shuffle="${TRAIN_DATALOADER_SHUFFLE:-true}"
per_device_train_batch_size="${PER_DEVICE_TRAIN_BATCH_SIZE:-4}"
gradient_accumulation_steps="${GRADIENT_ACCUMULATION_STEPS:-1}"
# if VRAM < 120GB per GPU, use this:
# per_device_train_batch_size="${PER_DEVICE_TRAIN_BATCH_SIZE:-1}"
# gradient_accumulation_steps="${GRADIENT_ACCUMULATION_STEPS:-4}"

# Distributed/NCCL timeout settings for slow rollout initialization and long-tail CoT steps.
export DEEPSPEED_TIMEOUT="${DEEPSPEED_TIMEOUT:-${ddp_timeout}}"
export NCCL_TIMEOUT="${NCCL_TIMEOUT:-${ddp_timeout}}"
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC="${TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC:-${ddp_timeout}}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
export TORCH_DISTRIBUTED_DEBUG="${TORCH_DISTRIBUTED_DEBUG:-INFO}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}" \
NPROC_PER_NODE="${NPROC_PER_NODE:-8}" \
COMPRESSOR_TYPE="${COMPRESSOR_TYPE:-mlp}" \
MTP_LOSS_WEIGHT="${MTP_LOSS_WEIGHT:-1}" \
PIPO_CONF_THRESHOLD="${pipo_conf_threshold}" \
OPD_KL_MODE="${opd_kl_mode}" \
OPD_TOPK_SOURCE="${opd_topk_source}" \
OPD_CONF_LOSS_WEIGHT="${opd_conf_loss_weight}" \
OPD_CONF_DETACH_INPUTS="${opd_conf_detach_inputs}" \
PIPO_OPD_CHUNK_SIZE="${PIPO_OPD_CHUNK_SIZE:-4096}" \
PIPO_EMPTY_CACHE_STEPS="${PIPO_EMPTY_CACHE_STEPS:-1}" \
swift rlhf \
  --use_hf=true \
  --rlhf_type gkd \
  --model "${student_ckpt}" \
  --model_type qwen3_5_mtp \
  --teacher_model "${teacher_model}" \
  --external_plugins "${PROJ_DIR}/pipo/trainer/swift_plugin.py" \
  --dataset "${dataset}" \
  --template qwen3_5 \
  --tuner_type lora \
  --lora_rank 64 \
  --lora_alpha 128 \
  --target_modules q_proj k_proj v_proj o_proj gate_proj up_proj down_proj \
  --modules_to_save compressor confidence_head mtp.fc mtp.pre_fc_norm_hidden mtp.pre_fc_norm_embedding mtp.norm \
  --max_length "${opd_max_length}" \
  --max_completion_length "${opd_max_completion_length}" \
  --ddp_timeout "${ddp_timeout}" \
  --truncation_strategy right \
  --per_device_train_batch_size "${per_device_train_batch_size}" \
  --gradient_accumulation_steps "${gradient_accumulation_steps}" \
  --train_dataloader_shuffle "${train_dataloader_shuffle}" \
  --num_train_epochs "${num_train_epochs}" \
  --warmup_ratio 0.05 \
  --logging_steps 1 \
  --save_steps 10 \
  --output_dir "${out_dir}" \
  --attn_impl "${attn_impl}" \
  --deepspeed zero2 \
  --gradient_checkpointing true \
  --gradient_checkpointing_kwargs '{"use_reentrant": false}' \
  --lmbda 1.0 \
  --beta 1.0 \
  --temperature 1.0 \
  --top_p 0.95 \
  --top_k 20 \
  --presence_penalty "${presence_penalty}" \
  --use_vllm true \
  --vllm_mode colocate \
  --rollout_backend sglang \
  --sglang_tp_size "${sglang_tp_size}" \
  --sglang_mem_fraction_static "${sglang_mem_fraction_static}" \
  --sglang_enable_pipo true \
  --sglang_disable_radix_cache true \
  "${sglang_context_args[@]}" \
  --sleep_level "${sleep_level}" \
  --vllm_enable_lora false \
  --log_completions true \
  --add_non_thinking_prefix false \
  --add_version false \
  --offload_teacher_model false \
  "${gkd_logits_topk_args[@]}"
