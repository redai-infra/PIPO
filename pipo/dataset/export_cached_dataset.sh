jsonl_path="$1"
model="${2:-Qwen/Qwen3.5-4B}"
output_dir="${jsonl_path}.cache"
swift export \
--to_cached_dataset true \
--model "${model}" \
--use_hf true \
--dataset "${jsonl_path}" \
--output_dir "${output_dir}" \
--add_non_thinking_prefix false  # IMPORTANT!
