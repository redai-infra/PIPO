# download models
hf download Qwen/Qwen3.5-4B
hf download Qwen/Qwen3.5-9B

# download eval datasets
hf download opencompass/AIME2025 --repo-type dataset
hf download livecodebench/code_generation_lite --repo-type dataset  # Large dataset (~900GB)
hf download Idavidrein/gpqa --repo-type dataset
hf download zai-org/LongBench-v2 --repo-type dataset

# Optional: download training datasets
# hf download open-r1/DAPO-Math-17k-Processed --repo-type dataset
# hf download open-r1/codeforces --repo-type dataset
