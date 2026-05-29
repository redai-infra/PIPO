pip install uv
cd third_party/sglang && uv pip install -e "python" && cd ..
cd ms-swift && uv pip install -e . && cd ../..
uv pip install -r requirements.txt

# final fix
uv pip install nvidia-cudnn-cu12==9.16.0.29