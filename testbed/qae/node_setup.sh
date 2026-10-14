#!/usr/bin/env bash
# One-shot setup of a fresh Chameleon CC-Ubuntu22.04-CUDA node for QAE scoring/training.
# Run as user cc after `qae/` and `qa/` files were copied (see qae/node_push.sh). Idempotent.
set -euo pipefail
cd ~
sudo apt-get update -qq && sudo apt-get install -y -qq python3-venv rsync > /dev/null
[ -d ~/venv ] || python3 -m venv ~/venv
source ~/venv/bin/activate
pip install -q --upgrade pip
pip install -q torch torchvision transformers accelerate pillow numpy opencv-python-headless scipy
mkdir -p ~/qae ~/qa/qae_batch/precheck ~/qa/qae_batch/sweeps ~/qa_scoring ~/qa/feats_qae
cp ~/qa/jepa_qa.py ~/qa_scoring/ 2>/dev/null || true
python - <<'EOF'
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration, AutoModel, CLIPModel, CLIPTokenizer
import torch
AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")
Qwen2_5_VLForConditionalGeneration.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct", torch_dtype=torch.bfloat16)
AutoModel.from_pretrained("facebook/dinov2-small")
CLIPModel.from_pretrained("openai/clip-vit-base-patch32"); CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
print("models cached")
EOF
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
bash ~/qae/start_loop.sh
echo SETUP_DONE
