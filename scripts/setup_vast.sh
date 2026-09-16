#!/usr/bin/env bash
# vast.ai（RTX PRO 6000 Blackwell 等）一键环境准备。
# 建议镜像：pytorch/pytorch:*-cuda12.8-cudnn9-devel 或更新的 devel 版（需要 nvcc）。
#
#   git clone git@github.com:clveryang/Llm_infer.git && cd Llm_infer
#   bash scripts/setup_vast.sh                 # 默认下载 DeepSeek-V2-Lite-Chat
#   MODEL=deepseek-ai/DeepSeek-V2-Lite bash scripts/setup_vast.sh
set -euo pipefail

MODEL=${MODEL:-deepseek-ai/DeepSeek-V2-Lite-Chat}
MODEL_DIR=${MODEL_DIR:-checkpoints/$(basename "$MODEL")}

echo "==> GPU / 驱动"
nvidia-smi --query-gpu=name,memory.total,driver_version,compute_cap --format=csv
if ! command -v nvcc >/dev/null; then
  echo "!! 没有 nvcc：以后写 CUDA 内核需要 devel 镜像（当前仍可跑 CPU 内核 + torch 回退路径）"
else
  nvcc --version | tail -1
fi

echo "==> PyTorch"
CAP=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d .)
if ! python -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() and 'sm_$CAP' in torch.cuda.get_arch_list() else 1)" 2>/dev/null; then
  echo "当前 torch 不支持 sm_$CAP，安装 cu128 版本"
  pip install -U torch --index-url https://download.pytorch.org/whl/cu128
fi
python -c "import torch; print(torch.__version__, 'cuda', torch.version.cuda, torch.cuda.get_device_name(), torch.cuda.get_arch_list())"

echo "==> 依赖"
pip install -U transformers safetensors pytest ninja hf_transfer "huggingface_hub[cli]"

echo "==> 编译 llm_infer"
export TORCH_CUDA_ARCH_LIST="$(python -c "import torch; print('.'.join(map(str, torch.cuda.get_device_capability())))")"
MAX_JOBS=${MAX_JOBS:-$(nproc)} pip install -e . --no-build-isolation -v 2>&1 | tail -5

echo "==> 单元测试"
python -m pytest -q tests

echo "==> 下载 $MODEL 到 $MODEL_DIR（约 32GB）"
export HF_HUB_ENABLE_HF_TRANSFER=1
hf download "$MODEL" --local-dir "$MODEL_DIR"

echo "==> 和 transformers 对比真实权重"
python scripts/compare_hf.py --model-dir "$MODEL_DIR"

echo "完成。试试：python examples/generate.py --model-dir $MODEL_DIR --prompt '你好，介绍一下 MLA'"
