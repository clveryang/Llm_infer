#!/usr/bin/env bash
# 在租来的 GPU 机器（vast.ai 等）上一键准备环境并验证。
# 建议镜像：pytorch/pytorch:*-cuda12.8-cudnn9-devel 或更新的 devel 版（必须有 nvcc）。
#
#   bash scripts/setup_vast.sh                  # 全流程：编译 → 测试 → 跑分 → 下载模型 → 对比
#   SKIP_MODEL=1 bash scripts/setup_vast.sh     # 只到跑分为止，不下载 32GB 权重（几分钟就能验证内核）
#   MODEL=deepseek-ai/DeepSeek-V2-Lite bash scripts/setup_vast.sh
set -euo pipefail

MODEL=${MODEL:-deepseek-ai/DeepSeek-V2-Lite-Chat}
MODEL_DIR=${MODEL_DIR:-checkpoints/$(basename "$MODEL")}
SKIP_MODEL=${SKIP_MODEL:-0}
PY=${PY:-python}

step() { echo; echo "=========== $* ==========="; }

step "1/7 GPU 和驱动"
nvidia-smi --query-gpu=name,memory.total,driver_version,compute_cap --format=csv
CAP=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1)
CAP_NODOT=${CAP//./}
if ! command -v nvcc >/dev/null; then
  echo "错误：没有 nvcc，无法编译 CUDA 内核。请换成 devel 镜像（带 -devel 后缀）。"
  exit 1
fi
nvcc --version | tail -1

step "2/7 PyTorch"
if ! $PY -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() and 'sm_$CAP_NODOT' in torch.cuda.get_arch_list() else 1)" 2>/dev/null; then
  echo "当前 torch 不支持 sm_$CAP_NODOT，安装 cu128 版本"
  pip install -U torch --index-url https://download.pytorch.org/whl/cu128
fi
$PY -c "import torch; print(torch.__version__, 'cuda', torch.version.cuda, '|', torch.cuda.get_device_name(), '|', torch.cuda.get_arch_list())"

step "3/7 依赖"
pip install -q -U transformers safetensors pytest ninja hf_transfer "huggingface_hub[cli]"

step "4/7 编译（针对 sm_$CAP_NODOT）"
export TORCH_CUDA_ARCH_LIST="$CAP"
NPROC=$(nproc)
export MAX_JOBS=${MAX_JOBS:-$(( NPROC < 8 ? NPROC : 8 ))}   # 每个 nvcc 进程吃 1-2GB 内存，别开满
time pip install -e . --no-build-isolation 2>&1 | tail -3
$PY - <<'EOF'
import torch
from llm_infer import ops
names = ["rms_norm", "fused_add_rms_norm", "silu_and_mul", "rotary_embedding", "concat_and_cache_mla",
         "mla_prefill_attention", "mla_decode_attention", "topk_softmax", "moe_align",
         "moe_expert_forward", "moe_combine"]
missing = [n for n in names if not ops.has_kernel(n, torch.device("cuda"))]
print(f"CUDA 内核已注册 {len(names) - len(missing)}/{len(names)}")
if missing:
    raise SystemExit(f"缺少 CUDA 内核: {missing}")
EOF

step "5/7 单元测试（算子对拍 + 小模型对比 transformers）"
$PY -m pytest -q tests

step "6/7 性能对比（自写内核 vs PyTorch 参考实现）"
$PY scripts/bench.py

if [ "$SKIP_MODEL" = "1" ]; then
  echo; echo "SKIP_MODEL=1，到此为止。内核已验证可用。"
  exit 0
fi

step "7/7 下载 $MODEL（约 32GB）并和 transformers 对比"
export HF_HUB_ENABLE_HF_TRANSFER=1
hf download "$MODEL" --local-dir "$MODEL_DIR"
$PY scripts/compare_hf.py --model-dir "$MODEL_DIR"

echo
echo "全部完成。接下来可以："
echo "  $PY examples/generate.py --model-dir $MODEL_DIR --prompt '介绍一下 MLA'"
echo "  $PY scripts/bench.py --model-dir $MODEL_DIR --e2e"
