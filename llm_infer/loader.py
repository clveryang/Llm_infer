"""从 Hub 格式的 safetensors 加载 DeepSeek-V2 权重。

Hub 上每个专家是独立的 gate_proj/up_proj/down_proj，这里直接拷进堆叠好的
experts.gate_up_proj [E, 2I, D] / experts.down_proj [E, D, I]，不额外占一份显存。
"""

import glob
import os
import re

import torch
from safetensors import safe_open

from .config import ModelConfig
from .model import DeepseekV2ForCausalLM

_EXPERT_RE = re.compile(r"^(model\.layers\.\d+\.mlp\.experts)\.(\d+)\.(gate_proj|up_proj|down_proj)\.weight$")


def load_weights(model: DeepseekV2ForCausalLM, model_dir: str) -> None:
    params = dict(model.named_parameters())
    inter = model.config.moe_intermediate_size
    loaded = set()

    files = sorted(glob.glob(os.path.join(model_dir, "*.safetensors")))
    if not files:
        raise FileNotFoundError(f"{model_dir} 下没有 .safetensors 文件")

    with torch.no_grad():
        for path in files:
            with safe_open(path, framework="pt") as f:
                for name in f.keys():
                    m = _EXPERT_RE.match(name)
                    if m:
                        prefix, e, proj = m.group(1), int(m.group(2)), m.group(3)
                        if proj == "down_proj":
                            target_name = f"{prefix}.down_proj"
                            dst = params[target_name][e]
                        else:
                            target_name = f"{prefix}.gate_up_proj"
                            half = slice(0, inter) if proj == "gate_proj" else slice(inter, 2 * inter)
                            dst = params[target_name][e, half]
                        loaded.add(target_name)
                    elif name in params:
                        target_name, dst = name, params[name]
                        loaded.add(name)
                    else:
                        continue  # 例如 rotary 的 inv_freq
                    t = f.get_tensor(name)
                    if dst.shape != t.shape:
                        raise ValueError(f"{name}: shape {tuple(t.shape)} != {tuple(dst.shape)}")
                    dst.copy_(t)

    missing = sorted(set(params) - loaded)
    if missing:
        raise ValueError(f"缺少权重: {missing[:10]}{' ...' if len(missing) > 10 else ''}")


def load_model(
    model_dir: str,
    dtype: torch.dtype = torch.bfloat16,
    device: str | torch.device = "cuda",
    max_model_len: int = 32768,
) -> DeepseekV2ForCausalLM:
    cfg = ModelConfig.from_pretrained(model_dir)
    with torch.device("meta"):
        model = DeepseekV2ForCausalLM(cfg, max_model_len=max_model_len)
    # 先在 meta 上改 dtype 再分配，避免在显卡上先占一份 fp32（约 63GB）
    model = model.to(dtype).to_empty(device=device)
    load_weights(model, model_dir)
    model.init_rope_cache()
    return model.eval()
