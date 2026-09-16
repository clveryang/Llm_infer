"""DeepSeek-V2 系列模型配置。

兼容两种来源：
- Hub 上原始的 config.json（rope_theta + rope_scaling）；
- transformers 原生 DeepseekV2Config.to_dict()（rope_parameters）。
"""

import json
import os
from dataclasses import dataclass, field, fields


@dataclass
class ModelConfig:
    vocab_size: int = 102400
    hidden_size: int = 2048
    intermediate_size: int = 10944
    moe_intermediate_size: int = 1408
    num_hidden_layers: int = 27
    num_attention_heads: int = 16

    # MLA
    kv_lora_rank: int = 512
    q_lora_rank: int | None = None
    qk_nope_head_dim: int = 128
    qk_rope_head_dim: int = 64
    v_head_dim: int = 128

    # MoE
    n_routed_experts: int = 64
    n_shared_experts: int = 2
    num_experts_per_tok: int = 6
    first_k_dense_replace: int = 1
    moe_layer_freq: int = 1
    topk_method: str = "greedy"
    norm_topk_prob: bool = False
    routed_scaling_factor: float = 1.0

    # 其他
    rms_norm_eps: float = 1e-6
    max_position_embeddings: int = 163840
    rope_theta: float = 10000.0
    rope_scaling: dict | None = field(
        default_factory=lambda: {
            "type": "yarn",
            "factor": 40,
            "original_max_position_embeddings": 4096,
            "beta_fast": 32,
            "beta_slow": 1,
            "mscale": 0.707,
            "mscale_all_dim": 0.707,
        }
    )
    bos_token_id: int | None = 100000
    eos_token_id: int | None = 100001

    @property
    def qk_head_dim(self) -> int:
        return self.qk_nope_head_dim + self.qk_rope_head_dim

    def is_moe_layer(self, layer_idx: int) -> bool:
        return (
            self.n_routed_experts > 0
            and layer_idx >= self.first_k_dense_replace
            and layer_idx % self.moe_layer_freq == 0
        )

    @classmethod
    def from_dict(cls, d: dict) -> "ModelConfig":
        names = {f.name for f in fields(cls)}
        kwargs = {k: v for k, v in d.items() if k in names}

        rope = d.get("rope_parameters") or {}
        if "rope_theta" not in d and "rope_theta" in rope:
            kwargs["rope_theta"] = rope["rope_theta"]
        if "rope_scaling" not in d:
            rope_type = rope.get("rope_type", rope.get("type", "default"))
            kwargs["rope_scaling"] = None if rope_type == "default" else dict(rope)
        if isinstance(kwargs.get("eos_token_id"), list):
            kwargs["eos_token_id"] = kwargs["eos_token_id"][0]
        return cls(**kwargs)

    @classmethod
    def from_pretrained(cls, model_dir: str) -> "ModelConfig":
        with open(os.path.join(model_dir, "config.json")) as f:
            return cls.from_dict(json.load(f))
