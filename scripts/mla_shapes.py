"""打印 MLA 一层的权重形状，以及 1 个 token 走一遍时每步张量的形状。

    python scripts/mla_shapes.py                   # 默认 V2-Lite
    python scripts/mla_shapes.py checkpoints/xxx   # 用某个 config.json（比如完整版 V2）
"""

import sys

import torch

from llm_infer.config import ModelConfig
from llm_infer.model import MLAttention


def main():
    cfg = ModelConfig.from_pretrained(sys.argv[1]) if len(sys.argv) > 1 else ModelConfig()
    attn = MLAttention(cfg)
    D, H = cfg.hidden_size, cfg.num_attention_heads
    nope, rope, v_dim, R = cfg.qk_nope_head_dim, cfg.qk_rope_head_dim, cfg.v_head_dim, cfg.kv_lora_rank

    print("=== 权重矩阵 ===")
    total = 0
    for name, p in attn.named_parameters():
        total += p.numel()
        print(f"{name:<28} {str(list(p.shape)):<16} {p.numel() / 1e6:>6.2f}M")
    print(f"{'一层合计':<28} {'':<16} {total / 1e6:>6.2f}M   × {cfg.num_hidden_layers} 层 = "
          f"{total * cfg.num_hidden_layers / 1e9:.2f}B")

    print("\n=== 1 个 token 走一遍 ===")
    h = torch.randn(1, D)
    print(f"输入 h                        {list(h.shape)}")
    if cfg.q_lora_rank is None:
        q = attn.q_proj(h)
        print(f"q_proj(h)                     {list(q.shape)}   = {H} 头 × {cfg.qk_head_dim}")
    else:
        q = attn.q_b_proj(attn.q_a_layernorm(attn.q_a_proj(h)))
        print(f"q_a_proj → layernorm → q_b_proj {list(q.shape)}   = {H} 头 × {cfg.qk_head_dim}")
    q = q.view(1, H, cfg.qk_head_dim)
    q_nope, q_pe = q.split([nope, rope], dim=-1)
    print(f"  view 成多头                  {list(q.shape)}")
    print(f"  拆成 q_nope / q_pe           {list(q_nope.shape)} / {list(q_pe.shape)}")

    kv = attn.kv_a_proj_with_mqa(h)
    latent, k_pe = kv.split([R, rope], dim=-1)
    print(f"kv_a_proj_with_mqa(h)         {list(kv.shape)}    = {R} latent + {rope} k_pe")
    print(f"  拆成 latent / k_pe           {list(latent.shape)} / {list(k_pe.shape)}")
    latent = attn.kv_a_layernorm(latent)
    print(f"  kv_a_layernorm(latent)       {list(latent.shape)}   ← KV cache 里存的就是这个（+ k_pe）")

    kvb = attn.kv_b_proj(latent)
    print(f"kv_b_proj(latent)             {list(kvb.shape)}  = {H} 头 × {nope + v_dim}")
    k_nope, v = kvb.view(1, H, nope + v_dim).split([nope, v_dim], dim=-1)
    print(f"  拆成 k_nope / v              {list(k_nope.shape)} / {list(v.shape)}")

    o = torch.randn(1, H, v_dim).reshape(1, H * v_dim)
    print(f"注意力输出拼平                 {list(o.shape)}   = {H} × {v_dim}")
    print(f"o_proj(...)                   {list(attn.o_proj(o).shape)}   ← 回到主干道")

    per_token = (R + rope) * cfg.num_hidden_layers * 2
    mha = H * (cfg.qk_head_dim + v_dim) * cfg.num_hidden_layers * 2
    print(f"\nKV cache：MLA 每 token {per_token / 1024:.1f} KB（bf16），"
          f"同样头维度的 MHA 要 {mha / 1024:.1f} KB，省 {mha / per_token:.1f} 倍")


if __name__ == "__main__":
    main()
