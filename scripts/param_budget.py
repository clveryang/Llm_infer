"""打印 DeepSeek-V2-Lite 的参数分布（总参数 / 每个 token 激活参数）。

    python scripts/param_budget.py                      # 用默认的 V2-Lite 配置
    python scripts/param_budget.py checkpoints/xxx      # 用某个 config.json
"""

import sys

from llm_infer.config import ModelConfig


def main():
    cfg = ModelConfig.from_pretrained(sys.argv[1]) if len(sys.argv) > 1 else ModelConfig()
    D, H = cfg.hidden_size, cfg.num_attention_heads
    L, E, k = cfg.num_hidden_layers, cfg.n_routed_experts, cfg.num_experts_per_tok

    # MLA：q_proj + kv_a + kv_a_layernorm + kv_b + o_proj（Lite 没有 q_lora）
    mla = D * H * cfg.qk_head_dim + D * (cfg.kv_lora_rank + cfg.qk_rope_head_dim) + cfg.kv_lora_rank
    mla += cfg.kv_lora_rank * H * (cfg.qk_nope_head_dim + cfg.v_head_dim) + H * cfg.v_head_dim * D
    if cfg.q_lora_rank:  # 完整版 V2 会先把 Q 压缩
        mla += D * cfg.q_lora_rank + cfg.q_lora_rank + cfg.q_lora_rank * H * cfg.qk_head_dim - D * H * cfg.qk_head_dim

    dense = 3 * D * cfg.intermediate_size
    expert = 3 * D * cfg.moe_intermediate_size
    shared = cfg.n_shared_experts * expert
    moe_total = E * D + E * expert + shared          # 路由器 + 所有专家 + 共享专家
    moe_active = E * D + k * expert + shared         # 每个 token 只走 k 个路由专家
    emb = cfg.vocab_size * D

    n_moe = sum(cfg.is_moe_layer(i) for i in range(L))
    n_dense = L - n_moe
    rows = [
        ("embed_tokens", emb, 0),
        (f"MLA × {L}", mla * L, mla * L),
        (f"dense FFN × {n_dense}", dense * n_dense, dense * n_dense),
        (f"MoE × {n_moe}", moe_total * n_moe, moe_active * n_moe),
        ("lm_head", 0 if cfg.vocab_size * D == 0 else emb, emb),
    ]
    total = sum(r[1] for r in rows)
    active = sum(r[2] for r in rows)

    print(f"{'部件':<22}{'总参数':>12}{'激活参数':>14}")
    for name, t, a in rows:
        print(f"{name:<22}{t / 1e9:>10.3f}B{a / 1e9:>13.3f}B")
    print(f"{'合计':<22}{total / 1e9:>10.2f}B{active / 1e9:>13.2f}B")
    print(f"\n单个专家 {expert / 1e6:.1f}M，dense FFN {dense / 1e6:.1f}M（= {dense / expert:.0f} 个专家）")

    latent = cfg.kv_lora_rank + cfg.qk_rope_head_dim
    print(f"KV cache：每 token 每层 {latent} 维，全模型 bf16 下每 token {L * latent * 2 / 1024:.1f} KB")


if __name__ == "__main__":
    main()
