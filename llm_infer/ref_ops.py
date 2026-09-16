"""纯 PyTorch 参考实现。

每个函数和 csrc 里同名 C++ 算子的语义完全一致，用途有两个：
1. 单测时作为对照（tests/test_ops.py）；
2. 设备或 dtype 还没有 C++ 内核时（比如 GPU 上的 bf16）作为回退路径。
"""

import torch
import torch.nn.functional as F


# ---------------- 通用 ----------------
def _upcast(x: torch.Tensor) -> torch.Tensor:
    """半精度提升到 float32 计算（和 HF 一致），float32/float64 保持不变。"""
    return x.to(torch.promote_types(x.dtype, torch.float32))


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    dtype = x.dtype
    xf = _upcast(x)
    xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    return weight * xf.to(dtype)


def fused_add_rms_norm(x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor, eps: float) -> None:
    residual.add_(x)
    x.copy_(rms_norm(residual, weight, eps))


def silu_and_mul(x: torch.Tensor) -> torch.Tensor:
    d = x.shape[-1] // 2
    return F.silu(x[..., :d]) * x[..., d:]


# ---------------- MLA ----------------
def rotary_embedding(
    positions: torch.Tensor, query: torch.Tensor, key: torch.Tensor, cos_sin_cache: torch.Tensor, is_neox: bool
) -> None:
    half = cos_sin_cache.shape[1] // 2
    cs = cos_sin_cache[positions.long()]
    cos, sin = cs[:, None, :half], cs[:, None, half:]
    for x in (query, key):
        if is_neox:
            x1, x2 = x[..., :half], x[..., half:]
        else:
            x1, x2 = x[..., 0::2], x[..., 1::2]
        o1 = x1 * cos - x2 * sin
        o2 = x2 * cos + x1 * sin
        if is_neox:
            x.copy_(torch.cat([o1, o2], dim=-1))
        else:
            x.copy_(torch.stack([o1, o2], dim=-1).flatten(-2))


def concat_and_cache_mla(
    kv_c: torch.Tensor, k_pe: torch.Tensor, kv_cache: torch.Tensor, slot_mapping: torch.Tensor
) -> None:
    mask = slot_mapping >= 0
    flat = kv_cache.view(-1, kv_cache.shape[-1])
    flat[slot_mapping[mask].long()] = torch.cat([kv_c, k_pe], dim=-1)[mask]


def mla_prefill_attention(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, cu_seqlens: torch.Tensor, scale: float
) -> torch.Tensor:
    out = q.new_empty(q.shape[0], q.shape[1], v.shape[2])
    bounds = cu_seqlens.tolist()
    for b, e in zip(bounds[:-1], bounds[1:]):
        # [S, H, D] -> [H, S, D]
        o = F.scaled_dot_product_attention(
            q[b:e].transpose(0, 1), k[b:e].transpose(0, 1), v[b:e].transpose(0, 1), is_causal=True, scale=scale
        )
        out[b:e] = o.transpose(0, 1)
    return out


def mla_decode_attention(
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    kv_lora_rank: int,
    scale: float,
) -> torch.Tensor:
    bsz, h, dim = q.shape
    block_size = kv_cache.shape[1]
    out = q.new_zeros(bsz, h, kv_lora_rank)
    for i, length in enumerate(seq_lens.tolist()):
        nblk = (length + block_size - 1) // block_size
        rows = kv_cache[block_tables[i, :nblk].long()].reshape(-1, dim)[:length]  # [L, R+P]
        scores = (q[i] @ rows.T) * scale  # [H, L]
        attn = _upcast(scores).softmax(dim=-1).to(q.dtype)
        out[i] = attn @ rows[:, :kv_lora_rank]
    return out


# ---------------- MoE ----------------
def topk_softmax(gating_logits: torch.Tensor, topk: int, renormalize: bool) -> tuple[torch.Tensor, torch.Tensor]:
    scores = gating_logits.float().softmax(dim=-1)
    weights, ids = torch.topk(scores, topk, dim=-1)
    if renormalize:
        weights = weights / weights.sum(dim=-1, keepdim=True)
    return weights, ids.long()


def moe_align(topk_ids: torch.Tensor, num_experts: int) -> tuple[torch.Tensor, torch.Tensor]:
    flat = topk_ids.flatten().long()
    sorted_idx = torch.argsort(flat, stable=True)
    counts = torch.bincount(flat, minlength=num_experts)
    offsets = torch.zeros(num_experts + 1, dtype=torch.long, device=flat.device)
    offsets[1:] = torch.cumsum(counts, dim=0)
    return sorted_idx, offsets


def moe_expert_forward(
    hidden_sorted: torch.Tensor, expert_offsets: torch.Tensor, w_gate_up: torch.Tensor, w_down: torch.Tensor
) -> torch.Tensor:
    out = hidden_sorted.new_empty(hidden_sorted.shape[0], w_down.shape[1])
    offs = expert_offsets.tolist()
    for e in range(w_gate_up.shape[0]):
        s, t = offs[e], offs[e + 1]
        if s == t:
            continue
        act = silu_and_mul(F.linear(hidden_sorted[s:t], w_gate_up[e]))
        out[s:t] = F.linear(act, w_down[e])
    return out


def moe_combine(
    expert_out: torch.Tensor, sorted_idx: torch.Tensor, topk_weights: torch.Tensor, num_tokens: int
) -> torch.Tensor:
    k = topk_weights.shape[1]
    w = topk_weights.flatten()[sorted_idx].to(expert_out.dtype)
    out = expert_out.new_zeros(num_tokens, expert_out.shape[1])
    out.index_add_(0, sorted_idx // k, expert_out * w[:, None])
    return out
