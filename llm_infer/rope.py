"""YaRN RoPE 的 cos/sin 表和注意力 scale，公式与 DeepSeek 官方 modeling_deepseek.py 一致。"""

import math

import torch


def yarn_get_mscale(scale: float = 1.0, mscale: float = 1.0) -> float:
    if scale <= 1:
        return 1.0
    return 0.1 * mscale * math.log(scale) + 1.0


def _correction_dim(num_rotations: float, dim: int, base: float, max_pos: int) -> float:
    return (dim * math.log(max_pos / (num_rotations * 2 * math.pi))) / (2 * math.log(base))


def _correction_range(low_rot: float, high_rot: float, dim: int, base: float, max_pos: int) -> tuple[int, int]:
    low = math.floor(_correction_dim(low_rot, dim, base, max_pos))
    high = math.ceil(_correction_dim(high_rot, dim, base, max_pos))
    return max(low, 0), min(high, dim - 1)


def _rope_type(scaling: dict | None) -> str:
    if not scaling:
        return "default"
    return scaling.get("rope_type", scaling.get("type", "default"))


def build_cos_sin_cache(
    rope_dim: int, max_pos: int, base: float, scaling: dict | None, dtype=torch.float32, device=None
) -> torch.Tensor:
    """返回 [max_pos, rope_dim]，每行前半是 cos、后半是 sin（每半 rope_dim/2 维）。"""
    exponent = torch.arange(0, rope_dim, 2, dtype=torch.float32, device=device) / rope_dim
    freq_extra = 1.0 / (base**exponent)
    mscale = 1.0

    rope_type = _rope_type(scaling)
    if rope_type == "yarn":
        factor = scaling["factor"]
        freq_inter = 1.0 / (factor * base**exponent)
        low, high = _correction_range(
            scaling.get("beta_fast", 32),
            scaling.get("beta_slow", 1),
            rope_dim,
            base,
            scaling["original_max_position_embeddings"],
        )
        if low == high:
            high += 0.001
        ramp = ((torch.arange(rope_dim // 2, dtype=torch.float32, device=device) - low) / (high - low)).clamp(0, 1)
        extra_mask = 1.0 - ramp  # 高频维度保持原频率（外推），低频维度插值
        inv_freq = freq_inter * (1 - extra_mask) + freq_extra * extra_mask
        mscale = yarn_get_mscale(factor, scaling.get("mscale", 1)) / yarn_get_mscale(
            factor, scaling.get("mscale_all_dim", 0)
        )
    elif rope_type == "default":
        inv_freq = freq_extra
    else:
        raise NotImplementedError(f"rope type {rope_type}")

    t = torch.arange(max_pos, dtype=torch.float32, device=device)
    freqs = torch.outer(t, inv_freq)
    cache = torch.cat([freqs.cos() * mscale, freqs.sin() * mscale], dim=-1)
    return cache.to(dtype)


def attention_scale(qk_head_dim: int, scaling: dict | None) -> float:
    scale = qk_head_dim**-0.5
    if _rope_type(scaling) != "default":
        mscale_all_dim = scaling.get("mscale_all_dim", 0)
        if mscale_all_dim:
            m = yarn_get_mscale(scaling["factor"], mscale_all_dim)
            scale *= m * m
    return scale
