"""算子分发层：模型代码只调用这里的函数。

后端由环境变量 LLM_INFER_BACKEND（或 set_backend）控制：
- auto（默认）：该设备注册了 C++ 内核且 dtype 支持时用 C++，否则用 ref_ops；
- cpp：强制 C++，不支持时直接报错（测试用）；
- torch：全部走 ref_ops。

新写了 CUDA 内核后，只要在 C++ 里 TORCH_LIBRARY_IMPL(llm_infer, CUDA, m) 注册，
这里会自动切换过去，模型代码不用改。
"""

import os

import torch

from . import ref_ops

try:
    from . import _C  # noqa: F401  触发 TORCH_LIBRARY 静态注册
    HAS_CPP = True
except ImportError:
    HAS_CPP = False

_BACKEND = os.environ.get("LLM_INFER_BACKEND", "auto")
# CPU 内核只实现了这些 dtype；GPU 内核由各自实现检查。
_CPU_DTYPES = (torch.float32, torch.float64)


def set_backend(name: str) -> None:
    global _BACKEND
    assert name in ("auto", "cpp", "torch"), name
    _BACKEND = name


def get_backend() -> str:
    return _BACKEND


def has_kernel(name: str, device: torch.device) -> bool:
    if not HAS_CPP:
        return False
    key = {"cpu": "CPU", "cuda": "CUDA"}.get(device.type)
    return key is not None and torch._C._dispatch_has_kernel_for_dispatch_key(f"llm_infer::{name}", key)


def _use_cpp(name: str, t: torch.Tensor, check_dtype: bool = True) -> bool:
    if _BACKEND == "torch":
        return False
    ok = has_kernel(name, t.device)
    if ok and check_dtype and t.device.type == "cpu":
        ok = t.dtype in _CPU_DTYPES
    if not ok and _BACKEND == "cpp":
        raise RuntimeError(f"no C++ kernel for llm_infer::{name} on {t.device} / {t.dtype}")
    return ok


_C_OPS = torch.ops.llm_infer


# ---------------- 通用 ----------------
def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    if _use_cpp("rms_norm", x):
        return _C_OPS.rms_norm(x.contiguous(), weight, eps)
    return ref_ops.rms_norm(x, weight, eps)


def fused_add_rms_norm(x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor, eps: float) -> None:
    """原地：residual += x；x = rms_norm(residual)。x 和 residual 必须是 contiguous。"""
    if _use_cpp("fused_add_rms_norm", x):
        _C_OPS.fused_add_rms_norm(x, residual, weight, eps)
    else:
        ref_ops.fused_add_rms_norm(x, residual, weight, eps)


def silu_and_mul(x: torch.Tensor) -> torch.Tensor:
    if _use_cpp("silu_and_mul", x):
        return _C_OPS.silu_and_mul(x.contiguous())
    return ref_ops.silu_and_mul(x)


# ---------------- MLA ----------------
def rotary_embedding(
    positions: torch.Tensor, query: torch.Tensor, key: torch.Tensor, cos_sin_cache: torch.Tensor, is_neox: bool
) -> None:
    """原地 RoPE。query/key 必须是 contiguous 的 [N, H, rot_dim]。"""
    if _use_cpp("rotary_embedding", query):
        _C_OPS.rotary_embedding(positions, query, key, cos_sin_cache, is_neox)
    else:
        ref_ops.rotary_embedding(positions, query, key, cos_sin_cache, is_neox)


def concat_and_cache_mla(
    kv_c: torch.Tensor, k_pe: torch.Tensor, kv_cache: torch.Tensor, slot_mapping: torch.Tensor
) -> None:
    if _use_cpp("concat_and_cache_mla", kv_cache):
        _C_OPS.concat_and_cache_mla(kv_c.contiguous(), k_pe.contiguous(), kv_cache, slot_mapping)
    else:
        ref_ops.concat_and_cache_mla(kv_c, k_pe, kv_cache, slot_mapping)


def mla_prefill_attention(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, cu_seqlens: torch.Tensor, scale: float
) -> torch.Tensor:
    if _use_cpp("mla_prefill_attention", q):
        return _C_OPS.mla_prefill_attention(q.contiguous(), k.contiguous(), v.contiguous(), cu_seqlens, scale)
    return ref_ops.mla_prefill_attention(q, k, v, cu_seqlens, scale)


def mla_decode_attention(
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    kv_lora_rank: int,
    scale: float,
) -> torch.Tensor:
    if _use_cpp("mla_decode_attention", q):
        return _C_OPS.mla_decode_attention(q.contiguous(), kv_cache, block_tables, seq_lens, kv_lora_rank, scale)
    return ref_ops.mla_decode_attention(q, kv_cache, block_tables, seq_lens, kv_lora_rank, scale)


# ---------------- MoE ----------------
def topk_softmax(gating_logits: torch.Tensor, topk: int, renormalize: bool) -> tuple[torch.Tensor, torch.Tensor]:
    # C++ 内核内部统一转 float32，不限制输入 dtype
    if _use_cpp("topk_softmax", gating_logits, check_dtype=False):
        return _C_OPS.topk_softmax(gating_logits, topk, renormalize)
    return ref_ops.topk_softmax(gating_logits, topk, renormalize)


def moe_align(topk_ids: torch.Tensor, num_experts: int) -> tuple[torch.Tensor, torch.Tensor]:
    if _use_cpp("moe_align", topk_ids, check_dtype=False):
        return _C_OPS.moe_align(topk_ids.contiguous(), num_experts)
    return ref_ops.moe_align(topk_ids, num_experts)


def moe_expert_forward(
    hidden_sorted: torch.Tensor, expert_offsets: torch.Tensor, w_gate_up: torch.Tensor, w_down: torch.Tensor
) -> torch.Tensor:
    if _use_cpp("moe_expert_forward", hidden_sorted):
        return _C_OPS.moe_expert_forward(hidden_sorted.contiguous(), expert_offsets, w_gate_up, w_down)
    return ref_ops.moe_expert_forward(hidden_sorted, expert_offsets, w_gate_up, w_down)


def moe_combine(
    expert_out: torch.Tensor, sorted_idx: torch.Tensor, topk_weights: torch.Tensor, num_tokens: int
) -> torch.Tensor:
    if _use_cpp("moe_combine", expert_out):
        return _C_OPS.moe_combine(expert_out.contiguous(), sorted_idx, topk_weights.contiguous(), num_tokens)
    return ref_ops.moe_combine(expert_out, sorted_idx, topk_weights, num_tokens)
