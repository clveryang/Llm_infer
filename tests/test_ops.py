"""逐个算子对比 C++ 内核和 ref_ops 参考实现。

默认在 CPU 上测；在有 CUDA 内核的机器上，DEVICES 会自动加入 cuda。
"""

import math

import pytest
import torch

from llm_infer import ops, ref_ops

C = torch.ops.llm_infer

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
DTYPES = [torch.float32, torch.float64]


def tol(dtype):
    return dict(rtol=1e-4, atol=1e-5) if dtype == torch.float32 else dict(rtol=1e-10, atol=1e-12)


def need_kernel(name, device):
    if not ops.has_kernel(name, torch.device(device)):
        pytest.skip(f"{name} 在 {device} 上还没有 C++ 内核")


@pytest.fixture(autouse=True)
def seed():
    torch.manual_seed(0)


# ---------------- 通用 ----------------
@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", DTYPES)
def test_rms_norm(device, dtype):
    need_kernel("rms_norm", device)
    x = torch.randn(7, 5, 33, dtype=dtype, device=device)
    w = torch.randn(33, dtype=dtype, device=device)
    torch.testing.assert_close(C.rms_norm(x, w, 1e-6), ref_ops.rms_norm(x, w, 1e-6), **tol(dtype))


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", DTYPES)
def test_fused_add_rms_norm(device, dtype):
    need_kernel("fused_add_rms_norm", device)
    x = torch.randn(9, 64, dtype=dtype, device=device)
    r = torch.randn(9, 64, dtype=dtype, device=device)
    w = torch.randn(64, dtype=dtype, device=device)
    x1, r1, x2, r2 = x.clone(), r.clone(), x.clone(), r.clone()
    C.fused_add_rms_norm(x1, r1, w, 1e-6)
    ref_ops.fused_add_rms_norm(x2, r2, w, 1e-6)
    torch.testing.assert_close(x1, x2, **tol(dtype))
    torch.testing.assert_close(r1, r2, **tol(dtype))


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", DTYPES)
def test_silu_and_mul(device, dtype):
    need_kernel("silu_and_mul", device)
    x = torch.randn(4, 6, 2 * 17, dtype=dtype, device=device) * 10  # 覆盖大正/大负值
    torch.testing.assert_close(C.silu_and_mul(x), ref_ops.silu_and_mul(x), **tol(dtype))


# ---------------- MLA ----------------
@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("is_neox", [False, True])
def test_rotary_embedding(device, dtype, is_neox):
    need_kernel("rotary_embedding", device)
    from llm_infer.rope import build_cos_sin_cache

    scaling = {"type": "yarn", "factor": 40, "original_max_position_embeddings": 4096, "beta_fast": 32,
               "beta_slow": 1, "mscale": 0.707, "mscale_all_dim": 0.707}
    cache = build_cos_sin_cache(64, 256, 10000.0, scaling, dtype, device)
    pos = torch.randint(0, 256, (13,), device=device)
    q = torch.randn(13, 16, 64, dtype=dtype, device=device)
    k = torch.randn(13, 1, 64, dtype=dtype, device=device)
    q1, k1, q2, k2 = q.clone(), k.clone(), q.clone(), k.clone()
    C.rotary_embedding(pos, q1, k1, cache, is_neox)
    ref_ops.rotary_embedding(pos, q2, k2, cache, is_neox)
    torch.testing.assert_close(q1, q2, **tol(dtype))
    torch.testing.assert_close(k1, k2, **tol(dtype))


def test_interleaved_rope_matches_complex_form():
    """交错 RoPE 等价于把相邻两维看成复数再乘 e^{iθ}，这是 transformers 里 DeepSeek-V2 的写法。"""
    from llm_infer.rope import build_cos_sin_cache

    cache = build_cos_sin_cache(8, 32, 10000.0, None, torch.float64)
    pos = torch.arange(5)
    q = torch.randn(5, 2, 8, dtype=torch.float64)
    k = torch.randn(5, 1, 8, dtype=torch.float64)
    freqs_cis = torch.complex(cache[pos, None, :4], cache[pos, None, 4:])  # cos + i·sin
    expected = torch.view_as_real(torch.view_as_complex(q.reshape(5, 2, 4, 2)) * freqs_cis).flatten(-2)
    C.rotary_embedding(pos, q, k, cache, False)
    torch.testing.assert_close(q, expected)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", DTYPES)
def test_concat_and_cache_mla(device, dtype):
    need_kernel("concat_and_cache_mla", device)
    num_blocks, block_size, R, P = 6, 4, 12, 4
    c = torch.randn(10, R, dtype=dtype, device=device)
    pe = torch.randn(10, P, dtype=dtype, device=device)
    slots = torch.randperm(num_blocks * block_size, device=device)[:10]
    slots[3] = -1  # 跳过
    cache1 = torch.zeros(num_blocks, block_size, R + P, dtype=dtype, device=device)
    cache2 = cache1.clone()
    C.concat_and_cache_mla(c, pe, cache1, slots)
    ref_ops.concat_and_cache_mla(c, pe, cache2, slots)
    torch.testing.assert_close(cache1, cache2)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", DTYPES)
def test_mla_prefill_attention(device, dtype):
    need_kernel("mla_prefill_attention", device)
    lens = [5, 1, 9]
    n, h, dqk, dv = sum(lens), 4, 24, 16
    q = torch.randn(n, h, dqk, dtype=dtype, device=device)
    k = torch.randn(n, h, dqk, dtype=dtype, device=device)
    v = torch.randn(n, h, dv, dtype=dtype, device=device)
    cu = torch.tensor([0] + list(torch.tensor(lens).cumsum(0)), dtype=torch.int32, device=device)
    scale = 1.3 / math.sqrt(dqk)
    torch.testing.assert_close(
        C.mla_prefill_attention(q, k, v, cu, scale), ref_ops.mla_prefill_attention(q, k, v, cu, scale), **tol(dtype)
    )


def _random_paged_cache(lens, block_size, dim, dtype, device):
    num_blocks = sum((n + block_size - 1) // block_size for n in lens) + 3
    cache = torch.randn(num_blocks, block_size, dim, dtype=dtype, device=device)
    perm = torch.randperm(num_blocks).tolist()
    max_blocks = max((n + block_size - 1) // block_size for n in lens)
    tables = torch.zeros(len(lens), max_blocks, dtype=torch.int32)
    for i, n in enumerate(lens):
        for b in range((n + block_size - 1) // block_size):
            tables[i, b] = perm.pop()
    return cache, tables.to(device), torch.tensor(lens, dtype=torch.int32, device=device)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", DTYPES)
def test_mla_decode_attention(device, dtype):
    need_kernel("mla_decode_attention", device)
    R, P, H = 20, 6, 3
    cache, tables, lens = _random_paged_cache([1, 7, 16, 23], 4, R + P, dtype, device)
    q = torch.randn(4, H, R + P, dtype=dtype, device=device) * 0.3
    args = (q, cache, tables, lens, R, 0.7)
    torch.testing.assert_close(C.mla_decode_attention(*args), ref_ops.mla_decode_attention(*args), **tol(dtype))


def test_latent_decode_equals_decompressed_attention():
    """验证 decode 的 latent 空间技巧：和"先用 kv_b_proj 解压再做标准注意力"结果一致。"""
    dtype = torch.float64
    H, nope, rope, vdim, R, L = 4, 12, 8, 10, 16, 11
    w = torch.randn(H * (nope + vdim), R, dtype=dtype)
    c = torch.randn(L, R, dtype=dtype)
    k_pe = torch.randn(L, rope, dtype=dtype)
    q_nope = torch.randn(1, H, nope, dtype=dtype)
    q_pe = torch.randn(1, H, rope, dtype=dtype)
    scale = 0.37

    # 解压路径
    k_nope, v = (c @ w.T).view(L, H, nope + vdim).split([nope, vdim], dim=-1)
    k = torch.cat([k_nope, k_pe[:, None].expand(-1, H, -1)], dim=-1)  # [L, H, 20]
    qf = torch.cat([q_nope, q_pe], dim=-1)[0]  # [H, 20]
    attn = (torch.einsum("hd,lhd->hl", qf, k) * scale).softmax(-1)
    expected = torch.einsum("hl,lhv->hv", attn, v)

    # latent 路径
    wv = w.view(H, nope + vdim, R)
    q_lat = torch.einsum("nhd,hdr->nhr", q_nope, wv[:, :nope])
    cache = torch.cat([c, k_pe], dim=-1).view(1, L, R + rope)
    out_lat = C.mla_decode_attention(
        torch.cat([q_lat, q_pe], -1), cache, torch.zeros(1, 1, dtype=torch.int32), torch.tensor([L]), R, scale
    )
    got = torch.einsum("nhr,hvr->nhv", out_lat, wv[:, nope:])[0]
    torch.testing.assert_close(got, expected)


# ---------------- MoE ----------------
@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("renormalize", [False, True])
def test_topk_softmax(device, renormalize):
    need_kernel("topk_softmax", device)
    logits = torch.randn(50, 64, device=device)  # bf16 会出现并列值，top-k 选择不唯一
    w1, i1 = C.topk_softmax(logits, 6, renormalize)
    w2, i2 = ref_ops.topk_softmax(logits, 6, renormalize)
    # top-k 内部顺序不重要，排序后比较
    i1s, o1 = i1.sort(-1)
    i2s, o2 = i2.sort(-1)
    torch.testing.assert_close(i1s, i2s)
    torch.testing.assert_close(w1.gather(-1, o1), w2.gather(-1, o2))


@pytest.mark.parametrize("device", DEVICES)
def test_moe_align(device):
    need_kernel("moe_align", device)
    ids = torch.randint(0, 8, (20, 3), device=device)
    s1, o1 = C.moe_align(ids, 8)
    s2, o2 = ref_ops.moe_align(ids, 8)
    torch.testing.assert_close(s1, s2)
    torch.testing.assert_close(o1, o2)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", DTYPES)
def test_moe_expert_forward_and_combine(device, dtype):
    need_kernel("moe_expert_forward", device)
    need_kernel("moe_combine", device)
    N, E, k, D, I = 17, 8, 3, 24, 10
    x = torch.randn(N, D, dtype=dtype, device=device)
    w_gu = torch.randn(E, 2 * I, D, dtype=dtype, device=device)
    w_d = torch.randn(E, D, I, dtype=dtype, device=device)
    weights, ids = ref_ops.topk_softmax(torch.randn(N, E, device=device), k, False)
    sorted_idx, offsets = ref_ops.moe_align(ids, E)
    hs = x.index_select(0, sorted_idx // k)

    e1 = C.moe_expert_forward(hs, offsets, w_gu, w_d)
    e2 = ref_ops.moe_expert_forward(hs, offsets, w_gu, w_d)
    torch.testing.assert_close(e1, e2, **tol(dtype))
    torch.testing.assert_close(
        C.moe_combine(e1, sorted_idx, weights, N), ref_ops.moe_combine(e1, sorted_idx, weights, N), **tol(dtype)
    )


# ---------------- CUDA 半精度 ----------------
# CUDA 内核内部把 fp16/bf16 提升到 float32 计算，所以对照组是「同样输入转 float32 后跑参考实现」，
# 差异只来自最后写回半精度时的舍入。
HALF_TOL = {torch.float16: dict(rtol=5e-3, atol=5e-3), torch.bfloat16: dict(rtol=2e-2, atol=2e-2)}
cuda_only = pytest.mark.skipif(not torch.cuda.is_available(), reason="需要 CUDA")


def _assert_half(got, expected_f32, dtype):
    assert got.dtype == dtype
    torch.testing.assert_close(got.float(), expected_f32.float(), **HALF_TOL[dtype])


@cuda_only
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_cuda_half_norm_act(dtype):
    need_kernel("rms_norm", "cuda")
    x = torch.randn(33, 128, device="cuda").to(dtype)
    r = torch.randn(33, 128, device="cuda").to(dtype)
    w = (torch.rand(128, device="cuda") + 0.5).to(dtype)
    _assert_half(C.rms_norm(x, w, 1e-6), ref_ops.rms_norm(x.float(), w.float(), 1e-6), dtype)

    x1, r1, x2, r2 = x.clone(), r.clone(), x.float(), r.float()
    C.fused_add_rms_norm(x1, r1, w, 1e-6)
    ref_ops.fused_add_rms_norm(x2, r2, w.float(), 1e-6)
    _assert_half(x1, x2, dtype)
    _assert_half(r1, r2, dtype)

    g = torch.randn(9, 2 * 64, device="cuda").to(dtype)
    _assert_half(C.silu_and_mul(g), ref_ops.silu_and_mul(g.float()), dtype)


@cuda_only
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_cuda_half_mla(dtype):
    need_kernel("mla_decode_attention", "cuda")
    from llm_infer.rope import build_cos_sin_cache

    cache = build_cos_sin_cache(64, 128, 10000.0, None, torch.float32, "cuda")
    pos = torch.randint(0, 128, (11,), device="cuda")
    q = torch.randn(11, 16, 64, device="cuda").to(dtype)
    k = torch.randn(11, 1, 64, device="cuda").to(dtype)
    q1, k1, q2, k2 = q.clone(), k.clone(), q.float(), k.float()
    C.rotary_embedding(pos, q1, k1, cache.to(dtype), False)
    ref_ops.rotary_embedding(pos, q2, k2, cache, False)
    _assert_half(q1, q2, dtype)
    _assert_half(k1, k2, dtype)

    lens = [6, 1, 13]
    n = sum(lens)
    qq, kk, vv = (torch.randn(n, 4, dd, device="cuda").to(dtype) for dd in (24, 24, 16))
    cu = torch.tensor([0, 6, 7, 20], device="cuda")
    _assert_half(
        C.mla_prefill_attention(qq, kk, vv, cu, 0.2),
        ref_ops.mla_prefill_attention(qq.float(), kk.float(), vv.float(), cu, 0.2),
        dtype,
    )

    R, P = 20, 6
    kv, tables, sl = _random_paged_cache([1, 9, 17], 4, R + P, torch.float32, "cuda")
    qd = (torch.randn(3, 4, R + P, device="cuda") * 0.3).to(dtype)
    _assert_half(
        C.mla_decode_attention(qd, kv.to(dtype), tables, sl, R, 0.5),
        ref_ops.mla_decode_attention(qd.float(), kv.to(dtype).float(), tables, sl, R, 0.5),
        dtype,
    )


@cuda_only
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_cuda_half_moe(dtype):
    need_kernel("moe_combine", "cuda")
    N, E, k, D, I = 21, 8, 3, 32, 12
    x = torch.randn(N, D, device="cuda").to(dtype)
    w_gu = (torch.randn(E, 2 * I, D, device="cuda") * 0.2).to(dtype)
    w_d = (torch.randn(E, D, I, device="cuda") * 0.2).to(dtype)
    weights, ids = C.topk_softmax(torch.randn(N, E, device="cuda").to(dtype), k, False)
    sorted_idx, offsets = C.moe_align(ids, E)
    hs = x.index_select(0, sorted_idx // k)
    eo = C.moe_expert_forward(hs, offsets, w_gu, w_d)
    _assert_half(eo, ref_ops.moe_expert_forward(hs.float(), offsets, w_gu.float(), w_d.float()), dtype)
    _assert_half(C.moe_combine(eo, sorted_idx, weights, N), ref_ops.moe_combine(eo.float(), sorted_idx, weights, N), dtype)
