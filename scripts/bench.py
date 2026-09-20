"""逐个算子对比自写内核和 PyTorch 参考实现的速度，并可选地跑端到端吞吐。

    python scripts/bench.py                       # 用 V2-Lite 的真实形状
    python scripts/bench.py --seq 4096 --batch 64 # 自定义规模
    python scripts/bench.py --model-dir checkpoints/DeepSeek-V2-Lite-Chat --e2e

在 CPU 上会自动缩小规模，否则朴素内核太慢。
"""

import argparse
import time

import torch

from llm_infer import ops, ref_ops
from llm_infer.config import ModelConfig

C = torch.ops.llm_infer


def sync(device):
    if device == "cuda":
        torch.cuda.synchronize()


def timeit(fn, device, warmup=3, iters=10):
    for _ in range(warmup):
        fn()
    sync(device)
    t = time.perf_counter()
    for _ in range(iters):
        fn()
    sync(device)
    return (time.perf_counter() - t) / iters * 1000  # ms


def build_cases(cfg, args, device, dtype):
    """返回 [(算子名, 自写内核调用, 参考实现调用, 备注)]。形状取自 V2-Lite。"""
    D, H = cfg.hidden_size, cfg.num_attention_heads
    R, P = cfg.kv_lora_rank, cfg.qk_rope_head_dim
    qk, v = cfg.qk_head_dim, cfg.v_head_dim
    E, k, MI = cfg.n_routed_experts, cfg.num_experts_per_tok, cfg.moe_intermediate_size
    N, B, L, bs = args.seq, args.batch, args.ctx, 16
    opts = dict(device=device, dtype=dtype)
    cases = []

    x = torch.randn(N, D, **opts)
    w = torch.rand(D, **opts) + 0.5
    res = torch.randn(N, D, **opts)
    cases.append(("rms_norm", lambda: C.rms_norm(x, w, 1e-6), lambda: ref_ops.rms_norm(x, w, 1e-6), f"[{N}, {D}]"))
    cases.append((
        "fused_add_rms_norm",
        lambda: C.fused_add_rms_norm(x.clone(), res.clone(), w, 1e-6),
        lambda: ref_ops.fused_add_rms_norm(x.clone(), res.clone(), w, 1e-6),
        f"[{N}, {D}]",
    ))

    gu = torch.randn(N, 2 * MI, **opts)
    cases.append(("silu_and_mul", lambda: C.silu_and_mul(gu), lambda: ref_ops.silu_and_mul(gu), f"[{N}, {2 * MI}]"))

    from llm_infer.rope import build_cos_sin_cache

    cache = build_cos_sin_cache(P, 8192, cfg.rope_theta, cfg.rope_scaling, dtype, device)
    pos = torch.randint(0, 8192, (N,), device=device)
    q_pe, k_pe = torch.randn(N, H, P, **opts), torch.randn(N, 1, P, **opts)
    cases.append((
        "rotary_embedding",
        lambda: C.rotary_embedding(pos, q_pe.clone(), k_pe.clone(), cache, False),
        lambda: ref_ops.rotary_embedding(pos, q_pe.clone(), k_pe.clone(), cache, False),
        f"q [{N}, {H}, {P}]",
    ))

    kv_blocks = max(B * (L // bs + 1), N // bs + 1) + 8
    kv_cache = torch.randn(kv_blocks, bs, R + P, **opts)
    latent, kpe2 = torch.randn(N, R, **opts), torch.randn(N, P, **opts)
    slots = torch.randperm(kv_blocks * bs, device=device)[:N]
    cases.append((
        "concat_and_cache_mla",
        lambda: C.concat_and_cache_mla(latent, kpe2, kv_cache, slots),
        lambda: ref_ops.concat_and_cache_mla(latent, kpe2, kv_cache, slots),
        f"{N} token",
    ))

    qa, ka, va = torch.randn(N, H, qk, **opts), torch.randn(N, H, qk, **opts), torch.randn(N, H, v, **opts)
    cu = torch.tensor([0, N], device=device)
    cases.append((
        "mla_prefill_attention",
        lambda: C.mla_prefill_attention(qa, ka, va, cu, 0.1147),
        lambda: ref_ops.mla_prefill_attention(qa, ka, va, cu, 0.1147),
        f"1 条 × {N} token",
    ))

    nb = L // bs + 1
    tables = torch.arange(B * nb, device=device).reshape(B, nb) % kv_blocks
    lens = torch.full((B,), L, device=device)
    qd = torch.randn(B, H, R + P, **opts) * 0.2
    cases.append((
        "mla_decode_attention",
        lambda: C.mla_decode_attention(qd, kv_cache, tables, lens, R, 0.1147),
        lambda: ref_ops.mla_decode_attention(qd, kv_cache, tables, lens, R, 0.1147),
        f"batch {B} × 上下文 {L}",
    ))

    logits = torch.randn(N, E, device=device)
    cases.append((
        "topk_softmax",
        lambda: C.topk_softmax(logits, k, False),
        lambda: ref_ops.topk_softmax(logits, k, False),
        f"[{N}, {E}] 选 {k}",
    ))

    _, ids = ref_ops.topk_softmax(logits, k, False)
    cases.append((
        "moe_align",
        lambda: C.moe_align(ids, E),
        lambda: ref_ops.moe_align(ids, E),
        f"{N * k} 个 (token, 专家)",
    ))

    weights, _ = ref_ops.topk_softmax(logits, k, False)
    sorted_idx, offsets = ref_ops.moe_align(ids, E)
    hs = x.index_select(0, sorted_idx // k)
    w_gu = torch.randn(E, 2 * MI, D, **opts) * 0.02
    w_d = torch.randn(E, D, MI, **opts) * 0.02
    cases.append((
        "moe_expert_forward",
        lambda: C.moe_expert_forward(hs, offsets, w_gu, w_d),
        lambda: ref_ops.moe_expert_forward(hs, offsets, w_gu, w_d),
        f"{E} 专家 / {N * k} 行",
    ))

    eo = ref_ops.moe_expert_forward(hs, offsets, w_gu, w_d)
    cases.append((
        "moe_combine",
        lambda: C.moe_combine(eo, sorted_idx, weights, N),
        lambda: ref_ops.moe_combine(eo, sorted_idx, weights, N),
        f"[{N}, {D}]",
    ))
    return cases


def bench_e2e(args, device):
    """端到端：prefill 延迟 + decode 吞吐，两种后端都跑一遍。"""
    from llm_infer.engine import Engine, Sequence
    from llm_infer.loader import load_model

    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    model = load_model(args.model_dir, dtype=dtype, device=device, max_model_len=8192)
    prompt = list(range(100, 100 + args.prompt_len))

    print(f"\n端到端（batch {args.batch}，prompt {args.prompt_len} token，生成 {args.gen} token）")
    print(f"{'后端':<8}{'prefill (ms)':>14}{'decode (tok/s)':>16}")
    for backend in ("auto", "torch"):
        ops.set_backend(backend)
        engine = Engine(model, num_blocks=4096, block_size=16)
        seqs = [Sequence(list(prompt), len(prompt)) for _ in range(args.batch)]
        t_pre = timeit(lambda: engine.prefill(seqs), device, warmup=1, iters=3)

        engine = Engine(model, num_blocks=4096, block_size=16)
        sync(device)
        t = time.perf_counter()
        engine.generate([prompt] * args.batch, max_new_tokens=args.gen)
        sync(device)
        dt = time.perf_counter() - t
        print(f"{backend:<8}{t_pre:>14.1f}{args.batch * args.gen / dt:>16.1f}")
    ops.set_backend("auto")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", type=int, default=0, help="prefill 的 token 数（0 = 按设备自动选）")
    ap.add_argument("--batch", type=int, default=0, help="decode 的并发序列数")
    ap.add_argument("--ctx", type=int, default=0, help="decode 时每条序列的上下文长度")
    ap.add_argument("--dtype", default="auto", choices=["auto", "float32", "float16", "bfloat16"])
    ap.add_argument("--model-dir")
    ap.add_argument("--e2e", action="store_true", help="额外跑端到端吞吐，需要 --model-dir")
    ap.add_argument("--prompt-len", type=int, default=128)
    ap.add_argument("--gen", type=int, default=32)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.dtype == "auto":
        dtype = torch.bfloat16 if device == "cuda" else torch.float32
    else:
        dtype = getattr(torch, args.dtype)
    # CPU 上朴素内核很慢，默认用小规模
    defaults = (512, 32, 1024) if device == "cuda" else (128, 8, 256)
    args.seq = args.seq or defaults[0]
    args.batch = args.batch or defaults[1]
    args.ctx = args.ctx or defaults[2]

    name = torch.cuda.get_device_name() if device == "cuda" else "CPU"
    print(f"设备: {name}　dtype: {dtype}　后端可用: C++={ops.HAS_CPP}")
    print(f"prefill {args.seq} token / decode batch {args.batch} × 上下文 {args.ctx}\n")

    cfg = ModelConfig.from_pretrained(args.model_dir) if args.model_dir else ModelConfig()
    cases = build_cases(cfg, args, device, dtype)

    print(f"{'算子':<22}{'自写 (ms)':>12}{'PyTorch (ms)':>14}{'倍数':>8}  规模")
    for name, mine, ref, note in cases:
        if not ops.has_kernel(name, torch.device(device)):
            print(f"{name:<22}{'无内核':>12}")
            continue
        t_mine = timeit(mine, device)
        t_ref = timeit(ref, device)
        print(f"{name:<22}{t_mine:>12.3f}{t_ref:>14.3f}{t_ref / t_mine:>7.2f}×  {note}")

    if args.e2e:
        if not args.model_dir:
            raise SystemExit("--e2e 需要 --model-dir")
        bench_e2e(args, device)


if __name__ == "__main__":
    main()
