"""在 GPU 上用真实 DeepSeek-V2-Lite 权重对比本项目和 transformers 原生实现。

    python scripts/compare_hf.py --model-dir checkpoints/DeepSeek-V2-Lite-Chat

两份 bf16 模型各约 32GB，96GB 显存可以同时放下。
bf16 下两边算子实现不同，logits 会有 1e-2 量级的差异，贪心生成可能在几十个 token 后分叉，属于正常现象；
top-1 一致、差异不随层数爆炸才是关键。
"""

import argparse
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from llm_infer import ops
from llm_infer.engine import Engine, Sequence
from llm_infer.loader import load_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--prompt", default="DeepSeek-V2 uses Multi-head Latent Attention, which")
    ap.add_argument("--new-tokens", type=int, default=32)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model_dir)
    ids = tok(args.prompt).input_ids
    print(f"prompt tokens: {len(ids)}")

    t = time.time()
    ours = load_model(args.model_dir, dtype=torch.bfloat16, device="cuda", max_model_len=4096)
    print(f"ours loaded in {time.time() - t:.1f}s, backend={ops.get_backend()}")

    t = time.time()
    hf = AutoModelForCausalLM.from_pretrained(args.model_dir, dtype=torch.bfloat16, device_map="cuda").eval()
    print(f"hf loaded in {time.time() - t:.1f}s, mem={torch.cuda.memory_allocated() / 2**30:.1f}GiB")

    engine = Engine(ours, num_blocks=512, block_size=16)
    ours_logits = engine.prefill([Sequence(list(ids), len(ids))], all_logits=True)
    with torch.no_grad():
        hf_logits = hf(torch.tensor([ids], device="cuda")).logits[0].float()

    diff = (ours_logits - hf_logits).abs()
    top1_match = (ours_logits.argmax(-1) == hf_logits.argmax(-1)).float().mean().item()
    print(f"prefill logits: max|Δ|={diff.max().item():.4f}  mean|Δ|={diff.mean().item():.5f}  top1一致率={top1_match:.3f}")

    ours_out = Engine(ours, num_blocks=512, block_size=16).generate([ids], max_new_tokens=args.new_tokens)[0]
    hf_out = hf.generate(torch.tensor([ids], device="cuda"), max_new_tokens=args.new_tokens, do_sample=False)
    hf_out = hf_out[0, len(ids) :].tolist()
    same = next((i for i, (a, b) in enumerate(zip(ours_out, hf_out)) if a != b), len(ours_out))
    print(f"greedy 前 {same}/{args.new_tokens} 个 token 一致")
    print("ours:", tok.decode(ours_out))
    print("hf  :", tok.decode(hf_out))


if __name__ == "__main__":
    main()
