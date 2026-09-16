"""用本项目的引擎跑 DeepSeek-V2-Lite(-Chat)。

    python examples/generate.py --model-dir checkpoints/DeepSeek-V2-Lite-Chat --prompt "介绍一下 MLA"
"""

import argparse
import time

import torch
from transformers import AutoTokenizer

from llm_infer import ops
from llm_infer.engine import Engine
from llm_infer.loader import load_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--prompt", action="append", required=True, help="可以重复传多次，一起 batch")
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--backend", default="auto", choices=["auto", "cpp", "torch"])
    args = ap.parse_args()

    ops.set_backend(args.backend)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    tok = AutoTokenizer.from_pretrained(args.model_dir)
    model = load_model(args.model_dir, dtype=dtype, device=device, max_model_len=8192)
    engine = Engine(model, num_blocks=2048, block_size=16)

    prompts = []
    for p in args.prompt:
        if tok.chat_template:
            ids = tok.apply_chat_template([{"role": "user", "content": p}], add_generation_prompt=True)
            prompts.append(list(ids["input_ids"] if isinstance(ids, dict) else ids))
        else:
            prompts.append(tok(p).input_ids)

    if device == "cuda":
        torch.cuda.synchronize()
    t = time.time()
    outs = engine.generate(prompts, args.max_new_tokens, args.temperature, eos_token_id=tok.eos_token_id)
    if device == "cuda":
        torch.cuda.synchronize()
    dt = time.time() - t

    n = sum(len(o) for o in outs)
    for p, o in zip(args.prompt, outs):
        print(f"\n>>> {p}\n{tok.decode(o, skip_special_tokens=True)}")
    print(f"\n{n} tokens in {dt:.2f}s ({n / dt:.1f} tok/s, 含 prefill)")


if __name__ == "__main__":
    main()
