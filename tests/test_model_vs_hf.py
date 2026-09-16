"""用随机初始化的小号 DeepSeek-V2，对比本项目模型和 transformers 原生实现的 logits。

覆盖：多序列 packed prefill、分页 cache 跨块的多步 decode、V2-Lite（无 q_lora）和 V2（有 q_lora）两种注意力。
"""

import pytest
import torch
from transformers import DeepseekV2Config, DeepseekV2ForCausalLM as HFModel

from llm_infer import ops
from llm_infer.config import ModelConfig
from llm_infer.engine import Engine, Sequence
from llm_infer.model import DeepseekV2ForCausalLM

BACKENDS = ["cpp", "torch"] if ops.HAS_CPP else ["torch"]
DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


def tiny_hf_model(q_lora_rank):
    cfg = DeepseekV2Config(
        vocab_size=97,
        hidden_size=64,
        intermediate_size=96,
        moe_intermediate_size=24,
        num_hidden_layers=3,
        num_attention_heads=4,
        num_key_value_heads=4,
        kv_lora_rank=16,
        q_lora_rank=q_lora_rank,
        qk_nope_head_dim=12,
        qk_rope_head_dim=8,
        v_head_dim=10,
        n_routed_experts=8,
        n_shared_experts=2,
        num_experts_per_tok=3,
        first_k_dense_replace=1,
        max_position_embeddings=256,
        initializer_range=0.2,
        rope_parameters={
            "rope_type": "yarn",
            "rope_theta": 10000.0,
            "factor": 4.0,
            "original_max_position_embeddings": 64,
            "beta_fast": 32,
            "beta_slow": 1,
            "mscale": 0.707,
            "mscale_all_dim": 0.707,
        },
        attn_implementation="eager",
    )
    torch.manual_seed(0)
    return HFModel(cfg).eval()


def ours_from_hf(hf):
    cfg = ModelConfig.from_dict(hf.config.to_dict())
    model = DeepseekV2ForCausalLM(cfg, max_model_len=256)
    model.load_state_dict(hf.state_dict(), strict=True)
    model.init_rope_cache()
    return model.eval()


@torch.no_grad()
def hf_logits(hf, ids):
    return hf(torch.tensor([ids])).logits[0].float()


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("q_lora_rank", [None, 24])
def test_prefill_and_decode_match_hf(backend, q_lora_rank, device):
    ops.set_backend(backend)
    try:
        hf = tiny_hf_model(q_lora_rank)  # 参考模型始终在 CPU 上
        ours = ours_from_hf(hf).to(device)
        engine = Engine(ours, num_blocks=64, block_size=4)

        g = torch.Generator().manual_seed(1)
        prompts = [torch.randint(0, 97, (n,), generator=g).tolist() for n in (7, 3, 12)]
        seqs = [Sequence(list(p), len(p)) for p in prompts]

        # prefill：所有位置的 logits 都要对上
        got = engine.prefill(seqs, all_logits=True).cpu()
        start = 0
        for p in prompts:
            torch.testing.assert_close(got[start : start + len(p)], hf_logits(hf, p), rtol=1e-4, atol=1e-4)
            start += len(p)

        # decode：teacher forcing 喂 6 个 token，跨越多个 cache 块
        for _ in range(6):
            for s in seqs:
                s.token_ids.append(int(torch.randint(0, 97, (1,), generator=g)))
            got = engine.decode(seqs).cpu()
            for i, s in enumerate(seqs):
                torch.testing.assert_close(got[i], hf_logits(hf, s.token_ids)[-1], rtol=1e-4, atol=1e-4)
    finally:
        ops.set_backend("auto")


def test_generate_greedy_matches_hf():
    hf = tiny_hf_model(None)
    ours = ours_from_hf(hf)
    prompt = [5, 17, 42, 8]
    out = Engine(ours, num_blocks=32, block_size=4).generate([prompt], max_new_tokens=10)[0]
    ref = hf.generate(torch.tensor([prompt]), max_new_tokens=10, do_sample=False)[0, len(prompt) :].tolist()
    assert out == ref


def test_load_hub_format_checkpoint(tmp_path):
    """把小模型存成 Hub 原始格式（每个专家单独的权重、rope_scaling 风格的 config），用 load_model 读回来。"""
    import json

    from safetensors.torch import save_file

    from llm_infer.loader import load_model

    hf = tiny_hf_model(None)
    c = hf.config
    inter = c.moe_intermediate_size
    tensors = {}
    for name, t in hf.state_dict().items():
        if name.endswith("mlp.experts.gate_up_proj"):
            prefix = name[: -len(".gate_up_proj")]
            for e in range(t.shape[0]):
                tensors[f"{prefix}.{e}.gate_proj.weight"] = t[e, :inter].clone()
                tensors[f"{prefix}.{e}.up_proj.weight"] = t[e, inter:].clone()
        elif name.endswith("mlp.experts.down_proj"):
            prefix = name[: -len(".down_proj")]
            for e in range(t.shape[0]):
                tensors[f"{prefix}.{e}.down_proj.weight"] = t[e].clone()
        else:
            tensors[name] = t.contiguous()
    names = sorted(tensors)
    save_file({k: tensors[k] for k in names[: len(names) // 2]}, tmp_path / "model-00001-of-00002.safetensors")
    save_file({k: tensors[k] for k in names[len(names) // 2 :]}, tmp_path / "model-00002-of-00002.safetensors")

    rope = dict(c.rope_parameters)
    hub_cfg = {
        k: getattr(c, k)
        for k in ["vocab_size", "hidden_size", "intermediate_size", "moe_intermediate_size", "num_hidden_layers",
                  "num_attention_heads", "kv_lora_rank", "q_lora_rank", "qk_nope_head_dim", "qk_rope_head_dim",
                  "v_head_dim", "n_routed_experts", "n_shared_experts", "num_experts_per_tok",
                  "first_k_dense_replace", "rms_norm_eps", "max_position_embeddings"]
    }
    hub_cfg.update(moe_layer_freq=1, topk_method="greedy", norm_topk_prob=False, routed_scaling_factor=1.0,
                   rope_theta=rope.pop("rope_theta"), rope_scaling={"type": rope.pop("rope_type"), **rope})
    (tmp_path / "config.json").write_text(json.dumps(hub_cfg))

    ours = load_model(str(tmp_path), dtype=torch.float32, device="cpu", max_model_len=256)
    ids = [3, 1, 4, 1, 5, 9, 2, 6]
    got = Engine(ours, num_blocks=16, block_size=4).prefill([Sequence(list(ids), len(ids))], all_logits=True)
    torch.testing.assert_close(got, hf_logits(hf, ids), rtol=1e-4, atol=1e-4)
