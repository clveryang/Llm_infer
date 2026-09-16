"""DeepSeek-V2 / V2-Lite 推理模型。

参数命名和 transformers 原生 DeepseekV2ForCausalLM 保持一致（专家权重堆叠成 3D），
所以可以直接 load_state_dict(hf_model.state_dict())。
所有计算都走 llm_infer.ops，token 按序列拼成一维（packed），和 vLLM 的组织方式相同。
"""

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from . import ops
from .config import ModelConfig
from .rope import attention_scale, build_cos_sin_cache


@dataclass
class AttnMetadata:
    is_prefill: bool
    slot_mapping: torch.Tensor  # [N]，每个 token 写入 KV cache 的位置
    cu_seqlens: torch.Tensor | None = None  # prefill：[B+1]
    block_tables: torch.Tensor | None = None  # decode：[B, max_blocks]
    seq_lens: torch.Tensor | None = None  # decode：[B]，包含当前 token


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor, residual: torch.Tensor | None = None):
        if residual is None:
            return ops.rms_norm(x, self.weight, self.eps)
        ops.fused_add_rms_norm(x, residual, self.weight, self.eps)
        return x, residual


class MLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_up = torch.cat([self.gate_proj(x), self.up_proj(x)], dim=-1)
        return self.down_proj(ops.silu_and_mul(gate_up))


class MLAttention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.num_heads = cfg.num_attention_heads
        self.nope_dim = cfg.qk_nope_head_dim
        self.rope_dim = cfg.qk_rope_head_dim
        self.v_dim = cfg.v_head_dim
        self.kv_lora_rank = cfg.kv_lora_rank
        self.qk_head_dim = cfg.qk_head_dim
        self.scale = attention_scale(cfg.qk_head_dim, cfg.rope_scaling)
        H, D = self.num_heads, cfg.hidden_size

        if cfg.q_lora_rank is None:  # V2-Lite
            self.q_proj = nn.Linear(D, H * self.qk_head_dim, bias=False)
        else:  # V2 / V3 会先压缩 Q
            self.q_a_proj = nn.Linear(D, cfg.q_lora_rank, bias=False)
            self.q_a_layernorm = RMSNorm(cfg.q_lora_rank, cfg.rms_norm_eps)
            self.q_b_proj = nn.Linear(cfg.q_lora_rank, H * self.qk_head_dim, bias=False)
        self.kv_a_proj_with_mqa = nn.Linear(D, self.kv_lora_rank + self.rope_dim, bias=False)
        self.kv_a_layernorm = RMSNorm(self.kv_lora_rank, cfg.rms_norm_eps)
        self.kv_b_proj = nn.Linear(self.kv_lora_rank, H * (self.nope_dim + self.v_dim), bias=False)
        self.o_proj = nn.Linear(H * self.v_dim, D, bias=False)

    def forward(
        self,
        positions: torch.Tensor,
        hidden: torch.Tensor,
        meta: AttnMetadata,
        kv_cache: torch.Tensor,
        cos_sin_cache: torch.Tensor,
    ) -> torch.Tensor:
        N, H = hidden.shape[0], self.num_heads

        if self.cfg.q_lora_rank is None:
            q = self.q_proj(hidden)
        else:
            q = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(hidden)))
        q = q.view(N, H, self.qk_head_dim)
        q_nope, q_pe = q.split([self.nope_dim, self.rope_dim], dim=-1)
        q_pe = q_pe.contiguous()

        latent, k_pe = self.kv_a_proj_with_mqa(hidden).split([self.kv_lora_rank, self.rope_dim], dim=-1)
        latent = self.kv_a_layernorm(latent)
        k_pe = k_pe.unsqueeze(1).contiguous()  # [N, 1, P]，所有 head 共享

        ops.rotary_embedding(positions, q_pe, k_pe, cos_sin_cache, False)
        ops.concat_and_cache_mla(latent, k_pe.squeeze(1), kv_cache, meta.slot_mapping)

        if meta.is_prefill:
            # prefill：把 latent 解压成完整的 k/v，再做标准的因果注意力
            k_nope, v = self.kv_b_proj(latent).view(N, H, -1).split([self.nope_dim, self.v_dim], dim=-1)
            q_full = torch.cat([q_nope, q_pe], dim=-1)
            k_full = torch.cat([k_nope, k_pe.expand(-1, H, -1)], dim=-1)
            out = ops.mla_prefill_attention(q_full, k_full, v, meta.cu_seqlens, self.scale)
        else:
            # decode：不解压 cache，把 query 投影到 latent 空间
            #   q_nope·(W_k c) = (W_kᵀ q_nope)·c,   Σ a·(W_v c) = W_v (Σ a·c)
            w = self.kv_b_proj.weight.view(H, self.nope_dim + self.v_dim, self.kv_lora_rank)
            w_k, w_v = w[:, : self.nope_dim], w[:, self.nope_dim :]
            q_latent = torch.einsum("nhd,hdr->nhr", q_nope, w_k)
            out_latent = ops.mla_decode_attention(
                torch.cat([q_latent, q_pe], dim=-1),
                kv_cache,
                meta.block_tables,
                meta.seq_lens,
                self.kv_lora_rank,
                self.scale,
            )
            out = torch.einsum("nhr,hvr->nhv", out_latent, w_v)

        return self.o_proj(out.reshape(N, H * self.v_dim))


class Experts(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        E, D, I = cfg.n_routed_experts, cfg.hidden_size, cfg.moe_intermediate_size
        self.gate_up_proj = nn.Parameter(torch.empty(E, 2 * I, D))
        self.down_proj = nn.Parameter(torch.empty(E, D, I))


class Router(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(cfg.n_routed_experts, cfg.hidden_size))


class MoE(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        if cfg.topk_method != "greedy":
            raise NotImplementedError(f"topk_method={cfg.topk_method}（V2-Lite 只用 greedy）")
        self.cfg = cfg
        self.gate = Router(cfg)
        self.experts = Experts(cfg)
        self.shared_experts = (
            MLP(cfg.hidden_size, cfg.moe_intermediate_size * cfg.n_shared_experts) if cfg.n_shared_experts else None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        cfg, N = self.cfg, x.shape[0]
        k = cfg.num_experts_per_tok

        logits = F.linear(x.float(), self.gate.weight.float())
        weights, ids = ops.topk_softmax(logits, k, cfg.norm_topk_prob)
        weights = weights * cfg.routed_scaling_factor

        sorted_idx, offsets = ops.moe_align(ids, cfg.n_routed_experts)
        hidden_sorted = x.index_select(0, sorted_idx // k)
        expert_out = ops.moe_expert_forward(
            hidden_sorted, offsets, self.experts.gate_up_proj, self.experts.down_proj
        )
        out = ops.moe_combine(expert_out, sorted_idx, weights, N)

        if self.shared_experts is not None:
            out = out + self.shared_experts(x)
        return out


class DecoderLayer(nn.Module):
    def __init__(self, cfg: ModelConfig, layer_idx: int):
        super().__init__()
        self.self_attn = MLAttention(cfg)
        self.mlp = MoE(cfg) if cfg.is_moe_layer(layer_idx) else MLP(cfg.hidden_size, cfg.intermediate_size)
        self.input_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)

    def forward(self, positions, hidden, residual, meta, kv_cache, cos_sin_cache):
        if residual is None:
            residual = hidden
            hidden = self.input_layernorm(hidden)
        else:
            hidden, residual = self.input_layernorm(hidden, residual)
        hidden = self.self_attn(positions, hidden, meta, kv_cache, cos_sin_cache)
        hidden, residual = self.post_attention_layernorm(hidden, residual)
        return self.mlp(hidden), residual


class DeepseekV2Model(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.layers = nn.ModuleList(DecoderLayer(cfg, i) for i in range(cfg.num_hidden_layers))
        self.norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)


class DeepseekV2ForCausalLM(nn.Module):
    def __init__(self, cfg: ModelConfig, max_model_len: int | None = None):
        super().__init__()
        self.config = cfg
        self.max_model_len = max_model_len or cfg.max_position_embeddings
        self.model = DeepseekV2Model(cfg)
        self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
        self.register_buffer("cos_sin_cache", torch.empty(0), persistent=False)

    def init_rope_cache(self) -> None:
        """构建 cos/sin 表。模型 .to(device, dtype) 之后调用。"""
        cfg, w = self.config, self.lm_head.weight
        self.cos_sin_cache = build_cos_sin_cache(
            cfg.qk_rope_head_dim, self.max_model_len, cfg.rope_theta, cfg.rope_scaling, w.dtype, w.device
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        meta: AttnMetadata,
        kv_caches: list[torch.Tensor],
    ) -> torch.Tensor:
        """input_ids/positions 为 [N]，返回最后一层 hidden [N, D]。"""
        if self.cos_sin_cache.numel() == 0:
            self.init_rope_cache()
        hidden = self.model.embed_tokens(input_ids)
        residual = None
        for layer, kv_cache in zip(self.model.layers, kv_caches):
            hidden, residual = layer(positions, hidden, residual, meta, kv_cache, self.cos_sin_cache)
        hidden, _ = self.model.norm(hidden, residual)
        return hidden

    def compute_logits(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.lm_head(hidden).float()
