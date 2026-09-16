// DeepSeek-V2-Lite 推理算子声明。
// 所有算子通过 TORCH_LIBRARY 注册到 torch.ops.llm_infer 命名空间，
// schema 定义在 torch_bindings.cpp，各设备实现各自用 TORCH_LIBRARY_IMPL 注册。
#pragma once

#include <torch/torch.h>

#include <tuple>

// ---------------- 通用 ----------------
// out = x / sqrt(mean(x^2) + eps) * weight
torch::Tensor rms_norm(const torch::Tensor& input, const torch::Tensor& weight, double eps);

// 原地：residual = input + residual; input = rms_norm(residual)
void fused_add_rms_norm(torch::Tensor& input, torch::Tensor& residual, const torch::Tensor& weight,
                        double eps);

// input [..., 2d] -> silu(input[..., :d]) * input[..., d:]，输出 [..., d]
torch::Tensor silu_and_mul(const torch::Tensor& input);

// ---------------- MLA ----------------
// 原地 RoPE。query [N, Hq, rot_dim]，key [N, Hk, rot_dim]，positions [N]，
// cos_sin_cache [max_pos, rot_dim]，每行前半是 cos、后半是 sin。
// is_neox=false 为交错排列 (x0,x1),(x2,x3)...，DeepSeek 用这种。
void rotary_embedding(const torch::Tensor& positions, torch::Tensor& query, torch::Tensor& key,
                      const torch::Tensor& cos_sin_cache, bool is_neox);

// 把压缩 latent 和 k_pe 写入分页 KV cache。
// kv_c [N, R]，k_pe [N, P]，kv_cache [num_blocks, block_size, R + P]，
// slot_mapping [N]，slot = block_id * block_size + offset，<0 表示跳过。
void concat_and_cache_mla(const torch::Tensor& kv_c, const torch::Tensor& k_pe, torch::Tensor& kv_cache,
                          const torch::Tensor& slot_mapping);

// 变长因果注意力（prefill）。q/k [N, H, Dqk]，v [N, H, Dv]，cu_seqlens [B+1]。
// 返回 [N, H, Dv]。
torch::Tensor mla_prefill_attention(const torch::Tensor& q, const torch::Tensor& k, const torch::Tensor& v,
                                    const torch::Tensor& cu_seqlens, double scale);

// latent 空间里的 decode 注意力，不解压 KV cache。
// q [B, H, R + P]：前 R 维是 W_kᵀ·q_nope（query 投影到 latent 空间），后 P 维是 q_pe。
// kv_cache [num_blocks, block_size, R + P]，block_tables [B, max_blocks]，
// seq_lens [B]（包含当前 token）。
// 返回 [B, H, R]，即 Σ softmax(q·kv) · c，调用方再乘 W_v 得到 value。
torch::Tensor mla_decode_attention(const torch::Tensor& q, const torch::Tensor& kv_cache,
                                   const torch::Tensor& block_tables, const torch::Tensor& seq_lens,
                                   int64_t kv_lora_rank, double scale);

// ---------------- MoE ----------------
// softmax(logits) 后取 top-k。返回 (weights [N, k] float32, ids [N, k] int64)。
std::tuple<torch::Tensor, torch::Tensor> topk_softmax(const torch::Tensor& gating_logits, int64_t topk,
                                                      bool renormalize);

// 按专家做计数排序。返回 (sorted_idx [N*k]：topk_ids 的扁平下标,
// expert_offsets [E+1])，专家 e 的 token 位于 sorted_idx[offsets[e]:offsets[e+1]]。
std::tuple<torch::Tensor, torch::Tensor> moe_align(const torch::Tensor& topk_ids, int64_t num_experts);

// 分组 SwiGLU 专家计算。hidden_sorted [M, D]，expert_offsets [E+1]，
// w_gate_up [E, 2I, D]，w_down [E, D, I]。返回 [M, D]。
torch::Tensor moe_expert_forward(const torch::Tensor& hidden_sorted, const torch::Tensor& expert_offsets,
                                 const torch::Tensor& w_gate_up, const torch::Tensor& w_down);

// 专家输出按 top-k 权重加权并还原到 token 顺序。返回 [num_tokens, D]。
torch::Tensor moe_combine(const torch::Tensor& expert_out, const torch::Tensor& sorted_idx,
                          const torch::Tensor& topk_weights, int64_t num_tokens);
