// MoE 相关算子的 CPU 实现：top-k 路由、按专家排序、分组专家计算、结果合并。
#include <algorithm>
#include <cmath>
#include <numeric>
#include <vector>

#include "../ops.h"
#include "common.h"

std::tuple<torch::Tensor, torch::Tensor> topk_softmax(const torch::Tensor& gating_logits, int64_t topk,
                                                      bool renormalize) {
  // 路由统一在 float32 里算，和 HF 实现保持一致。
  auto logits = gating_logits.to(torch::kFloat32).contiguous();
  check_cpu_contig(logits, "gating_logits");
  TORCH_CHECK(logits.dim() == 2, "gating_logits must be [N, E]");
  const int64_t n = logits.size(0), num_experts = logits.size(1);
  TORCH_CHECK(topk > 0 && topk <= num_experts, "invalid topk");

  auto weights = torch::empty({n, topk}, logits.options());
  auto ids = torch::empty({n, topk}, logits.options().dtype(torch::kInt64));
  const float* lp = logits.data_ptr<float>();
  float* wp = weights.data_ptr<float>();
  int64_t* ip = ids.data_ptr<int64_t>();

  at::parallel_for(0, n, 64, [&](int64_t b, int64_t e) {
    std::vector<float> prob(num_experts);
    std::vector<int64_t> order(num_experts);
    for (int64_t i = b; i < e; ++i) {
      const float* row = lp + i * num_experts;
      const float mx = *std::max_element(row, row + num_experts);
      double sum = 0;
      for (int64_t j = 0; j < num_experts; ++j) sum += std::exp(static_cast<double>(row[j] - mx));
      for (int64_t j = 0; j < num_experts; ++j) prob[j] = static_cast<float>(std::exp(row[j] - mx) / sum);

      std::iota(order.begin(), order.end(), 0);
      std::partial_sort(order.begin(), order.begin() + topk, order.end(),
                        [&](int64_t a, int64_t c) { return prob[a] > prob[c]; });
      float wsum = 0;
      for (int64_t j = 0; j < topk; ++j) {
        ip[i * topk + j] = order[j];
        wp[i * topk + j] = prob[order[j]];
        wsum += prob[order[j]];
      }
      if (renormalize) {
        for (int64_t j = 0; j < topk; ++j) wp[i * topk + j] /= wsum;
      }
    }
  });
  return {weights, ids};
}

std::tuple<torch::Tensor, torch::Tensor> moe_align(const torch::Tensor& topk_ids, int64_t num_experts) {
  check_index_tensor(topk_ids, "topk_ids");
  const int64_t total = topk_ids.numel();
  auto offsets = torch::zeros({num_experts + 1}, torch::dtype(torch::kInt64));
  auto sorted_idx = torch::empty({total}, torch::dtype(torch::kInt64));
  int64_t* off = offsets.data_ptr<int64_t>();
  int64_t* sp = sorted_idx.data_ptr<int64_t>();

  // 计数排序，同一专家内保持原顺序（稳定）。
  for (int64_t i = 0; i < total; ++i) {
    const int64_t eid = read_index(topk_ids, i);
    TORCH_CHECK(eid >= 0 && eid < num_experts, "expert id out of range: ", eid);
    off[eid + 1]++;
  }
  for (int64_t e = 0; e < num_experts; ++e) off[e + 1] += off[e];
  std::vector<int64_t> cursor(off, off + num_experts);
  for (int64_t i = 0; i < total; ++i) sp[cursor[read_index(topk_ids, i)]++] = i;
  return {sorted_idx, offsets};
}

torch::Tensor moe_expert_forward(const torch::Tensor& hidden_sorted, const torch::Tensor& expert_offsets,
                                 const torch::Tensor& w_gate_up, const torch::Tensor& w_down) {
  check_cpu_contig(hidden_sorted, "hidden_sorted");
  check_index_tensor(expert_offsets, "expert_offsets");
  TORCH_CHECK(w_gate_up.dim() == 3 && w_down.dim() == 3, "expert weights must be 3D");
  const int64_t num_experts = w_gate_up.size(0);
  TORCH_CHECK(expert_offsets.numel() == num_experts + 1 && w_down.size(0) == num_experts, "num_experts mismatch");
  TORCH_CHECK(read_index(expert_offsets, num_experts) == hidden_sorted.size(0), "offsets[-1] must equal M");

  auto out = torch::empty({hidden_sorted.size(0), w_down.size(1)}, hidden_sorted.options());
  for (int64_t e = 0; e < num_experts; ++e) {
    const int64_t s = read_index(expert_offsets, e), cnt = read_index(expert_offsets, e + 1) - s;
    if (cnt == 0) continue;
    auto x = hidden_sorted.narrow(0, s, cnt);
    // 矩阵乘走 torch 自带的 BLAS，激活用我们自己的 silu_and_mul。
    auto gate_up = torch::mm(x, w_gate_up[e].t());
    auto act = silu_and_mul(gate_up);
    out.narrow(0, s, cnt).copy_(torch::mm(act, w_down[e].t()));
  }
  return out;
}

torch::Tensor moe_combine(const torch::Tensor& expert_out, const torch::Tensor& sorted_idx,
                          const torch::Tensor& topk_weights, int64_t num_tokens) {
  check_cpu_contig(expert_out, "expert_out");
  check_index_tensor(sorted_idx, "sorted_idx");
  check_cpu_contig(topk_weights, "topk_weights");
  TORCH_CHECK(topk_weights.dtype() == torch::kFloat32, "topk_weights must be float32");
  TORCH_CHECK(topk_weights.dim() == 2 && topk_weights.size(0) == num_tokens, "topk_weights must be [N, k]");
  const int64_t k = topk_weights.size(1), total = num_tokens * k, d = expert_out.size(1);
  TORCH_CHECK(sorted_idx.numel() == total && expert_out.size(0) == total, "size mismatch");

  // 反查表：扁平下标 -> 在 expert_out 中的行号，这样可以按 token 并行、没有写竞争。
  std::vector<int64_t> inv(total);
  for (int64_t i = 0; i < total; ++i) inv[read_index(sorted_idx, i)] = i;

  auto out = torch::zeros({num_tokens, d}, expert_out.options());
  LLM_DISPATCH_FLOAT(expert_out.scalar_type(), "moe_combine", {
    const T* ep = expert_out.data_ptr<T>();
    const float* wp = topk_weights.data_ptr<float>();
    T* op = out.data_ptr<T>();
    at::parallel_for(0, num_tokens, 16, [&](int64_t b, int64_t e) {
      for (int64_t t = b; t < e; ++t) {
        T* ot = op + t * d;
        for (int64_t j = 0; j < k; ++j) {
          const T w = static_cast<T>(wp[t * k + j]);
          const T* row = ep + inv[t * k + j] * d;
          for (int64_t c = 0; c < d; ++c) ot[c] += w * row[c];
        }
      }
    });
  });
  return out;
}

TORCH_LIBRARY_IMPL(llm_infer, CPU, m) {
  m.impl("topk_softmax", &topk_softmax);
  m.impl("moe_align", &moe_align);
  m.impl("moe_expert_forward", &moe_expert_forward);
  m.impl("moe_combine", &moe_combine);
}
