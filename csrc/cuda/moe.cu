// MoE 相关算子的 CUDA 实现：top-k 路由、按专家排序、分组专家计算、结果合并。
#include "common.cuh"

namespace llm_cuda {

torch::Tensor silu_and_mul(const torch::Tensor& input);  // norm_act.cu

namespace {

constexpr int64_t kMaxTopK = 32;

// 每行一个线程：softmax 单调，所以直接在 logits 上选 top-k，再用 logsumexp 算概率。
__global__ void topk_softmax_fwd(const float* logits, float* weights, int64_t* ids, const int64_t n,
                                   const int64_t num_experts, const int64_t topk, const bool renormalize) {
  const int64_t i = LLM_CUDA_INDEX();
  if (i >= n) return;
  const float* row = logits + i * num_experts;

  double lse = -INFINITY;
  for (int64_t e = 0; e < num_experts; ++e) lse = log_add_exp(lse, static_cast<double>(row[e]));

  // k 很小（DeepSeek 是 6），重复 k 次取最大值即可
  double wsum = 0;
  for (int64_t j = 0; j < topk; ++j) {
    int64_t best = -1;
    for (int64_t e = 0; e < num_experts; ++e) {
      bool taken = false;
      for (int64_t p = 0; p < j; ++p) taken = taken || ids[i * topk + p] == e;
      if (!taken && (best < 0 || row[e] > row[best])) best = e;
    }
    ids[i * topk + j] = best;
    const double w = std::exp(static_cast<double>(row[best]) - lse);
    weights[i * topk + j] = static_cast<float>(w);
    wsum += w;
  }
  if (renormalize) {
    for (int64_t j = 0; j < topk; ++j) weights[i * topk + j] = static_cast<float>(weights[i * topk + j] / wsum);
  }
}

// 计数排序只需要 O(N·k) 的顺序扫描，用一个 GPU 线程做，避免并发写计数器；
// 并行版本可以参考 vLLM 的 moe_align_block_size。
__global__ void moe_align_fwd(const int64_t* ids, int64_t* sorted_idx, int64_t* offsets, int64_t* cursor,
                                const int64_t total, const int64_t num_experts) {
  if (LLM_CUDA_INDEX() != 0) return;
  for (int64_t e = 0; e <= num_experts; ++e) offsets[e] = 0;
  for (int64_t i = 0; i < total; ++i) offsets[ids[i] + 1]++;
  for (int64_t e = 0; e < num_experts; ++e) offsets[e + 1] += offsets[e];
  for (int64_t e = 0; e < num_experts; ++e) cursor[e] = offsets[e];
  for (int64_t i = 0; i < total; ++i) sorted_idx[cursor[ids[i]]++] = i;
}

__global__ void invert_perm_fwd(const int64_t* sorted_idx, int64_t* inv, const int64_t total) {
  const int64_t i = LLM_CUDA_INDEX();
  if (i >= total) return;
  inv[sorted_idx[i]] = i;
}

// 每个 token 一个线程，读取它的 k 个专家输出加权求和，没有写竞争。
template <typename T, typename CT>
__global__ void moe_combine_fwd(T* out, const T* expert_out, const float* weights, const int64_t* inv,
                                  const int64_t n, const int64_t k, const int64_t d) {
  const int64_t t = LLM_CUDA_INDEX();
  if (t >= n) return;
  T* o = out + t * d;
  for (int64_t c = 0; c < d; ++c) {
    CT acc = 0;
    for (int64_t j = 0; j < k; ++j) {
      acc += static_cast<CT>(weights[t * k + j]) * static_cast<CT>(expert_out[inv[t * k + j] * d + c]);
    }
    o[c] = static_cast<T>(acc);
  }
}

}  // namespace

std::tuple<torch::Tensor, torch::Tensor> topk_softmax(const torch::Tensor& gating_logits, int64_t topk,
                                                      bool renormalize) {
  TORCH_CHECK(gating_logits.is_cuda(), "gating_logits must be a CUDA tensor");
  auto logits = gating_logits.to(torch::kFloat32).contiguous();
  TORCH_CHECK(logits.dim() == 2, "gating_logits must be [N, E]");
  const int64_t n = logits.size(0), num_experts = logits.size(1);
  TORCH_CHECK(topk > 0 && topk <= num_experts && topk <= kMaxTopK, "invalid topk");
  auto weights = torch::empty({n, topk}, logits.options());
  auto ids = torch::empty({n, topk}, logits.options().dtype(torch::kInt64));
  if (n == 0) return {weights, ids};

  const c10::cuda::CUDAGuard guard(logits.device());
  const auto stream = at::cuda::getCurrentCUDAStream();
  topk_softmax_fwd<<<grid_for(n), block_dim(), 0, stream.stream()>>>(logits.data_ptr<float>(), weights.data_ptr<float>(),
                                                         ids.data_ptr<int64_t>(), n, num_experts, topk, renormalize);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {weights, ids};
}

std::tuple<torch::Tensor, torch::Tensor> moe_align(const torch::Tensor& topk_ids, int64_t num_experts) {
  auto ids = as_index(topk_ids, "topk_ids").view(-1);
  const int64_t total = ids.numel();
  auto opts = ids.options();
  auto sorted_idx = torch::empty({total}, opts);
  auto offsets = torch::zeros({num_experts + 1}, opts);
  auto cursor = torch::empty({num_experts}, opts);
  if (total == 0) return {sorted_idx, offsets};

  const c10::cuda::CUDAGuard guard(ids.device());
  const auto stream = at::cuda::getCurrentCUDAStream();
  moe_align_fwd<<<dim3(1), dim3(1), 0, stream.stream()>>>(ids.data_ptr<int64_t>(), sorted_idx.data_ptr<int64_t>(),
                                             offsets.data_ptr<int64_t>(), cursor.data_ptr<int64_t>(), total,
                                             num_experts);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {sorted_idx, offsets};
}

torch::Tensor moe_expert_forward(const torch::Tensor& hidden_sorted, const torch::Tensor& expert_offsets,
                                 const torch::Tensor& w_gate_up, const torch::Tensor& w_down) {
  check_cuda(hidden_sorted, "hidden_sorted");
  TORCH_CHECK(w_gate_up.dim() == 3 && w_down.dim() == 3, "expert weights must be 3D");
  const int64_t num_experts = w_gate_up.size(0);
  TORCH_CHECK(expert_offsets.numel() == num_experts + 1 && w_down.size(0) == num_experts, "num_experts mismatch");

  // 每层同步一次把 offsets 拷回 CPU，然后每个专家做两次 cuBLAS 矩阵乘。
  // 进一步优化：cublasGemmGroupedBatchedEx 或 Triton grouped GEMM，一次 launch 算完所有专家。
  auto offs = expert_offsets.to(torch::kCPU, torch::kInt64).contiguous();
  const int64_t* op = offs.data_ptr<int64_t>();
  TORCH_CHECK(op[num_experts] == hidden_sorted.size(0), "offsets[-1] must equal M");

  const c10::cuda::CUDAGuard guard(hidden_sorted.device());
  auto out = torch::empty({hidden_sorted.size(0), w_down.size(1)}, hidden_sorted.options());
  for (int64_t e = 0; e < num_experts; ++e) {
    const int64_t s = op[e], cnt = op[e + 1] - op[e];
    if (cnt == 0) continue;
    auto x = hidden_sorted.narrow(0, s, cnt);
    auto act = silu_and_mul(torch::mm(x, w_gate_up[e].t()));
    auto dst = out.narrow(0, s, cnt);
    torch::mm_out(dst, act, w_down[e].t());
  }
  return out;
}

torch::Tensor moe_combine(const torch::Tensor& expert_out, const torch::Tensor& sorted_idx,
                          const torch::Tensor& topk_weights, int64_t num_tokens) {
  check_cuda(expert_out, "expert_out");
  check_cuda(topk_weights, "topk_weights");
  auto idx = as_index(sorted_idx, "sorted_idx");
  TORCH_CHECK(topk_weights.dtype() == torch::kFloat32, "topk_weights must be float32");
  TORCH_CHECK(topk_weights.dim() == 2 && topk_weights.size(0) == num_tokens, "topk_weights must be [N, k]");
  const int64_t k = topk_weights.size(1), total = num_tokens * k, d = expert_out.size(1);
  TORCH_CHECK(idx.numel() == total && expert_out.size(0) == total, "size mismatch");
  auto out = torch::empty({num_tokens, d}, expert_out.options());
  if (num_tokens == 0) return out;

  const c10::cuda::CUDAGuard guard(expert_out.device());
  const auto stream = at::cuda::getCurrentCUDAStream();
  auto inv = torch::empty({total}, idx.options());
  invert_perm_fwd<<<grid_for(total), block_dim(), 0, stream.stream()>>>(idx.data_ptr<int64_t>(), inv.data_ptr<int64_t>(), total);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  LLM_CUDA_DISPATCH(expert_out.scalar_type(), "moe_combine",
                    moe_combine_fwd<T, CT><<<grid_for(num_tokens), block_dim(), 0, stream.stream()>>>(
                        out.data_ptr<T>(), expert_out.data_ptr<T>(), topk_weights.data_ptr<float>(),
                        inv.data_ptr<int64_t>(), num_tokens, k, d));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

}  // namespace llm_cuda

TORCH_LIBRARY_IMPL(llm_infer, CUDA, m) {
  m.impl("topk_softmax", &llm_cuda::topk_softmax);
  m.impl("moe_align", &llm_cuda::moe_align);
  m.impl("moe_expert_forward", &llm_cuda::moe_expert_forward);
  m.impl("moe_combine", &llm_cuda::moe_combine);
}
