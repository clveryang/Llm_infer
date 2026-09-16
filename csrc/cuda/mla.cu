// MLA 相关算子的 CUDA 实现：RoPE、分页 latent cache 写入、prefill / decode 注意力。
//
// 两个注意力内核都是「每个 (query, head) 一个 GPU 线程」，线程内顺序扫描所有 key，
// 用两遍扫描完成 softmax，不需要额外显存：
//   第 1 遍：用 double 累加 logsumexp(score)
//   第 2 遍：out += exp(score - lse) · value
// 这是便于理解和对拍的朴素版本，性能远不如 FlashAttention / FlashMLA 这类分块 kernel。
#include "common.cuh"

namespace llm_cuda {
namespace {

template <typename T, typename CT>
inline C10_HOST_DEVICE void rotate(T* x, const T* cos, const T* sin, const int64_t half, const bool is_neox) {
  for (int64_t i = 0; i < half; ++i) {
    const int64_t i1 = is_neox ? i : 2 * i;
    const int64_t i2 = is_neox ? i + half : 2 * i + 1;
    const CT x1 = static_cast<CT>(x[i1]);
    const CT x2 = static_cast<CT>(x[i2]);
    const CT c = static_cast<CT>(cos[i]);
    const CT s = static_cast<CT>(sin[i]);
    x[i1] = static_cast<T>(x1 * c - x2 * s);
    x[i2] = static_cast<T>(x2 * c + x1 * s);
  }
}

template <typename T, typename CT>
__global__ void rotary_fwd(const int64_t* positions, T* q, T* k, const T* cache, const int64_t n, const int64_t hq,
                             const int64_t hk, const int64_t rot_dim, const bool is_neox) {
  const int64_t t = LLM_CUDA_INDEX();
  if (t >= n) return;
  const int64_t half = rot_dim / 2;
  const T* cos = cache + positions[t] * rot_dim;
  const T* sin = cos + half;
  for (int64_t h = 0; h < hq; ++h) rotate<T, CT>(q + (t * hq + h) * rot_dim, cos, sin, half, is_neox);
  for (int64_t h = 0; h < hk; ++h) rotate<T, CT>(k + (t * hk + h) * rot_dim, cos, sin, half, is_neox);
}

template <typename T>
__global__ void cache_mla_fwd(const T* c, const T* pe, T* cache, const int64_t* slots, const int64_t n,
                                const int64_t r, const int64_t p) {
  const int64_t i = LLM_CUDA_INDEX();
  if (i >= n) return;
  const int64_t slot = slots[i];
  if (slot < 0) return;
  T* dst = cache + slot * (r + p);
  for (int64_t j = 0; j < r; ++j) dst[j] = c[i * r + j];
  for (int64_t j = 0; j < p; ++j) dst[r + j] = pe[i * p + j];
}

// 线程编号 = token * H + head
template <typename T, typename CT>
__global__ void prefill_attn_fwd(CT* out, const T* q, const T* k, const T* v, const int64_t* seq_start,
                                   const int64_t n, const int64_t h, const int64_t dqk, const int64_t dv,
                                   const CT scale) {
  const int64_t idx = LLM_CUDA_INDEX();
  if (idx >= n * h) return;
  const int64_t i = idx / h, hh = idx % h;
  const int64_t s0 = seq_start[i];
  const T* qi = q + idx * dqk;

  // 第 1 遍：logsumexp
  double lse = -INFINITY;
  for (int64_t j = s0; j <= i; ++j) {
    const T* kj = k + (j * h + hh) * dqk;
    CT dot = 0;
    for (int64_t d = 0; d < dqk; ++d) dot += static_cast<CT>(qi[d]) * static_cast<CT>(kj[d]);
    lse = log_add_exp(lse, static_cast<double>(dot * scale));
  }
  // 第 2 遍：加权求和
  CT* o = out + idx * dv;
  for (int64_t j = s0; j <= i; ++j) {
    const T* kj = k + (j * h + hh) * dqk;
    CT dot = 0;
    for (int64_t d = 0; d < dqk; ++d) dot += static_cast<CT>(qi[d]) * static_cast<CT>(kj[d]);
    const CT a = static_cast<CT>(std::exp(static_cast<double>(dot * scale) - lse));
    const T* vj = v + (j * h + hh) * dv;
    for (int64_t d = 0; d < dv; ++d) o[d] += a * static_cast<CT>(vj[d]);
  }
}

// 线程编号 = seq * H + head
template <typename T, typename CT>
__global__ void decode_attn_fwd(CT* out, const T* q, const T* cache, const int64_t* block_tables,
                                  const int64_t* seq_lens, const int64_t bsz, const int64_t h, const int64_t dim,
                                  const int64_t r, const int64_t block_size, const int64_t max_blocks,
                                  const CT scale) {
  const int64_t idx = LLM_CUDA_INDEX();
  if (idx >= bsz * h) return;
  const int64_t s = idx / h;
  const int64_t len = seq_lens[s];
  const int64_t* table = block_tables + s * max_blocks;
  const T* qi = q + idx * dim;

  double lse = -INFINITY;
  for (int64_t t = 0; t < len; ++t) {
    const T* row = cache + (table[t / block_size] * block_size + t % block_size) * dim;
    CT dot = 0;
    for (int64_t d = 0; d < dim; ++d) dot += static_cast<CT>(qi[d]) * static_cast<CT>(row[d]);
    lse = log_add_exp(lse, static_cast<double>(dot * scale));
  }
  CT* o = out + idx * r;
  for (int64_t t = 0; t < len; ++t) {
    const T* row = cache + (table[t / block_size] * block_size + t % block_size) * dim;
    CT dot = 0;
    for (int64_t d = 0; d < dim; ++d) dot += static_cast<CT>(qi[d]) * static_cast<CT>(row[d]);
    const CT a = static_cast<CT>(std::exp(static_cast<double>(dot * scale) - lse));
    for (int64_t d = 0; d < r; ++d) o[d] += a * static_cast<CT>(row[d]);  // 只累加 latent 部分
  }
}

// 累加结果用的 dtype：半精度提升到 float32，避免逐步累加时的舍入误差。
inline torch::Dtype compute_dtype(torch::Dtype dt) { return dt == torch::kFloat64 ? torch::kFloat64 : torch::kFloat32; }

}  // namespace

void rotary_embedding(const torch::Tensor& positions, torch::Tensor& query, torch::Tensor& key,
                      const torch::Tensor& cos_sin_cache, bool is_neox) {
  check_cuda(query, "query");
  check_cuda(key, "key");
  check_cuda(cos_sin_cache, "cos_sin_cache");
  auto pos = as_index(positions, "positions");
  TORCH_CHECK(query.dim() == 3 && key.dim() == 3, "query/key must be [N, H, rot_dim]");
  const int64_t n = query.size(0), rot_dim = cos_sin_cache.size(1);
  TORCH_CHECK(rot_dim % 2 == 0, "rot_dim must be even");
  TORCH_CHECK(query.size(2) == rot_dim && key.size(2) == rot_dim, "query/key last dim must equal rot_dim");
  TORCH_CHECK(key.size(0) == n && pos.numel() == n, "batch size mismatch");
  TORCH_CHECK(query.dtype() == key.dtype() && cos_sin_cache.dtype() == query.dtype(), "dtype mismatch");
  if (n == 0) return;

  const c10::cuda::CUDAGuard guard(query.device());
  const auto stream = at::cuda::getCurrentCUDAStream();
  LLM_CUDA_DISPATCH(query.scalar_type(), "rotary_embedding",
                    rotary_fwd<T, CT><<<grid_for(n), block_dim(), 0, stream.stream()>>>(
                        pos.data_ptr<int64_t>(), query.data_ptr<T>(), key.data_ptr<T>(),
                        cos_sin_cache.data_ptr<T>(), n, query.size(1), key.size(1), rot_dim, is_neox));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void concat_and_cache_mla(const torch::Tensor& kv_c, const torch::Tensor& k_pe, torch::Tensor& kv_cache,
                          const torch::Tensor& slot_mapping) {
  check_cuda(kv_c, "kv_c");
  check_cuda(k_pe, "k_pe");
  check_cuda(kv_cache, "kv_cache");
  auto slots = as_index(slot_mapping, "slot_mapping");
  const int64_t n = kv_c.size(0), r = kv_c.size(1), p = k_pe.size(1);
  TORCH_CHECK(k_pe.size(0) == n && slots.numel() == n, "batch size mismatch");
  TORCH_CHECK(kv_cache.dim() == 3 && kv_cache.size(2) == r + p, "kv_cache last dim must be kv_lora_rank + rope_dim");
  TORCH_CHECK(kv_c.dtype() == kv_cache.dtype() && k_pe.dtype() == kv_cache.dtype(), "dtype mismatch");
  if (n == 0) return;

  const c10::cuda::CUDAGuard guard(kv_cache.device());
  const auto stream = at::cuda::getCurrentCUDAStream();
  LLM_CUDA_DISPATCH(kv_cache.scalar_type(), "concat_and_cache_mla",
                    cache_mla_fwd<T><<<grid_for(n), block_dim(), 0, stream.stream()>>>(
                        kv_c.data_ptr<T>(), k_pe.data_ptr<T>(), kv_cache.data_ptr<T>(), slots.data_ptr<int64_t>(), n,
                        r, p));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

torch::Tensor mla_prefill_attention(const torch::Tensor& q, const torch::Tensor& k, const torch::Tensor& v,
                                    const torch::Tensor& cu_seqlens, double scale) {
  check_cuda(q, "q");
  check_cuda(k, "k");
  check_cuda(v, "v");
  auto cu = as_index(cu_seqlens, "cu_seqlens");
  TORCH_CHECK(q.dim() == 3 && k.sizes() == q.sizes() && v.dim() == 3, "q/k must be [N, H, Dqk], v [N, H, Dv]");
  TORCH_CHECK(v.size(0) == q.size(0) && v.size(1) == q.size(1), "v shape mismatch");
  TORCH_CHECK(q.dtype() == k.dtype() && q.dtype() == v.dtype(), "dtype mismatch");
  const int64_t n = q.size(0), h = q.size(1), dqk = q.size(2), dv = v.size(2);
  auto out = torch::zeros({n, h, dv}, q.options().dtype(compute_dtype(q.scalar_type())));
  if (n == 0) return out.to(q.dtype());

  // 每个 token 所属序列的起点：seq_id = searchsorted(cu, token, right) - 1
  auto tokens = torch::arange(n, cu.options());
  auto seq_id = torch::searchsorted(cu, tokens, /*out_int32=*/false, /*right=*/true) - 1;
  auto seq_start = cu.index_select(0, seq_id).contiguous();

  const c10::cuda::CUDAGuard guard(q.device());
  const auto stream = at::cuda::getCurrentCUDAStream();
  LLM_CUDA_DISPATCH(q.scalar_type(), "mla_prefill_attention",
                    prefill_attn_fwd<T, CT><<<grid_for(n * h), block_dim(), 0, stream.stream()>>>(
                        out.data_ptr<CT>(), q.data_ptr<T>(), k.data_ptr<T>(), v.data_ptr<T>(),
                        seq_start.data_ptr<int64_t>(), n, h, dqk, dv, static_cast<CT>(scale)));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out.to(q.dtype());
}

torch::Tensor mla_decode_attention(const torch::Tensor& q, const torch::Tensor& kv_cache,
                                   const torch::Tensor& block_tables, const torch::Tensor& seq_lens,
                                   int64_t kv_lora_rank, double scale) {
  check_cuda(q, "q");
  check_cuda(kv_cache, "kv_cache");
  auto tables = as_index(block_tables, "block_tables");
  auto lens = as_index(seq_lens, "seq_lens");
  TORCH_CHECK(q.dim() == 3 && kv_cache.dim() == 3, "q must be [B, H, R+P], kv_cache [blocks, block_size, R+P]");
  TORCH_CHECK(q.dtype() == kv_cache.dtype(), "dtype mismatch");
  const int64_t bsz = q.size(0), h = q.size(1), dim = q.size(2), r = kv_lora_rank;
  TORCH_CHECK(kv_cache.size(2) == dim && r > 0 && r < dim, "latent dim mismatch");
  TORCH_CHECK(tables.dim() == 2 && tables.size(0) == bsz && lens.numel() == bsz, "block_tables/seq_lens batch mismatch");
  auto out = torch::zeros({bsz, h, r}, q.options().dtype(compute_dtype(q.scalar_type())));
  if (bsz == 0) return out.to(q.dtype());

  const c10::cuda::CUDAGuard guard(q.device());
  const auto stream = at::cuda::getCurrentCUDAStream();
  LLM_CUDA_DISPATCH(q.scalar_type(), "mla_decode_attention",
                    decode_attn_fwd<T, CT><<<grid_for(bsz * h), block_dim(), 0, stream.stream()>>>(
                        out.data_ptr<CT>(), q.data_ptr<T>(), kv_cache.data_ptr<T>(), tables.data_ptr<int64_t>(),
                        lens.data_ptr<int64_t>(), bsz, h, dim, r, kv_cache.size(1), tables.size(1),
                        static_cast<CT>(scale)));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out.to(q.dtype());
}

}  // namespace llm_cuda

TORCH_LIBRARY_IMPL(llm_infer, CUDA, m) {
  m.impl("rotary_embedding", &llm_cuda::rotary_embedding);
  m.impl("concat_and_cache_mla", &llm_cuda::concat_and_cache_mla);
  m.impl("mla_prefill_attention", &llm_cuda::mla_prefill_attention);
  m.impl("mla_decode_attention", &llm_cuda::mla_decode_attention);
}
