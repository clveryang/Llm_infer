// MLA 相关算子的 CPU 实现：RoPE、分页 latent cache 写入、prefill / decode 注意力。
#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

#include "../ops.h"
#include "common.h"

namespace {

// x 为一个 head 的 rot_dim 维向量，原地旋转。
template <typename T>
inline void rotate(T* x, const T* cos, const T* sin, int64_t half, bool is_neox) {
  for (int64_t i = 0; i < half; ++i) {
    const int64_t i1 = is_neox ? i : 2 * i;
    const int64_t i2 = is_neox ? i + half : 2 * i + 1;
    const T x1 = x[i1];
    const T x2 = x[i2];
    x[i1] = x1 * cos[i] - x2 * sin[i];
    x[i2] = x2 * cos[i] + x1 * sin[i];
  }
}

// 数值稳定的 softmax，原地作用于 s[0:n]。
template <typename T>
inline void softmax_inplace(T* s, int64_t n) {
  T mx = -std::numeric_limits<T>::infinity();
  for (int64_t i = 0; i < n; ++i) mx = std::max(mx, s[i]);
  T sum = 0;
  for (int64_t i = 0; i < n; ++i) {
    s[i] = std::exp(s[i] - mx);
    sum += s[i];
  }
  for (int64_t i = 0; i < n; ++i) s[i] /= sum;
}

}  // namespace

void rotary_embedding(const torch::Tensor& positions, torch::Tensor& query, torch::Tensor& key,
                      const torch::Tensor& cos_sin_cache, bool is_neox) {
  check_index_tensor(positions, "positions");
  check_cpu_contig(query, "query");
  check_cpu_contig(key, "key");
  check_cpu_contig(cos_sin_cache, "cos_sin_cache");
  TORCH_CHECK(query.dim() == 3 && key.dim() == 3, "query/key must be [N, H, rot_dim]");
  const int64_t n = query.size(0);
  const int64_t rot_dim = cos_sin_cache.size(1);
  TORCH_CHECK(rot_dim % 2 == 0, "rot_dim must be even");
  TORCH_CHECK(query.size(2) == rot_dim && key.size(2) == rot_dim, "query/key last dim must equal rot_dim");
  TORCH_CHECK(key.size(0) == n && positions.numel() == n, "batch size mismatch");
  TORCH_CHECK(query.dtype() == key.dtype() && cos_sin_cache.dtype() == query.dtype(), "dtype mismatch");
  const int64_t hq = query.size(1), hk = key.size(1), half = rot_dim / 2;
  const int64_t max_pos = cos_sin_cache.size(0);

  LLM_DISPATCH_FLOAT(query.scalar_type(), "rotary_embedding", {
    T* q = query.data_ptr<T>();
    T* k = key.data_ptr<T>();
    const T* cache = cos_sin_cache.data_ptr<T>();
    at::parallel_for(0, n, 16, [&](int64_t b, int64_t e) {
      for (int64_t t = b; t < e; ++t) {
        const int64_t pos = read_index(positions, t);
        TORCH_CHECK(pos >= 0 && pos < max_pos, "position out of range: ", pos);
        const T* cos = cache + pos * rot_dim;
        const T* sin = cos + half;
        for (int64_t h = 0; h < hq; ++h) rotate(q + (t * hq + h) * rot_dim, cos, sin, half, is_neox);
        for (int64_t h = 0; h < hk; ++h) rotate(k + (t * hk + h) * rot_dim, cos, sin, half, is_neox);
      }
    });
  });
}

void concat_and_cache_mla(const torch::Tensor& kv_c, const torch::Tensor& k_pe, torch::Tensor& kv_cache,
                          const torch::Tensor& slot_mapping) {
  check_cpu_contig(kv_c, "kv_c");
  check_cpu_contig(k_pe, "k_pe");
  check_cpu_contig(kv_cache, "kv_cache");
  check_index_tensor(slot_mapping, "slot_mapping");
  const int64_t n = kv_c.size(0);
  const int64_t r = kv_c.size(1), p = k_pe.size(1);
  const int64_t block_size = kv_cache.size(1);
  const int64_t num_slots = kv_cache.size(0) * block_size;
  TORCH_CHECK(k_pe.size(0) == n && slot_mapping.numel() == n, "batch size mismatch");
  TORCH_CHECK(kv_cache.size(2) == r + p, "kv_cache last dim must be kv_lora_rank + rope_dim");
  TORCH_CHECK(kv_c.dtype() == kv_cache.dtype() && k_pe.dtype() == kv_cache.dtype(), "dtype mismatch");

  LLM_DISPATCH_FLOAT(kv_cache.scalar_type(), "concat_and_cache_mla", {
    const T* c = kv_c.data_ptr<T>();
    const T* pe = k_pe.data_ptr<T>();
    T* cache = kv_cache.data_ptr<T>();
    for (int64_t i = 0; i < n; ++i) {
      const int64_t slot = read_index(slot_mapping, i);
      if (slot < 0) continue;
      TORCH_CHECK(slot < num_slots, "slot out of range: ", slot);
      T* dst = cache + slot * (r + p);
      std::copy(c + i * r, c + (i + 1) * r, dst);
      std::copy(pe + i * p, pe + (i + 1) * p, dst + r);
    }
  });
}

torch::Tensor mla_prefill_attention(const torch::Tensor& q, const torch::Tensor& k, const torch::Tensor& v,
                                    const torch::Tensor& cu_seqlens, double scale) {
  check_cpu_contig(q, "q");
  check_cpu_contig(k, "k");
  check_cpu_contig(v, "v");
  check_index_tensor(cu_seqlens, "cu_seqlens");
  TORCH_CHECK(q.dim() == 3 && k.sizes() == q.sizes() && v.dim() == 3, "q/k must be [N, H, Dqk], v [N, H, Dv]");
  TORCH_CHECK(v.size(0) == q.size(0) && v.size(1) == q.size(1), "v shape mismatch");
  TORCH_CHECK(q.dtype() == k.dtype() && q.dtype() == v.dtype(), "dtype mismatch");
  const int64_t n = q.size(0), h = q.size(1), dqk = q.size(2), dv = v.size(2);
  const int64_t num_seqs = cu_seqlens.numel() - 1;
  TORCH_CHECK(num_seqs >= 1 && read_index(cu_seqlens, num_seqs) == n, "cu_seqlens[-1] must equal N");

  // 每个 token 所属序列的起始位置，便于按 token 并行。
  std::vector<int64_t> seq_start(n);
  for (int64_t s = 0; s < num_seqs; ++s) {
    const int64_t b = read_index(cu_seqlens, s), e = read_index(cu_seqlens, s + 1);
    TORCH_CHECK(b <= e, "cu_seqlens must be non-decreasing");
    for (int64_t t = b; t < e; ++t) seq_start[t] = b;
  }

  auto out = torch::zeros({n, h, dv}, q.options());
  LLM_DISPATCH_FLOAT(q.scalar_type(), "mla_prefill_attention", {
    const T* qp = q.data_ptr<T>();
    const T* kp = k.data_ptr<T>();
    const T* vp = v.data_ptr<T>();
    T* op = out.data_ptr<T>();
    const T sc = static_cast<T>(scale);
    at::parallel_for(0, n, 1, [&](int64_t b, int64_t e) {
      std::vector<T> scores;
      for (int64_t i = b; i < e; ++i) {
        const int64_t s0 = seq_start[i];
        const int64_t len = i - s0 + 1;  // 因果：只看 [s0, i]
        scores.resize(len);
        for (int64_t hh = 0; hh < h; ++hh) {
          const T* qi = qp + (i * h + hh) * dqk;
          for (int64_t j = 0; j < len; ++j) {
            const T* kj = kp + ((s0 + j) * h + hh) * dqk;
            T dot = 0;
            for (int64_t d = 0; d < dqk; ++d) dot += qi[d] * kj[d];
            scores[j] = dot * sc;
          }
          softmax_inplace(scores.data(), len);
          T* oi = op + (i * h + hh) * dv;
          for (int64_t j = 0; j < len; ++j) {
            const T* vj = vp + ((s0 + j) * h + hh) * dv;
            const T a = scores[j];
            for (int64_t d = 0; d < dv; ++d) oi[d] += a * vj[d];
          }
        }
      }
    });
  });
  return out;
}

torch::Tensor mla_decode_attention(const torch::Tensor& q, const torch::Tensor& kv_cache,
                                   const torch::Tensor& block_tables, const torch::Tensor& seq_lens,
                                   int64_t kv_lora_rank, double scale) {
  check_cpu_contig(q, "q");
  check_cpu_contig(kv_cache, "kv_cache");
  check_index_tensor(block_tables, "block_tables");
  check_index_tensor(seq_lens, "seq_lens");
  TORCH_CHECK(q.dim() == 3 && kv_cache.dim() == 3, "q must be [B, H, R+P], kv_cache [blocks, block_size, R+P]");
  TORCH_CHECK(q.dtype() == kv_cache.dtype(), "dtype mismatch");
  const int64_t bsz = q.size(0), h = q.size(1), dim = q.size(2);
  const int64_t r = kv_lora_rank;
  const int64_t num_blocks = kv_cache.size(0), block_size = kv_cache.size(1);
  const int64_t max_blocks = block_tables.size(1);
  TORCH_CHECK(kv_cache.size(2) == dim && r > 0 && r < dim, "latent dim mismatch");
  TORCH_CHECK(block_tables.dim() == 2 && block_tables.size(0) == bsz && seq_lens.numel() == bsz,
              "block_tables/seq_lens batch mismatch");

  auto out = torch::zeros({bsz, h, r}, q.options());
  LLM_DISPATCH_FLOAT(q.scalar_type(), "mla_decode_attention", {
    const T* qp = q.data_ptr<T>();
    const T* cache = kv_cache.data_ptr<T>();
    T* op = out.data_ptr<T>();
    const T sc = static_cast<T>(scale);
    at::parallel_for(0, bsz, 1, [&](int64_t b, int64_t e) {
      std::vector<T> scores;
      std::vector<const T*> rows;
      for (int64_t s = b; s < e; ++s) {
        const int64_t len = read_index(seq_lens, s);
        TORCH_CHECK(len > 0 && len <= max_blocks * block_size, "invalid seq_len: ", len);
        // 先把这条序列在 cache 里的每一行位置解析出来，所有 head 共用。
        rows.resize(len);
        for (int64_t t = 0; t < len; ++t) {
          const int64_t blk = read_index(block_tables, s * max_blocks + t / block_size);
          TORCH_CHECK(blk >= 0 && blk < num_blocks, "block id out of range: ", blk);
          rows[t] = cache + (blk * block_size + t % block_size) * dim;
        }
        scores.resize(len);
        for (int64_t hh = 0; hh < h; ++hh) {
          const T* qi = qp + (s * h + hh) * dim;
          // score = (W_kᵀ q_nope)·c + q_pe·k_pe，正好是 q 和 cache 行在 R+P 维上的点积
          for (int64_t t = 0; t < len; ++t) {
            const T* row = rows[t];
            T dot = 0;
            for (int64_t d = 0; d < dim; ++d) dot += qi[d] * row[d];
            scores[t] = dot * sc;
          }
          softmax_inplace(scores.data(), len);
          // value 在 latent 空间累加：Σ a_t · c_t，调用方再乘 W_v
          T* oi = op + (s * h + hh) * r;
          for (int64_t t = 0; t < len; ++t) {
            const T* row = rows[t];
            const T a = scores[t];
            for (int64_t d = 0; d < r; ++d) oi[d] += a * row[d];
          }
        }
      }
    });
  });
  return out;
}

TORCH_LIBRARY_IMPL(llm_infer, CPU, m) {
  m.impl("rotary_embedding", &rotary_embedding);
  m.impl("concat_and_cache_mla", &concat_and_cache_mla);
  m.impl("mla_prefill_attention", &mla_prefill_attention);
  m.impl("mla_decode_attention", &mla_decode_attention);
}
