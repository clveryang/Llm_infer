// RMSNorm、fused add + RMSNorm、SiLU-and-mul 的 CPU 实现。
#include <cmath>

#include "../ops.h"
#include "common.h"

namespace {

template <typename T>
inline void rms_norm_row(T* out, const T* in, const T* w, int64_t d, double eps) {
  double ss = 0.0;
  for (int64_t j = 0; j < d; ++j) ss += static_cast<double>(in[j]) * in[j];
  const T inv = static_cast<T>(1.0 / std::sqrt(ss / d + eps));
  for (int64_t j = 0; j < d; ++j) out[j] = in[j] * inv * w[j];
}

}  // namespace

torch::Tensor rms_norm(const torch::Tensor& input, const torch::Tensor& weight, double eps) {
  check_cpu_contig(input, "input");
  check_cpu_contig(weight, "weight");
  const int64_t d = input.size(-1);
  TORCH_CHECK(weight.numel() == d && weight.dtype() == input.dtype(), "weight shape/dtype mismatch");
  const int64_t n = input.numel() / d;
  auto out = torch::empty_like(input);

  LLM_DISPATCH_FLOAT(input.scalar_type(), "rms_norm", {
    const T* in = input.data_ptr<T>();
    const T* w = weight.data_ptr<T>();
    T* o = out.data_ptr<T>();
    at::parallel_for(0, n, 64, [&](int64_t b, int64_t e) {
      for (int64_t i = b; i < e; ++i) rms_norm_row(o + i * d, in + i * d, w, d, eps);
    });
  });
  return out;
}

void fused_add_rms_norm(torch::Tensor& input, torch::Tensor& residual, const torch::Tensor& weight,
                        double eps) {
  check_cpu_contig(input, "input");
  check_cpu_contig(residual, "residual");
  check_cpu_contig(weight, "weight");
  TORCH_CHECK(input.sizes() == residual.sizes() && input.dtype() == residual.dtype(), "input/residual mismatch");
  const int64_t d = input.size(-1);
  TORCH_CHECK(weight.numel() == d && weight.dtype() == input.dtype(), "weight shape/dtype mismatch");
  const int64_t n = input.numel() / d;

  LLM_DISPATCH_FLOAT(input.scalar_type(), "fused_add_rms_norm", {
    T* x = input.data_ptr<T>();
    T* r = residual.data_ptr<T>();
    const T* w = weight.data_ptr<T>();
    at::parallel_for(0, n, 64, [&](int64_t b, int64_t e) {
      for (int64_t i = b; i < e; ++i) {
        T* xi = x + i * d;
        T* ri = r + i * d;
        for (int64_t j = 0; j < d; ++j) ri[j] += xi[j];
        rms_norm_row(xi, ri, w, d, eps);
      }
    });
  });
}

torch::Tensor silu_and_mul(const torch::Tensor& input) {
  check_cpu_contig(input, "input");
  const int64_t d2 = input.size(-1);
  TORCH_CHECK(d2 % 2 == 0, "silu_and_mul: last dim must be even");
  const int64_t d = d2 / 2;
  const int64_t n = input.numel() / d2;
  auto out_sizes = input.sizes().vec();
  out_sizes.back() = d;
  auto out = torch::empty(out_sizes, input.options());

  LLM_DISPATCH_FLOAT(input.scalar_type(), "silu_and_mul", {
    const T* in = input.data_ptr<T>();
    T* o = out.data_ptr<T>();
    at::parallel_for(0, n, 16, [&](int64_t b, int64_t e) {
      for (int64_t i = b; i < e; ++i) {
        const T* gate = in + i * d2;
        const T* up = gate + d;
        T* oi = o + i * d;
        for (int64_t j = 0; j < d; ++j) {
          const T g = gate[j];
          const T sig = g >= 0 ? T(1) / (T(1) + std::exp(-g)) : std::exp(g) / (T(1) + std::exp(g));
          oi[j] = g * sig * up[j];
        }
      }
    });
  });
  return out;
}

TORCH_LIBRARY_IMPL(llm_infer, CPU, m) {
  m.impl("rms_norm", &rms_norm);
  m.impl("fused_add_rms_norm", &fused_add_rms_norm);
  m.impl("silu_and_mul", &silu_and_mul);
}
