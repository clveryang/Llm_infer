// RMSNorm、fused add + RMSNorm、SiLU-and-mul 的 CUDA 实现。
// 每个 GPU 线程处理一个 token（一行），行内按 hidden 维度循环。
#include "common.cuh"

namespace llm_cuda {
namespace {

template <typename T, typename CT>
__global__ void rms_norm_fwd(T* out, const T* in, const T* w, const int64_t n, const int64_t d, const CT eps) {
  const int64_t i = LLM_CUDA_INDEX();
  if (i >= n) return;
  const T* x = in + i * d;
  CT ss = 0;
  for (int64_t j = 0; j < d; ++j) {
    const CT v = static_cast<CT>(x[j]);
    ss += v * v;
  }
  const CT inv = CT(1) / std::sqrt(ss / static_cast<CT>(d) + eps);
  T* o = out + i * d;
  for (int64_t j = 0; j < d; ++j) {
    o[j] = static_cast<T>(static_cast<CT>(x[j]) * inv * static_cast<CT>(w[j]));
  }
}

template <typename T, typename CT>
__global__ void fused_add_rms_norm_fwd(T* x, T* res, const T* w, const int64_t n, const int64_t d,
                                         const CT eps) {
  const int64_t i = LLM_CUDA_INDEX();
  if (i >= n) return;
  T* xi = x + i * d;
  T* ri = res + i * d;
  CT ss = 0;
  for (int64_t j = 0; j < d; ++j) {
    ri[j] = static_cast<T>(static_cast<CT>(xi[j]) + static_cast<CT>(ri[j]));
    const CT v = static_cast<CT>(ri[j]);
    ss += v * v;
  }
  const CT inv = CT(1) / std::sqrt(ss / static_cast<CT>(d) + eps);
  for (int64_t j = 0; j < d; ++j) {
    xi[j] = static_cast<T>(static_cast<CT>(ri[j]) * inv * static_cast<CT>(w[j]));
  }
}

template <typename T, typename CT>
__global__ void silu_and_mul_fwd(T* out, const T* in, const int64_t n, const int64_t d) {
  const int64_t i = LLM_CUDA_INDEX();
  if (i >= n) return;
  const T* gate = in + i * 2 * d;
  const T* up = gate + d;
  T* o = out + i * d;
  for (int64_t j = 0; j < d; ++j) {
    const CT g = static_cast<CT>(gate[j]);
    const CT sig = g >= 0 ? CT(1) / (CT(1) + std::exp(-g)) : std::exp(g) / (CT(1) + std::exp(g));
    o[j] = static_cast<T>(g * sig * static_cast<CT>(up[j]));
  }
}

}  // namespace

torch::Tensor rms_norm(const torch::Tensor& input, const torch::Tensor& weight, double eps) {
  check_cuda(input, "input");
  check_cuda(weight, "weight");
  const int64_t d = input.size(-1);
  TORCH_CHECK(weight.numel() == d && weight.dtype() == input.dtype(), "weight shape/dtype mismatch");
  const int64_t n = input.numel() / d;
  auto out = torch::empty_like(input);
  if (n == 0) return out;

  const c10::cuda::CUDAGuard guard(input.device());
  const auto stream = at::cuda::getCurrentCUDAStream();
  LLM_CUDA_DISPATCH(input.scalar_type(), "rms_norm",
                    rms_norm_fwd<T, CT><<<grid_for(n), block_dim(), 0, stream.stream()>>>(
                        out.data_ptr<T>(), input.data_ptr<T>(), weight.data_ptr<T>(), n, d, static_cast<CT>(eps)));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

void fused_add_rms_norm(torch::Tensor& input, torch::Tensor& residual, const torch::Tensor& weight,
                        double eps) {
  check_cuda(input, "input");
  check_cuda(residual, "residual");
  check_cuda(weight, "weight");
  TORCH_CHECK(input.sizes() == residual.sizes() && input.dtype() == residual.dtype(), "input/residual mismatch");
  const int64_t d = input.size(-1);
  TORCH_CHECK(weight.numel() == d && weight.dtype() == input.dtype(), "weight shape/dtype mismatch");
  const int64_t n = input.numel() / d;
  if (n == 0) return;

  const c10::cuda::CUDAGuard guard(input.device());
  const auto stream = at::cuda::getCurrentCUDAStream();
  LLM_CUDA_DISPATCH(input.scalar_type(), "fused_add_rms_norm",
                    fused_add_rms_norm_fwd<T, CT><<<grid_for(n), block_dim(), 0, stream.stream()>>>(
                        input.data_ptr<T>(), residual.data_ptr<T>(), weight.data_ptr<T>(), n, d,
                        static_cast<CT>(eps)));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

torch::Tensor silu_and_mul(const torch::Tensor& input) {
  check_cuda(input, "input");
  const int64_t d2 = input.size(-1);
  TORCH_CHECK(d2 % 2 == 0, "silu_and_mul: last dim must be even");
  const int64_t d = d2 / 2;
  const int64_t n = input.numel() / d2;
  auto out_sizes = input.sizes().vec();
  out_sizes.back() = d;
  auto out = torch::empty(out_sizes, input.options());
  if (n == 0) return out;

  const c10::cuda::CUDAGuard guard(input.device());
  const auto stream = at::cuda::getCurrentCUDAStream();
  LLM_CUDA_DISPATCH(input.scalar_type(), "silu_and_mul",
                    silu_and_mul_fwd<T, CT><<<grid_for(n), block_dim(), 0, stream.stream()>>>(
                        out.data_ptr<T>(), input.data_ptr<T>(), n, d));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

}  // namespace llm_cuda

TORCH_LIBRARY_IMPL(llm_infer, CUDA, m) {
  m.impl("rms_norm", &llm_cuda::rms_norm);
  m.impl("fused_add_rms_norm", &llm_cuda::fused_add_rms_norm);
  m.impl("silu_and_mul", &llm_cuda::silu_and_mul);
}
