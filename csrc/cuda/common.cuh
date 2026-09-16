#pragma once

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/torch.h>

#include <cmath>

// CUDA 内核支持 float32 / float16 / bfloat16 / float64。
//   T  ：张量里存的类型（c10::Half / c10::BFloat16 可以和 float 互相转换，主机和设备上都能用）
//   CT ：实际做算术的类型。半精度统一提升到 float32 计算，写回时再转成 T。
// 用可变参数宏，因为 body 里会出现 kernel<T, CT> 这种顶层逗号。
#define LLM_CUDA_DISPATCH(dtype, name, ...)                                                  \
  switch (dtype) {                                                                          \
    case torch::kFloat32: {                                                                 \
      using T = float;                                                                      \
      using CT = float;                                                                     \
      __VA_ARGS__;                                                                          \
      break;                                                                                \
    }                                                                                       \
    case torch::kFloat16: {                                                                 \
      using T = c10::Half;                                                                  \
      using CT = float;                                                                     \
      __VA_ARGS__;                                                                          \
      break;                                                                                \
    }                                                                                       \
    case torch::kBFloat16: {                                                                \
      using T = c10::BFloat16;                                                              \
      using CT = float;                                                                     \
      __VA_ARGS__;                                                                          \
      break;                                                                                \
    }                                                                                       \
    case torch::kFloat64: {                                                                 \
      using T = double;                                                                     \
      using CT = double;                                                                    \
      __VA_ARGS__;                                                                          \
      break;                                                                                \
    }                                                                                       \
    default:                                                                                \
      TORCH_CHECK(false, name, ": CUDA kernel does not support dtype ", dtype);             \
  }

namespace llm_cuda {

constexpr int kThreadsPerBlock = 256;

inline dim3 grid_for(int64_t n) {
  return dim3(static_cast<unsigned int>((n + kThreadsPerBlock - 1) / kThreadsPerBlock));
}

inline dim3 block_dim() { return dim3(kThreadsPerBlock); }

inline void check_cuda(const torch::Tensor& t, const char* name) {
  TORCH_CHECK(t.is_cuda(), name, " must be a CUDA tensor");
  TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
}

// 索引张量统一转成 int64 + contiguous（已经满足时不拷贝）。
inline torch::Tensor as_index(const torch::Tensor& t, const char* name) {
  TORCH_CHECK(t.is_cuda(), name, " must be a CUDA tensor");
  TORCH_CHECK(t.dtype() == torch::kInt32 || t.dtype() == torch::kInt64, name, " must be int32 or int64");
  return t.to(torch::kInt64).contiguous();
}

// 当前线程在一维 launch 里的全局编号。
#define LLM_CUDA_INDEX() (static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x)

// 数值稳定的 log(exp(a) + exp(b))，用 double 累加，长序列也不会积累误差。
inline C10_HOST_DEVICE double log_add_exp(double a, double b) {
  if (a < b) {
    const double t = a;
    a = b;
    b = t;
  }
  if (b == -INFINITY) return a;
  return a + std::log1p(std::exp(b - a));
}

}  // namespace llm_cuda
