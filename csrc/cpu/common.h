#pragma once

#include <ATen/Parallel.h>
#include <torch/torch.h>

// CPU 内核支持 float32 / float64。bf16/fp16 在 CPU 上没有原生算术类型，
// Python 端遇到这些 dtype 会回退到 PyTorch 参考实现。
#define LLM_DISPATCH_FLOAT(dtype, name, body)                              \
  switch (dtype) {                                                         \
    case torch::kFloat32: {                                                \
      using T = float;                                                     \
      body;                                                                \
      break;                                                               \
    }                                                                      \
    case torch::kFloat64: {                                                \
      using T = double;                                                    \
      body;                                                                \
      break;                                                               \
    }                                                                      \
    default:                                                               \
      TORCH_CHECK(false, name, ": CPU kernel only supports float32/float64, got ", dtype); \
  }

// 读取整型张量（int32 或 int64）的第 i 个元素。
inline int64_t read_index(const torch::Tensor& t, int64_t i) {
  return t.dtype() == torch::kInt32 ? static_cast<int64_t>(t.data_ptr<int32_t>()[i]) : t.data_ptr<int64_t>()[i];
}

inline void check_cpu_contig(const torch::Tensor& t, const char* name) {
  TORCH_CHECK(t.device().is_cpu(), name, " must be a CPU tensor");
  TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
}

inline void check_index_tensor(const torch::Tensor& t, const char* name) {
  check_cpu_contig(t, name);
  TORCH_CHECK(t.dtype() == torch::kInt32 || t.dtype() == torch::kInt64, name, " must be int32 or int64");
}
