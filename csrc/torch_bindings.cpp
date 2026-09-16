// 算子 schema 定义。Python 端 import llm_infer._C 之后即可通过 torch.ops.llm_infer.xxx 调用。
// 新增设备实现（如 CUDA）时只需在对应文件里 TORCH_LIBRARY_IMPL(llm_infer, CUDA, m)。
#include <torch/extension.h>

TORCH_LIBRARY(llm_infer, m) {
  // 通用
  m.def("rms_norm(Tensor input, Tensor weight, float eps) -> Tensor");
  m.def("fused_add_rms_norm(Tensor(a!) input, Tensor(b!) residual, Tensor weight, float eps) -> ()");
  m.def("silu_and_mul(Tensor input) -> Tensor");

  // MLA
  m.def("rotary_embedding(Tensor positions, Tensor(a!) query, Tensor(b!) key, Tensor cos_sin_cache, "
        "bool is_neox) -> ()");
  m.def("concat_and_cache_mla(Tensor kv_c, Tensor k_pe, Tensor(a!) kv_cache, Tensor slot_mapping) -> ()");
  m.def("mla_prefill_attention(Tensor q, Tensor k, Tensor v, Tensor cu_seqlens, float scale) -> Tensor");
  m.def("mla_decode_attention(Tensor q, Tensor kv_cache, Tensor block_tables, Tensor seq_lens, "
        "int kv_lora_rank, float scale) -> Tensor");

  // MoE
  m.def("topk_softmax(Tensor gating_logits, int topk, bool renormalize) -> (Tensor, Tensor)");
  m.def("moe_align(Tensor topk_ids, int num_experts) -> (Tensor, Tensor)");
  m.def("moe_expert_forward(Tensor hidden_sorted, Tensor expert_offsets, Tensor w_gate_up, "
        "Tensor w_down) -> Tensor");
  m.def("moe_combine(Tensor expert_out, Tensor sorted_idx, Tensor topk_weights, int num_tokens) -> Tensor");
}

// 空的 Python 模块，只为了让 `import llm_infer._C` 触发上面的静态注册。
PYBIND11_MODULE(_C, m) {}
