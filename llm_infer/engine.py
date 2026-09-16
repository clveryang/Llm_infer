"""最小推理引擎：分页 MLA KV cache + 批量 prefill / decode + 贪心或温度采样。"""

from dataclasses import dataclass, field

import torch

from .model import AttnMetadata, DeepseekV2ForCausalLM


class PagedKVCache:
    """每层一个 [num_blocks, block_size, kv_lora_rank + rope_dim] 的张量，块在各层间共用同一套编号。"""

    def __init__(self, num_layers, num_blocks, block_size, latent_dim, dtype, device):
        self.block_size = block_size
        self.caches = [
            torch.zeros(num_blocks, block_size, latent_dim, dtype=dtype, device=device) for _ in range(num_layers)
        ]
        self._free = list(range(num_blocks - 1, -1, -1))

    def allocate(self) -> int:
        if not self._free:
            raise RuntimeError("KV cache 块用完了，调大 num_blocks")
        return self._free.pop()

    def release(self, blocks: list[int]) -> None:
        self._free.extend(blocks)

    @staticmethod
    def bytes_per_token(num_layers: int, latent_dim: int, dtype: torch.dtype) -> int:
        return num_layers * latent_dim * torch.empty(0, dtype=dtype).element_size()


@dataclass
class Sequence:
    token_ids: list[int]
    prompt_len: int
    block_ids: list[int] = field(default_factory=list)
    finished: bool = False

    @property
    def output_ids(self) -> list[int]:
        return self.token_ids[self.prompt_len :]


class Engine:
    def __init__(self, model: DeepseekV2ForCausalLM, num_blocks: int = 1024, block_size: int = 16):
        cfg = model.config
        w = model.lm_head.weight
        self.model = model
        self.device = w.device
        self.block_size = block_size
        self.kv = PagedKVCache(
            cfg.num_hidden_layers, num_blocks, block_size, cfg.kv_lora_rank + cfg.qk_rope_head_dim, w.dtype, w.device
        )

    def _slot(self, seq: Sequence, pos: int) -> int:
        while len(seq.block_ids) * self.block_size <= pos:
            seq.block_ids.append(self.kv.allocate())
        return seq.block_ids[pos // self.block_size] * self.block_size + pos % self.block_size

    def _tensor(self, data) -> torch.Tensor:
        return torch.tensor(data, dtype=torch.long, device=self.device)

    @torch.inference_mode()
    def prefill(self, seqs: list[Sequence], all_logits: bool = False) -> torch.Tensor:
        """对整段 prompt 做一次前向。默认只返回每条序列最后位置的 logits [B, V]。"""
        tokens, positions, slots, cu = [], [], [], [0]
        for s in seqs:
            n = len(s.token_ids)
            tokens += s.token_ids
            positions += range(n)
            slots += [self._slot(s, p) for p in range(n)]
            cu.append(cu[-1] + n)
        meta = AttnMetadata(is_prefill=True, slot_mapping=self._tensor(slots), cu_seqlens=self._tensor(cu))
        hidden = self.model(self._tensor(tokens), self._tensor(positions), meta, self.kv.caches)
        if not all_logits:
            hidden = hidden[self._tensor(cu[1:]) - 1]
        return self.model.compute_logits(hidden)

    @torch.inference_mode()
    def decode(self, seqs: list[Sequence]) -> torch.Tensor:
        """每条序列的最后一个 token 还没进 cache，本步把它写进去并返回下一个 token 的 logits [B, V]。"""
        tokens, positions, slots, lens = [], [], [], []
        for s in seqs:
            pos = len(s.token_ids) - 1
            tokens.append(s.token_ids[-1])
            positions.append(pos)
            slots.append(self._slot(s, pos))
            lens.append(pos + 1)
        max_blocks = max(len(s.block_ids) for s in seqs)
        tables = [s.block_ids + [0] * (max_blocks - len(s.block_ids)) for s in seqs]
        meta = AttnMetadata(
            is_prefill=False,
            slot_mapping=self._tensor(slots),
            block_tables=self._tensor(tables),
            seq_lens=self._tensor(lens),
        )
        hidden = self.model(self._tensor(tokens), self._tensor(positions), meta, self.kv.caches)
        return self.model.compute_logits(hidden)

    @staticmethod
    def _sample(logits: torch.Tensor, temperature: float) -> list[int]:
        if temperature <= 0:
            return logits.argmax(dim=-1).tolist()
        probs = torch.softmax(logits / temperature, dim=-1)
        return torch.multinomial(probs, 1).squeeze(-1).tolist()

    def generate(
        self,
        prompts: list[list[int]],
        max_new_tokens: int = 64,
        temperature: float = 0.0,
        eos_token_id: int | None = None,
    ) -> list[list[int]]:
        seqs = [Sequence(list(p), len(p)) for p in prompts]
        logits = self.prefill(seqs)
        for step in range(max_new_tokens):
            active = [s for s in seqs if not s.finished]
            for s, tok in zip(active, self._sample(logits, temperature)):
                s.token_ids.append(tok)
                if tok == eos_token_id or step == max_new_tokens - 1:
                    s.finished = True
                    self.kv.release(s.block_ids)
                    s.block_ids = []
            active = [s for s in seqs if not s.finished]
            if not active:
                break
            logits = self.decode(active)
        return [s.output_ids for s in seqs]
