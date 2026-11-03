# ADR-0008: Static sparse attention subset (opt-in)

- Status: accepted (opt-in, default off; log §50)
- Date: 2026-11-03

## Context

Latent Sparse Attention (arXiv 2610.42147, §§3.1–3.2) selects a kept set
per query — sink + selected long-range + window — and attends only those.
The paper's indexer (Ward clustering over a distilled latent space, GPU
kernels, 10M context) cannot land here: no trainer, no CUDA path, 32k hard
cap, mmap-first weights (rejections in log §50 with swarm receipts).
The transferable residue is the kept-set geometry, with stride sampling
standing in for scored buckets.

## Decision

1. `SparseConfig{enabled,sink=4,window=512,stride=4}` on `Qwen2`
   (`qwen2.hpp`); subset gathered in `attention_head` for f32 KV only into
   thread-local scratch, then the unchanged scalar/AVX2 kernel.
2. Precedence `uses_sparse` before `uses_gqa` in `attention()`; every other
   condition (short span, half/int8 KV, Scalar oracle) keeps the exact,
   bit-identical dense path.
3. CLI `--sparse [--sparse-window N] [--sparse-stride N]` on `generate`
   and `fidelity` only; default off; `--sparse` with f16/i8 KV is an error,
   not a silent fallback.
4. Priced like the VNNI/i8 precedents: full-scalar fidelity (mean KL +
   top-1) plus identical-shape generate A/B at 512 and 8192 ctx.

## Consequences

- Kept opt-in on log §50 numbers: prefill attn phase consistently lower at
  7k ctx with identical free-running ids; 48/1023 top-1 flips on natural
  text at ctx 1024, so natural-text long context stays dense by default.
- Scored (content-aware) selection is future work, admitted only with the
  same fidelity + A/B pricing; tree/beam search stays out (single-stream
  decode is flat per paper §4.1).
- Shared-file protocol: `qwen2.*`/`main.cpp` changes are additive
  (new struct, setter, branches, flags); the Vulkan bring-up's dispatch
  hooks compose by checking their conditions independently.
