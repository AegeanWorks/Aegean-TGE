# Roadmap

Milestones follow the project specification. A milestone is done only when its
deliverable is compiled, tested and measured on the target machine.

## 0.1 — Native CPU foundation (in progress)

Delivered:

- [x] CMake + Ninja build, GCC 16.2, static executable
- [x] CPUID detection with XCR0 verification
- [x] Windows topology: P/E core classification, cache hierarchy, NUMA
- [x] Scalar reference tensor operations
- [x] AVX2 + FMA GEMM with runtime dispatch
- [x] Numerical correctness tests against the reference
- [x] System RAM detection
- [x] Memory budget manager with pool limits
- [x] Benchmark harness (`tge bench`)
- [x] Hardware audit report (`tge info`)
- [x] Packing and K blocking in the AVX2 GEMM, measured before and after
- [x] Multithreaded GEMM with dynamic work claiming across P and E cores
- [x] Memory bandwidth benchmark (`tge bench mem`), single and multi-threaded
- [x] llama.cpp CPU baseline on identical hardware
- [x] GEMV decode kernel with measured thread scaling (`tge bench gemv`)
- [x] Fixed unguarded AVX2 calls in the bandwidth benchmark

Remaining for 0.1:

- [ ] AVX-VNNI quantized dot product tier

## 0.2 — Model loading and CPU inference

- [x] GGUF header, metadata and tensor-info parsing, memory mapped
- [x] Q8_0 quantized matrix-vector kernel with runtime dispatch
- [x] Qwen2 decoder: RMSNorm, NeoX RoPE, GQA attention, SwiGLU, KV cache
- [x] Greedy decode loop validated against llama.cpp token for token (pre-vectorisation; see log section 19 for near-ties)
- [x] End-to-end decode at parity with llama.cpp 11624 at short context (45.5 vs 45.1 t/s); behind at 2k (38.3 vs 46.9)
- [x] Batched prompt processing (261-266 t/s against llama.cpp 11624's 352.2 at 512)
- [x] Sampling: temperature, top-k, top-p, min-p, repetition penalty, seed
- [x] Nucleus draw over the kept mass; penalty once per distinct token
- [x] Top-p and min-p without a full-vocabulary sort
- [x] Reference pre-tokenizer splits; USER_DEFINED tokens; opt-out for user content
- [x] Stop on the end-of-generation set (eos, eot, eos_token_ids, Qwen end-of-text)
- [x] RoPE sin/cos precomputed per position
- [x] Spin-then-sleep fork-join pool (handoff 39 us -> 0.5 us)
- [x] Parallel attention across query heads and prefill chunk tokens
- [x] Vectorised attention inner loops (AVX2 dot and value sum)
- [x] Teacher-forced fidelity harness (`tge fidelity`, KL and top-1)
- [x] GQA-aware decode attention over positions and KV groups
- [x] Quantized formats: Q8_0, Q5_0, Q4_0, Q4_K, Q5_K, Q6_K
- [x] Detokenizer: token ids to text
- [x] Tokenizer encoding: text to ids via BPE merges, validated against llama.cpp
- [x] Context selectable up to 32768, default 4096
- [x] Integer-domain Q8 x Q8 inner loop — measured, no gain, reverted (ADR-0006)
- [ ] Weight packing and register blocking across output rows for prefill
- [ ] Fuse QKV and gate/up into single parallel regions to cut per-token barriers

## 0.3 — System RAM and memory manager

- Memory-mapped GGUF weights (`CreateFileMapping` / `MapViewOfFile`)
- Weight prepacking and shared immutable weights between sessions
- Resident-set monitoring so a mapped model does not page-fault in the hot path
- KV-cache manager with paged allocation and eviction
- Memory pressure monitoring and admission control

## 0.4 — AMD GPU inference

Blocked until an MSVC Build Tools install and a HIP SDK with `gfx1200` support
are verified. Not started, and not claimed.

## 0.5 — Hybrid RAM/VRAM engine

Layer-based placement, transfer scheduling, pinned host buffers, memory-aware
planner. Depends on 0.4.

## 0.6 — Vulkan backend

Requires the Vulkan SDK (headers and `glslc`); the loader and AMD ICD are
already present.

## 0.7 — Advanced inference

Continuous batching, concurrent sessions, speculative decoding, adaptive
planning.

## 0.8 — Windows inference platform

OpenAI-compatible server, streaming, CLI, Windows service, packaging.

## Constraints that shape the order

The target machine has 16 GiB RAM in a single DIMM and only ~2 GiB free during
the audit. Memory-constrained and hybrid modes are therefore the ones that
decide whether this engine is useful on its own development hardware, and RAM
management (0.3) cannot be deferred behind GPU work (0.4).
