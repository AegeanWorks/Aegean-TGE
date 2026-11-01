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

- GGUF header, metadata and tensor-info parsing
- Tokenizer integration
- Quantized tensor loading (Q4, Q5, Q6, Q8, F16)
- Transformer execution for Llama-compatible decoders
- KV cache
- Autoregressive decoding with greedy, temperature, top-k, top-p, min-p
- CPU memory pools backed by the budget manager

Deliverable: independent CPU inference engine, measured in tokens/sec.

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
