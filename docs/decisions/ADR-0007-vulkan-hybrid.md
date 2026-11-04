# ADR-0007: Vulkan hybrid backend (portable CPU+GPU)

- Status: accepted (bring-up [1/4] landed; offload dispatch follows)
- Date: 2026-11-03

## Context

Decode is DRAM-bound on this box (Q8_0 GEMV at ~22 GB/s against a ~28 GB/s
wall; 45 t/s short-context is ~84% of the 53.3 t/s ceiling) while prefill is
compute-bound (CPU batch ~350-400 GFLOPS vs GPU FP32 ~25.6 TFLOPS on the
RX 9060 XT). The GPU needs a backend that runs on the installed AMD card:
CUDA is a failure (NVIDIA-only), HIP needs MSVC + an unverified `gfx1200`
SDK, OpenCL is deprecated with no measured INT8-accelerated path here.
`vulkaninfo` reports Vulkan 1.4.344, an AMD discrete GPU, heap0 ~15.9 GiB
host-visible, heap1 ~15.9 GiB `DEVICE_LOCAL`, and
`integerDotProduct8Bit*Accelerated=true` with cooperative-matrix subgroups
32-64. The loader + AMD ICD exist; the SDK (headers, `glslc`) does not.

## Decision

1. **Vulkan Compute 1.3** (`VK_KHR_cooperative_matrix`, `shaderInt8`,
   `dotProduct`). Three shaders only: `q8_gemm_comp` (prefill/verify/batch),
   `q8_gemv_comp` (decode output region), `attn_gqa_comp` (long-decode GQA).
   Norms/RoPE/SiLU ride as elementwise twins of the AVX2 paths, fidelity-
   priced like them.
2. **Static thresholds, not autotuning**: batch offload at `tokens >= 8`
   (so the 6-token speculative verifier keeps the pinned per-head
   arithmetic), GQA offload at `span >= 256` with `threads > 1` (mirrors
   `uses_gqa()`), decode output norm+projection as one single-sync GPU
   region (H2D 3.5 KiB + D2H 608 KiB = 611328 bytes at 896/151936, saving
   4.5-7 ms of 22.2 ms/token).
3. **Exactly one fallback**: `use_gpu()` false (no SDK build, `vkInit`
   fail, VRAM check fail, `--gpu off`, scalar/head/threads<=1/K-quant/
   VNNI-int8 paths) lands on the same `cpuDispatch` label. `--gpu on`
   hard-fails instead of silently benchmarking the CPU.
4. **mmap-first preserved**: `gguf::File` stays the sole weight owner;
   the GPU holds one derived device-local copy uploaded once
   (`rows x row_bytes` per slice, never `info.bytes` for fused views).
   KV stays CPU-authoritative with one gathered H2D delta per token.
5. **SDK-optional build**: `TGE_ENABLE_VULKAN` + `find_package(Vulkan
   QUIET)`; without it `TGE_HAS_VULKAN=0`, `noop.cpp` reports
   Unavailable, everything runs CPU-only. No link dependency without SDK.

## Consequences

- Bring-up ([1/4]) is build + detect + fallback only: no numerics move,
  no fidelity price owed, `tge info` unchanged.
- Each offloaded op ([2/4]+) carries its transfer accounting beside the
  speedup claim in `docs/measurement-log.md`; a faster hybrid that
  answers differently is reverted, not kept.
- Null path kept: if submit latency exceeds ~40 us, GQA offload is
  dropped and the hybrid is prefill-GPU + output-GPU + CPU-everything-
  else, which still wins prefill ~50x and short decode ~15-25%.
