# ADR-0004: Decode is GEMV, and it is a bandwidth problem

- Status: accepted
- Date: 2026-11-01

## Context

The engine needs to be fast at token generation, not just at prompt processing.
Those are different operations:

- Prefill pushes a whole prompt through the model at once: a matrix-matrix
  product with a large M, which is arithmetic-bound and benefits from packing,
  blocking and register tiling.
- Decode pushes **one** activation vector through the model per token: a
  matrix-vector product, M = 1, which reads every weight exactly once and reuses
  each of them zero times within the call.

Using the GEMM for decode wastes five sixths of a 6x16 register tile and streams
the same data through a loop structure designed for reuse that cannot happen.

Measurement on this machine settled what decode is actually limited by:

- It is not arithmetic. Decode does 2 floating-point operations per weight byte,
  far below the ~845 GFLOPS peak of the six P-cores.
- It is memory bandwidth. The measured ceiling is 25-29 GB/s for a threaded read
  and 28-35 GB/s for the GEMV kernel itself.
- Single-threaded bandwidth is only 13-18 GB/s, because one core cannot sustain
  enough outstanding misses to saturate DRAM.

## Decision

1. Maintain a dedicated GEMV kernel as the decode primitive, separate from GEMM.
2. Thread it over **contiguous row bands**, not interleaved rows, so each worker
   streams sequentially and the TLB and hardware prefetcher both behave.
3. Do not add software prefetch. It was measured with and without and lost on the
   median; the hardware prefetcher already covers a contiguous band.
4. Keep the kernel as simple as the memory system allows. Once it is at the
   bandwidth wall, further instruction-level work is wasted effort.

## Consequences

- Sustained decode bandwidth goes from 18.4 GB/s on one thread to 35.4 GB/s on
  sixteen for a 64 MiB matrix. Threading is the largest single lever available.
- Larger streams lose bandwidth (35.4 GB/s at 64 MiB down to 28.7 GB/s at
  256 MiB) from TLB pressure. Large pages would attack this and are blocked on a
  privilege this account does not have.
- The implied ceiling for a 1 GiB model is 27-33 tokens/s. The measured llama.cpp
  baseline is 7.22 tokens/s on a 1.04 GiB model, so real engines run at roughly a
  quarter of available bandwidth.
- This redirects the optimization roadmap: faster GEMM cannot help decode.
  Fewer bytes per weight (quantization), fewer bytes per token (AVX-VNNI), and
  better page behaviour are what move the number.
