# Architecture

Aegean TGE is a native inference engine. It owns model loading, memory placement
and kernel execution; it does not wrap another runtime.

This document describes what exists today. Planned components are listed in
[roadmap.md](roadmap.md) and are not described here as if they were built.

## Layering

```
apps/cli            tge info | tge bench        (no inference yet)
      |
engine core         detection, dispatch, budgets
      |
tensor + kernels    reference ops, AVX2 GEMM
      |
memory              system RAM, aligned allocation, budget pools
```

The dependency direction is one-way. Kernels never read hardware state
directly; they receive it from the dispatch layer, which makes them testable on
any machine.

## Components as built (v0.1.0)

### `tge::cpu` — capability detection

`src/core/cpu.cpp` reads CPUID directly and cross-checks the result against the
operating system before reporting a feature.

- Feature bits come from leaf 1 and leaf 7 (including the leaf 7.1 vector
  extensions such as AVX-VNNI and AMX).
- A vector feature is reported true only if XCR0 has the corresponding state
  enabled. Reporting AVX2 without checking XCR0 is the classic way to crash a
  process on a hypervisor that masks the upper register file.
- Topology comes from `GetLogicalProcessorInformationEx`, not from CPUID counts,
  because Windows is the authority on which logical processors this process may
  actually run on.
- Efficiency classes distinguish performance and efficiency cores on hybrid
  parts. The development machine reports 6 P-cores and 4 E-cores, which matters:
  a worker pool that treats all 10 cores as equivalent will over-subscribe the
  E-cores and lose aggregate throughput.

### `tge` — kernel dispatch

`select_kernels` maps a feature set to concrete function pointers exactly once.
There is no fallback path that silently downgrades at call time: if AVX2 is
absent the reference kernel is chosen up front and reported by `tge info`.

Every ISA-specific translation unit is compiled with its own flags
(`tge_set_isa_flags` in CMake). The rest of the engine is baseline code, so the
binary starts on any x86-64 machine.

### `tge::ops` — reference tensor operations

`gemm_ref`, `add_inplace`, `scale_inplace`, `dot`, `max_abs_diff`.

The reference GEMM is written in `i, p, j` order so the innermost loop streams
contiguous memory. It is the correctness oracle for every optimized kernel and
is deliberately not micro-optimized.

### `cpu::gemm_avx2` — packed, blocked, threaded

Layout: six rows by sixteen columns per register tile. Twelve YMM accumulators
plus two operand registers fit the sixteen architectural YMM registers without
spilling.

Loop structure, outermost first: column block (NC=256) → depth block (KC=256) →
row block (MC=144) → column tile → row tile. A and B are packed into aligned
panels per block, so every operand load inside the microkernel is contiguous.

Three deliberate choices:

1. **The first depth pass stores, later passes accumulate.** C is written by the
   first pass over every element, so the initial clear is unnecessary and the
   first pass can store instead of read-modify-write. Later depth passes read
   and add.
2. **Packed panels are 64-byte aligned**, allocated through the engine's own
   `AlignedBuffer`, so B-panel loads never split a cache line.
3. **A panels are not padded.** Padding the 6-float stride to 8 was measured and
   was slower; see `docs/benchmarks.md` section 2.

Threading: the row-block loop is the parallel dimension, because each block
writes a disjoint set of C rows. Work is claimed dynamically, so a fast P-core
takes more row blocks than an E-core instead of everyone waiting for the slowest
worker.

Edges (M not a multiple of 6, N not a multiple of 16) are handled by storing the
tile to a stack buffer and copying the valid subrange, not by masked vector
code. Partial K and partial panels are handled by zero padding at pack time.

Known limitations, in the order they should be fixed:

1. No prefetching.
2. No AVX-VNNI or AVX-512 tier.
3. Packing allocates its panels on every call; a per-thread scratch arena would
   remove that.

### `core::ThreadPool` — work-stealing pool

Fixed-size worker pool with a blocking `parallel_for(count, grain, body)`. Work
is claimed through an atomic counter rather than partitioned up front.

The reason is the target machine: an i5-14400F has six P-cores and four E-cores
with materially different throughput. A static split makes the slowest worker
decide when the call finishes; dynamic claiming lets fast cores absorb the
remainder.

The calling thread participates as one of the workers, so `ThreadPool(1)` spawns
nothing and runs every body inline. Pools are cached per thread count for the
process lifetime so benchmark iterations do not pay thread creation cost.

### `cpu::gemv_avx2` — the decode kernel

Autoregressive decoding multiplies one activation vector by every weight matrix.
That is a matrix-vector product, and using the GEMM for it would waste five
sixths of every register tile and every load.

`gemv_avx2` streams the matrix once. Four accumulators per output row give the
load units enough independent work to cover DRAM latency; the activation vector
is re-read per row but stays in L1.

Threading is over contiguous row bands, four bands per worker. Contiguous bands
are what the TLB and the hardware prefetcher both want. This is the single
largest decode lever in the engine: measured bandwidth goes from 18.4 GB/s on one
thread to 35.4 GB/s on sixteen for a 64 MiB matrix.

There is deliberately **no software prefetch**. It was implemented and measured
with and without; the hardware prefetcher already handles a contiguous band and
the explicit hint cost more than it saved. See `docs/benchmarks.md` section 3.

### Streaming kernels and the dispatch rule

`copy_stream` and `sum_stream` measure the bandwidth ceiling and are used by the
benchmark harness. They are selected through the same `KernelSet` as everything
else, with portable scalar fallbacks.

They were originally called directly from the CLI, which meant a machine without
AVX2 would execute an illegal instruction during a measurement. That was a bug
against ADR-0002: an unsupported instruction must never be reachable, in the
inference path or in diagnostics. Every kernel entry point in `KernelSet` is now
non-null and feature-selected.

### `tge::memory` — RAM as a first-class tier

- `query_system_memory` reports installed, visible and available physical memory,
  plus commit limit and page size.
- `AlignedBuffer` is the only allocation primitive the engine uses for tensor
  storage; alignment is normalized to a power of two and defaults to the 64-byte
  cache line.
- `MemoryBudget` divides a usable budget across weights, KV cache, scratch and
  sessions, and refuses reservations that would cross a pool limit.

The budget derives its ceiling from *available* physical memory, not installed
memory, and subtracts an OS reserve. Commit limit is read but never used for
placement: pagefile-backed execution is not a supported tier, and treating it as
one is how an engine ends up thrashing.

## Design decisions

Recorded under [decisions/](decisions/).
