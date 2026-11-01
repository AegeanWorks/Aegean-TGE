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

### Model layer: GGUF and a Qwen2 decoder

`gguf::File` maps a model read-only and parses the header, metadata and tensor
table. Weights are never copied: `tensor_data` points straight into the mapping,
so the OS page cache is the only thing deciding what is resident. That is the
memory architecture the project intends to build on, used from the start rather
than bolted on later.

`models::Qwen2` runs the decoder graph: embedding lookup, RMSNorm, QKV
projections with biases, NeoX rotary embedding, grouped-query attention with a
KV cache, SwiGLU feed-forward, and the output projection.

It exists to make the speed claim measurable rather than to be the finished
model layer. It is deliberately narrow: one token at a time, greedy sampling, no
tokenizer, no batching, 2048-token context.

Two correctness details that are easy to get wrong and were not:

- Qwen2 rotates the **first half of a head against the second half** (NeoX
  layout), not adjacent pairs. Getting this backwards still produces plausible
  numbers and complete nonsense text.
- Qwen2 carries **QKV biases**, unlike Llama.

Both were confirmed by reproducing llama.cpp's greedy output token for token on
the same model.

### `cpu::gemv_q8_0` — the quantized decode kernel

Q8_0 stores each 32 weights as a 34-byte block: one fp16 scale and 32 signed
bytes. The kernel widens each group of eight int8 values to int32, converts to
float, and accumulates with FMA, applying the block scale once.

That widening chain looked like the bottleneck and was investigated: single
threaded decode reaches about 8.3 GB/s against 18.4 GB/s for the f32 GEMV. An
integer-domain variant (quantize the activation too, then `maddubs` with no
widening) was implemented and measured at 31.0 t/s median against 30.3-34.0 for
this version. No gain, and it broke token parity with llama.cpp, so it was
reverted. Decode here is limited by locality and per-call barriers, not by the
conversion. See `docs/decisions/ADR-0006-q8-stays-in-float.md`.

### Prompt processing and sampling

`gemv_q8_batch_avx2` reuses one weight row across a group of four tokens. That
is the whole difference between prefill and decode: decoding one token reads
every weight once, so processing P tokens one at a time reads them P times. The
kernel is the same data in the same layout, folded over four activation rows
before the weight block is dropped.

`Qwen2::prefill` runs a prompt in 64-token chunks. Chunk size is a cache
decision, not a numerical one: the batch kernel re-reads the activation rows once
per output row, so the chunk has to stay resident. Attention still runs per
token, because every token has a different causal prefix.

`models::sample_token` draws from the logits with temperature, top-k, top-p,
min-p and a repetition penalty. Defaults are greedy, which is what every
benchmark used, so adding sampling did not change any measured number.

### Quantized formats and tokenization

Q8_0 keeps a fused kernel: 34-byte blocks, int8 widened to float inside the
inner loop. The legacy formats (Q4_0, Q5_0) and the super-block formats (Q4_K,
Q5_K, Q6_K) share one dequantize-then-dot path, because a decoded row is cheap
to produce once and reuse across every token in a batch.

`quant_row_bytes` derives the stride from the type. That matters because
"Q4_K_M" is not one format: it is a recipe that mixes Q5_0, Q4_K, Q6_K and Q8_0
in the same file, so assuming any single layout for the embedding table silently
decodes garbage.

`models::Tokenizer` converts both ways. GGUF stores vocabulary entries in the
GPT-2 byte-level representation, so decoding is a table lookup per character.
Encoding runs the other direction: split the text into pre-tokens, map each byte
onto the byte-level alphabet, then repeatedly apply the lowest-ranked adjacent
merge from `tokenizer.ggml.merges`.

Pre-tokenization is the approximate part. It implements the split rules
procedurally — contractions, optional leading space with a letter run, single
digits, punctuation runs with trailing newlines, whitespace runs — rather than
through the family's Unicode-aware regular expression. ASCII text matches the
reference tokenizer exactly; exotic Unicode punctuation is folded into the
letter run instead of being split, which changes the ids but not the text.

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
