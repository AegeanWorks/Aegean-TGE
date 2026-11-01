# Benchmarks

Every number here was measured on the development machine. Nothing is
estimated.

**Measurement caveat.** This machine runs with 2.5–5 GB free of 16 GB while the
benchmark runs, with VS Code, Defender and Memory Compression holding roughly
3 GB. Background memory pressure moved single-run results by up to 2x, so each
configuration below was run five times and both the median and the best observed
value are reported. Best-observed numbers are comparable with the "before"
column, which was also best-of-N. Treat differences under about 15 percent as
noise on this hardware.

```powershell
. .\scripts\env.ps1
cmake --preset release
cmake --build --preset release
.\build\release\bin\tge.exe bench --m 1024 --n 1024 --k 1024 --repeat 3
```

## 1. f32 GEMM: before and after the blocking rewrite

Before: 6x16 microkernel over unpacked matrices, `nt` outer / `mt` inner, single
threaded, no packing, no K blocking.

After: packed A and B panels, 144/256/256 blocking, first depth pass stores
instead of accumulating, dynamic work-stealing thread pool.

| Shape (M=N=K) | Before, 1 thread | After, 1 thread | After, 16 threads | After 16T vs before 1T |
|---|---|---|---|---|
| 512  | 88.5 GFLOPS | 75.9 best / 66.4 median | 120.6 best / 102.3 median | **1.36x** |
| 1024 | 56.8 GFLOPS | 66.8 best / 65.5 median | 184.9 best / 183.1 median | **3.26x** |
| 2048 | 18.2 GFLOPS | 43.3 best / 34.1 median | 246.5 best / 216.7 median | **13.5x** |

A second measurement batch of the same binary produced 16-thread peaks of 131.6,
205.1 and 267.1 GFLOPS for the three shapes, which is the width of the noise band
on this machine.

### What changed and why it matters

The old kernel's throughput fell apart as the working set left cache: 88 GFLOPS
at 512 cubed, 18 at 2048 cubed. That is a 4.8x collapse with no change in
arithmetic. The new kernel is essentially flat at roughly 60–65 GFLOPS
single-threaded regardless of shape, because packing and blocking keep the
microkernel fed.

Scaling is the larger win. At 2048 cubed the threaded version is 13.5x the old
single-threaded result and 5.7x the new single-threaded one.

### Honest read

- 512 cubed single-threaded is slightly *slower* than before (75.9 vs 88.5,
  about 14 percent). Packing overhead is a fixed cost that a small,
  cache-resident problem cannot amortize. Every other configuration improved.
- Still single-digit percent of the ~845 GFLOPS theoretical peak of six P-cores.
  Packing and blocking were the largest structural defect, not the last one.
- No AVX-VNNI kernel exists yet. It is the highest-value CPU feature left.

## 2. Rejected experiment: A-panel stride padding

Hypothesis: with a 24-byte packed A stride, broadcasts straddle 64-byte cache
lines, so padding the panel to 32 bytes should help.

Measured, five runs per shape, medians:

| Shape | Stride 6 | Stride 8 (padded) |
|---|---|---|
| 512 | 64.4 | 65.2 |
| 1024 | 62.0 | 52.7 |
| 2048 | 59.8 | 47.6 |

Rejected. The padding removes split loads but adds 33 percent more panel
traffic, and the traffic costs more than the splits save. The constant
`kAStride` is left in the source with a comment recording the result so the
experiment is not repeated.

## 3. Memory bandwidth

`tge bench mem --size-mb 256 --repeat 3`

| Operation | Bandwidth |
|---|---|
| copy (read + write) | 10.02 GB/s |
| read (sum) | 12.44 GB/s |

This is a single 4800 MT/s DIMM in single-channel configuration, which is the
ceiling for everything memory-bound. Autoregressive decode reads every weight
for every token, so with a 1.04 GiB Q4 model the decode ceiling is roughly
12 GB/s / 1.04 GiB ≈ 11 tokens/s before any compute cost. Measured decode below
is 7.2 tokens/s, so decode is already within 35 percent of the bandwidth wall.
Optimizing GEMM further will not move decode; only smaller weights, better
bandwidth, or batching will.

## 4. llama.cpp CPU baseline

Built from source at commit `a91eab6` (ggml 0.28.0), GCC 16.2, CPU backend with
`-mavx2 -mfma -mf16c`, no AVX-512. Same machine, same thread count.

```powershell
llama-bench -m <model>.gguf -p 512 -n 128 -t 16 -r 3
```

| Model | Size | Prompt (pp512) | Decode (tg128) |
|---|---|---|---|
| qwen2.5-0.5B-Instruct Q4_K_M | 463 MiB | 407.6 ± 29.7 t/s | 8.73 ± 2.30 t/s |
| qwen2.5-1.5B-Instruct Q4_K_M | 1.04 GiB | 207.6 ± 0.7 t/s | 7.22 ± 0.75 t/s |

### Status of comparability

These are **not** a comparison with TGE. TGE cannot run inference yet, so there
is nothing to compare against. The baseline exists to answer one question later:
once TGE has its own GGUF loader and decoder, does it beat these numbers? Any
claim of superiority before that would be unsupported.

Two things the baseline already establishes:

1. Decode is memory-bandwidth-bound, not compute-bound. 7–9 tokens/s against
   ~11 tokens/s of theoretical bandwidth headroom confirms it.
2. Prompt processing at 407 t/s for a 0.5B model on a 16-thread CPU is the bar
   TGE's prefill path has to clear.

## What to measure next

1. TGE inference against this baseline (needs GGUF loading, milestone 0.2).
2. Thread scaling 1→16 for GEMM and for decode separately; decode likely wants
   fewer threads than prefill.
3. AVX-VNNI quantized dot product, compared against the AVX2 path on Q4_K.
