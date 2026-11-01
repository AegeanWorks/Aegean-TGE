# Benchmarks

Every number here was measured on the development machine. Nothing is
estimated.

**Measurement caveat.** This machine runs with 2–5 GB free of 16 GB while the
benchmark runs, with VS Code, Defender and Memory Compression holding roughly
3 GB. Background memory pressure moved single-run results by up to 2x. Every
configuration was run several times; medians and ranges are reported. Treat
differences under about 15 percent as noise on this hardware.

```powershell
. .\scripts\env.ps1
cmake --preset release
cmake --build --preset release
.\build\release\bin\tge.exe bench gemv --rows 4096 --cols 4096 --repeat 5
```

## 1. GEMV: the decode primitive

Autoregressive decoding is a matrix-vector product, not a matrix-matrix one. One
token pushes one activation vector through every weight matrix. The matrix is
streamed once and never reused inside the call, so decode is a pure memory
bandwidth problem, and it is benchmarked as one.

`tge bench gemv` measures exactly that shape: scan an M x K weight matrix once,
one activation vector, one output vector.

| Matrix | Size | Threads | Bandwidth | Implied 1 GiB model |
|---|---|---|---|---|
| 4096 x 4096 | 64 MiB | 1 | 18.4 GB/s (median) | 17 t/s |
| 4096 x 4096 | 64 MiB | 16 | **35.4 GB/s** (median; best 35.7) | **33 t/s** |
| 8192 x 8192 | 256 MiB | 1 | 13.4 GB/s | 13 t/s |
| 8192 x 8192 | 256 MiB | 16 | 28.7 GB/s | 27 t/s |

Two things fall out of this:

1. **Threading is the single largest decode lever available.** A memory-bound
   stream is limited by how many outstanding misses one core can sustain, not by
   arithmetic. Going from one thread to sixteen nearly doubles sustained
   bandwidth (18.4 to 35.4 GB/s at 64 MiB).
2. **Larger streams are slower per byte.** Bandwidth drops from 35.4 GB/s at
   64 MiB to 28.7 GB/s at 256 MiB. That is TLB pressure: a 256 MiB buffer walks
   far more 4 KiB pages. Large pages would attack this directly (see section 4).

The kernel is at the measured hardware wall: `bench mem` reports 25.6-29.4 GB/s
for a plain threaded read of 256 MiB, and GEMV on a 256 MiB matrix reports
28.7 GB/s. There is nothing left to win in the kernel itself; the remaining
levers are bytes read and page size.

## 2. f32 GEMM: prompt processing

Gemv covers decode; GEMM covers prefill. Before/after the blocking rewrite:

| Shape | Before, 1 thread | After, 1 thread | After, 16 threads |
|---|---|---|---|
| 512  | 88.5 GFLOPS | 75.9 best / 66.4 median | 120.6 best |
| 1024 | 56.8 GFLOPS | 66.8 best / 65.5 median | **206.6** |
| 2048 | 18.2 GFLOPS | 43.3 best / 34.1 median | 246.5 best |

The old kernel collapsed 4.8x as the working set left cache. The new kernel is
flat at roughly 60-70 GFLOPS single-threaded regardless of shape. At 512 cubed
single-threaded it is about 14 percent slower than before: packing overhead is a
fixed cost that a small cache-resident problem cannot amortize.

GEMM matters much less than GEMV for token generation. Prefill is one burst;
decode is the long tail.

## 3. Rejected experiments

Each of these was implemented, measured, and removed. They are recorded so the
same idea is not tried twice on intuition alone.

| Experiment | Result | Decision |
|---|---|---|
| Pad packed A stride from 6 to 8 floats | 1024: 62.0 -> 52.7 GFLOPS; 2048: 59.8 -> 47.6 | Rejected: 33 percent more panel traffic outweighed fewer line splits |
| Software prefetch in GEMV, distance 512 | 1T: ~19.6 with vs ~18.4-20.0 without; 16T median 28.4 with vs 32.1 without | Rejected: hardware prefetcher already handles a contiguous per-thread band |
| `_MM_HINT_NTA` on the GEMV prefetch | 16T median 29.8 vs 28.4 for T0, inside the noise band | Rejected: no measurable difference |

## 4. Memory bandwidth

`tge bench mem --size-mb 256 --repeat 5`

| Operation | 1 thread | 16 threads |
|---|---|---|
| copy (read + write) | 10.3 GB/s | 23.8 GB/s |
| read (sum) | 13.5 GB/s | 25.6-29.4 GB/s |

This is one 4800 MT/s DDR5 DIMM in single-channel configuration: 38.4 GB/s is
the theoretical peak, and 16 threads reach 25-29 GB/s of it, roughly 75 percent.
A single core reaches only 35 percent, which is why decode cannot be
single-threaded no matter how good the kernel is.

Large pages (2 MiB) would cut the TLB pressure visible in the 256 MiB GEMV
result. They are **not implemented**, because `SeLockMemoryPrivilege` is not
granted to this account at all (`whoami /priv` does not list it), so the
mechanism cannot be verified here. It requires a Local Security Policy change by
an administrator. Recorded as an available lever, not a claim.

## 5. llama.cpp CPU baseline

Built from source at commit `a91eab6` (ggml 0.28.0), GCC 16.2, CPU backend with
`-mavx2 -mfma -mf16c`, no AVX-512. Same machine, same thread count.

```powershell
llama-bench -m <model>.gguf -p 512 -n 128 -t 16 -r 3
```

| Model | Size | Prompt (pp512) | Decode (tg128) |
|---|---|---|---|
| qwen2.5-0.5B-Instruct Q4_K_M | 463 MiB | 407.6 ± 29.7 t/s | 8.73 ± 2.30 t/s |
| qwen2.5-1.5B-Instruct Q4_K_M | 1.04 GiB | 207.6 ± 0.7 t/s | 7.22 ± 0.75 t/s |

### What the two datasets say together

For the 1.04 GiB model, the measured bandwidth wall is about 27-29 t/s and
llama.cpp achieves 7.22. **It is running at roughly a quarter of the available
memory bandwidth.** That is the most useful number this milestone produced: it
says decode is not closed off by the hardware, and a well-built decoder with
quantized weights has a large gap to close.

It also says what to optimize. Faster GEMM will not help. Smaller weights,
better bandwidth utilisation, and keeping the KV cache resident will.

## What to measure next

1. TGE inference against this baseline (needs GGUF loading and a decoder).
2. Whether decode wants fewer threads than prefill once real work is interleaved
   with the weight scan.
3. AVX-VNNI quantized dot product against the AVX2 path on Q4_K weights, which
   quarters the bytes that must cross the bus.
