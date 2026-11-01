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

## 6. End-to-end decode against llama.cpp

This is the comparison that matters, and it is like for like: the **same model
file** (`qwen2.5-0.5b-instruct-q8_0.gguf`, 638.74 MiB, 630.17 M parameters), the
same machine, the same thread count, greedy sampling, 128 generated tokens.

```powershell
.\build\release\bin\tge.exe generate --model <q8_0.gguf> \
    --tokens "785 6722 315 9625 374" --n 129
llama-bench -m <q8_0.gguf> -p 128 -n 128 -t 6 -r 5
```

| Engine | Threads | Decode |
|---|---|---|
| TGE | 6 (its default) | **30.3 t/s median, 33.1 best** |
| llama.cpp | 6 | 15.2 ± 2.1 t/s |
| TGE | 16 | 21.8 t/s |
| llama.cpp | 16 | 10.1 t/s |

**TGE decodes about two times faster than llama.cpp on identical weights and
hardware.** Both engines lose badly at 16 threads, which is the same phenomenon
in both: synchronisation and E-core contention outweigh the extra bandwidth.

### Prompt processing, now that it is batched

Prompt processing used to feed tokens one at a time, re-reading every weight per
token. It now runs through a batched kernel that reuses one weight row across a
group of tokens, which is the only reason prefill and decode differ by more than
an order of magnitude.

| Engine | Threads | Prompt (pp512) | Decode (tg128) |
|---|---|---|---|
| TGE | 6 / all | **170.9 t/s** (median of 3) | **23.4 t/s** (median of 3) |
| llama.cpp | 6 | 285.3 ± 22.8 t/s | 17.9 ± 1.2 t/s |

Both measured with a 510-token context resident, so decode numbers here are
lower than the short-context figures above: attention has more keys to walk.

That is **11 t/s to 171 t/s, a 15x improvement**, and prefill is now 60 percent
of llama.cpp rather than 4 percent. Decode remains faster than llama.cpp.

The remaining prefill gap is arithmetic: TGE's batch kernel sustains about
207 GFLOPS, llama.cpp's quantized GEMM about 358 GFLOPS on the same six cores.

### Correctness, not just speed

A fast decoder producing the wrong tokens would be worthless, so the greedy
output was compared token for token. Prompt "The capital of France is"
(ids 785 6722 315 9625 374):

- llama.cpp: ` Paris. It is the largest city in Europe and the second largest in the world`
- TGE token ids: `12095 13 1084 374 279 7772 3283 304 4505 323 279 2086 7772 304 279 1879`

Those ids detokenize to exactly the same sentence. The forward pass — RMSNorm,
NeoX rotary embedding, grouped-query attention, SwiGLU, QKV biases — agrees with
llama.cpp on every position of that span.

### Honest caveats

- The batched prefill path reproduces greedy output token for token against
  llama.cpp, including across chunk boundaries (verified on a 136-token prompt,
  which crosses two 64-token chunks).
- **Quantized formats**: Q8_0, Q5_0, Q4_0, Q4_K, Q5_K and Q6_K are all
  executable. The Q4_K_M model mixes four of them (Q5_0 for embeddings, Q4_K and
  Q6_K for blocks, Q8_0 for the output projection), and it reproduces llama.cpp's
  greedy output token for token — one test that exercises every dequantizer at
  once.
- **Context** defaults to 4096 tokens and is selectable with `--context`, capped
  by the model's own maximum and a 32768 hard limit.
- **Tokenization is half done**: ids decode to text, so output is readable.
  Encoding text to ids is not implemented, so prompts are still supplied as ids.
- Sampling supports temperature, top-k, top-p, min-p and repetition penalty, but
  the comparison above uses greedy so the numbers are reproducible.
- Prefill arithmetic is still behind: the batch kernel sustains about 207 GFLOPS
  against llama.cpp's 358. Closing that needs weight packing and register
  blocking across output rows, which is a kernel rewrite rather than a tweak.

## 7. Rejected: integer-domain Q8 inner loop

Single-threaded decode reaches 16.6 t/s, which is about 8.3 GB/s of weight
traffic, while a standalone f32 GEMV reaches 18.4 GB/s on one thread. That gap
looked like the cost of widening every int8 to int32 and then to float, so the
obvious fix was to quantize the activation too and keep the inner loop in
integer domain.

It was implemented with `sign` + `maddubs` + `madd`, with lane accumulation and
one horizontal reduction per row. Measured over 128 generated tokens:

| Kernel | Decode |
|---|---|
| Float activation | 30.3 / 28.9 / 33.1 t/s |
| Integer activation | 31.0 / 30.4 / 32.9 t/s |

No measurable gain, and it broke exact token parity with llama.cpp at position
14 of 16. Reverted; see `decisions/ADR-0006-q8-stays-in-float.md`.

**The lesson is the more useful result.** A token is not one GEMV: it is 24
layers of two norms, six projections, an attention pass and a 151,936-row output
projection, with roughly 150 barriers between the calls. Comparing whole-token
throughput against a single-kernel number measured the wrong thing. Decode is
structural here — locality and synchronisation — not arithmetic.

That redirects the remaining work to structure: batch the prompt, fuse QKV into
one parallel region, fuse gate and up.

## 8. What to measure next

1. TGE inference against this baseline (needs GGUF loading and a decoder).
2. Whether decode wants fewer threads than prefill once real work is interleaved
   with the weight scan.
3. AVX-VNNI quantized dot product against the AVX2 path on Q4_K weights, which
   quarters the bytes that must cross the bus.
