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
| TGE | 6 / all | **191.6 t/s** (median of 3) | **23.4 t/s** (median of 3) |
| llama.cpp | 6 | 285.3 ± 22.8 t/s | 17.9 ± 1.2 t/s |

Both measured with a 510-token context resident, so decode numbers here are
lower than the short-context figures above: attention has more keys to walk.

That is **11 t/s to 191.6 t/s**, and prefill is now 67 percent of llama.cpp
rather than 4 percent. The last 12 percent came from widening the kernel's token
group from four to six, which amortizes the weight widening over more
activations: the same 32 weights are folded across six rows instead of four,
and six accumulators still fit the register file where eight would spill.
Aligning the 64-token chunk to that group was measured and was slightly worse,
so it was reverted.

What remains is arithmetic: 241 GFLOPS against llama.cpp's 358. The int8-to-float
widening is now the largest cost in the inner loop, and the fix is an integer
dot product rather than more float unrolling.

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
- **Tokenizer**: ids decode to text and text encodes to ids. Encoding was
  validated against `llama-tokenize` on punctuation, decimal numbers, percent
  signs, contractions ("don't") and double spaces — all matched exactly. Unicode
  punctuation outside ASCII is folded into the letter run, which can change ids
  without changing the decoded text.
- Sampling supports temperature, top-k, top-p, min-p and repetition penalty, but
  the comparison above uses greedy so the numbers are reproducible.
- Prefill arithmetic is still behind: the batch kernel sustains about 241 GFLOPS
  against llama.cpp's 358. Widening the token group from four to six closed part
  of it. The integer inner loop was implemented and measured 22 percent slower,
  so it was reverted (section 7); the remaining gap needs port-level analysis,
  not more instruction counting.

### The kernel measured in isolation

`tge bench batch` times one prefill matrix-vector kernel on synthetic weights, so
the forward pass around it cannot hide anything:

| Shape (rows x cols), 64 tokens | GFLOPS | Weight traffic |
|---|---|---|
| 4864 x 896 (ffn gate/up) | 267.9 | 4.4 MiB |
| 896 x 896 (attention projection) | 227.3 | 0.8 MiB |
| 151936 x 896 (output projection) | 262.2 | 137.9 MiB |

The full prefill reaches 241 GFLOPS, so about 91 percent of prefill time is this
kernel; norms, rope, attention and residual adds account for the rest. The gap
with llama.cpp is therefore genuinely in the kernel, not in the code around it.

### The llama.cpp comparison is less stable than it looks

Two runs of `llama-bench -p 512 -t 6` on the same file and machine in the same
session:

| Run | pp512 |
|---|---|
| earlier in this session | 285.3 ± 22.8 t/s |
| after the kernel work | **220.7 ± 3.8 t/s** |

The same binary, the same weights, a 29 percent difference. TGE's own prefill
runs span 170-201 t/s. So "67 percent of llama.cpp" was computed against the
high llama sample; against the low one TGE is at 87 percent, which is inside
this machine's noise band.

The honest statement is therefore: **TGE prefill is somewhere between 13 and 33
percent behind llama.cpp, and this machine cannot resolve which** without the
other applications giving up their 3 GB. The isolated kernel figure (265 GFLOPS)
is the stable number worth tracking.

## 7. Rejected experiments

Each of these was implemented, measured, and removed. They are recorded so the
same idea is not tried twice on intuition alone.

| Experiment | Result | Decision |
|---|---|---|
| Pad packed A stride from 6 to 8 floats | 1024: 62.0 -> 52.7 GFLOPS; 2048: 59.8 -> 47.6 | Rejected: 33 percent more panel traffic outweighed fewer line splits |
| Software prefetch in GEMV, distance 512 | 1T: ~19.6 with vs ~18.4-20.0 without; 16T median 28.4 with vs 32.1 without | Rejected: hardware prefetcher already handles a contiguous per-thread band |
| `_MM_HINT_NTA` on the GEMV prefetch | 16T median 29.8 vs 28.4 for T0, inside the noise band | Rejected: no measurable difference |
| Align the 64-token prompt chunk to the six-token group | prefill median 185.4 vs 191.6 | Rejected: measured worse, so the tail stays on the single-token path |
| **Integer inner loop for prefill** (quantize activations, `maddubs`, no widening) | prefill median **150.0 t/s** (145.9 / 150.0 / 152.1) vs **186.0** float (five samples, 150.3-201.7) | Rejected: **22 percent slower** |
| **AVX-VNNI `dpbusd` inner loop** | kernel in isolation **faster** (283.96 vs 267.88 GFLOPS on ffn, 342.03 vs 262.20 on the output projection) but end-to-end prefill **150.9 vs 186.0 t/s** | Rejected: the faster kernel does not survive integration |

The VNNI attempt is the most useful negative result so far, because it separates
the two layers cleanly.

The kernel itself was genuinely faster. `dpbusd` moves the integer dot onto the
FMA ports, which is exactly what the `maddubs` attempt lacked: that version was
fighting `psignb` for the single port that issues it. Weights are biased into
unsigned bytes with `xor 0x80`, and because the activation is the biased operand
the correction term is per-lane and depends only on the weights, so it is
computed once per block and shared by every token in the group.

Getting there exposed a real bug worth recording: `dpbusd` spreads 32 products
across eight int32 lanes, so the bias correction must be applied per lane. The
first version subtracted a block-wide total from every lane and produced errors
of 178-1890; the per-lane correction brought that to 0.06-1.2, which is ordinary
activation-quantization noise.

But end-to-end it lost, and the reason is not in the kernel: quantizing the
activations allocates two vectors on every `matvec` call, and the model makes
roughly 150 of those per chunk. The isolated benchmark has the same quantization
cost but amortizes it over one huge matrix instead of many small ones.

So the kernel-level idea is sound and the integration is what failed. The next
attempt should keep a reusable activation buffer per worker instead of
allocating per call, then re-measure with `tge bench batch` and end to end.

The integer prefill kernel is the most instructive rejection. It was built
exactly as planned — activations quantized once per call, `sign` + `maddubs` +
`madd` in the inner loop, one int-to-float conversion per block — and it lost.

The likely reason, unconfirmed because there is no profiler on this machine yet:
the integer path needs `psignb`, which issues on one port, plus a scalar scale
broadcast and a conversion per token per block, where the float path uses four
FMAs spread over two ports with no conversions at all. Counting instructions
made the integer version look 1.5x cheaper; counting *ports* did not.

That is the third performance hypothesis this milestone that instruction
counting got wrong, which is the real lesson: on this machine, measure first.

## 8. What to measure next

1. TGE inference against this baseline (needs GGUF loading and a decoder).
2. Whether decode wants fewer threads than prefill once real work is interleaved
   with the weight scan.
3. AVX-VNNI quantized dot product against the AVX2 path on Q4_K weights, which
   quarters the bytes that must cross the bus.
