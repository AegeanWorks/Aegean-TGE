# Measurement Log

Every change to this engine is recorded with the number that justified it
(project guideline G-002). This file is the canonical record: the change table,
the measurements behind it, and the changes that lost.

The point is not ceremony. Five of the optimisations below looked obviously
correct on paper and measured worse, and one headline number was 29 percent
noise. Without the numbers, the plausible story survives.

## Machine and method

| | |
|---|---|
| CPU | Intel Core i5-14400F, 10 physical / 16 logical (6 P + 4 E) |
| ISA | AVX2, FMA, AVX-VNNI. **No AVX-512** (fused off), no AMX |
| RAM | 16 GiB, one 4800 MT/s DIMM, single channel |
| GPU | AMD Radeon RX 9060 XT, 15.67 GiB VRAM (Vulkan only, no HIP SDK) |
| OS | Windows 11 Pro 10.0.26200 |
| Compiler | MinGW GCC 16.2, `-O3`, no `-march=native` |

**Read every absolute number as ±20 percent.** The machine runs with 2-4 GB free
of 16; VS Code, Defender and Memory Compression hold the rest. Two back-to-back
runs of the same binary differ by up to 29 percent. Ratios measured within one
run are reliable; absolute values across sessions are not.

A baseline snapshot: `.\scripts\measure.ps1`. A change is measured by diffing
one before and one after, repeated two or three times when the delta is small.

## Change log

| Date | Change | Metric | Before | After | Verdict |
|---|---|---|---|---|---|
| 2026-11-01 | Packed panels, 144/256/256 blocking, work-stealing pool | GEMM 2048³, 16 threads | 18.2 GFLOPS | 246.5 GFLOPS | **kept** |
| 2026-11-01 | Pad packed A stride from 6 to 8 floats | GEMM 1024³ | 62.0 GFLOPS | 52.7 GFLOPS | reverted |
| 2026-11-01 | Software prefetch in GEMV, distance 512 | GEMV 4096², 16 threads | 32.1 GB/s | 28.4 GB/s | reverted |
| 2026-11-01 | `_MM_HINT_NTA` on that prefetch | GEMV 4096², 16 threads | 28.4 GB/s | 29.8 GB/s | reverted (noise) |
| 2026-11-01 | Dedicated threaded GEMV decode kernel | GEMV 4096² | 18.4 GB/s (1T) | 35.4 GB/s (16T) | **kept** |
| 2026-11-01 | Decode threads: all logical to P-cores | decode 0.5B Q8_0 | 21.8 t/s (16T) | 32.5 t/s (6T) | **kept** |
| 2026-11-01 | Per-matrix thread sizing for decode | decode 0.5B Q8_0 | 32.8 t/s | 27.5 t/s | reverted |
| 2026-11-01 | Batched prefill (weight row reused across tokens) | prefill, 510 tokens | 11 t/s | 170.9 t/s | **kept** |
| 2026-11-01 | Prefill token group 4 to 6 | prefill, 510 tokens | 170.9 t/s | 191.6 t/s | **kept** |
| 2026-11-01 | Align 64-token chunk to the group | prefill, 510 tokens | 191.6 t/s | 185.4 t/s | reverted |
| 2026-11-01 | Integer `maddubs` prefill inner loop | prefill, 510 tokens | 186.0 t/s | 150.0 t/s | reverted |
| 2026-11-01 | AVX-VNNI `dpbusd` prefill inner loop | prefill, 510 tokens | 186.0 t/s | 150.9 t/s | reverted |
| 2026-11-01 | Skip metadata arrays while loading the model | load, id prompt | 65 ms | 24 ms | **kept** |
| 2026-11-01 | Same, text prompt | load + tokenizer | 207-219 ms | 183-205 ms | **kept** |

## 1. f32 GEMM

Reference kernel is the scalar oracle; `avx2-fma` is the shipped kernel.

| Shape | Reference (1T) | Shipped (1T) | Shipped (16T) |
|---|---|---|---|
| 512³ | 16.8-17.2 GFLOPS | 51.4-75.9 | 120.6-122.5 |
| 1024³ | 17.4-20.6 | 62.0-66.8 | 184.9-206.6 |
| 2048³ | 3.2-8.7 | 27.2-43.3 | 180.4-246.5 |

Best observed 246.5 GFLOPS at 2048³; the same config measured 180.4 later under
heavier background load. The reference kernel's own variance (3.2 to 8.7) shows
how much the machine moves.

### Ablation: which of the three changes did it

The 18.2 to 246.5 entry bundles three changes *and* a 1-to-16-thread comparison.
`bench gemm --kernel unpacked|unblocked|packed` separates them.

| 2048³, GFLOPS | 1 thread | 16 threads |
|---|---|---|
| unpacked (no packing, no blocking) | 18.80 | n/a (control is single-threaded) |
| + packing, no blocking | 43.16 | 33.14 |
| + blocking (shipped) | 27.23 | **180.37** |

| 512³, GFLOPS | 1 thread | 16 threads |
|---|---|---|
| unpacked | 72.75 | n/a |
| + packing, no blocking | 33.59 | 24.36 |
| + blocking (shipped) | 51.36 | **122.51** |

- **The pool is the largest single factor**: 27.23 to 180.37 at 2048³, 6.6x.
- **Blocking is the precondition for it.** Without blocking, more threads make
  the kernel *slower* (43.16 to 33.14): the panel spans the whole matrix and
  every worker re-reads it from DRAM. Blocking alone is a single-thread
  *regression* (43.16 to 27.23) and only pays at 16 threads.
- **Packing trades small for large**: 18.80 to 43.16 at 2048³, but 72.75 to
  33.59 at 512³. Blocking recovers most of the small-shape loss.

## 2. GEMV f32 - the decode primitive

| Matrix | 1 thread | 16 threads |
|---|---|---|
| 4096² (64 MiB) | 18.4 GB/s (15.9-20.8) | **35.4 GB/s** (29.6-35.7) |
| 8192² (256 MiB) | 13.4 GB/s | 28.7 GB/s |

Against a plain threaded read of 256 MiB (25.6-29.4 GB/s) the kernel is at the
hardware wall. The larger matrix loses bandwidth to TLB pressure; large pages
would attack it and are blocked on a privilege this account does not have.

## 3. Memory bandwidth

`bench mem --size-mb 256`

| Operation | 1 thread | 16 threads |
|---|---|---|
| copy (read + write) | 10.3 GB/s | 23.8 GB/s |
| read (sum) | 13.5 GB/s | 25.6-29.4 GB/s |

One DDR5-4800 DIMM single-channel is 38.4 GB/s theoretical; 16 threads reach
roughly 75 percent of it and a single core about 35 percent. That is why decode
cannot be single-threaded.

## 4. End-to-end decode vs llama.cpp

Same file (`qwen2.5-0.5b-instruct-q8_0.gguf`, 638.74 MiB), same machine, greedy,
128 generated tokens.

| Engine | Threads | Decode |
|---|---|---|
| TGE | 6 | **30.3-33.1 t/s** |
| llama.cpp | 6 | 15.2-17.9 t/s (three runs) |
| TGE | 16 | 21.8 t/s |
| llama.cpp | 16 | 10.1 t/s |

Thread sweep for TGE decode: 1T 16.6, 4T 28.7, 6T 27.0-33.0, 8T 30.5-32.5,
10T 29.5, 12T 22.4, 16T 21.8. Both engines fall off a cliff at 16 threads.

Output is **token-identical to llama.cpp** on both models tested.

## 5. End-to-end prefill vs llama.cpp

510-token prompt, 64-token chunks.

| Engine | Prompt (pp512) |
|---|---|
| TGE | **170.9 then 191.6 t/s** (before and after the token-group change) |
| llama.cpp | 285.3 ± 22.8, and **220.7 ± 3.8** on a later run |

Two llama runs on the same file, same binary, same session, differ by 29
percent. TGE's own prefill spans 170-201. So TGE is **13-33 percent behind
llama.cpp and this machine cannot resolve which**; the "67 percent" figure
quoted earlier was computed against the high sample.

Correctness: output is token-identical to llama.cpp on a 136-token prompt that
crosses two chunk boundaries.

## 6. Prefill kernel in isolation

`bench batch --rows R --cols C --tokens 64 --repeat 3`

| Shape | float | VNNI (`--kernel vnni`) |
|---|---|---|
| 128 x 896 (k/v) | 85.9 GFLOPS | 28.9 |
| 896 x 896 (q/o) | 337.7 | 141.6 |
| 4864 x 896 (gate/up) | 379.0 | 353.8 |
| 896 x 4864 (down) | 347.4 | 182.7 |
| 151936 x 896 (output) | 396.1 | **448.2** |

The full prefill reaches 241 GFLOPS, so about 91 percent of prefill time is this
kernel; the surrounding norms, rope and attention are the rest.

VNNI wins on one shape and loses on four. It stays behind the benchmark flag:
using it for the output projection alone would be worth roughly 11 percent, but
it changes the logits and breaks token parity, which is a deliberate trade.

## 7. Quantized formats

Validated by running `qwen2.5-0.5b-instruct-q4_k_m.gguf`, which mixes Q5_0,
Q4_K, Q6_K and Q8_0 in one file: output is token-identical to llama.cpp.

| Model | Format | Decode |
|---|---|---|
| 0.5B | Q8_0 (fused kernel) | 30.3-33.1 t/s |
| 0.5B | Q4_K_M (dequantize-then-dot) | 10.8 t/s |

The K-quant path is correct but not yet fast: it expands a row to float before
the dot. A fused K-quant dot is the obvious follow-up.

## 8. Load and startup

`generate` reports its own load and tokenizer timings.

| Measurement | Value |
|---|---|
| GGUF header + tensor table parse | ~2 ms |
| KV cache allocation | ~23 ms per 4096 tokens (40.8 / 59.0 / 83.7 at 512 / 4096 / 8192) |
| Vocabulary: 151,936 tokens + 151,387 merges | 60-180 ms depending on path |
| Model load, metadata skipped | **24 ms** |
| Model load, metadata eager | 65-90 ms |

`--vocabulary auto|eager|lazy`, three runs each:

| Path | lazy | eager |
|---|---|---|
| id prompt, load | **32.9 ms** | 86.9 ms |
| text prompt, load + tokenizer | 207-219 ms | **183-205 ms** |

Auto picks per invocation and is right both ways: 2.6x on id prompts, about 10
percent on text. All three modes produce identical ids.

A manifest caching the parse would save ~2 ms. The cost is the vocabulary, so a
cached vocabulary sidecar - not a weight manifest - is the artefact worth
building, and only for repeated CLI invocations.

Dequantising K-quants into a prepared form is a trap: Q4_K_M at 638 MiB becomes
~2.5 GiB as float, which this machine cannot hold.

## 9. llama.cpp baseline, other models

Built from source at `a91eab6` (ggml 0.28.0), GCC 16.2, `-mavx2 -mfma -mf16c`.

| Model | Prompt (pp512) | Decode (tg128) |
|---|---|---|
| qwen2.5-0.5B Q4_K_M | 407.6 ± 29.7 t/s | 8.73 ± 2.30 t/s |
| qwen2.5-0.5B Q8_0 | 278.6-285.3 t/s | 10.1-17.9 t/s |
| qwen2.5-1.5B Q4_K_M | 207.6 ± 0.7 t/s | 7.22 ± 0.75 t/s |

Decode on this hardware is bandwidth-bound: a 1.04 GiB model at 28-35 GB/s has a
ceiling near 27-33 t/s, and llama.cpp reaches 7.2. That is roughly a quarter of
available bandwidth, which is why decode has headroom while prefill does not.

## 10. Planning before allocating

`tge plan` computes the whole memory budget from the GGUF tensor table without
reading weight data, then decides whether the request fits. Measured on the 0.5B
Q8_0 model:

| | |
|---|---|
| weights (all tensors) | 638.7 MiB |
| read per decode token | 500.8 MiB (embedding row excluded) |
| KV per token | 24 KiB (24 layers x 128 values x 2 x 4 bytes) |
| KV at 8192 tokens | 192.0 MiB per session |
| usable budget here | 1.74 GiB |
| headroom after weights | 1.11 GiB |
| verdict, 8192 x 2 sessions | fits |

The projection is the useful part: **500.8 MiB per token against 28 GB/s gives a
decode ceiling of 53.3 t/s.** The engine measures ~30 t/s, so decode runs at
**56 percent of its bandwidth bound** — the other 44 percent is attention,
normalisation and the per-call barriers, not memory.

**The loader now applies the same rule** rather than only reporting it: asking
for 32768 tokens on the 1.5B Q4_K_M model loads at 14178 with a printed notice,
instead of allocating and failing. Cost of consulting the plan: load went from
24.0 to 28.5 ms warm (+4 ms). A cold file reads 232 ms once.

Analysis costs **2.3 ms warm**, 50 ms on a cold file (the first read of the
metadata region faults it in). Either way it is far cheaper than loading a model,
and it makes "does this fit?" answerable before committing to an allocation —
which is the whole point, since the failure it prevents is allocating first and
discovering afterwards.

## 11. Correctness checks that back these numbers

- Greedy output matches llama.cpp token for token on Q8_0 and on Q4_K_M.
- Prefill matches across chunk boundaries (136-token prompt, two boundaries).
- Q4_K/Q5_K/Q6_K/Q5_0/Q4_0 dequantizers agree with their scalar references.
- Batch kernels match scalar references across token counts hitting both the
  group and its tail.
- Tokenizer encoding matches `llama-tokenize` on punctuation, decimals, percent
  signs, contractions and double spaces.
- 45 unit cases, all passing.

## 11. Rejected, with margins

Recorded so the same idea is not rebuilt on intuition. Four of these six looked
correct on paper.

| Change | Lost by | Reason, where known |
|---|---|---|
| Pad packed A stride 6 to 8 | 1024³ 62.0 to 52.7 GFLOPS | 33 percent more panel traffic beat fewer line splits |
| Software prefetch in GEMV | 32.1 to 28.4 GB/s | hardware prefetcher already covers a contiguous band |
| `_MM_HINT_NTA` on it | 28.4 to 29.8, inside noise | no measurable difference |
| Align chunk to token group | 191.6 to 185.4 t/s | measured worse; tail stays on the single-token path |
| Integer `maddubs` prefill | 186.0 to 150.0 t/s | `psignb` contends for the single port that issues it |
| AVX-VNNI `dpbusd` prefill | 186.0 to 150.9 t/s end to end | faster per kernel (284/342 GFLOPS) but per-call activation quantization costs more than it saves |

VNNI is the instructive one: it separates the kernel from the integration
cleanly. The kernel is genuinely faster on the two large shapes; the model calls
it ~150 times per chunk and the fixed cost per call eats the gain.

## 12. What to measure next

1. A profiler. Four performance hypotheses here were wrong because
   instruction counting was used instead of port or memory analysis. WPA and
   AMD uProf are not installed.
2. Fused K-quant dot, to lift Q4_K_M decode from 10.8 toward the Q8_0 figure.
3. A cached vocabulary sidecar, worth ~100 ms per repeated CLI invocation.
4. Large pages, once `SeLockMemoryPrivilege` is available.
