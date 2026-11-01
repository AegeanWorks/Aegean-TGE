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

**Read every absolute number as Ã‚Â±20 percent.** The machine runs with 2-4 GB free
of 16; VS Code, Defender and Memory Compression hold the rest. Two back-to-back
runs of the same binary differ by up to 29 percent. Ratios measured within one
run are reliable; absolute values across sessions are not.

A baseline snapshot: `.\scripts\measure.ps1`. A change is measured by diffing
one before and one after, repeated two or three times when the delta is small.

## Change log

| Date | Change | Metric | Before | After | Verdict |
|---|---|---|---|---|---|
| 2026-11-01 | Packed panels, 144/256/256 blocking, work-stealing pool | GEMM 2048Ã‚Â³, 16 threads | 18.2 GFLOPS | 246.5 GFLOPS | **kept** |
| 2026-11-01 | Pad packed A stride from 6 to 8 floats | GEMM 1024Ã‚Â³ | 62.0 GFLOPS | 52.7 GFLOPS | reverted |
| 2026-11-01 | Software prefetch in GEMV, distance 512 | GEMV 4096Ã‚Â², 16 threads | 32.1 GB/s | 28.4 GB/s | reverted |
| 2026-11-01 | `_MM_HINT_NTA` on that prefetch | GEMV 4096Ã‚Â², 16 threads | 28.4 GB/s | 29.8 GB/s | reverted (noise) |
| 2026-11-01 | Dedicated threaded GEMV decode kernel | GEMV 4096Ã‚Â² | 18.4 GB/s (1T) | 35.4 GB/s (16T) | **kept** |
| 2026-11-01 | Decode threads: all logical to P-cores | decode 0.5B Q8_0 | 21.8 t/s (16T) | 32.5 t/s (6T) | **kept** |
| 2026-11-01 | Per-matrix thread sizing for decode | decode 0.5B Q8_0 | 32.8 t/s | 27.5 t/s | reverted |
| 2026-11-01 | Batched prefill (weight row reused across tokens) | prefill, 510 tokens | 11 t/s | 170.9 t/s | **kept** |
| 2026-11-01 | Prefill token group 4 to 6 | prefill, 510 tokens | 170.9 t/s | 191.6 t/s | **kept** |
| 2026-11-01 | Align 64-token chunk to the group | prefill, 510 tokens | 191.6 t/s | 185.4 t/s | reverted |
| 2026-11-01 | Integer `maddubs` prefill inner loop | prefill, 510 tokens | 186.0 t/s | 150.0 t/s | reverted |
| 2026-11-01 | AVX-VNNI `dpbusd` prefill inner loop | prefill, 510 tokens | 186.0 t/s | 150.9 t/s | reverted |
| 2026-11-01 | Skip metadata arrays while loading the model | load, id prompt | 65 ms | 24 ms | **kept** |
| 2026-11-01 | Same, text prompt | load + tokenizer | 207-219 ms | 183-205 ms | **kept** |
| 2026-11-01 | Stop at end-of-sequence/end-of-turn | ChatML, `--n 128` | 127 tokens, 75 after the end token, invented turn | 52 tokens, stops at 151645 | **kept** |
| 2026-11-01 | Match GGUF control tokens in text | `<\|im_start\|>user` ids | 8 | 2 (151644 872) | **kept** |
| 2026-11-01 | Skip control tokens when detokenizing | ChatML `--text` | `2+2 equals 4.<\|im_end\|>` | `2+2 equals 4.` | **kept** |
| 2026-11-01 | Nucleus draw over the kept mass | 1M draws, p=(.50,.30,.15,.05), top-p 0.9 | .500/.299/.200/0 | .527/.315/.158/0 | **kept** |
| 2026-11-01 | Penalty once per distinct token, window 64 | logit 2.0, 10 occurrences | 0.77 | 1.82 | **kept** |
| 2026-11-01 | No full sort when no cut is requested | 151,936 logits, T=0.8 | 16.75 ms/draw | 0.61 ms/draw | **kept** |
| 2026-11-01 | Precompute RoPE sin/cos per position | trig pairs per token | 12,288 | 0 | **kept** |
| 2026-11-01 | Spin-then-sleep fork-join pool | empty `parallel_for`, 4T/6T | 39.4/64.3 us | 0.51/0.80 us | **kept** |
| 2026-11-01 | Same, end-to-end decode | short-context decode | 31.2/28.1 t/s | 36.9/43.1 t/s | **kept** |
| 2026-11-01 | Parallel attention, heads and chunk tokens | 2048-token prefill | 63.3/78.1 t/s | 144.0/203.6 t/s | **kept** |
| 2026-11-01 | Same, decode at 2048 context | decode | 20.3/19.9 t/s | 22.9/23.8 t/s | **kept** |
| 2026-11-01 | Same, short-context decode | decode | 36.9/43.1 t/s | 41.9/47.3 t/s | **kept** |
| 2026-11-01 | Reference pre-tokenizer splits | `x = y + z`, `[i`, `.com` ids | 7 / 2 / 2 | 5 / 1 / 1 | **kept** |
| 2026-11-01 | Match USER_DEFINED tokens and allow opt-out | `<tool_call>` ids | 5 | 1 | **kept** |
| 2026-11-01 | Stop on the end-of-generation set | Qwen eos/eot/bos | eos+eot only | adds 151643 and `eos_token_ids` | **kept** |
| 2026-11-01 | top-p/min-p without a full sort | 151,936 logits, T=0.8 | 18.2 / 19.2 ms | 4.1 / 4.1 ms | **kept** |
| 2026-11-01 | Teacher-forced fidelity harness | 1023 positions, context 1024 | no harness | mean KL 6.0e-12, top-1 1023/1023 | **kept** |
| 2026-11-01 | Vectorised attention (AVX2 dot + value sum) | 2048-token decode | 21.97 t/s | 31.38 t/s | **kept** |
| 2026-11-01 | Same, 2048-token prefill | prefill | 182.0 t/s | 218.5 t/s | **kept** |
| 2026-11-01 | GQA-aware decode attention (position chunks) | 2048-token decode | 35.11 t/s | 40.34 t/s | **kept** |
| 2026-11-01 | Same, fidelity against scalar | 511 positions | â€” | mean KL 5.0e-12, top-1 511/511 | **kept** |
| 2026-11-01 | Re-measured against llama.cpp build 11624 | 6 threads, Q8_0 | old log: llama tg128 15.2 t/s | TGE tg128 45.5 vs llama 45.1 t/s; prefill 261-266 vs 352; parity ~95 tokens | reference |
| 2026-11-01 | Per-session thread pools for concurrent sessions | 4 sessions × 6 threads | refused / 0xC0000005 | 46.2 t/s aggregate, transcripts identical | **kept** |
| 2026-11-01 | Prompt-lookup speculative decoding (greedy) | 128-token story prompt | 30.10 t/s | 49.92 t/s (1.66x), ids identical | **kept** |
| 2026-11-01 | Token-outer group kernel + prefill GQA | 2048 prefill / decode | 157.3 / 28.61 t/s | 140.2 / 24.88 t/s | **reverted** (2k prefill -5%, decode -13%) |
| 2026-11-01 | Speculation pins the per-head schedule | 2048, 64 tokens | normal head 31.47 t/s | spec 32.71 t/s, ids equal to head and auto | **kept** |
| 2026-11-01 | Session benchmark tail forward | 4x32 tokens, 6 threads | wall 3.746 s, 34.17 t/s | wall 2.329 s, 53.24 t/s | **kept** |
| 2026-11-01 | Story speculative decode (recheck) | 128 tokens | 28.13 t/s | 56.38 t/s (2.00x), ids identical | **kept** |
| 2026-11-01 | VNNI Q8 batch, pre-quantized activations, `--vnni` | 510 / 2048 prefill | 163.1 / 165.6 t/s | 194.5 / 188.2 t/s (+19% / +14%) | **kept opt-in** (KL 1.568e-3, top-1 485/497) |
| 2026-11-01 | f16 KV cache (`--kv f16`) | 2048 prefill / decode | 145.7 / 21.7 t/s | 165.9 / 35.4 t/s | **kept opt-in** (KL 1.658e-6, top-1 497/497, ids identical) |
| 2026-11-01 | Continuous batching (4 KV slots, 1 model) | 8 requests × 32 tokens | sessions 42.21 t/s aggregate | 58.28 t/s (+38%), ids identical | **kept** |

## 1. f32 GEMM

Reference kernel is the scalar oracle; `avx2-fma` is the shipped kernel.

| Shape | Reference (1T) | Shipped (1T) | Shipped (16T) |
|---|---|---|---|
| 512Ã‚Â³ | 16.8-17.2 GFLOPS | 51.4-75.9 | 120.6-122.5 |
| 1024Ã‚Â³ | 17.4-20.6 | 62.0-66.8 | 184.9-206.6 |
| 2048Ã‚Â³ | 3.2-8.7 | 27.2-43.3 | 180.4-246.5 |

Best observed 246.5 GFLOPS at 2048Ã‚Â³; the same config measured 180.4 later under
heavier background load. The reference kernel's own variance (3.2 to 8.7) shows
how much the machine moves.

### Ablation: which of the three changes did it

The 18.2 to 246.5 entry bundles three changes *and* a 1-to-16-thread comparison.
`bench gemm --kernel unpacked|unblocked|packed` separates them.

| 2048Ã‚Â³, GFLOPS | 1 thread | 16 threads |
|---|---|---|
| unpacked (no packing, no blocking) | 18.80 | n/a (control is single-threaded) |
| + packing, no blocking | 43.16 | 33.14 |
| + blocking (shipped) | 27.23 | **180.37** |

| 512Ã‚Â³, GFLOPS | 1 thread | 16 threads |
|---|---|---|
| unpacked | 72.75 | n/a |
| + packing, no blocking | 33.59 | 24.36 |
| + blocking (shipped) | 51.36 | **122.51** |

- **The pool is the largest single factor**: 27.23 to 180.37 at 2048Ã‚Â³, 6.6x.
- **Blocking is the precondition for it.** Without blocking, more threads make
  the kernel *slower* (43.16 to 33.14): the panel spans the whole matrix and
  every worker re-reads it from DRAM. Blocking alone is a single-thread
  *regression* (43.16 to 27.23) and only pays at 16 threads.
- **Packing trades small for large**: 18.80 to 43.16 at 2048Ã‚Â³, but 72.75 to
  33.59 at 512Ã‚Â³. Blocking recovers most of the small-shape loss.

## 2. GEMV f32 - the decode primitive

| Matrix | 1 thread | 16 threads |
|---|---|---|
| 4096Ã‚Â² (64 MiB) | 18.4 GB/s (15.9-20.8) | **35.4 GB/s** (29.6-35.7) |
| 8192Ã‚Â² (256 MiB) | 13.4 GB/s | 28.7 GB/s |

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

> Superseded by section 19. The installed llama.cpp build 11624 measures
> 45.1 t/s tg128; the a91eab6 build below measured 15.2-17.9.

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

> Superseded by section 19. Build 11624 measures 352.2 t/s pp512 and 287.9
> pp2048; the a91eab6 numbers below are historical.

510-token prompt, 64-token chunks.

| Engine | Prompt (pp512) |
|---|---|
| TGE | **170.9 then 191.6 t/s** (before and after the token-group change) |
| llama.cpp | 285.3 Ã‚Â± 22.8, and **220.7 Ã‚Â± 3.8** on a later run |

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
the dot. `bench batch --format q4k` isolates why. At 896x4864:

| Kernel | 64 tokens | 1 token |
|---|---|---|
| Q8_0 | 369.3 GFLOPS, 3.07 GB/s | 41.1 GFLOPS, **21.85 GB/s** |
| Q4_K | 198.7 GFLOPS, 0.87 GB/s | 18.4 GFLOPS, **5.18 GB/s** |

Decode is the damning case: Q4_K sustains 5.18 GB/s against a ~28 GB/s wall, so
it runs at **18 percent of available bandwidth** while Q8_0 reaches 21.85. It is
compute-bound on the dequantisation, not limited by how few bytes it reads -
which is why a 28 percent smaller file decodes three times slower. A fused
K-quant dot has to make the unpack cheaper, not just avoid the scratch buffer.

## 7b. Vectorised Q4_K dequant (+29.6%, kept)

The dequant was a scalar loop in a non-AVX2 translation unit. It is now explicit
AVX2: nibble split, `cvtepi8_epi32`, multiply then subtract - never fused, so the
result is bit-identical to the scalar path rather than merely close.

| 896x4864 Q4_K | before | after |
|---|---|---|
| 1 token | 18.41 GFLOPS, 5.18 GB/s | **23.87 GFLOPS, 6.71 GB/s** |

**+29.6 percent on decode**, and token parity on 1.5B Q4_K_M is unchanged:
`12095 13 576 6722 315 15344`, " Paris. The capital of Italy".

This was attempted once before and reverted, because it produced
`23990 92795 34163 7 79368 95880` instead of that. The bug: the scalar path packs
the **low** nibbles of 32 values into all 32 bytes of the chunk, then the same 32
bytes' high nibbles into the next 32. The first attempt split the 16-byte range
between the two halves instead, so the scales were right and the values were in
the wrong order - plausible enough to survive a glance.

Reopening it properly meant writing the differential test first: AVX2 against
scalar on identical bytes, `memcmp` for exact equality, at one block and across
three. That test now guards the kernel, and it is why this attempt could be
trusted in one pass.

The path is still 6.71 GB/s against Q8_0's 21.85, so the dequant was not the
whole cost - the remaining gap is the per-row scratch round-trip and the dot.
Fusing the two is the next step, and it needs the same treatment: test first,
then time.

**No end-to-end gain is claimed.** Decode on 0.5B Q4_K_M measured 10.55 and 6.63
t/s in consecutive runs on the same binary, against a logged 10.8 before the
change - indistinguishable. The Q8_0 control fell from ~30 to 21.34 t/s in the
same session, which is the size of the noise: the machine was loaded. A kernel
worth 29.6 percent in isolation is invisible here, and stating that is more
useful than quoting whichever run looked best.

## 7c. Concurrent sessions

0.5B Q8_0, 32 tokens each, context 2048, greedy. The pool is per session
(section 20), and the timer starts after every session has warmed it.

| Sessions | Threads each | Per session | Aggregate |
|---|---|---|---|
| 1 | 6 | 37.42 t/s | 37.42 t/s |
| 2 | 6 | 23.68 t/s | 47.36 t/s |
| 4 | 1 | 10.75 t/s | 43.00 t/s |
| 4 | 6 | 11.55 t/s | 46.20 t/s |

Aggregate plateaus near 47 t/s: one session on six threads leaves bandwidth
unused, and additional sessions fill it until the weight stream is the wall.
The earlier 1 x 6 figure (17.37) was dominated by pool creation and thread
start-up, which the benchmark now warms outside the timed region. All
transcripts are identical, including the >1-thread rows that used to crash.

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

> Build 11624, the current installed baseline, is in section 19. These a91eab6
> numbers are historical.

Built from source at `a91eab6` (ggml 0.28.0), GCC 16.2, `-mavx2 -mfma -mf16c`.

| Model | Prompt (pp512) | Decode (tg128) |
|---|---|---|
| qwen2.5-0.5B Q4_K_M | 407.6 Ã‚Â± 29.7 t/s | 8.73 Ã‚Â± 2.30 t/s |
| qwen2.5-0.5B Q8_0 | 278.6-285.3 t/s | 10.1-17.9 t/s |
| qwen2.5-1.5B Q4_K_M | 207.6 Ã‚Â± 0.7 t/s | 7.22 Ã‚Â± 0.75 t/s |

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
**56 percent of its bandwidth bound** Ã¢â‚¬â€ the other 44 percent is attention,
normalisation and the per-call barriers, not memory.

**The loader now applies the same rule** rather than only reporting it: asking
for 32768 tokens on the 1.5B Q4_K_M model loads at 14178 with a printed notice,
instead of allocating and failing. Cost of consulting the plan: load went from
24.0 to 28.5 ms warm (+4 ms). A cold file reads 232 ms once.

Analysis costs **2.3 ms warm**, 50 ms on a cold file (the first read of the
metadata region faults it in). Either way it is far cheaper than loading a model,
and it makes "does this fit?" answerable before committing to an allocation Ã¢â‚¬â€
which is the whole point, since the failure it prevents is allocating first and
discovering afterwards.

## 11. Correctness checks that back these numbers

- Greedy output matched llama.cpp token for token on the pre-vectorisation
  arithmetic. On the current schedules it matches for ~95 of 128 tokens and
  then diverges at a near-tie; section 19 records the current comparison.
- Prefill matches across chunk boundaries (136-token prompt, two boundaries).
- Q4_K/Q5_K/Q6_K/Q5_0/Q4_0 dequantizers agree with their scalar references.
- Batch kernels match scalar references across token counts hitting both the
  group and its tail.
- Tokenizer encoding matches `llama-tokenize` on punctuation, decimals, percent
  signs, contractions and double spaces.
- 53 unit cases, all passing.

## 12. Rejected, with margins

Recorded so the same idea is not rebuilt on intuition. Four of these six looked
correct on paper.

| Change | Lost by | Reason, where known |
|---|---|---|
| Pad packed A stride 6 to 8 | 1024Ã‚Â³ 62.0 to 52.7 GFLOPS | 33 percent more panel traffic beat fewer line splits |
| Software prefetch in GEMV | 32.1 to 28.4 GB/s | hardware prefetcher already covers a contiguous band |
| `_MM_HINT_NTA` on it | 28.4 to 29.8, inside noise | no measurable difference |
| Align chunk to token group | 191.6 to 185.4 t/s | measured worse; tail stays on the single-token path |
| Integer `maddubs` prefill | 186.0 to 150.0 t/s | `psignb` contends for the single port that issues it |
| AVX-VNNI `dpbusd` prefill | 186.0 to 150.9 t/s end to end | faster per kernel (284/342 GFLOPS) but per-call activation quantization costs more than it saves |

VNNI is the instructive one: it separates the kernel from the integration
cleanly. The kernel is genuinely faster on the two large shapes; the model calls
it ~150 times per chunk and the fixed cost per call eats the gain.

## 13. What to measure next

1. A profiler. Four performance hypotheses here were wrong because
   instruction counting was used instead of port or memory analysis. WPA and
   AMD uProf are not installed.
2. Fused K-quant dot, to lift Q4_K_M decode from 10.8 toward the Q8_0 figure.
3. A cached vocabulary sidecar, worth ~100 ms per repeated CLI invocation.
4. Large pages, once `SeLockMemoryPrivilege` is available.

## 14. Quality handoffs (2026-11-01)

The engine matches its arithmetic reference; these defects were at the other
handoffs. Each was reproduced before the change, on the 0.5B Q8_0 file.

### Termination: stop where the model stops

A ChatML question was answered in 52 tokens ending in `151645` (`<|im_end|>`).
The old loop fed that id back and generated 75 more tokens, inventing a
`Human:` turn. `generate` now stops on the model's `eos_token_id`/`eot_token_id`,
which `read_config` takes from `tokenizer.ggml.eos_token_id` and
`tokenizer.ggml.eot_token_id`.

| ChatML prompt, `--n 128` | before | after |
|---|---|---|
| generated tokens | 127 | 52 |
| tokens after the first end id | 75 | 0 |
| invented next turn | yes | no |

### Input: control tokens are one id

`<|im_start|>user` tokenised to `27 91 318 62 2468 91 29 872` (8 ids) because
the tokenizer never consulted `tokenizer.ggml.token_type`. Encoding now matches
CONTROL entries literally before BPE.

| prompt | before | after |
|---|---|---|
| `<\|im_start\|>user` | 8 ids | 2 ids (151644 872) |
| `<\|im_start\|>user<\|im_end\|><\|im_start\|>assistant` | 14 ids | 5 ids (151644 872 151645 151644 77091) |

### Output: control tokens are not text

`--text` printed the literal control token because decoding treated every
vocabulary entry as text. Entries marked CONTROL are now skipped, matching the
reference tokenizer's default; `--print-ids` still shows them.

| ChatML answer | before | after |
|---|---|---|
| `--text` | `2+2 equals 4.<\|im_end\|>` | `2+2 equals 4.` |

### Selection: nucleus mass and one penalty per token

A/B of the old and new sampler on 1M draws from p=(.50,.30,.15,.05), top-p 0.9:

| | token 0 | token 1 | token 2 | excluded tail |
|---|---|---|---|---|
| before | .500 | .299 | **.200** | .000 |
| after | .527 | .315 | **.158** | .000 |
| renormalised target | .526 | .316 | .158 | 0 |

The old draw used the total mass, so the 0.05 tail fell through to the weakest
kept token. The draw now uses the kept mass.

The repetition penalty applied once per occurrence over the whole history, so a
logit of 2.0 became 0.77 after ten occurrences. It now penalises each distinct
token once within a 64-token window: 2.0 -> 1.82. `--penalty-window N` exposes
the window; 0 means the whole history.

With a temperature and no top-k/top-p/min-p the sampler sorted all 151,936
entries. That path now exponentiates and scans in index order, no ordering.

| 151,936 logits, T=0.8 | before | after |
|---|---|---|
| time per draw | 16.75 ms | 0.61 ms |

### RoPE: one table instead of per-head trigonometry

`apply_rope` computed `cos`/`sin` for every head in every layer: 12,288 pairs
per token for this model. The table is built once at load (context x head_dim/2)
and indexed by position. The values are the same `std::cos`/`std::sin` results
on the same angles, so greedy output is unchanged; decode was 32.3 t/s after
against 29.9-31.5 before, which is inside this machine's noise and is not
claimed as a gain.

### Tests

48 -> 53 cases. New cases cover the nucleus mass, the once-per-token penalty,
the penalty window, control-token encoding and the end-token ids.

## 15. Thread pool and attention parallelism (2026-11-01)

### Spin-then-sleep fork-join

The pool parked every worker on a condition variable for each of the 169
parallel calls in a decode step. It now publishes an atomic generation, spins
for a bounded number of pauses, and parks only when a call is long enough to
make spinning wasteful. `parallel_for` with an empty body, 50,000 calls:

| threads | before | after |
|---|---|---|
| 4 | 39.4 us | 0.51 us |
| 6 | 64.3 us | 0.80 us |

End-to-end, same 11-token prompt and 128 generated tokens:

| | before | after |
|---|---|---|
| decode | 31.2 / 28.1 t/s | 36.9 / 43.1 t/s |
| ids | identical | identical |

The first version of the handshake used release/acquire on the sleeper count
and the generation. That is a store-buffer pattern: on x86 both sides can read
stale values, so a parked worker missed a generation and the caller waited
forever. It reproduced reliably at 15 workers and not at 3; the handshake is now
sequentially consistent, and `thread_pool_many_threads_survives_many_rounds`
covers it with 16 workers and 200 rounds.

### Parallel attention

Attention ran all query heads on the calling thread. It now writes one score
slice per task (per head for decode, per chunk token for prefill) and spreads
the tasks over the pool. Below 64 positions of history the decode path keeps the
heads inline, because the pool round trip costs more than the work.

| 2048-token prompt | before | after |
|---|---|---|
| prefill | 63.3 / 78.1 t/s | 144.0 / 203.6 t/s |
| decode at 2048 | 20.3 / 19.9 t/s | 22.9 / 23.8 t/s |

| short context (11-token prompt) | before | after |
|---|---|---|
| decode | 36.9 / 43.1 t/s | 41.9 / 47.3 t/s |

The 510-token prompt measured 152.3 / 198.7 t/s prefill and 22.0 / 20.7 t/s
decode after both changes. Greedy ids are unchanged on the 127-token run and on
the ChatML end-to-end check. The machine was heavily loaded during some later
runs (11-token prefill of 1 s against 0.14 s idle), so short-context numbers are
stated as measured pairs rather than as a single figure.

Tests: 53 -> 54 cases, adding the 16-worker pool stress case.

## 16. Input, termination and sampling cuts (2026-11-01)

The second audit's remaining exact defects, each reproduced against 7a500c7.

### Input: the reference splitter's two optional-character rules

`x = y + z` tokenised to 7 ids because a space before a punctuation run was its
own piece, and `[i`, `.com` took two ids because one non-letter, non-digit
character did not lead the following letter run. Both rules are now implemented
in the pre-tokenizer.

| string | before | after |
|---|---|---|
| `x = y + z` | 7 ids | 5 ids |
| `[i` | 2 ids | 1 id |
| `.com` | 2 ids | 1 id |
| `<tool_call>` | 5 ids | 1 id |

USER_DEFINED entries are matched with CONTROL entries, and
`Tokenizer::encode(text, false)` leaves both as ordinary text so a server can
tokenise user content without letting it forge a template or a tool call.

### Termination: the end-of-generation set

The config now collects the file's eos and eot ids, every id in
`tokenizer.ggml.eos_token_ids`, and for Qwen the BOS id, which is
`<|endoftext|>` (151643 here) and a valid end-of-text token. Previously only
151645 could end a turn.

### Sampling: no full sort for top-p or min-p

Both cuts sorted all 151,936 entries. The sampler now collects the tokens that
could survive the cut, lowers the floor geometrically until a top-p nucleus fits
inside the collected set, and sorts only that set. At a floor of zero every
token is a candidate, which is the exact full-sort fallback.

| 151,936 logits, T=0.8 | before | after |
|---|---|---|
| top-p 0.95 | 18.2 ms per draw | 4.1 ms |
| min-p 0.05 | 19.2 ms per draw | 4.1 ms |

End to end, an 11-token prompt and 64 generated tokens: top-p 0.95 ran at
47.1 t/s and min-p 0.05 at 49.0 t/s, against 24-25 t/s with the old sampler
(measured on a different prompt, so treat that pair as directional; the
151,936-logit pair above is the paired measurement).

Tests: 54 -> 55 cases.

A later regression covers the top-p floor-lowering path; the suite is at 58
cases.

## 17. Fidelity harness and vectorised attention (2026-11-01)

### The harness

`tge fidelity --model PATH --corpus PATH` runs two models in lockstep,
teacher-forced over the corpus: one forced to the scalar attention kernel and
one on the dispatch-selected kernel. At each position it reports mean and worst
KL in double precision, the largest logit difference, and top-1 agreement.
`generate --attention scalar|auto` provides the same escape hatch for A/B
timing.

### Vectorised attention

`attention_head_scalar` is the oracle: the exact sequential loops the engine
used. `attention_head_avx2` vectorises the query-key dot product and the
weighted value sum with FMA. The softmax exponential stays scalar in both
paths, because a vectorised approximation changes the distribution and needs
its own price first.

| 1023 positions, context 1024 | value |
|---|---|
| mean KL | 6.0e-12 nats |
| worst KL | 9.1e-10 nats |
| max logit difference | 1.8e-4 |
| top-1 agreement | 1023/1023 |
| 256 free-running greedy tokens | identical |

| 2048-token prompt | scalar | AVX2 |
|---|---|---|
| prefill | 182.0 t/s | 218.5 t/s |
| decode | 21.97 t/s | 31.38 t/s |

The kernel is selected by dispatch (`tge info` prints it) and the scalar path
remains the reference the harness compares against.

Tests: 55 -> 56 cases, adding the attention differential across spans and head
widths.

## 18. GQA-aware decode attention (2026-11-01)

Decode split 14 query heads over 6 workers and re-read each KV row once per
query head. `attention_gqa` splits each KV group's positions into chunks; every
task reads each K and V row once for the seven query heads that share it and
keeps an online softmax (running maximum, running sum, running value sum) per
head. The caller merges the chunk states. `Auto` uses it from 256 positions of
history; below that the per-head path is cheaper. `--attention head|gqa` forces
either for A/B timing, and `scalar` remains the oracle.

| 2048-token decode | per-head | GQA |
|---|---|---|
| decode | 35.11 t/s | 40.34 t/s |

The online softmax is priced by the harness, not assumed: 511 positions gave
mean KL 5.0e-12, worst 1.4e-10, max logit difference 1.4e-4, top-1 511/511, and
256 free-running greedy tokens were identical to the per-head schedule.

Tests: 56 -> 57 cases, adding the grouped-kernel differential.

## 19. llama.cpp comparison, build 11624 (2026-11-01)

The installed llama.cpp is build 11624 (commit af0579c78), CPU backend
`alderlake`, which is much faster than the a91eab6 build the earlier sections
measured. Same file (`qwen2.5-0.5b-instruct-q8_0.gguf`, 638.74 MiB), same
machine, 6 threads, greedy. TGE runs with `--threads 6 --attention auto`;
llama-bench is `-t 6 -ngl 0`. llama-cli 11624 has no raw-completion switch; it
wraps every prompt in the model's chat template, so the output comparison used a
pass-through template, `{{ messages[0]['content'] }}`, to give both engines the
same raw text.

| test | TGE | llama.cpp |
|---|---|---|
| pp512 | 261.4 / 265.6 t/s | 352.17 +/- 2.48 |
| tg128 | 45.04 / 45.94 t/s | 45.10 +/- 1.22 |
| pp2048 | 225.2 / 246.1 t/s | 287.93 +/- 9.30 |
| tg32 at 2048 | 38.33 t/s (GQA) | 46.92 +/- 2.17 |

TGE is at decode parity at short context, at ~75-85% of llama's prefill, and at
~82% of its 2k decode. llama-bench uses an f16 KV cache by default while TGE
keeps f32, and llama.cpp's CPU path quantises activations to 8 bits before its
dot products (ADR-0006). Both differences favour llama on speed.

### Output parity

A 128-token greedy run from the same raw prompt is identical for the first ~95
tokens, then a near-tie flips: llama continues "and a constant", TGE "with a
variable". Both are plausible; after that token the traces separate. The
fidelity harness prices TGE's arithmetic at mean KL 5.0e-12 against the scalar
reference, while llama.cpp's 8-bit activation path is a larger deviation from
exact f32. The token-identical claims in sections 4, 5, 7 and 11 were measured
on the pre-vectorisation arithmetic and the older llama build; they are
superseded here, not withdrawn as observations.

## 20. Per-session thread pools for concurrent sessions (2026-11-01)

`acquire_pool` was a process-wide registry keyed by size. Two sessions on
different threads that asked for the same size received the same `ThreadPool`;
both published into its `body_`/`count_`/`grain_` and corrupted each other
(0xC0000005). The sessions benchmark refused that combination.

The pools are now owned by the model: `Qwen2` keeps one decode pool and one
prefill pool, and `GemvConfig`/`GemmConfig` carry a `pool` pointer through every
kernel. The process-wide registry remains only as a fallback for the standalone
benchmarks and tests, which never run two sessions. Nothing is stored in
thread-local storage, so no worker join happens from a TLS destructor.

The sessions benchmark also warms each pool and starts the timer after a
barrier, so pool creation and thread start-up are no longer counted as decode
time. The earlier 1 x 6 figure was mostly start-up.

| sessions | threads each | per session | aggregate |
|---|---|---|---|
| 1 | 6 | 37.42 t/s | 37.42 t/s |
| 2 | 6 | 23.68 t/s | 47.36 t/s |
| 4 | 1 | 10.75 t/s | 43.00 t/s |
| 4 | 6 | 11.55 t/s | 46.20 t/s |

All transcripts identical (same seed), including the rows above one thread per
session. The q8 kernel tests now pass an explicit pool, and the standalone
benchmarks keep using the fallback registry.

Tests: 57 -> 58 cases; the explicit pool path is covered by the q8 kernel tests
and the sessions benchmark.

## 21. Prompt-lookup speculative decoding (2026-11-01)

The verifier is the batched prefill path. `verify_draft` feeds five drafted
tokens plus one padding token at the six-token kernel group, computes logits for
all six rows with the batched output projection, and returns the greedy argmax
after each position. The key/value rows are committed to the cache; a rejected
suffix is overwritten by later tokens, so no rollback pass is needed.

The drafter needs no second model: it finds the latest earlier match of the last
one to three tokens in prompt+output and proposes the five tokens that followed
it. Speculation runs only for pure greedy (temperature 0, repetition penalty 1);
a sampled run takes the normal path. It also pins the per-head attention
schedule (section 22) so the verifier and a sequential decode agree.

Story prompt, 128 tokens, 6 threads, greedy:

| | decode | speculation |
|---|---|---|
| normal | 30.10 t/s | — |
| speculative | 49.92 t/s | 20 steps, 79 draft tokens accepted (3.95/step) |
| recheck after the pin | 28.13 t/s | 56.38 t/s (2.00x), 99 tokens emitted |

Ids are identical over all 127 decode steps. The short ChatML answer (8 tokens)
also matches, but speculation costs slightly there: 2 steps, 3 accepted, 24.56
vs 27.15 t/s. It is an explicit `--speculative` flag, not the default.

On a 2048-token prompt the pinned run measures 32.71 t/s against 31.47 for the
per-head run and 37.24 for auto; all three id traces agree over 63 tokens. The
pin costs the grouped schedule's advantage at long context, which is why the
flag is opt-in.

Tests: 58/58; parity is checked end to end on the story and ChatML prompts.

## 22. Prefill-side GQA attempt, verifier pin, session tail (2026-11-01)

Reviewing the uncommitted speculation work and the last three pushes surfaced
three defects:

1. The verifier ran the per-head prefill attention while decode switches to GQA
   at 256 positions. Below 256 the schedules coincide; above it they do not, so
   a near tie could accept a draft the sequential decode would not.
2. `attention_group` looped heads outside positions, so its comment ("each K/V
   row once") was false: every head re-read the same rows.
3. The sessions benchmark forwarded once after its last token, wasting one
   forward per session and mixing the numerator with the timed work.

The first was implemented as prefill-side GQA: tokens at `position >= 256`
dispatch to the same chunked GQA kernel as decode, and the group kernel was
transposed so a position reads its K/V row once for the group. That was
measured and rejected.

| run (6 threads) | before | token-outer + prefill GQA | final |
|---|---|---|---|
| 510 prefill | 244.4 t/s | 170.7 auto / 128.6 head | 166.0 / 254.6 |
| 510 decode (auto / head) | 33.98 / 34.76 | 31.26 / 34.16 | 38.52 / 42.56 |
| 2048 prefill | 157.3 t/s | 140.2 auto / 148.2 head | 200.4 / 170.1 |
| 2048 decode (auto / head) | 28.61 / 18.24 | 24.88 / 28.38 | 39.37 / 29.95 |

The 510 prefill gained 33 percent over the per-head run, but the 2048 prefill
lost 5 percent and the decode lost 8-13 percent against the per-head run. At
these lengths the K/V rows already live in L2; the token-outer state reads and
merges cost more than the traffic they saved. The kernel stays head-outer, and
its comment now says what it does. The absolute columns move with machine load
(the final runs were the fastest run of the session), so the within-run
auto/head pairs are the evidence, not the columns.

The verifier gap is closed in the CLI instead: `--speculative` pins the model to
the per-head schedule (scalar keeps its scalar path), so the verifier and a
sequential decode take identical arithmetic. Normal auto keeps the grouped
schedule.

| 2048 prompt, 64 tokens | t/s | ids |
|---|---|---|
| normal auto (GQA) | 37.24 | reference |
| normal head | 31.47 | reference |
| speculative (pinned head) | 32.71 | equal to both |

Speculation is 4 percent ahead of the per-head run and 12 percent behind auto at
2048. The story prompt remains the win: 28.13 -> 56.38 t/s (2.00x this run;
1.66x earlier), 20 steps, 79 accepted, 99 emitted, ids identical.

The session tail is gone: the loop forwards only when another token is due.

| 4 x 6 threads, 32 tokens | before | after |
|---|---|---|
| wall | 3.746 s | 2.329 s |
| per session / aggregate | 8.54 / 34.17 t/s | 13.31 / 53.24 t/s |

Part of that is machine load, but the tail forward is gone and the rate uses the
same `tokens - 1` the normal decoder reports. All transcripts remain identical.

Fidelity is unchanged by the reverted kernel: 497 positions, mean KL 5.104e-12,
worst 1.426e-10, top-1 497/497. Tests: 58/58; gates: 4/4.

## 23. AVX-VNNI with pre-quantized activations (2026-11-01)

The report's VNNI lost end to end because `gemv_q8_batch_vnni` quantized the
activation rows again on every call, and the model calls it six to seven times
per layer. The model now quantizes each distinct activation once (q/k/v share
`batch_h_`, gate/up share it after the FFN norm, and so on) and hands the
pre-quantized rows to the kernel. `--vnni` selects it and
`tge fidelity --vnni` prices it.

| run (6 threads) | f32 | `--vnni` |
|---|---|---|
| 510 prefill | 163.1 t/s | **194.5 t/s (+19%)** |
| 2048 prefill | 165.6 t/s | **188.2 t/s (+14%; a later run 155.5, inside noise)** |
| 510 decode | 36.88 t/s | 26.20 t/s |
| 2048 decode | 36.33 t/s | 36.93 t/s |
| 2048 speculative prefill | 149.9 t/s | **178.1 t/s (+19%)** |

Pre-quantizing flips the prefill from the report's loss to a win. The decode's
single-token path is not a win (one 510 run lost 29 percent), so the option is
for prompt processing.

The accuracy price is real: against the scalar/f32 reference over 497 positions,
`--vnni` gives mean KL **1.568e-3**, worst 2.254e-2, max |logit| 0.754, and
top-1 **485/497** (12 flips). Free-running ids differ at 128 tokens, at 2048 and
under speculation. That is the ADR-0006 parity trade, now measured rather than
assumed.

The VNNI is therefore opt-in and the default stays f32. The 2048 prefill gains
14 percent on the best run (still about 65 percent of llama.cpp's 288 t/s); the
default's output and numbers are unchanged. `tge info` reports whether the CPU
has AVX-VNNI; `--vnni` refuses to run without it.

## 24. f16 KV cache (2026-11-01)

The KV cache is now optionally IEEE half (`--kv f16`), read by the attention
kernels with F16C; the default stays f32. `tge fidelity --kv f16` prices it.

| run (6 threads) | f32 | f16 |
|---|---|---|
| 510 prefill | 139.7 t/s | **220.8 t/s (+58%)** |
| 510 decode | 35.5 t/s | 30.4 t/s |
| 2048 prefill | 145.7 / 154.7 t/s | 165.9 / 133.6 t/s |
| 2048 decode | 21.7 / 32.7 t/s | 35.4 / 24.9 t/s |
| story decode | 26.3 t/s | 22.8 t/s |
| 2048 speculative decode | 33.5 t/s | 29.4 t/s |

The absolute columns move with this machine (Â±20%); the f16 is not a uniform
speed win. Its case is memory and accuracy: the KV halves (192 -> 96 MiB at
8192 tokens), and against the f32 reference over 497 positions it gives mean KL
**1.658e-6**, worst 2.446e-5, max |logit| 5.3e-2 and top-1 **497/497**.
Free-running ids are identical on the 2048 prompt, the story prompt and under
speculation.

The default stays f32; `--kv f16` is for long-context or multi-session memory.
`--kv f16` refuses to run on a CPU without F16C, which `tge info` reports.
Tests: 58 -> 60 (half round-trip, f16 attention differential).

## 25. Continuous batching (2026-11-01)

The model now owns `sequences` KV slots and `forward_batch` decodes one token
per sequence in a single weight pass. `bench continuous` keeps N slots active,
admits a new request into each freed slot, prefills its seed token, and decodes
the active batch. The paged/mixed prefill-decode scheduler is not shipped.

| 8 requests x 32 tokens, 6 threads | aggregate |
|---|---|
| 4 concurrent sessions (4 models) | 42.21 t/s |
| continuous, 4 slots (1 model) | **58.28 t/s (+38%)** |
| single request | 26.29 t/s |

The first transcript is token-identical to `generate` on the same seed token,
and all eight are identical, so the per-slot KV cache does not leak. The 1.38x
over concurrent sessions comes from sharing one weight stream across the four
decodes; the wall is 4.256 s for 248 decode tokens. The prefill of a newly
admitted request still reads the weights separately, and the KV is not paged or
evicted.

Tests: 60/60 unchanged; the batch parity is checked through the CLI transcript.
`bench continuous` also accepts `--vnni` and `--kv f16` (the flags are guarded by
the CPU checks above), so the trades can be combined with batching.
