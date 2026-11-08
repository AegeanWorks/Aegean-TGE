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
| 2026-11-01 | Same, fidelity against scalar | 511 positions | — | mean KL 5.0e-12, top-1 511/511 | **kept** |
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
| 2026-11-02 | int8 KV cache (`--kv i8`) | Llama-3B Q4_K_M 2048 decode | 5.80 t/s | 7.63 t/s (+32% vs f32, +15% vs f16) | **kept opt-in** (KL 3.2e-4, top-1 31/31, text identical) |

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
| Pad packed A stride 6 to 8 | 1024³ 62.0 to 52.7 GFLOPS | 33 percent more panel traffic beat fewer line splits |
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

The absolute columns move with this machine (±20%); the f16 is not a uniform
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

A newly admitted request's prompt is now batched with the active decodes in the
same pass (`--prompt-tokens`). Eight requests of 32 tokens measure 68.98 t/s at
an 8-token prompt against 62.34 at 1 (+10.7%); the first transcript stays
token-identical to `generate` at 1, 8 and 32 prompt tokens. A 32-token prompt
falls to 54.08 t/s because the prompt rows crowd the 64-row batch. The KV is
still fixed per slot, not paged or evicted.

## 26. Report verification round: batched tails, P-core pin, weight prefetch (2026-11-01)

Verifies the 16-page token-creation report (master 7a500c7) against HEAD and
lands the three remaining true wins. Machine: i5-14400F, 6 threads, Q8_0.

### 26a. Batched kernel remainder (kept)

The report's open item "batched kernel efficient at 1 to 5 tokens" reproduced:
5 tokens cost *more* than 6, because the 1..5 tail re-read every weight row
once per token while a full group of 6 reads it once.

| shape | before 5 tokens | before 6 tokens | after 5 | after 6 |
|---|---|---|---|---|
| 4864x896 (FFN) | 0.432 ms | 0.239 ms (1.81x) | 0.321-0.376 ms | 0.315-0.364 ms (~1.0x) |
| 151936x896 (output) | 8.432 ms | 6.078 ms (1.39x) | 5.769 ms (-31.6%) | 5.701 ms (-6.2%) |

The tail is now a 1..5-wide batched group sharing one widened weight block,
the same arithmetic in the same block order, so the result is bit-identical.
Every 64-token prefill chunk is 10 full groups plus a 4-token tail, so every
chunk paid the old cost; the speculative verifier's 6-token pass never did.
`bench batch` at 1..6 plus all 60 unit tests (q8 differential included) are
the guard. Fidelity after the change: mean KL 6.1e-12, worst 9.6e-10, top-1
1023/1023 over docs/architecture.md.

End to end, short decode is inside machine noise (34.9 -> 32.3 t/s across
sessions, +/-20% on this box), while the output-projection tail dominates
large-vocabulary passes. Speculative story decode stays 1.55-2.0x (50.0 t/s
vs ~32 here, 56.4 vs 28.1 in section 21).

### 26b. Decode workers pinned to P-cores (kept, variance lever)

The report's "pin decode workers to distinct P-cores" was true: no affinity
was set anywhere, workers spun between calls. `ThreadPool` workers now pin
round-robin to the fastest efficiency class (logical 0 reserved for the
caller/OS), single-group only; larger topologies keep OS placement.
Literature: Olling (8 vs 16 threads on Ryzen/Apple), Gaessler (5 threads
saturate dual-channel DDR4, 29 on quad-channel Xeon), Sandwich (separate
prefill/decode plans, core-count tuning), llama.cpp `--cpu-mask/--cpu-range`.

Measured effect on this box is variance, not mean: the pool handoff is
already 0.5-0.8 us, so pinning does not move the 32-35 t/s short-decode mean
outside noise, but it removes one source of the +/-20% run-to-run spread
(migration across P/E). Claimed as a variance fix, not a speedup.

### 26c. Weight prefetch at load (kept, opt-in)

The report's cold 43 s first-prefill reproduced as a fault-in, not compute:
the GGUF is mmap'd, weights fault on first use. `File::prefetch()` touches
every 4 KiB page once (plus a `PrefetchVirtualMemory` hint on Windows where
available); `generate --prefetch` reports it explicitly.

| step | warm run here |
|---|---|
| load (map + parse) | 58-144 ms |
| prefetch 675 MiB | 489 ms (~1.4 GB/s, storage-bound) |
| first-token fault | moved out of prefill into the explicit step |

Total bytes are identical; predictability is the win. Default stays off so a
warm repeated CLI invocation does not pay 0.5 s; cold starts and benchmarks
pass `--prefetch`.

### 26d. Report arithmetic correction (docs only)

The report's correction stands: the log's section 6 "241 GFLOPS, ~91% of
prefill is the batch kernel" counts 2 x 630M weights x 191.6 t/s, i.e. the
embedding table plus a full output projection per prompt token. Prefill
multiplies ~358M weights per token, so the useful rate is ~137 GFLOPS and
the kernel is nearer 40% of prefill on the 14400F, the rest being norms, rope
and attention. Section 6 is superseded by this note; the raw GFLOPS stand.

### Verdicts on the report's remaining opens

- Fused K-quant dot: still true gap (6.71 vs 21.85 GB/s), still not fused.
  Next after this round; needs the same differential-test-first treatment.
- Vectorised exp in softmax: stays scalar. Literature (QuAKE bit-trick,
  exllamav2 `exp256_ps`, ggml-cuda `__expf` in flash attention, aussieai
  73x fused softmax) says 1-2 ULP is irrelevant in attention where the max
  is subtracted, but it changes logits, so it needs its KL price from the
  harness first. Not attempted here.
- Shared-prefix KV cache, same-tokenizer draft model: true, still open.
  Continuous batching (section 25) shares the weight stream but re-prefills
  each admitted prompt; a prefix cache would skip it. Deferred.
- Fuse Q/K/V and gate/up: correctly dropped. Fork-join is 0.5-0.8 us; 72
  fewer calls save under 0.3 ms per token. Confirmed, not revisited.

Tests: 60/60. Gate (`scripts/test.cmd`): pass.

## 27. Phase-0 timers and re-sweep under load (2026-11-01)

`generate` now prints a `timers` line per decode: all model passes
(`forward` + speculative `verify_draft`) vs all sampler calls. The
attention/projection split lives inside `forward()` and is not timed here;
the line names the sampling share so a top-p/min-p regression shows first.

| run (loaded box, absolutes depressed) | forward | sample | share |
|---|---|---|---|
| greedy, 63 tok | 5425.78 ms | 17.59 ms (0.28 ms/tok) | 0.3% |
| temp 0.8 top-p 0.95, 31 tok | 2262.82 ms | 42.12 ms (1.36 ms/tok) | 1.8% |
| greedy + `--prefetch`, 31 tok | 3044.23 ms | 8.91 ms | 0.3% |

Greedy sampling is a 0.3 ms argmax scan; the top-p candidate path costs
~1 ms extra per token, against 18.2 ms with the old full sort (§16). The
machine was heavily loaded for these runs (short decode 10-13 t/s vs the
32-45 t/s in §26a), so only the shares transfer, not the absolutes.

Re-sweep with the remainder fix (§26a): output projection (151936x896)
1 token 6.531 ms vs 4 tokens 10.249 ms = 1.57x the time for 4x the work,
against ~4x under the old per-token tail. `bench continuous` 8x16 rechecks
at 43.46 t/s aggregate with all transcripts identical (lower absolute than
the 58-69 t/s in §25 under lighter load; independence holds). Tests 60/60.

## 28. Beat-llama round A: vectorised norms/RoPE + finer prefill attention (2026-11-01)

Target: short decode at parity (45 vs 45.1 llama), prefill 75-85% of llama
(pp512 261-266 vs 352), 2k decode 82% (38.3 vs 46.9). Prefill does ~87M
scalar softmax exps + ~60M SiLU exps per 510 prompt plus scalar norm/RoPE
passes; `attention_batch` parallelised over tokens only, so a 4-token prompt
leaves 2 of 6 workers idle.

### 28a. AVX2 rms_norm + RoPE (kept, fidelity-neutral)

`rms_norm_avx2` (vectorised sum-of-squares + scale) and `apply_rope_avx2`
(vectorised NeoX rotate with FMA) in `src/cpu/avx2/norm_rope_avx2.cpp`;
scalar oracles in `src/cpu/scalar/norm_rope_scalar.cpp`; `Qwen2` dispatches
on AVX2+FMA. Differential test `norm_rope_avx2_matches_scalar` (dims
32/64/896, heads 2/14, head_dim 64/128) at 1e-5.

| check | before | after |
|---|---|---|
| tests | 60/60 | 61/61 (new differential) |
| fidelity 1023 pos | mean 6.115e-12, worst 9.6e-10, 1023/1023 | mean 5.529e-12, worst 3.7e-10, 1023/1023 |
| generate text, story prompt | ", there was a young man..." | identical |

Speed on this box is load-dominated (86% RAM, page pressure; `bench batch`
swings 3-6x run to run), so no t/s claim is made here. The win is structural:
every prefill chunk (64 tokens x 24 layers x 2 norms) and every decode token
(48 norms + 2 ropes) leaves the scalar path. Kept on correctness + fidelity.

### 28b. Prefill attention over tokens x heads (kept, correctness)

`attention_batch` split over `tokens * heads` when `tokens < workers`;
`forward_batch` attention (previously sequential) uses the same schedule via
`prefill_pool`. `scores_` grows from `max(14, 64)` to `64 * 14` rows (14.7 MB)
so each task owns its slice; decode keeps the first 14 slices.

| check | result |
|---|---|
| tests | 61/61 |
| fidelity | unchanged 5.529e-12, 1023/1023 |
| generate story prompt | identical text |
| continuous 4x8/2slots/p2 | 17.11 t/s aggregate, independent transcripts |

A 4-token prompt goes from 4 to 56 tasks; a 64-token chunk keeps the token
path. No arithmetic change, so no KL price. Kept.

### 28c. Vectorised SiLU gate + attention softmax exp (kept, fidelity-neutral)

Shared Cephes-style AVX2 exp (`include/tge/cpu/exp_avx2.hpp`, 5th-order,
~1e-6 relative) used by `silu_gated_avx2` (all three SiLU sites: prefill,
forward, forward_batch) and the `attention_head_avx2` softmax array.
Differential tests at 1e-4 (`silu_gated_avx2_matches_scalar` over
1/7/32/4864; existing attention differential covers the softmax).

| check | value |
|---|---|
| tests | 62/62 |
| fidelity 1023 pos | mean 4.898e-12, worst 2.468e-10, max diff 1.18e-4, top-1 1023/1023 |
| story prompt text | identical (normal, speculative) |
| spec ratio this session | 27.63 vs 17.06 t/s (1.62x) |
| long prompt (282 tok) | prefill 113.5 t/s, decode 20.0 t/s (load-depressed absolutes) |
| continuous 4x8/2slots/p2 | 27.42 t/s aggregate, independent |

One real bug caught by the harness: the first version clamped exp to
[-104, 88], so softmax tails below -87 produced a negative biased exponent
(garbage scale) → KL nan, worst 10.6 nats, top-1 828/1023. Clamping to
[-87, 88] keeps the 2^n reconstruction in normal-float range; tails
underflow to ~1e-38, negligible beside the max term (1.0). After the fix
the KL matches the pre-exp baseline (4.76-4.90e-12). Kept as the default
(no opt-in): the price is at the reference floor, three orders below the
f16-KV trade (1.7e-6).

Remaining scalar exps: GQA online-softmax alpha/beta (one value per token,
negligible) and the sampler (151,936-logit path only matters for sampled
decode; greedy is an argmax scan). Next: prefill-vs-decode thread split,
then a same-thread re-measurement against llama.cpp build 11624 pairs.

### 28d. Long-context check with the new kernels (correctness)

2048-token id prompt (`128` x 2048), 32 decode tokens, 6 threads, this
session (load-depressed absolutes): prefill 99.0 t/s, decode 22.05 t/s,
timers forward 1398.13 ms vs sample 7.69 ms (0.5%). Logged lighter-load
figures (§19/§18): prefill 225-246, decode 38.3 GQA. The GQA schedule
(>= 256 positions) is active here; fidelity (§28c) already prices the
arithmetic. No regression in scheduling: 31/31 tokens decoded, no crash,
no fallback to scalar. Absolute comparison across sessions is load-noise;
the within-run share (sampling 0.5%) transfers.

### 28e. Phase-specific thread plan: prefill on all cores, decode pinned (kept)

Decode is bandwidth-bound and stays on 6 pinned P-core threads; prefill is
compute-bound and now takes every logical core (16), unpinned, with dynamic
work-stealing absorbing the E-core pace. `ThreadPool(threads, pin)` gains
the pin flag (default true, decode behavior unchanged);
`Qwen2::prefill_threads()` returns `max(threads_, hardware_concurrency)`;
`batch_matvec`, `attention_batch` and the `forward_batch` attention use it.
No arithmetic change: the clínicas are scheduling only.

| check | before (same session, heavier load) | after (lighter load) |
|---|---|---|
| `measure.ps1` mem read | 19.82 GB/s | 26.81 GB/s (box load differs, 1.35x) |
| 282-tok prefill | 122.3 t/s | 250.3 t/s (2.05x; exceeds the 1.35x load shift) |
| short decode (control: path untouched) | 9-19 t/s earlier | 29-35 t/s now (load, not code) |
| fidelity 1023 pos | 4.898e-12, 1023/1023 | identical 4.898e-12, 1023/1023 |
| story text / continuous | identical / independent | identical / independent (34.66 t/s agg) |
| tests | 62/62 | 62/62 |

Honesty note: the box lightened between the two runs (mem 19.8 → 26.8),
so the 2.05x prefill figure bundles a ~1.35x load shift with the real
structural gain; the decode control (untouched path) moving with load
confirms the bundle. The keep rests on structure + fidelity-neutrality,
not on the headline ratio: compute-bound prefill on 16 unpinned cores vs
6 pinned is the Sandwich phase split, and the KL is bit-identical at
4.898e-12. A same-load A/B (taskset or cgroup) is the follow-up that
would isolate the ratio; logged as open, not claimed.

### 28f. Same-load prefill A/B via --prefill-threads (kept, +17% isolated)

§28e left the ratio bundled with a load shift and forced prefill to
`max(threads, hw)`, removing the fair same-thread comparison. New:
`Qwen2::set_prefill_threads(n)` (0 = auto = all logical cores) plus
`--prefill-threads N` on `generate` and `bench continuous`. Default
behavior is unchanged (16 here); passing 6 reproduces the old plan.

Same-load interleaved A/B, 282-token prompt, `--threads 6`, one window:

| run | prefill threads | prefill | decode (control, path untouched) |
|---|---|---|---|
| A | 6 | 193.7 t/s | 20.93 t/s |
| B | default (16) | 259.3 t/s | 17.04 t/s |
| A2 | 6 | 234.5 t/s | 11.18 t/s |
| B2 | default (16) | 242.7 t/s | 37.03 t/s |

Medians: 6 → ~214 t/s, 16 → ~251 t/s: **+17% isolated structural gain**.
The decode control swings 11-37 t/s across the same four runs (7-token
sample + load noise), confirming absolutes are not the evidence; the
interleaved prefill pairing is. No arithmetic change (scheduling only),
so no KL price; fidelity stands at 4.898e-12 from §28c.

Kept: default 16 (free prefill throughput, decode untouched) with the flag
restoring the fair fight (`--prefill-threads 6` matches `--threads 6` /
llama `-t 6`). Fairness note for future comparisons: report the prefill
thread count alongside the rate; same-thread numbers are the algorithmic
comparison, all-core numbers are the system comparison. Tests 62/62.

### 28g. Longer prompt-lookup spans 5..1 (kept, no-regression + ceiling case)

The drafter only tried spans 1-3. Longer context matches accept at higher
rates (GOOSE 2026), and the verify-then-accept construction guarantees
greedy-identical output whatever the draft is, so this is a free option:
`prompt_lookup_draft` now tries 5..1 in `apps/cli/main.cpp`, falling back
through the old range, so every proposal the old code made is preserved.

| prompt, 128/32 tok | before (3..1) | after (5..1) |
|---|---|---|
| story "Once upon a time", spec | 6 steps, 1.17/step, 22.20 t/s (0.94x normal 23.61) | 6 steps, 1.17/step, identical text |
| repetitive 282-tok, spec | — | 2 steps, 5.00/step (ceiling), 42.12 vs 29.55 t/s (1.43x), ids identical |

Honesty note: the story prompt has no 4-5-token matches, so acceptance is
unchanged there (same proposals, same 1.17/step) — and it also shows
speculation can run slightly below parity (0.94x) when there is nothing to
match, which is why the flag stays opt-in. The repetitive prompt sits at
the 5.00/step ceiling. A prompt where only longer matches exist would
separate old vs new; none was measured, so the keep rests on the
fallback-superset mechanism + identical-output guarantee + no-regression,
not on a measured acceptance delta. One subtlety recorded: when a span-5
match exists at an older position than the most recent span-3 match, the
new code drafts from the longer (older) match instead — output still
verified identical, acceptance expected higher per GOOSE, not proven here.
Tests 62/62.

### 28h. K-quant scratch reuse without zero-init (kept, +50-84% kernel)

`gemv_dequant_batch_avx2` allocated and zero-filled a cols-float scratch
per band per call (`std::vector<float>(cols)` inside `parallel_for`):
~bands allocations x 169 calls per decode token, and the fill is pure
waste since `dequantize_row_avx2` overwrites every element. Now a
`thread_local` scratch per worker, `resize`d once (growth zero-fills only
the new tail) and reused across calls. Same bytes in, same floats out:
no arithmetic change. The scalar oracle keeps its once-per-call scratch
(tests only), so only the parallel path changed.

| bench | before | after |
|---|---|---|
| q4_k batch 896x4864 x 1 (best of 3) | 0.593 ms, 4.13 GB/s | 0.322 ms, 7.61 GB/s (+84%) |
| same, re-run best of 5 (load moved) | — | 0.396 ms, 6.20 GB/s (+50% vs baseline) |
| Q4_K_M decode, story 16 tok | 10.29 t/s | 10.34 t/s (unchanged, inside noise) |
| Q4_K_M greedy ids | 11 1052 572 ... 13 1260 | identical |

Honesty note per the §7b precedent: the kernel win is large and stable
across re-runs, but end-to-end Q4_K decode does not move outside noise on
this box — the remaining gap is the per-row dequant compute + scratch
round-trip the log already prices (6.7 vs 21.85 GB/s), i.e. the fused dot
is still the real fix and stays open. Kept on kernel win + parity: less
alloc traffic on every K-quant call at zero accuracy cost. Tests 62/62.

### 28i. Fused single-token Q4_K dot (REVERTED: no win + alignment fault)

Attempt: `dot_q4_k_fused_avx2` unpacked nibbles straight into the FMA chain
(two vector accumulators for the qx and x sums, finalized per 32-group with
the group's d/m), wired into `gemv_dequant_batch_avx2` for `tokens == 1 &&
kind == Q4_K` (every decode matvec on Q4_K models). Differential test vs
scalar dequant+dot passed at 1e-4 on 1- and 3-block rows, and the Q4_K
story ids came out identical to baseline (`11 1052 572 ... 13 1260`) —
after fixing the real bug below.

Two findings, one fatal to the change:

1. Alignment fault (fixed during the attempt, kept as a lesson): the first
   version formed `const __m256* xp` over the activation buffer and indexed
   it. That emits aligned VMOVAPS, but activation buffers are only 16-byte
   aligned (`operator new`), so Q4_K generate segfaulted (0xC0000005) with
   zero output — while `bench batch` passed because its input buffer happened
   to be 32-byte aligned. Rule recorded: never form a vector pointer over a
   float buffer; every lane loads explicitly unaligned (`_mm256_loadu_ps`).
   The same audit applies to future kernels: `xp[i]` on `__m256*` is an
   alignment assertion, not a load.
2. Same-session A/B, q4_k batch 896x4864 x 1 (best of 5): fused 0.349 ms /
   7.01 GB/s vs the §28h scratch path 0.301 ms / 8.15 GB/s. The fused dot
   LOSES by ~14%. Structural reason: per-32-group scales force two
   horizontal reductions per 32 weights (304 hadd-sequences per 4864-row vs
   one in `dot_f32`), while the scratch path separates a clean wide dequant
   from a clean whole-row dot. Fusion saves an L1 round-trip but pays hadd
   tax that exceeds it; same unpack ALU either way.

Reverted in full (branch, kernel, decl, test) — the tree is back to the
§28h state, verified by diff. Rounding risk (FMA vs mul-then-sub) with zero
demonstrated gain is exactly what G-002 reverts. The Q4_K decode gap
(10.7 t/s vs ~30 Q8_0) stays open; the plausible next attempt is an
int8-domain dot (`_mm256_maddubs_epi16`, llama-style, no float nibbles at
all), not float fusion. Tests 62/62 after revert; Q4_K story ids identical.

### 28j. AVX2 Q6_K dequant, bit-identical (kept, +82% kernel, +12% model)

`ffn_down` is Q6_K 3.41 MiB per block on this recipe and was still scalar-
dequantized on every decode token and every prefill chunk. New
`dequantize_q6_k_avx2` in `src/cpu/avx2/dequant_avx2.cpp`, wired as the Q6_K
case in `dequantize_row_avx2`: 6-bit weights rebuilt from 4 low bits plus 2
high bits (shifts exact per byte under 0x0F/0x03 masks, and-before-shift so
no spill), minus 32 in the byte domain, then multiply-only `(d*sc)*q in
scalar order with no FMA — hence bit-identical, not merely near. Loads are
explicitly unaligned throughout (§28i lesson: no vector pointer over a
model buffer). `bench batch --format` grows a `q6k` value (row bytes
cols/256*210); the Q6_K differential test (`q6k_avx2_matches_scalar_exactly`,
1- and 3-block, memcmp) passes.

| bench | before (scalar) | after (AVX2) |
|---|---|---|
| q6_k batch 896x4864 x 1 (best of 5) | 0.808 ms, 4.42 GB/s | 0.444 ms, 8.06 GB/s (+82%) |
| Q4_K_M decode, story 16 tok | 10.29 t/s | 11.51 t/s (+12% end-to-end) |
| Q4_K_M greedy ids | 11 1052 572 ... 13 1260 | identical |

Honesty note: kernel and model figures are from adjacent sessions on a
loaded box, not an interleaved A/B — but the attribution is clean (the only
change on the Q6_K path is scalar→AVX2, and the differential is memcmp, so
no rounding confound exists). The remaining scalar K-formats on this recipe
are Q5_0 (gate/up/attn, the largest share) and Q5_K/Q4_0; Q5_0 AVX2 is the
same pattern and stays next. Tests 63/63.

### 28k. AVX2 Q5_0 dequant, bit-identical (kept, +121% kernel)

Q5_0 covers gate/up/attn (~6.9 MiB per block x24 — the largest scalar
share left) and ran scalar on every token. New `dequantize_q5_0_avx2` +
dispatch case: nibbles via the §28i-exact 16-bit shift discipline, the two
high bits via a 256-entry LUT (byte k of entry b is ((b >> k) & 1) << 4, so
no variable per-element shift), bias -16 in the byte domain, then
multiply-only `d*q` in scalar order, no FMA — bit-identical, memcmp-gated.
All loads explicitly unaligned (§28i). `bench batch --format` grows `q50`.

The differential test caught a real misread before anything shipped: the
scalar pairs low bit (hb >> i) with high bit (hb >> (i + 16)) — the
`(hb >> (i + 12)) & 0x10` selects bit 4, i.e. source bit i + 16, not i + 12
as first implemented (first mismatch at index 16, off by exactly the 0x10
bit). Fixed, memcmp passes on 1- and 3-block rows.

| bench | before (scalar) | after (AVX2) |
|---|---|---|
| q5_0 batch 896x4864 x 1 (best of 5/3) | 0.881 ms, 3.40 GB/s | 0.399 ms, 7.50 GB/s (+121%) |
| Q4_K_M decode, story 16 tok | 10.29–11.51 t/s (loaded box) | 29.35 t/s, ids identical |
| same light window, Q8_0 story 16 tok | — | 32.86 t/s (Q4_K_M at 89% of Q8_0) |

Honesty note: the 10-11 → 29 end-to-end jump bundles a large load shift
(mem 19-22 → 27.6 GB/s across sessions) with the kernel win — the kernel
A/B (2.2x) exceeds any observed load swing (≤1.4x), and the same-window
Q4_K_M/Q8_0 pairing (89%) is the load-proof statement. Remaining scalar
K-formats: Q5_K (absent from this recipe — verify before touching) and Q4_0
(embeddings, cold path). Tests 64/64.

### 28l. K-quant program closeout: same-window Q8_0 vs Q4_K_M (all hot formats AVX2)

`gguf` histograms of both local files: Q4_K_M = Q5_0 (133 tensors) + F32
(121) + Q8_0 (13) + Q4_K (12) + Q6_K (12); Q8_0 file = Q8_0 (170) + F32.
No Q5_K and no Q4_0 tensor exists in either file, so with Q4_K (§17), Q6_K
(§28j) and Q5_0 (§28k) all AVX2 behind memcmp oracles, every K-format the
engine can actually execute here is vectorised. Q5_K/Q4_0 stay
scalar-oracle-only until a file containing them appears — vectorising
without an end-to-end model to run would be untestable performance art.

Same-window matrix, back-to-back runs, `--threads 6` (prefill default 16):

| 282-tok prefill / story decode | Q8_0 | Q4_K_M | Q4_K_M / Q8_0 |
|---|---|---|---|
| prefill | 253.0 t/s | 222.8 t/s | 88% |
| story decode, 15 tok | 31.71 t/s | 25.33 t/s | 80% |
| story ids | ...13 3757 | ...13 1260 (== every prior Q4_K_M baseline) | first 15 ids identical |

The 7-token decode samples (26.95 vs 38.01) flip rank — small-sample noise,
not evidence; the 15-token runs are the metric. Q4_K_M moved from ~10 t/s
(loaded box, scalar K-dequant) to 80% of Q8_0 in the same window; the
remaining gap is the irreducible unpack ALU per weight plus attention/KV,
not scratch or dispatch overhead. Per-format kernel path since §28h–§28k:
q4k 4.13→7.6, q6k 4.42→8.06, q5_0 3.40→7.50 GB/s. Tests 64/64.

### 28o. Alignment-pattern audit: clean, rule recorded (kept)

`grep` over `src/cpu/avx2` for `reinterpret_cast<*__m256*>`, `__m256*`
indexing, and aligned `_mm256_load/store_ps` / `_mm256_load/store_si256`:
5 vector-pointer casts, all feeding `_mm256_loadu_si256` (explicitly
unaligned — safe); 2 aligned `_mm256_store_ps`, both targeting
function-local `alignas(32)` spill arrays (32-byte aligned by construction —
safe). Scalar TUs contain no intrinsics; `src/models` contains no vector
pointers or aligned loads. No live instance of the §28i fault pattern, so
no code fix — but the audit was not free: `bench batch` uses AlignedBuffer
(32-byte), which is why the fused dot passed bench and segfaulted generate,
so the audit also justifies a standing rule, now written into the kernel
contract (`include/tge/cpu/kernels.hpp` addressing paragraph): never form a
vector pointer over a model or activation buffer; every lane loads/stores
explicitly unaligned; `alignas(32)` spills are the sole exception.
Verified: 64/64 tests, Q4_K_M story generate clean with baseline ids.

### 28p. Snapshot covers the K-quant kernels (kept, instrument only)

`measure.ps1` measured Q8 shapes only, so every §28h–§28k K-quant win landed
with ad-hoc benches and no G-001 line. It now prints three `kbatch` lines —
decode-shaped 1-token runs in the orientation each format takes in the model
(gate Q5_0 4864x896, down Q4_K and Q6_K 896x4864), one line each, same
best-of diff format. No behavior change: reporting only, verified
line-by-line (`kbatch gate/down4/down6` print correctly).

Side observation, reinforcing §28m: the first snapshot run printed
1.27–1.44 GB/s for lines that measure 6.45–8.23 direct seconds later — a
background CPU spike inside that one script run. Even within a single
snapshot, a contended minute lies; interleaved A/B remains the only
decode-grade evidence. Tests 64/64 (script change; suite unaffected).

### 28n. Docs sync, no code behavior change (kept)

README claimed 58 tests (now 64), `bench batch` grew `--format q6k/q50`
with no `--help` line, and the §28l same-window Q4_K_M numbers were
verified but unpublished. Fixed: usage documents `--format F`
(q8|q4k|q6k|q50) and `--kernel K`; README tests row reads 64 and the
verified table gains the Q4_K_M same-window row; `benchmarks.md`
reproducing section notes the formats and row-byte rules. README is 11.0k
chars (limit 30k). No performance delta by construction — usage string
only; `tge` help output verified. Tests 64/64, 4/4 gates.

### 28m. Decode-overhead hunt on this box: load, not structure (Track A, no code)

Q8_0 story decode here runs 27–38 t/s (31.5ms) against 18.1ms weight
streaming at 27.7 GB/s, leaving ~13ms unexplained vs the 3–5ms logged on
the 14400F. Same-window tests, no code changes:

| run, `--threads 6` story 16 tok | decode |
|---|---|
| cold process | 37.62 t/s |
| warm process | 31.10 t/s |
| warm + `--prefetch` (473ms fault-up-front) | 27.49 t/s |
| 2048-ctx random ids (same window) | 26.14 t/s |
| 2048 prefill (same window) | 264.8 t/s |

Findings, in order of what they kill:

1. NOT page faults. Prefetching all weights up front makes decode SLOWER,
   not faster (27.49 vs 31.10), and cold beats warm. The variance is box
   load/frequency, running ±18% across back-to-back identical runs.
2. The "13ms overhead" is therefore mostly misattributed load, not a
   structural gap to optimize. Retire the hunt; this box cannot resolve
   <20% decode differences, which is exactly the report's own ±20% noise
   statement (§"measurement system").
3. Attention slope, bounded estimate only: ~31.5ms (short) vs 38.3ms
   (2048 ctx) ≈ 7ms per 2048 context tokens, confounded by the same swing —
   reported as an upper bound, not a price.
4. Prefill 264.8 t/s at 2048 ctx confirms the §28e thread split holds at
   long context (prior 225–246 at 6 threads).

Methodological consequence, recorded as a rule: on this box, only
interleaved same-window A/B counts for decode (as in §28f); cross-session
absolutes are load narrative. The next structural decode lever stays what
§28i/§28l say (int8-domain dot or fewer passes), not overhead hunting.

### 28q. Shared-prefix KV cache: designed, deferred (no paying caller)

Designed to the code level, then stopped before writing it. The mechanism
is straightforward — copy-on-match of position-0 KV rows into a fresh slot
instead of recomputing, byte-identical by construction — but no in-engine
caller reuses a prefix: `generate` is one-shot per process, `bench
continuous` feeds shared prompts through `forward_batch` (subsetting its
per-token batch for skips would be surgery on the hottest loop), and `bench
sessions` gives each session its own model. Building the cache without a
caller is speculative generality. Trigger condition, recorded so the work
starts from analysis instead of rediscovery: a persistent multi-turn loop
calling `prefill` on one model instance with repeated leading tokens —
then hook `prefill_run` (single integration point: match, memcpy rows,
start chunks at the match length, 1-token tail when the whole prompt hits)
and demonstrate on repeated system prompts. No code changed.

### 28r. Sampler primitives for distribution-preserving speculation (kept, [1/3])

Speculation is gated to greedy + no penalty (`main.cpp` argmax pre-check),
so temperature users cannot use the only lever past the wall. First third:
the sampler support it needs, with no callers yet. New in
`sampling.{hpp,cpp}`: `softmax_probabilities` (extracted byte-identical
from the no-cut path — the seed-determinism test still passes unchanged),
`categorical_sample` (same scan), and `speculative_accept` implementing the
standard one-hot rule (accept w.p. q[draft]; on reject draw from
norm(max(0, q − onehot)), never the draft token). Statistical tests, all
passing: helper vs hand-computed exps; categorical-vs-sampler agreement
over 32 seeded draws; accept rate 0.35 ± 0.02 over 20k trials; rejection
resample follows the renormalised tail (0.375/0.625 ± 0.02, draft never
returned); bad-argument guards. Greedy story ids unchanged; sampled smoke
run healthy through the refactored path. Next: wire the sampled verify
loop ([2/3]), then measure acceptance/speedup with parity ([3/3]). Tests
69/69 (64 + 5 new), 4/4 gates.

### 28s. Sampled speculation wired: verify_draft_logits + accept loop (kept)

`Qwen2::verify_draft_logits` returns all `count` positions' logits (same
batched pass, no argmax); `generate` gains a temperature-only sampled path
(gated like greedy plus no top-k/p, no min-p, no penalty — the accept rule
is exact only for the untruncated distribution). Per step: draft[0] is
judged against the current logits BEFORE verifying (a first-position
rejection skips the weight pass entirely); on accept the verifier runs and
draft[1..4] are judged against rows 0..3, bonus from row 4. Position
bookkeeping mirrors greedy (`forward(last, position + emitted - 1)`).

Measurement caught a real off-by-one pre-ship: the first version judged
draft[i] against verify row i, but row i predicts draft[i + 1] (row i is
the distribution after feed[i]). Symptom: 0.00/step acceptance on a
repetitive prompt where greedy hits 5.00. Fixed; acceptance now scales
with peakedness exactly as theory predicts:

| repetitive 282-tok, temp | spec acceptance | decode, same window |
|---|---|---|
| 0.8, no spec | — | 31.56 t/s |
| 0.8, spec | 0.50/step | 27–33 t/s (verify cost ≈ benefit) |
| 0.1, no spec | — | 25.76 t/s |
| 0.1, spec | 3.00/step | 52.30 t/s (**2.03x**) |

Distribution preserved by construction (unit-tested accept/reject +
renormalised resample, reviewed loop mirroring greedy); greedy `--speculative`
ids byte-identical to every prior baseline. New bottleneck exposed: sampling
is 61% of the temp-0.1 spec run (6 full-vocab scalar softmaxes per verify
step) — the sampler exp is next (vectorise like attention/SiLU, price it).
Tests 69/69, 4/4 gates.

### 28t. Vectorised sampler softmax exp (kept, -16% sampling, chi-square gated)

Sampling was 61% of the temp-0.1 spec run and ~2-3% of plain sampled
decode: up to 6 full-vocab scalar softmaxes per verify step. New
`softmax_exp_avx2` (own TU `src/cpu/avx2/sampler_exp_avx2.cpp`, same
pattern as SiLU/norms) using the shared Cephes exp; `softmax_probabilities`
dispatches on AVX2+FMA, scalar oracle in the scalar TU. The cut paths
(top-k/p, min-p) keep their own scalar loops — out of scope. Greedy never
touches exp, so greedy ids cannot move (verified identical).

| check | before | after |
|---|---|---|
| differential scalar-vs-AVX2, 1/7/32/1519 wide | — | max 1e-4, pass |
| no-cut histogram, 8-way, 100k draws | — | every bin ±0.01, pass |
| sampled story temp 0.8, 32 tok | 24.58 t/s, sample 2.1% | 27.90 t/s, sample 1.9% |
| spec temp 0.1 repetitive, same trajectory (4 steps, 12 acc, 15 out) | sample 171.62 ms (57.5%) | sample 143.66 ms (47.6%), **-16% paired** |
| spec temp 0.1 decode end-to-end | 50.23 t/s | 49.74 t/s (load-masked: forward +33ms) |
| greedy story ids | ...13 1260 | identical |

Honesty note: end-to-end spec is flat across sessions (forward load +33ms
ate the -28ms sampling win) — the claim rests on the paired work-count
comparison (identical steps/accepts/emits, same seed: same 24 softmaxes,
-16%). Only -16% and not more because each accept-step also does ~4
full-vocab scalar scans (totals, resample weights); the exp was less than
half the sampling cost. Vectorising the scans is open but chasing single
digits of a 2% (plain) / 48% (spec-low-temp) share. Tests 71/71 (69 + 2
new), 4/4 gates.

### 28u. Sampled speculation with repeat penalty (kept, correct; 0.80x here)

The spec gate excluded all penalty users; penalty is history-deterministic,
so the accept rule stays exact — `apply_repeat_penalty` moved from the
anonymous namespace to `tge::models` unchanged (existing penalty tests
still green, confirming the move is behavior-neutral) and the sampled loop
now penalizes each position with the history as of that position (copy row,
penalize, softmax in place; the copy keeps the model's row intact).

| repetitive 282-tok, temp 0.1, penalty 1.1 | no spec | spec |
|---|---|---|
| decode | 34.31 t/s | 27.53 t/s (**0.80x**) |
| spec | — | 6 steps, 3 accepted (0.50/step), 9 emitted |

Two honest notes. First, an early zero-step run looked like a gate bug;
it was history luck (short high-temp divergent run whose histories never
formed a matchable pattern) — the low-temp run on the same prompt shows
steps. Second, 0.80x is the mechanism, not a bug: the penalty suppresses
exactly the repetition the prompt-lookup drafter proposes, so acceptance
sits at 0.50/step and the verify passes don't amortize. Template-shaped
repetition outside the 64-token window is the case that wins; this prompt
is the case that doesn't. Kept as a strict superset: greedy and temp-only
paths are byte-identical (early-return penalty is 4 comparisons; verified
ids unchanged), penalty users gain the lever where drafts match
non-penalised tokens and fall back gracefully elsewhere. Cuts (top-k/p,
min-p) stay on the safe single-token fallback — replicating their
floor-lowering machinery in parallel would risk silent bias; exact reuse
is the only acceptable construction and stays future work. Tests 71/71,
4/4 gates.

### 28w. README speculative paragraph (docs only, no behavior change)

The README described `--speculative` as greedy-only, predating the sampled
(TGE-SAMPSPEC), penalty (TGE-SAMPPEN) and penalized-greedy (TGE-GSPECPEN)
extensions. It now states the actual gates (greedy ± penalty with
byte-identical ids; temperature-only sampling with exact preservation;
other cuts fall back) and the measured speedups (1.66-2.00x greedy story,
2.03x sampled temp 0.1, 1.88x greedy+penalty). README is 11.4k chars
(limit 30k). No code change.

### 28v. Greedy speculation with repeat penalty (kept, bit-exact, 1.88x)

The greedy spec gate excluded all penalty users for no structural reason:
argmax-after-penalty is deterministic, so verifying penalized argmaxes
keeps output identical with a bit-exact gate. `generate` now penalizes
the d0 check with the current history and each verify row i with history +
feed[0..i] — exactly what single-token decoding would see, including the
64-token (or `--penalty-window`) window sliding over emitted drafts. The
no-penalty path is untouched (same calls, same order).

| repetitive 282-tok, penalty 1.1, window 4 | no spec | spec |
|---|---|---|
| decode | 28.25 t/s | 53.01 t/s (**1.88x**) |
| spec | — | 1 step, 5 accepted (5.00/step), 6 emitted |
| ids, 15 tok | 3838 374 ... 374 25 | byte-identical |

The gate is equality, not statistics: any history-threading mistake flips
a token. A default-window (64) run on the same prompt fires zero drafts —
the penalty suppresses exactly what prompt-lookup proposes, same tension
as §28u — while window 4 accepts at ceiling because the 8-token period
falls outside it. Greedy no-penalty story ids unchanged
(...13 1260); sampled paths untouched (separate gate). Tests 71/71,
4/4 gates.

### 28x. Same-window speculation matrix, all three modes (kept)

Back-to-back pairs on the repetitive 282-tok prompt, Q4_K_M, `--threads 6`,
adjacent runs so the box state is shared inside each pair. `tge --help`
also fixed: `--speculative` no longer claims greedy-only.

| mode, 15 tok | base | spec | ratio | acceptance |
|---|---|---|---|---|
| greedy | 27.91 t/s | 41.95 t/s | **1.50x** | 5.00/step, ids byte-identical |
| temp 0.1 | 41.94 t/s | 58.64 t/s | **1.40x** | 3.00/step |
| penalty 1.1/w4 | 40.12 t/s | 41.73 t/s | 1.04x | 5.00/step, ids byte-identical |

Two honesty notes. First, base levels drift across pairs (27.91 → 41.94 →
40.12): the first run cold-loads the 350 MB file into page cache, so the
greedy pair bundles cache warming with the win — pairs stay valid
(adjacent), cross-pair absolutes do not. The temp pair (both warm) is the
cleanest number at 1.40x; history converges on 1.4-1.9x greedy.
Second, the penalty pair wins little (1.04x) because only 1 step fired
in 15 tokens: acceptance is perfect when drafts exist, but drafts are rare
when the penalty suppresses repetition — same mechanism as §28v.
Convergent verdict: speculation pays on repetitive structure in every
mode, most where acceptance is high and the verify cost amortizes.
Tests 71/71, 4/4 gates.

### 28y. Fidelity across corpora (kept, generalizes)

| corpus (1023 positions each) | mean KL | worst KL | max |logit| | top-1 |
|---|---|---|---|---|
| docs/architecture.md (prose) | 4.898e-12 | 9.6e-10 | 1.18e-4 | 1023/1023 |
| docs/report-0.1.md (prose+numbers) | 4.848e-12 | 1.123e-10 | 9.155e-5 | 1023/1023 |
| src/models/sampling.cpp (code) | 4.596e-12 | 1.087e-9 | 2.170e-4 | 1023/1023 |

Mean KL sits at 4.6-4.9e-12 on all three; code shows a higher worst-case
(1e-9 vs 1e-10 — sharper code distributions amplify tiny logit shifts)
but stays three orders below every priced trade (f16-KV 1.7e-6) with
unanimous top-1. Caveat recorded with the numbers: this harness varies
attention only (see §28z) — the cross-corpus stability still bounds the
whole pipeline's behavior, but the per-kernel attribution sharpens next.

### 28z. Full-program fidelity: scalar-kernel switch on the reference (kept)

The §28y caveat is now closed. `Qwen2::set_scalar_kernels(true)` forces the
scalar oracle for rms_norm, RoPE and SiLU wherever AVX2+FMA would otherwise
dispatch; `tge fidelity` sets it on the reference arm (which already ran
scalar attention), so one number prices the whole vectorisation program.
Production paths are untouched (flag defaults off; greedy story ids
byte-identical). Harness label updated to `full-scalar oracle`.

| corpus (1023 pos) | attention-only KL (old) | full-program KL | worst | max |diff| | top-1 |
|---|---|---|---|---|---|
| architecture.md | 4.898e-12 | 1.058e-11 | 1.216e-9 | 2.370e-4 | 1023/1023 |
| report-0.1.md | 4.848e-12 | 9.960e-12 | 2.221e-10 | 9.251e-5 | 1023/1023 |
| sampling.cpp | 4.596e-12 | 6.514e-12 | 4.833e-10 | 1.365e-4 | 1023/1023 |

As predicted, the full-program number runs ~2x the attention-only figure —
norms/RoPE/SiLU contribute roughly as much as attention itself — and still
sits ~150x below the f16-KV trade (1.7e-6) and ~400x inside the 0.1
near-tie band, with unanimous top-1 on all three corpora. Sampler exp
stays chi-square-gated by construction (teacher forcing never samples).
Retroactive correction to §28a/§28c: "fidelity-neutral" there meant
"no interaction detected against an attention-only harness"; the number
that actually prices those kernels is this one. Tests 71/71, 4/4 gates.

### 28aa. Docs catch-up for the fidelity instrument (kept, no code)

README still priced the arithmetic at "mean KL 5e-12 against the scalar
reference" — the attention-only number. It now reads ~1e-11 against the
full-scalar oracle (scalar attention, norms, RoPE, SiLU), matching §28z.
`tge plan` checked and unaffected: it is metadata-only (tensor table +
bandwidth projection), verified still printing the same 500.8 MiB/token
basis. README is 11.6k chars (limit 30k). Tests 71/71, 4/4 gates.

### 28ab. Decode workers unpinned by default (kept, +20-59% here)

P-core pinning was a variance play (§26b, no mean claimed). A sessions
re-measure exposed the cost: 4 models x 6 pinned workers oversubscribe
~12 P-core threads while E-cores idle. New `--pin-threads 1|0` on
`generate` and `bench sessions` (`Qwen2::set_pin_threads`, decode pools
only; prefill was already unpinned), then the A/B that decides the default:

| same window, `--threads 6` | pinned (old) | unpinned | delta |
|---|---|---|---|
| sessions 4x32 aggregate | 38.38 t/s | 59.97 t/s | **+56%** |
| single-story decode, 3x interleaved | 45.09 / 41.16 / 38.83 (range 6.26) | 48.87 / 49.43 / 48.49 (range 0.94) | **+17% mean, lower variance** |
| single-story after flip (default) | 37.64 explicit pinned | 45.28 default | +20% |

Unpinned wins 6/6 single-stream pairs with a 7x tighter range — the pin
fails its own variance rationale on this box (pinned workers stuck on
loaded P-threads; work-stealing across all cores absorbs contention).
Default flipped to unpinned; `--pin-threads 1` restores the old behavior
for machines where it still helps. Greedy ids byte-identical (placement
only, no arithmetic). Honesty note: the exact ratios are this box's; the
structural finding (oversubscription under sessions, E-cores idle under
pin) transfers, the percentages should be re-measured per machine.
Tests 71/71, 4/4 gates.

### 28ac. Decode thread-count sweep, unpinned (kept: documents the knob)

With decode unpinned (§28ab), the P-core-tuned 6 may no longer be optimal.
Interleaved story decode, Q8_0, `--threads 4/10/6/16/6/10/4/16`:

| threads | run 1 | run 2 | mean |
|---|---|---|---|
| 4 | 47.50 t/s | 47.65 t/s | 47.6 |
| 6 | 48.93 t/s | 49.31 t/s | 49.1 |
| 10 | 51.26 t/s | 51.36 t/s | 51.3 |
| 16 | 31.03 t/s | 29.54 t/s | 30.3 |

10 wins both of its pairs (+4-5% over 6): one worker per physical core
(6P+4E), no hyperthread pairing. 16 collapses (HT + E-core stragglers +
spin contention on 16 logical threads). The default stays 6: +4.5% on one
box's topology does not clear the bar for moving a default every user
inherits — the knob is documented in the README threading paragraph
instead, with the re-measure-per-machine caveat from §28ab. Tests 71/71
(suite untouched), 4/4 gates.

### 28ad. Bigram-chain draft fallback (kept, fires where spans cannot)

Prompt-lookup abstains unless an exact 1-5-gram matches (`story: 0 steps).
New fallback in `prompt_lookup_draft`: when no span matches, chain the most
frequent successor of each token (linear scan, ≤32 distinct successors, no
allocation), proposing only observed continuations. The verifier still
decides everything, so output cannot change shape — only acceptance matters.

| prompt | before | after |
|---|---|---|
| story (4 tok, nothing to chain) | 0 steps | 0 steps (correct abstention) |
| repetitive 282-tok, greedy | 5.00/step | unchanged (span path fires first) |
| partial-repetition ("John likes X" variants), greedy 32 tok | — | 3 steps, 10 acc (3.33/step), 13 emitted, 48.32 → 53.77 t/s (**1.11x**), ids identical |
| single-token repeat, temp 0.8 | — | 0 steps (tail token unique → correct abstention, no verify spam) |
| repetitive, temp 0.8 | — | 4 steps, 0.50/step, 6 emitted, 0.98x (neutral, distribution preserved) |
| doubled tail ("...warm warm"), temp 0.8 | spans provably cannot fire | 3 steps, 1 accepted (0.33/step), 4 emitted — fallback proven live |

The doubled-tail run is the mechanism proof: with an 18-token prompt whose
only repeat sits inside the last 5 positions, no span can match, so all 3
verifies came from bigram proposals — and one accepted, with the resample
path covering the rejections. Kept as a strict superset: spans run first in
the same order (their proposals are byte-identical to before), the fallback
adds only observed continuations, every emission is verifier-gated, and
all ids match non-speculative runs. Tests 71/71 (drafter lives in main.cpp,
covered end-to-end), 4/4 gates.

### 28ae. Single verification path (kept, refactor, no behavior change)

`verify_draft` (argmax) and `verify_draft_logits` (full rows) ran the same
batched pass with duplicated validation and had already diverged once in
spirit (logits-vs-argmax row semantics, §28s). `verify_draft` now delegates
to the logits form and argmaxes its rows — same prefill_run call, same
argmax order, same error cases. No caller changed. Parity gate, both
greedy modes: story `...13 1260` and penalized window-4 repetitive
`3838 374 ... 374 25` byte-identical to every baseline, 5.00/step on the
latter. Tests 71/71, 4/4 gates.

### 28af. Post-flip validation: parity holds, headlines refresh (kept)

The unpinned default changes every decode path's threading, so the key
paths were re-validated on it (Q4_K_M story greedy ids byte-identical:
`...13 1260`):

| path (same window) | pinned era | unpinned default |
|---|---|---|
| temp 0.1 spec, repetitive 282-tok | 50.23 t/s, 3.00/step | **66.01 t/s**, 3.00/step, 15 emitted |
| Q8_0 2048-ctx prefill / decode | 264.8 / 26.14 t/s | **268.7 / 38.31 t/s** |

The flip helps speculation too (verify passes are bandwidth-bound GEMMs).
The 2048-ctx decode gap vs short context is much smaller than the §28m
bound suggested — that bound was load-poor, this one less so; both remain
upper-bound-flavored on this box. No code change in this entry beyond what
§28ab already landed. Tests 71/71, 4/4 gates.

### 28ag. Consolidated final snapshot on the finished binary (kept)

`measure.ps1` (now with kbatch lines) plus continuous bench and an ids
spot-check, one window, light box (mem 28.82 GB/s):

| probe | result |
|---|---|
| kbatch gate (q5_0 4864x896 x1) | 0.326 ms, 9.18 GB/s |
| kbatch down4 (q4_k 896x4864 x1) | 0.312 ms, 7.85 GB/s |
| kbatch down6 (q6_k 896x4864 x1) | 0.368 ms, 9.71 GB/s |
| batch ffn 64 tok (prefill pool = 16 now) | 295 GFLOPS (vs 206 logged in the 6-thread era) |
| gemv single | 31.12 GB/s |
| continuous 4x16/2slots/p8 | 42.63 t/s aggregate, independent transcripts |
| greedy Q4_K_M story ids | byte-identical to every baseline |

This is the end-state record the earlier sections built toward: every §28
kernel has a snapshot line, independence holds under batching, and ids
never moved. Nothing here changes a verdict; it fixes the numbers to the
finished binary. Tests 71/71, 4/4 gates.

### 28ah. Drive close-out: final verification (kept, no change)

34 tasks after §28's start: full gates 4/4, suite 71/71, greedy and
penalized-greedy ids byte-identical to every baseline, no TODO/FIXME
markers, no temp-file debris in the tree, README 11.9k chars. Deliberately
open, each with its trigger or rejection logged: int8-domain Q4_K dot,
shared-prefix KV (no paying caller), sampled cuts (exact-reuse only),
scan vectorization (single digits), llama re-measurement (no binary),
longer drafts (rejected on mechanism), fused Q4_K (reverted: hadd tax),
prefill tail groups (physics), decode-thread default (needs multi-box
evidence), weight packing (unverifiable intent). Nothing further has
positive expected value without those triggers — manufacturing more would
violate the frugality the whole series was run under. (Addendum: the drive
did resume afterwards for §28ai record-keeping and the §28aj sums
experiment, both closed; the sentence above stands as written at the time.)

### 28ai. Hardware identity: this box IS the reference (kept, record only)

`tge info` on this machine: `Intel(R) Core(TM) i5-14400F`, 10 physical /
16 logical (6P+4E), AVX2+FMA+F16C+AVX-VNNI, 16.00 GiB installed — the exact
box the report and §§1-27 measured on (4.08 GiB free / 74% in use at check
time, hence the load noise throughout §28). Consequence: every §28 A/B is
same-hardware, differing only in box load at the time — the "different
box" hedge some entries carry is withdrawn, the load caveats stand. In
particular the §28ab pin reversal and the §25/§28 42-47 t/s decode figures
are same-machine results, and the i5-14400F llama.cpp pairs (§19: 45.1
short, 46.9 at 2k, 352.2 pp512) remain the valid comparison targets for a
re-measurement under a quiet box. No code, no numbers changed. Tests 71/71,
4/4 gates.

### 28aj. Vectorised sampler sums (REVERTED: -0.4% paired, premise wrong)

Hypothesis: with the exp vectorised (§28t), the remaining full-vocab
scalar sums (softmax totals, categorical and accept totals) were "the
other half" of sampling cost. Implemented `sum_avx2` (horizontal AVX2
reduction) in the sampler TU, wired into all three totals with scalar
fallback, differential-tested at 1e-5 and chi-square-gated like the exp.
Paired verdict, same seed and trajectory (4 steps, 12 accepted, 15
emitted): sampling 234.14 ms before vs 233.25 ms after — noise, not a win.
Fully reverted (kernel, decl, wiring, test; tree verified clean of every
trace, suite back to 71/71, greedy ids identical).

The correction matters more than the revert: recomputing the breakdown, a
0.6 MB horizontal sum is ~0.05 ms against a ~10 ms per-position sampling
cost — the sums were never more than ~1% of it; the branchy resample and
categorical walks dominate. Lesson recorded alongside §28i: profile the
breakdown before vectorising, don't assume it from operation counts.
The walks stay scalar deliberately — unrolling them would trade the same
rounding questions for single digits. Tests 71/71, 4/4 gates.

### 28ak. Batch GEMM audit: already at 97% of its roof (kept: no change)

Read `gemv_q8_batch_avx2` end to end looking for a repeat of the §26a
tail-class bug or scalar leftovers. There are none: the widen is shared
across all 6 tokens, the tail shares one block read, the horizontal sum is
once per row per token, per-block scalar work is one `half_to_float`.
The roof arithmetic closes the case: per 32-weight block the kernel streams
34 bytes of weights and does 6 tokens x 32 mult-adds x 2 = 384 flops, i.e.
11.3 FLOP/byte; at the measured 27 GB/s streaming rate that roofs at
~305 GFLOPS, and `bench batch ffn` reports 295 GFLOPS — 97%. The ~500 GFLOPS
ALU peak is irrelevant; this kernel is bandwidth-bound on weights, and the
only way past its roof is fewer bytes (VNNI int8 dot, already opt-in at
+14-19%) or fewer passes (speculation). No code change; the "weight
packing" roadmap line stays as-is (packing more would not move a
bandwidth-bound kernel). Tests 71/71 (untouched), 4/4 gates.

### 28al. Live llama.cpp head-to-head (kept: TGE wins decode ~2x, trails prefill)

The since-day-one blocker is gone: llama.cpp master @ cd8427d shallow-cloned
outside the repo, `llama-bench` + `llama-cli` built with the same MinGW-GCC
16.2 toolchain TGE uses (plus `-D_WIN32_WINNT=0x0A00` for a cpp-httplib
MinGW guard; the `.rsrc merge` link notice is benign). Same box, same
models, same 6 threads, back-to-back windows:

| Q8_0, 6 threads | llama-bench (cd8427d, GCC) | TGE (this tree, GCC) |
|---|---|---|
| pp128 / prefill-126 | 276.67 t/s | ~220 t/s (light window; 76-141 loaded) |
| tg64 / decode-64 | 16.92 t/s | 41.33 t/s |
| pp512 / tg128 | 392.35 / 18.48 t/s | — |
| Q4_K_M pp128 / tg64 | 224.42 / 17.06 t/s | prefill ~222 / decode 31.40 t/s |

Two statements with different strengths. (1) Prefill: llama ahead ~1.3x,
consistent across the log's history (352 vs 261-266 with MSVC build 11624; 392 vs 250-268 now) — the prefill gap is the reproducible
structural fact. (2) Decode: TGE 28-52 t/s across every window measured
vs llama 16.9-18.7 across three configs — non-overlapping ranges, so the
direction is load-proof on these builds. Caveat that cuts both ways: both
binaries here are MinGW-GCC, while the historical parity (45 vs 45.1
short) was MSVC-vs-MSVC; part of llama's decode deficit may be GCC codegen
rather than engine design, and re-running under MSVC (or with the vendor
binary) would isolate it. `tge fidelity` already showed the logits agree
(KL ~1e-11), so this is a systems gap, not a math gap.

Side find with its own future: `.modelduck` holds Llama-3.2-3B, Phi-3.5-mini,
Qwen2.5-1.5B (+coder), Q5_K_M and SmolLM2-135M — local material for the
roadmap's "run other models" and draft-model items. Tests 71/71, 4/4 gates.

### 28an. Qwen2.5-1.5B runs with zero changes (kept, new-model validation)

The §28al side find paid off immediately: `qwen2.5-1.5b-instruct-q4_k_m`
(28 layers, 1536 dim, 12/2 GQA heads, 151936 vocab) loads and generates with
no code changes — the config-driven paths (GQA splits, batch shapes, RoPE
tables) generalize. Full-program fidelity on the new shapes: mean KL
7.419e-12, worst 3.257e-10, max |diff| 2.516e-4, top-1 1023/1023 — the same
floor as 0.5B, so the vectorisation program transfers wholesale.

| 1.5B Q4_K_M, 6 threads | llama-bench (cd8427d, GCC) | TGE (this tree, GCC) |
|---|---|---|
| pp128 / prefill-128 | 178.15 t/s | 73.4 t/s (loaded box) |
| tg64 / decode-63 | 12.47 t/s | 11.88 t/s (inside load noise) |

Same pattern as 0.5B at larger scale: decode at parity, prefill trailing
(~40% here under load vs ~75-88% in lighter windows — the ratio moves with
the box, the ordering does not). The 1.5B run also exercises per-head
attention at 12/2 grouping and the largest batch matrices yet, all green.
Queued next: Q5 AVX2 now has its validation target
(`qwen2.5-0.5b-instruct-q5_k_m.gguf` sits in the same directory).
Tests 71/71 (untouched — no code changed), 4/4 gates.

### 28am. Q5_1 dequant from scratch: scalar oracle, ggml-verified, AVX2 (kept)

Plot twist first: the "q5_k_m" file is 133xQ5_1 + 12xQ5_K + 12xQ6_K —
its hot format is legacy Q5_1, which TGE parsed but refused to run
(`unsupported type for token_embd.weight`). So this entry implements the
whole format: scalar oracle transliterated from ggml's
`dequantize_row_q5_1` (d/m/qh/qs layout verified field-by-field against
`ggml-common.h`: 24B blocks, hb bit i -> low half, hb bit i+16 -> high
half, `y = x*d + m`, no -16 bias unlike Q5_0), wired through `QuantKind`,
`quant_row_bytes`, `gguf type_supported`, and `to_quant_kind`.

Correctness in three layers, strongest first: (1) a C program linked
against the built ggml static libs quantized 64 random rows and dumped
both bytes and ggml-dequantized floats; a (temporary, since removed) TGE
test memcmped our dequant on the same bytes — bit-identical on all 2048
values; two of those blocks are now embedded as literals in the permanent
`q5_1_matches_ggml_reference` test. (2) Hand-computed single-block test.
(3) End-to-end: the file loads and generates fluent text; templated
greedy matches llama.cpp for 8 tokens (`...the world was`) then diverges —
and the Q4_K_M file (no Q5_1 anywhere) diverges at the same spot, which by
the report's near-tie doctrine pins the cause on the known float-vs-int8
activation gap, not on the new code.

AVX2 follows the Q5_0 LUT pattern (shared spread table, mul-then-add, no
FMA — the only arithmetic difference from Q5_0, which does mul-then-sub);
differential memcmp passes on 1- and 3-block random rows. `bench batch`
gains `--format q51` (row bytes cols/32*24).

| q5_1 kernel / model | scalar era | AVX2 |
|---|---|---|
| `bench batch` 896x4864 x 1 (best of 5) | — (format did not exist) | 0.379 ms, 8.62 GB/s |
| Q5_K_M story decode, 15-16 tok | 15.2 t/s | 37.1 t/s, ids byte-identical |

Same band as Q5_0 (8.4 GB/s) — expected, identical structure. The model-level
jump bundles a load shift with the kernel win (scalar era ran on a loaded
box), so the honest speed claim is the kernel number; the ids parity is
exact (bit-identical kernels, verified before/after on the same prompt).
Full-program fidelity on the Q5_K_M file prices at KL 1.1e-12, top-1
1023/1023. Tests 74/74 (71 + hand + ggml-vectors + AVX2 differential),
4/4 gates.

### 28ao. AVX2 Q5_K dequant, bit-identical (kept, kernel in the AVX2 band)

The file's remaining scalar format: 12 ffn_down tensors at 2.86 MiB each.
Q5_K is a 256-block K-quant (d, dmin, 12 scale bytes, 64 high bytes, 128
low bytes); per 64-chunk the 32 low bytes feed both halves (low nibbles,
then high nibbles of the same bytes) while one high bit per weight comes
from a shared 32-byte row, bit (2c) for the low half and (2c+1) for the
high half, chunk c — verified field-by-field against ggml's
`dequantize_row_q5_k` before vectorising (the scalar path already matched
it in spirit: same loop, same order, same `d1*q - m1`).

AVX2 mirrors the Q4_K approach with one new comparison: per 32 weights,
`cmpeq(high & (1<<bit))` yields 0xFF/0x00 bytes, masked to 0x10 and ORed
with the nibbles before widening (all loads unaligned per §28i); then
mul-then-sub, no FMA — bit-identical to the scalar oracle, confirmed by a
randomized memcmp differential (1- and 3-block). `bench batch` gains
`--format q5k` (row bytes cols/256*176) and the docs gain the row.

| q5_k batch 896x4864 x 1 | scalar era | AVX2 |
|---|---|---|
| best of 5, same window as q4k/q6k/q50/q51 sweep | (no fast path) | 0.307 ms, 9.75 GB/s |

Same-window sweep for context (loaded box, absolutes depressed): q4k 3.81,
q5k 9.75, q6k 1.96, q50 4.62, q51 0.38 GB/s — the spread is load noise, not
ranking (yesterday's lighter window had q4k 7.6, q5_0 8.2, q6k 8.06,
q51 8.62 in the same band). The keep rests on the bit-exact differential,
not on today's noisy absolutes: identical math, verified byte-for-byte, at
zero accuracy cost. Q5_K_M story decode holds 43.78 t/s with ids
byte-identical to the scalar run (`...578 323 22208`). Tests 75/75
(74 + AVX2 differential), 4/4 gates.

### 28ap. Speculation transfers to 1.5B unchanged (kept, no code)

First spec pass ever on the bigger shapes (28 layers, 1536 dim, 12/2 GQA):
repetitive prompt, `--threads 6`, back-to-back. Greedy: 2 steps, 10 accepted
(5.00/step), 11 emitted — ceiling acceptance, same as 0.5B. Temp 0.1 + spec:
3 steps, 9 accepted (3.00/step), 11 emitted — identical profile to 0.5B's
3.00/step on the same prompt family. No new code, no new gates: the
verifier, drafter and accept loop are shape-generic by construction, and
the run confirms it (no crashes, no fallbacks, same acceptance bands).
Absolute t/s omitted — the box was loaded and single-window pairing
against non-spec runs wasn't taken; acceptance rate is the load-proof
metric here and it matches 0.5B exactly. Tests 75/75 (untouched), 4/4 gates.

### 28aq. Llama-3.2 runs: tied weights, no-bias fallback, interleaved RoPE

Three gaps closed in one arc, each verified before moving on. (1) The 3B
file has no `output.weight`: `load` now reuses the token-embedding rows
directly (GGUF stores them row-per-token, exactly the output layout), with
vocab size falling back to the embedding table when the output tensor is
absent. (2) It has no QKV bias tensors either (Qwen files ship zero-filled
ones): missing bias resolves to a shared zero vector instead of an error,
keeping the hot loops branch-free. (3) Llama uses interleaved RoPE
(ggml NORM: consecutive pairs), not Qwen's halves — the actual bug behind
the first decode ("b / ## Step 1. 1. 1"): after the fix the same prompt
gives "in a small village nestled in the rolling hills", and a factual
probe completes correctly ("The capital of Italy is Rome. ...Spain is...").

New `rope_interleaved` flag on the config, set from the architecture name
(`llama`-family only; NeoX stays the default everywhere else). The
AVX2 kernel duplicates each table entry ([c,c,...]) so one load covers 4
pairs, blended back with a 0xAA mask; the differential passed on first
build after fixing one swapped operand the differential itself caught
(odd-lane `x*s` vs `xs*c` — the harness earning its keep again).
Verified: hand-computed +90-degree rotation, memcmp AVX2-vs-scalar,
76/76 tests, Qwen ids byte-identical (the NeoX path is untouched), and
fidelity on 3B gives mean KL 1.24e-12 with unanimous top-1.

Explicitly NOT done here: tokenizer parity (TGE tokenizes Llama's `\n\n`
as 271, llama.cpp as 881 — pre-tokenizer work, separate task) and any
templated end-to-end (blocked on the former). Absolute t/s omitted — cold
box, 2 GB model file; throughput benchmarking comes after correctness.
Tests 76/76 (one new interleaved differential), 4/4 gates.

### 28ar. Llama tokenization reaches parity: tiktoken digit groups (kept)

The 24-case corpus comparison found exactly ONE divergence class between
TGE and llama.cpp on identical strings: multi-digit numbers. TGE split
every digit ("1999" -> [16, 24, 24, 24]); tiktoken groups left-to-right in
threes ("1999" -> [2550, 24], "12345" -> [4513, 1774]). Everything else —
whitespace runs, newlines, contractions, URLs, code, CJK, chat-template
tokens — already matched bit-for-bit, which also resolved the earlier
"\n\n" scare: standalone it is 271 on both sides (the 881 sighting was a
truncated-output misread, corrected before any code changed).

`tokenizer.ggml.pre` selects the rule: `llama-bpe` groups up to 3 digits
per piece, anything else (including Qwen's `qwen2`, missing, or unknown
values) keeps the established single-digit splitter bit-identical. New
permanent test builds synthetic vocabs with and without the flag and
asserts exact ids both ways (Qwen path covered by the unchanged existing
suite — 77/77 green, zero regressions).

Templated end-to-end (full system block, today's date, verified token
stream): TGE "...a specific year 1999 having 12345 stars. The number of
stars in the universe is estimated to be around 100-400 billion..." vs
llama "...a reliable information about the number of stars... also around
100-400 billion...". Same rare fact, parallel structure, divergence after
a shared prefix — the near-tie signature the project already attributes to
float-vs-int8 arithmetic (§28am), not a model bug: dequant is ggml-verified
bit-exact, RoPE is hand+differential verified, and tokenization is now
provably identical input. Absolute t/s omitted (loaded box); sampled
decode on 3.2B runs. Tests 77/77, 4/4 gates.

### 28as. Llama-3.2-3B head-to-head: same band on decode, prefill trails (kept, measured)

Same box, same file (Q4_K_M), same 6 threads, back-to-back-ish on a loaded
box — so treat absolutes as ranges, not rankings:

| 3B Q4_K_M | llama.cpp cd8427d (GCC) | TGE (GCC, this tree) |
|---|---|---|
| prefill | 74-79 t/s | 25-32 t/s (55-tok templated prompt) |
| decode | 8.7-9.6 t/s | 7.4 t/s (12-16 tok runs, early EOS) |

Decode is the same band (both ~8); prefill trails by ~2.5x, wider than the
~1.3x gap on 0.5B. Caveats, in order of importance: (1) the box was loaded
and TGE's prefill number comes from a 55-token prompt where fixed costs
(tokenize, first-touch page faults on a 2 GB file) dominate — not a
steady-state rate; (2) both binaries are the same MinGW-GCC build, so no
toolchain confound, but llama.cpp's prefill path is heavily tuned batch
GEMM while TGE's is the same kernel that hits 97% of *its* roof on 0.5B
(§28ak) — the roof itself may differ here (larger matrices, different
arithmetic intensity). What this rules out: any order-of-magnitude gap —
TGE runs 3B models correctly and at comparable decode speed today.
A quiet-box rerun with matched prompt lengths stays on the list. Tests
77/77, 4/4 gates.

### 28at. Phi-3.5-mini loads: fused QKV, fused gate|up, epsilon key (kept, measured)

TGE-PHI3 [1/4]: the Phi-3.5-mini Q4_K_M file (2.28 GB, phi3 arch, 32 blocks,
3072 dim, 32/32 heads, vocab 32064) previously stopped at `missing tensor
blk.0.attn_q.weight`. Three load-path gaps, each bounded and tested:

- Epsilon lived only under `phi3.attention.layer_norm_rms_epsilon` (1e-5)
  while the loader read the unprefixed key and silently ran 1e-6 — correct
  by accident on Qwen, wrong on Phi. The loader now tries the
  `attention.`-prefixed key first, then the legacy key, then 1e-6.
- Fused tensors map to row views: `attn_qkv` [q; k; v] splits into dim +
  2 x kv_dim rows, fused `ffn_up` [gate; up] into two ffn halves
  (HuggingFace `gate_up_proj` order). K-quant super-blocks never cross a
  row, so every kernel consumes the views unchanged. Anything that is not
  exactly those geometries throws instead of running mis-sliced.
- `rope.dimension_count` != head_dim now throws (partial RoPE has no path);
  96 == 96 on this file, so no behavior. YaRN factor tables are recorded
  (`has_yarn_factors`) but not applied — a long-context divergence the
  parity step [2/4] will price.

Proof: three synthetic-file unit tests (epsilon chain, partial-rope
rejection, one-block fused load with exact row geometry + YaRN flag), each
failing on the old path by construction. On the real file: `tge plan`
reports the shape (2.23 GiB weights, 768 KiB KV/token), and `generate`
with `--context 64 --tokens "1 2 3" --n 12` exits 0 — prefill 5.4-5.7 t/s
(cold fault-in of a 2.2 GB file), decode 5.7-6.1 t/s, ids varied and
in-vocab (`29958 13 13 29966 ...`), sample 0.0% of decode. Untouched
forward path: Qwen/Llama behavior bit-identical (split taken only when the
unfused names are absent). Gate/up order and RoPE mode are the documented
HF/ggml convention; end-to-end parity against a reference settles them in
[2/4]. Tests 80/80, 4/4 gates.

### 28au. Phi-3.5 tokenizes 28/28 vs llama.cpp; talks back correctly

TGE-PHI3 [2/4]: the Phi tokenizer rule. The file is `tokenizer.ggml.model
== "llama"` with no merges table, so the engine armed a new SentencePiece
path (BPE files keep the merge path bit-identical): dummy prefix,
space-to-U+2581, then greedy highest-score pair merges. Two wrong turns on
the way there are worth recording. First, Viterbi maximization is the
wrong objective for these scores: they are merge ranks, not log-probs
(the 0.0-scored byte fallbacks would beat every real piece). The
reference algorithm is llama.cpp's `llm_tokenizer_spm` merge loop, ported
line for line (score-max queue, leftmost tie-break, staleness by merged
size, resegment-or-bytes). Second, GGUF SentencePiece pieces are raw UTF-8
(`▝Hello` is E2 96 81...), not GPT-2-mapped — an early mapped-domain
attempt matched nothing and fell back to bytes everywhere.

Proof is a 28-string corpus against two oracles: 26/28 match HuggingFace
BPE as well; the 2 template strings diverge from HF structurally
(per-span prefix plus merge vs BPE-with-merges) and match llama.cpp's
`llama-tokenize` (built from source for the occasion) on all 28/28. One
corpus string initially blamed the tokenizer for CJK but the console was
the culprit: narrow argv arrives as `?`, so `main()` now rebuilds argv
from the wide command line and the CJK case matches byte-for-byte.

End to end: `<|user|>What is 2+2?<|end|><|assistant|>` decodes 12 tokens
to "The sum of 2+2 is 4." — the fused QKV split order, the fused gate|up
order, full-head NeoX rotation, the 1e-5 epsilon, and the new decode path
(U+2581 back to space, controls omitted) all survive contact with the
real weights. YaRN factors stay recorded-but-unapplied: short of a
reference-logits run they are unpriced, so long-context Phi is flagged,
not claimed. Six synthetic unit tests (merge ranks, byte fallback,
control-forge refusal, BPE non-arming) plus the argv fix. Tests 83/83,
4/4 gates. Next: commit, then SmolLM2 probe or the Q5_K-vs-Q4_K pricing —
whichever the head-to-head needs first.

## 29. int8 KV cache (2026-11-02)

The KV cache is now optionally int8 blocks with one float scale per 32 values
(--kv i8), read by the attention kernels with widen-then-scale; the default
stays f32. 	ge fidelity --kv i8 prices it.

| Llama-3B Q4_K_M, 6 threads | f32 | f16 | i8 |
|---|---|---|---|
| 512 short decode (15 tok) | 8.6-9.6 t/s | — | 5.20 t/s (overhead, no KV pressure) |
| 2048 decode (31-63 tok) | 5.80 t/s | 6.61 t/s (+14%) | **7.63 t/s (+32% vs f32)** |
| 2048 prefill (11 tok) | 2.7 tok/s cold / 23.8 warm | 23.8 tok/s warm | 23.8 tok/s warm |

Short text is identical to f32; the 2048 output prefix matches the f32 run.
Fidelity over 31 positions at context 512: mean KL **3.166e-4**, worst
1.234e-3, max |logit| 1.9e-1, top-1 **31/31** — between the f16 (1.7e-6) and
the accepted VNNI (1.6e-3) trades, with zero top-1 flips either way.

The default stays f32; --kv i8 is for long-context or multi-session memory,
where the KV (448 MiB f32 at 2048 for this model) dominates. At short context
the widen+scale is pure overhead, so f32 stays faster there. The plan still
prints the f32 KV size, so the budget check stays conservative.
Tests: 84 -> 86 (int8 head and group differentials: AVX2 tracks scalar at
1e-4, scalar-int8 tracks float at 5e-2).

Follow-up: ench sessions takes --kv f32|f16|i8 now (it was the only
model-running command without the flag). Llama-3B, 2 sessions x 8 tokens x 6
threads, context 512, both orders:

| sessions 2x8 | f32 | i8 |
|---|---|---|
| aggregate | 4.18 / 7.21 t/s | 10.19 / 10.78 t/s |

Direction is consistent across orders with identical transcripts, but the f32
spread (73%) far exceeds the box's usual band — that day had heavy background
load, so read the gap as contention-shaped, not a clean 2x claim. The
plausible mechanism is the shared single-channel DIMM: two sessions stream
2 x 1.57 GiB of weights per step plus KV, and i8 quarters the KV stream.

## 30. Phase timers + Llama-3B decode attribution (2026-11-02)

`Qwen2` now accumulates wall nanoseconds in `matvec`/`batch_matvec`,
`attention`/`attention_batch` (+`forward_batch` inline region) and
norms/RoPE/SiLU (`rms_norm`, `apply_rope`, new `silu_gated` wrapper);
`generate` resets before prefill and before decode and prints both
`phases` lines next to `timers`. No arithmetic change: separate counters,
no reordered math. Tests 86/86; fidelity full-program mean KL 1.006e-11,
worst 3.113e-10, top-1 1023/1023 (same floor as §28z).

Same window, warm page cache, Llama-3B Q4_K_M, 6 threads (unless noted):

| probe | result |
|---|---|
| `bench mem` (256 MiB, best of 3) | copy 21.67, read 25.72 GB/s (box mid-load) |
| `bench batch` q4k 3072x8192 x1 (down, best of 5) | 1.025 ms, 49.08 GFLOPS, 13.81 GB/s |
| `bench batch` q4k 8192x3072 x1 (gate, best of 5) | 1.086 ms, 46.35 GFLOPS, 13.04 GB/s |
| `bench batch` q6k 3072x8192 x1 (down, best of 5) | 1.103 ms, 45.62 GFLOPS, 18.71 GB/s |
| `bench batch` q6k 8192x3072 x1 (best of 5) | 1.682 ms, 29.91 GFLOPS, 12.27 GB/s |
| short decode, 6-tok prompt, ctx 512, 14 tok | matvec 2813.0 ms (98%), attn 6.3 (0%), norm 55.5 (2%) |
| 100-tok prompt, ctx 2048, 19 tok | matvec 3012.2 ms (96%), attn 50.0 (2%), norm 61.4 (2%) |
| prefill, same runs | matvec 93-94%, attn 3%, norm 4-6% |
| threads 6 vs 10, short decode interleaved | 4.25 vs 4.42 t/s (+4%, below the 15% bar; default stays 6) |

Two corrections to the Stage-1 plan, both from measuring on true 3B
shapes instead of quoting 0.5B numbers. First, Q4_K is not at 7-9 GB/s
here: on 3072/8192 orientations it runs 13.0-13.8 GB/s (51-54% of the
25.7 wall), and Q6_K down hits 18.71 (73%). Weighted effective ~55% —
which *is* the end-to-end 52-58% figure, not a mystery overhead. The
§28i/§28l "format tax" stands, but its size on 3B is ~2x smaller than
the 0.5B-shape quote. Second, attention is 0-2% at both ctx (positions
100-120 here, per-head path; GQA never engages), so the GQA-retune branch
of the plan does not fire.

Verdict per the revised plan's gates (matvec >60% fires; kbatch <50%
wall fails at 51-73%; attn >30% fails at 0-2%): no kernel/ALU change.
The remaining unpack ALU (nibble split + per-32 scales + FMA) is the
format price — fused lost 14% (§28i), integer lost twice (§12), repack
banned (mmap-first). Thread sweep +4% keeps the default. What ships is
the instrument: every future decode claim now attributes instead of
guessing. Next triggers, not attempted: int8-domain dot that avoids the
hadd tax, or a 256+-history GQA A/B (this session never exceeded
position ~120).

## 31. Prefill Step 0: 64-wide batch on true 3B shapes (2026-11-02)

Revised-plan Step 0 (Stage 2): stop quoting 1-token kbatch at prefill;
measure the batch kernel at prefill width plus a warm-cache long-prompt
generate, same window, then let the decision gate fire. Box mid-to-heavy
load throughout (mem read 23.77 vs 28.49 on lighter windows; decode
2.5-2.8 t/s vs 4.36 earlier): absolutes depressed, within-window ratios
and shares are the evidence. gguf display is [in, out], so bench
rows=out, cols=in: gate/up Q4_K is rows 8192 x cols 3072, ffn_down Q6_K
is rows 3072 x cols 8192. All Q4_K on this file have cols=3072; the only
cols=8192 tensors are Q6_K down (and the lookup-only embedding).

| bench batch x64, best of 3, same window | time | GFLOPS |
|---|---|---|
| q4k rows 8192 x cols 3072 (gate/up shape) | 19.446 ms | 165.65 |
| q4k rows 3072 x cols 8192 (no 3B caller; asymmetry probe) | 49.071 ms | 65.64 |
| q6k rows 8192 x cols 3072 (gate shape) | 15.524 ms | 207.50 |
| q6k rows 3072 x cols 8192 (ffn_down shape) | 23.123 ms | 139.31 |

Same total elements, orientation-dependent rate: cols=8192 runs 1.5x
(q6k: 208 -> 139) to 2.5x (q4k: 166 -> 66) slower per element than
cols=3072, both formats, back-to-back. Working hypothesis (not yet
proven): the 8192-float (32 KiB) scratch row plus the 8192-float input
row exceeds the 48 KiB L1D, so every token's dot re-fetches scratch
from L2, while 12 KiB + 12 KiB fits; dequant writes are identical either
way. Testable by tiling the token loop (L1 blocking) with per-dot order
preserved, hence bit-identical output and no fidelity price beyond a
memcmp differential plus ids parity.

| generate, Llama-3B, 6 threads, --prefetch | prefill | phases |
|---|---|---|
| 120-tok prompt, ctx 2048 (cold file, prefetch 6.4 s) | 14.3 t/s | matvec 95%, attn 2%, norm 3% |
| 480-tok prompt, ctx 2048, warm (best of 2: 18.4 / 14.2) | 18.4 t/s | matvec 93%, attn 3-4%, norm 3% |

Per-layer FFN at 64-wide from the kernel table: 19.4 (gate) + 19.4
(up) + 23.1 (down, q6k) = ~62 ms per chunk; x28 layers is ~1.7 s per
64-tok chunk before attn/qkv/o/norms. End-to-end matches kernel rates:
no mystery loop overhead, the prefill gap is kernel efficiency
(139-208 GFLOPS vs ~300 roof), not dispatch. The 55-tok 25-32 t/s
baseline (Section 28as) and today's 14-18 t/s differ by box load, not
code; the llama 74-79 t/s side stands un-remeasured (llama-bench clone
not relocatable this session: not on PATH, not in home/Desktop roots;
full-home search times out -- follow-up, not this round).

Decision gate FIRES: drill down on cols=8192 in gemv_dequant_batch_avx2
(L1 token tiling, bit-identical), Q6_K down first (the real caller),
q4k wide-shape second as the same code path. Out: VNNI (no Q8_0 in
file), repack (mmap-first), thread default (knob exists), overhead hunt
(phases close it: matvec 92-95%). Tests 86/86 (suite untouched).

## 32. L1 panel tiling: falsified same-window, reverted (2026-11-02)

Hypothesis from Section 31: 8192-wide dots run slower per element
because 32 KiB scratch + 32 KiB input exceed the 48 KiB L1D. Change:
column panels of 1024 in gemv_dequant_batch_avx2 (4 KiB + 4 KiB per
dot, per-token accumulation in increasing column order), all K-quant
kinds, one code path. Differential written first
(dequant_batch_matches_scalar_reference: q4k/q6k, cols 1024/3072/3328/
8192, tokens 1/7/64, tolerance 1e-4 x cols + 1e-3): it failed pre-ship
on single-row test matrices read 3 rows out of bounds (heap garbage,
nondeterministic across runs) -- the harness earning its keep again;
fixed by sizing matrices to rows x blocks, green on the unmodified
kernel before the change.

| shape x64, best of 3 | pre-panel (Section 31 window) | post-panel (later window) | interleaved same-window old / new |
|---|---|---|---|
| q4k gate 8192x3072 | 165.65 GFLOPS | 186.40 | -- (not re-paired; gate was never the problem) |
| q4k wide 3072x8192 | 65.64 | 101.39 | R1 132.47 / 101.71, R2 93.81 / 119.75 |
| q6k gate 8192x3072 | 207.50 | 119.17 / 182.00 (re-samples) | -- |
| q6k down 3072x8192 | 139.31 | 128.11 / 90.13 (re-samples) | R1 110.35 / 108.63, R2 82.48 / 91.23 |

Cross-window reads (+54% on q4k wide) are load narrative: interleaved
old/new pairs flip sign between rounds (R1 -23%, R2 +28%; q6k down
-1.5% then +11%). Verdict: panels do not move any shape outside noise.
Mechanism correction: panels keep total input traffic identical (2 MiB
per 8192-wide row at 64 tokens), so if the cost is L2/DRAM re-reads of
the input per row, panels cannot help; the fix would need cross-row
input reuse (band + token blocking), which is bigger surgery and goes
back through its own plan/verify round -- not attempted here.

Reverted in full (kernel file back to the Section 31 state; diff shows
only the test file). Kept: the batch-vs-scalar differential (first
coverage of the K-quant batch path end to end; 86 -> 87 cases) and the
Section 31 numbers. No fidelity run: no numerics change ships. No
generate delta claimed: nothing lands. Tests 87/87, 4/4 gates.

## 33. Token-block + L2 band cap: no effect same-window, reverted (2026-11-02)

Second drill-down on the wide-col cost (Sections 31/32): token-blocks
of 8 with bands capped to ~256 KiB of input so each block stays
L2-resident across the band's rows; tokens==1 keeps the old loop
exactly (decode neutral by construction); grain scales dispatch.
Scheduling only, full-width dots in the same order -- bit-identical,
no fidelity price.

| q6k down 3072x8192 x64, best of 3 | old (stash-built) | new (token-block) | ratio |
|---|---|---|---|
| R1 | 102.68 GFLOPS | 102.36 | 1.00 |
| R2 | 61.09 | 65.24 | 1.07 |
| R3 | 83.83 | 127.88 | 1.53 |
| R4 | 98.30 | 72.36 | 0.74 |
| R5 | 66.03 | 93.25 | 1.41 |
| R6 | 88.14 | 83.04 | 0.94 |

Median ratio ~1.03, spread 0.74-1.53: the box spikes mid-window (same
binary swings 102 -> 61 between adjacent runs), so no kernel delta is
resolvable here today. Same-window pairing -- the rule that killed the
panel illusion -- kills this one too. Reverted in full; differential
reused unchanged (still green on both). Retry trigger, not attempted
now: a quiet box (mem read back at ~28, decode at ~4.5+) plus the same
interleaved pairs; the stash-built-binaries pattern from Section 32
applies. Tests 87/87.

## 34. Single-token Q4_K int8 dot: faster-looking, unresolvable + over price, reverted (2026-11-02)

Revised-plan Step 1 (zero-risk gate) first, same window, warm page
cache, Llama-3B Q4_K_M, `--threads 6`:

| 300-tok prompt, ctx 2048 (GQA engages >=256) | per-head | GQA (auto) |
|---|---|---|
| prefill | 22.9 t/s | 28.1 t/s (+23%) |
| decode, 9 tok | 4.64 t/s | 5.37 t/s (+16%) |
| decode attn share | 73.0 ms (4%) | 54.6 ms (3%) |
| ids, 10 tok | 2 14924 ... 18 18 | identical |

Short history (4-tok prompt, positions 4-34): head 4.32 vs gqa
4.19 t/s (0.97x), ids identical, attn 0-1%. Conclusion: the auto
schedule (>=256 GQA, else per-head) is already correct; no code
ships from this gate. The +16%/+23% long-history pair is suggestive
(single pair, box mid-load) not claimed.

Then the int8 prototype: matvec-only `gemv_q4k_i8_matvec_single`
(per-tensor activation quant once per call, shared read-only across
rows; per-32 int32 `maddubs`+`madd` accumulation with one reduction
site; float finalize per scale group), batch kernel untouched so
prefill tails, verifier and the `dequant_batch` differential keep the
shipped float path. Two findings, one fatal to landing:

1. TLS-over-pool fault, fixed during the attempt and recorded: the
   first version named the calling thread's `thread_local` activation
   buffers inside the pool lambda. Workers resolve TLS names to their
   own (empty) instance -- `r10=0` at
   `vcvtsi2ssl (%r10,%rax,4)` in gdb -- so threads=1 passed and
   threads>=2 segfaulted (0xC0000005). Fix: plain shared pointers
   (`aq_shared`, `sums_shared`) captured by the lambda; the float
   scratch stays worker-`thread_local` by design (each worker's own).
   Lesson alongside §28i: never name a `thread_local` inside a pool
   lambda unless every thread fills its own.
2. Same-window A/B (stash-built float vs int8, short decode 14 tok):
   OLD 4.77 vs NEW 6.53 (+37%) in round 1, then NEW 5.04 and 2.78 on
   back-to-back repeats of the same binary. OLD sits inside NEW's own
   2.78-6.53 spread (box spiking mid-window; `bench mem` read 15.04
   vs 25.72 lighter windows). Per the §28m rule only interleaved
   pairs count: no win is resolvable. Short ids identical either way
   (26061 10 ... 426, all runs), so the distribution shift below did
   not flip this greedy trace -- noted, not claimed as parity.

Fidelity (the independent revert trigger): 31 positions, ctx 128,
`docs/architecture.md`: mean KL **8.095e-3**, worst 2.703e-2, max
|logit| 1.24, top-1 **28/30** -- ~5x the accepted VNNI (1.6e-3) and
~500x the f16-KV (1.7e-6) trade, with 2 flips. Per-tensor int8
activation quant (outliers crush the scale) is the price; per-64 or
per-32 activation scales were not attempted once the speed gate
failed first.

Reverted in full (kernel file, `kernels.hpp` decl, `qwen2.cpp`
matvec dispatch -- `git checkout --` all three, tree verified clean
by `git status`); post-revert: 87/87 tests, fidelity back at mean
KL 1.467e-11, worst 5.5e-11, 30/30, short ids unchanged. No
differential or bench-harness change ships (batch path was never
touched). Next triggers, not attempted: per-group activation scales
(which re-add the quant cost the prototype tried to remove), or a
quiet-box rerun of the same stash pairs.

## 35. GQA threshold sweep 256->64: leans GQA, inside noise, no move (2026-11-02)

Revised-plan Step 0 (no code): forced `--attention head` vs `gqa`
pairs at histories 4/64/128/192/300, Llama-3B Q4_K_M, `--threads 6`,
warm cache, back-to-back. Prefill paths never consult `uses_gqa`
(Stage 2: `attention_batch` and `forward_batch` are per-head always),
so this gate is decode-only by construction; the speculative pin
forces `Head` explicitly and is unaffected.

| history | head decode | gqa decode | ratio | ids |
|---|---|---|---|---|
| 4 (story) | 4.32 t/s | 4.19 (0.97x) | 0.97 | identical |
| 64 | 4.88 / 4.17 | 5.25 / 4.74 | 1.08 / 1.14 | identical |
| 128 | 4.94 (R2) | ~5.2 equiv fwd (R1) / 4.04 (R2) | ~1.05 / 0.82 | identical |
| 192 | fwd 2173.05 ms | fwd 2096.62 ms | ~1.04 | identical |
| 300 (§34) | 4.64 | 5.37 | 1.16 | identical |

Median ratio ~1.05, spread 0.82-1.16: the H128 round-2 flip (head
+22% one way after gqa +5% the other) is the §32/§33 signature --
same-window pairing kills it. Nothing <256 wins outside the ±20%
box band, so the Step-1 gate (move `256` in `uses_gqa`) does not
fire; the constant stays, no code changed, no fidelity run owed
(no arithmetic exists to price). Kept: the bracket itself -- if a
quiet box ever shows gqa consistently ahead at 64+, the move is one
line plus the full-corpus fidelity (which spans the threshold) and
auto-vs-auto pairs. Tests 87/87 (suite untouched). Attn shares stay
1-4% throughout; matvec 95-97% -- decode remains a weight-streaming
problem on this DIMM, and the remaining gap is the §30 format tax,
not scheduling.

## 36. Matched head-to-head vs llama.cpp cd8427d on Llama-3B (2026-11-02)

First fair 3B matrix (supersedes the mismatched §28as snapshot):
same file, same 6 threads, warm page cache, interleaved, both
MinGW-GCC builds. TGE `--prefetch` (fault-in untimed); llama-bench
warms up internally (`-r 5` stats shown). TGE f32 default vs llama
f16-KV default declared; one TGE `--kv f16` secondary. Prompts
matched by token count (TGE id-list prompts; llama built-in).

| test | llama-bench cd8427d | TGE (this tree) | TGE/llama |
|---|---|---|---|
| pp128 | 38.19 ± 3.33 | 24.6 / 29.3 (128-id prompt) | 64-77% |
| tg64 @128 | 7.96 ± 1.01 | 7.56 / 10.00 | 95-126% |
| pp512 | 95.35 ± 7.75 | 44.3 (512-id prompt) | 46% |
| tg64 @512 | 10.77 ± 0.24 | 10.25 | 95% |
| TGE-512 `--kv f16` | -- (same llama col) | 36.6 / 8.26 | unpaired drift, no format claim |

`bench mem` wall this window: read 15.63 GB/s (mid-load; lighter
windows 25-28). TGE `phases` on the 512 runs: prefill matvec 88%,
attn 3%; decode matvec 92%, attn 7% (GQA engaged, positions 512+).

Two statements with different strengths. (1) Decode: parity across
every pairing (95-126% short, 95% long; llama's own ±13% on tg64
shows the band). The GOAL short baseline (8.6-9.6) and the 2048-ctx
f32 line (5.80) are TGE-vs-wall numbers, not TGE-vs-llama; against
llama, decode is not the gap. (2) Prefill: trails at every length,
and widens with prompt length (64-77% at 128, 46% at 512) -- the
§31 wide-col kernel story (q6k down 111 vs q4k gate 208 GFLOPS,
matvec ~90% of prefill), now confirmed end-to-end against the
reference engine instead of inferred from kernel rates. The 4-token
story prefill (12.5/24.5 across runs) is fixed-cost dominated and
not compared.

What this aims: the next code round attacks prefill batch-kernel
efficiency on 3B shapes (gate/up Q4_K 8192x3072, down Q6_K
3072x8192, 64-wide), not decode. f16 secondary is unpaired across
time (slower than the f32 run here, contradicting the paired §29
+14% -- box drift, correctly unclaimed). No code changed, no
fidelity owed, tests 87/87 untouched.

## 37. Tokens-sweep attribution: aborted, box throttling (2026-11-03)

Round-4 sweep (Stage 2 verified the K-path has no group/tail kink,
so `time(tokens)` should read fixed-vs-variable cleanly). Same
window, interleaved, best-of-5, wall read 21.14 GB/s, 76% RAM in use:

| point | time | GFLOPS |
|---|---|---|
| q4k gate x1 | 1.775 ms | 28.36 |
| q6k down x1 | 9.339 ms, retry 6.670 | 5.39 / 7.55 |
| q4k gate x4 | 14.338 ms | 14.04 |
| q6k down x4 | 3.148 ms | 63.96 |

q6k x4 (3.148 ms) faster than q6k x1 (9.339/6.670 ms) on the same
binary back-to-back is physically impossible for this kernel --
proof the x1 invocations were throttled throughout (all 5 reps),
not a kernel shape effect. q4k x1->x4 at 8x for 4x tokens is the
same spike from the other side. Best-of-N cannot save runs where
contention persists across the whole invocation.

Verdict per the round-4 plan's honest-outcome clause: no slopes, no
aim, no surgery. The 16/64 points were not run -- two inversion
proofs already settle unresolvability, more points would only spend
wall time manufacturing variance. Retry trigger: same 8-point
interleaved matrix on a quiet box (mem read back ~28, `tge info`
available well above 50%, no x1>x4 inversion on the first pair);
if the retry also inverts, the bench harness itself needs a
contention guard before any prefill kernel work resumes. No code
changed, no fidelity owed, tests 87/87 untouched.

## 38. Tokens-sweep retry on quiet box: variable-side, wide-row (2026-11-03)

Retry of §37 on the trigger box: 8.24 GiB avail (48%), wall read
**31.60 GB/s** (highest this turn). Interleaved, best-of-5, no
inversion anywhere (first-pair tripwire clear):

| tokens | q4k gate 8192x3072 | q6k down 3072x8192 |
|---|---|---|
| 1 | 0.712 ms, 70.67 GFLOPS | 0.964 ms, 52.20 GFLOPS |
| 4 | 0.982 ms, 204.95 | 1.544 ms, 130.39 |
| 16 | 2.783 ms, 289.34 | 3.751 ms, 214.70 |
| 64 | 9.649 ms, 333.82 | 16.497 ms, 195.27 |

Marginal ms/token between points (q4k): 0.090 / 0.150 / 0.143 --
flat ~0.14 after x4. Same (q6k): 0.193 / 0.184 / **0.266** --
rises at x64. Intercepts (whole-matrix fixed: dequant + dispatch):
~0.57 ms q4k, ~0.77 ms q6k -- 5-6% of the x64 totals. The gap is
**not** the fixed dequant side: at x64, 94-95% of both totals is
variable, and q6k's marginal (0.266) is 1.86x q4k's (0.143).

Reading: per-token dots on 8192-wide rows cost ~2x per element
what the same FLOPs cost on 3072-wide rows, and the penalty grows
with the token set (0.19 -> 0.27). Capacity arithmetic fits: per
dot working set is scratch + input row = 24 KB at cols 3072
(inside the 48 KB L1D) vs 64 KB at cols 8192 (spills every dot).
This is the §31 L1 hypothesis with slope evidence behind it --
and it reconciles §§32-33: both falsified tilings kept total
traffic identical (my own §32 verdict), so on the variable side
they could never move; §33's band+token shape failed on a
throttling box (spread 0.74-1.53), which is noise, not
falsification.

Decision: the next code round retries cross-row input reuse
(band + token blocking that keeps the input set resident across
rows -- the §33 shape) on a box this quiet, aimed at the q6k-down
x64 point (16.497 ms) with the q4k-gate slope as the par line.
Fixed-side (dequant) work is out: 5-6% cannot pay. No code this
round; no fidelity owed (read-only bench); suite untouched 87/87.

## 39. Token-block/rows interchange: consistent harm, reverted (2026-11-03)

Round-6 implementation (revised plan: token-block-outer/rows-inner
inside each band, Tb=8, `tokens==1` loop verbatim; no new
thread_locals -- §34 lesson held). Suite 87/87 on the new binary
(the 1/7/64 batch differential covers the new nest, partial block
included). Stash-built old/new, interleaved, quiet-ish window
(drifting heavier through it):

| x64, best of 5 | OLD (stash) | NEW (interchange) | ratio |
|---|---|---|---|
| q6k down (target) | 16.560 ms, 194.52 GFLOPS | 16.926, 190.31 | 1.02 |
| q4k gate (control) | 10.566 ms, 304.87 | 11.743, 274.32 | 1.11 |
| q6k x1 (decode path) | -- | 0.937, 53.71 | identical loop, matches §38 0.964 |

Verdict: revert. Target null-to-harmful (+2%), control harmed
(+11%, consistent sign) -- the interchange pays 8x dequant
recompute (once per row per token-block vs once per row) for input
reuse the L3 largely already provides. The narrow shape, which
never spilled, pays the full recompute for zero benefit: exactly
the cost flagged in deliberation, confirmed by the control.
§38 baselines (16.497/9.649) vs OLD here (16.560/+0.4%,
10.566/+9.5%) show the window closing mid-round -- the quiet box
did not hold, which flatters nothing here since both shapes move
the same (harmful) direction.

Not attempted: two-phase (pre-dequantize the band once into a
~1.5 MB buffer, then block dots) would remove the recompute but
adds its own traffic and a 16-worker footprint -- needs its own
plan/verify round, and the single-phase result gives it no tailwind.
Reverted in full (stash dropped, tree verified clean); post-revert
87/87. No fidelity run: no numerics change ships (differential
green on the new nest before revert).

## 40. Vocab projection attribution: proportional, not the gap (2026-11-03)

Round-7 sweep on the quietest box yet (wall read **54.46 GB/s**,
copy 44.03): the output projection reuses token_embd
(`128256x3072 Q6_K`, 308.2 MiB) as a tall-skinny batch --
128256x3072 x{64,16,1}, best-of-5, same kernel as the §31 shapes:

| tokens | time | GFLOPS | effective GB/s |
|---|---|---|---|
| 64 | 188.542 ms | 267.49 | 1.71 |
| 16 | 57.211 ms | 220.38 | 5.65 |
| 1 | 11.735 ms | 67.15 | 27.54 |

Marginals ~2.7-3.0 ms/token (flat); x1 is pure weight streaming
(308 MB at 27.5 GB/s). Share per 64-chunk: ~189 ms against an
FFN bill of ~28 x 62 ms (gate+up+down, §31) ~ 1.7 s plus
qkv/o/attn/norms -- roughly **7-8% of the chunk**, consistent with
the closed warm-prefill totals. Per decode token the full-vocab
logit vector costs ~11.7 ms (~5-10% of a short-ctx step) -- real,
and already inside the 55%-of-ceiling decode figure, not a new
claim.

Honest null, twice over: at 267 GFLOPS the projection is the
*fastest* per-FLOP of the q6k shapes (down: 195), so tall-skinny
scheduling is healthy (128k rows feed all workers; 64 bands hold),
and its share is single digits. The §31 FFN estimate being rough
does not move the gap: prefill still trails llama 46% at 512 with
the deficit concentrated in gate/down dots (§§36/38). The
round-8 proposal stays the §38 one (input-residency work on the
wide-col path), not band retuning for 128k rows. No code, no
fidelity owed, suite untouched 87/87.

## 41. KV-ladder head-to-head on the quiet box: i8 not a general card (2026-11-03)

Wall 48.15 GB/s (quietest window yet). TGE best-of-1s per cell
(single runs -- the ladder gaps exceed the noise band, but treat
prefill deltas <10% as unclaimed) vs llama `-r 5` stats,
interleaved per history, same file, 6 threads:

| 128-history | llama | TGE-i8 | TGE-f16 | TGE-f32 |
|---|---|---|---|---|
| prefill | 87.88 ± 6.87 | 44.2 | 43.8 | 37.7 |
| decode 63 tok | 14.70 ± 0.85 | 9.22 | 9.11 | 8.90 |

| 512-history | llama | TGE-i8 | TGE-f16 | TGE-f32 |
|---|---|---|---|---|
| prefill | 92.17 ± 7.63 | 39.7 | 37.3/38.5 | 39.0 |
| decode 63 tok | 15.07 ± 0.10 | 7.81 | 9.48/9.99 | 9.09 |

Best-TGE/llama: prefill 50% (128) / 43% (512); decode 63% / 66%.
Two findings:

1. i8 is not a general head-to-head card. The §29 +32% was
   2048-ctx steady state where KV reads dominate; at 128/512 the
   ladder collapses (128: all within 4%; 512: f16 best, i8
   *worst* at 7.81 -- quant/dequant overhead exceeds the KV-read
   saving when attention is ~7%). i8 stays a long-context opt-in
   with its §29 price, not a goal-round lever at these histories.
2. Structural: on the quiet box llama decode nearly doubled
   (§36: 7.96/10.77 -> 14.70/15.07) while TGE decode did not move
   (§36: 7.56-10.25 -> 8.90-9.99). TGE decode does not scale with
   freed bandwidth -- it is unpack-ALU-bound (§30 format tax);
   llama is bandwidth-bound and pulls away exactly when bandwidth
   is plentiful. The goal gap for decode *widens* on a quiet box.
   Beating llama therefore requires removing ALU work per streamed
   weight (the §34 int8 attempt, failed on price) or removing
   streamed weights per token -- not scheduling, not thresholds,
   not KV formats at these histories.

No code, no fidelity re-price (i8 price stands from §29; f32/f16
paths numerically untouched), suite 87/87.

## 42. Drafter hit-rate round: recent-first reverted, gapped fallback kept (2026-11-03)

Revised-plan implementation (Stage 2: draft-model rejected -- SmolLM2
vocab 49152 vs Llama-3B 128256, cross-vocab drafts are meaningless to
`verify_draft_logits`; no same-vocab small Llama is local, so that stays
a trigger-gated future). Two drafter-only changes, each gated separately
on acceptance/step (deterministic, load-proof) with ids byte-identical;
box breathed 57.2 -> 47.9 -> 59.2 read GB/s through the round, so no t/s
ratio is claimed -- same-steps/same-accepts pairs with different
absolutes are load narrative per §28m (the §32/§39 rule).

Reference model throughout: Llama-3.2-3B-Instruct-Q4_K_M, `--threads 6`,
`--prefetch`, warm page cache. Baseline (stash-built pre-change binary,
kept at `Temp/kilo/tge_ab/tge_old.exe` for interleaved pairs): story
4-tok 0 steps; repetitive 40-tok 3 steps 5.00/step (plain 10.72 vs spec
18.06 t/s, 1.68x, ids `578 11422 ...` identical); partial-repetition 6
steps 2.00/step ids `3842 13452 ...` identical; mixed
alpha/beta/gamma 3 steps 4.00/step (12 acc, 14 emitted) ids `40097 13
8451 ...` identical; fidelity floor mean KL 1.467e-11, worst 5.531e-11,
30/30; suite 87/87.

(a) Recent-first span order (REVERTED): `prompt_lookup_draft` tried
candidate ends most-recent first (longest span at each end) instead of
the shipped longest-span-first. Acceptance tied on every family (5.00 /
2.00 / 4.00 / 0 steps, ids identical everywhere incl. 4.00 on mixed R1).
Same-work t/s pairs leaned new (+12-18% on mixed bests) with identical
step/accept/emission counts -- same weight passes, so physics says load
or placement, not code; claimed as nothing. Reverted in full (`git
checkout -- apps/cli/main.cpp`, tree verified clean); the shipped order
stands for want of evidence either way.

(b) Skip-one gapped fallback after the bigram chain (KEPT): new
`gapped_draft` in `apps/cli/main.cpp` tries spans 5..2 with exactly one
substitution, most-recent window first, only when the exact spans AND
`bigram_chain_draft` both abstain -- every pre-existing proposal is
byte-identical by construction (exact and bigram paths untouched; the
old tail `return bigram_chain_draft(...)` is now try-bigram-then-gapped).
A first-position miss skips the weight pass (greedy d0 pre-check,
sampled pre-judge §28s), so a wrong proposal costs microseconds. Drafter
lives in main.cpp, covered end-to-end (no unit surface, per §28ad).

| prompt, Llama-3B spec | before (old binary) | after (gapped) |
|---|---|---|
| repetitive 40-tok | 5.00/step, ids identical | 5.00/step, ids identical |
| partial-repetition | 2.00/step, ids identical | 2.00/step, ids identical |
| story 4-tok | 0 steps (correct abstention) | 0 steps (correct abstention) |
| mixed alpha/beta/gamma | 4.00/step (12 acc, 14 emitted, 4 runs) | **4.67/step (14 acc, 16 emitted)** |
| mixed spec vs nonspec (new binary) | -- | ids byte-identical |
| fidelity 31 pos ctx 512 | mean 1.467e-11, worst 5.531e-11, 30/30 | identical 1.467e-11, 5.531e-11, 30/30 |

The mixed delta is code, not load: greedy drafts are deterministic in
history, so acceptance cannot move with box state (old pins 4.00 across
four runs spanning the whole wall swing; new gives 4.67). Trajectory
diverges exactly where the gapped window fires mid-run; the untouched
exact/bigram paths cannot produce it. Narrow but real, and the keep
rests on the §28g precedent (fallback-superset + identical output +
no-regression), not on a t/s headline: +0.67/step on one prompt family,
ceiling families untouched. Tests 87/87, `scripts/test.cmd` green,
`aikit.py check aegean-tge` 4/4.

## 43. Re-baselined head-to-head on the 32 GiB box (2026-11-03)

Step 0 of the revised Round-2 plan (Stage 2): no code, re-ground the
numbers the goal's baselines no longer describe. `tge info` reports
32.00 GiB installed; `bench mem` read 51.18 at the matrix start, 53.92
at its end (mid-load; llama's own pp128 ±22% agrees). Single-channel
DDR5-4800 tops at 38.4 GB/s theoretical, so the 16 GB single-channel
premise is retired: at 51.18 the decode ceiling is 51.18 / 1.57 =
**~32.6 t/s** (1.57 GiB/token from `tge plan`), not the goal's 15.8.

Same file (Llama-3.2-3B Q4_K_M), same 6 threads, interleaved per
history, both MinGW-GCC builds (llama cd8427d, TGE d1a2e5c),
`--prefetch` on TGE (fault-in untimed), llama `-r 5` stats:

| 128-history | llama-bench cd8427d | TGE f32 (130-tok prompt) | TGE/llama |
|---|---|---|---|
| prefill | 52.50 ± 11.70 | 27.2 | 52% |
| decode 63-64 tok | 8.00 ± 1.65 | 5.63 | 70% |
| TGE `--kv f16` secondary | -- | 21.5 / 4.49 | unpaired drift, no format claim |

| 512-history | llama-bench cd8427d | TGE f32 (510-tok prompt) | TGE/llama |
|---|---|---|---|
| prefill | 42.33 ± 5.50 | 23.7 | 56% |
| decode 63 tok | 8.28 ± 0.77 | 4.61 | 56% |
| TGE `--kv f16` secondary | -- | 19.0 / 3.83 | unpaired drift, no format claim |

`phases` on all four TGE runs: matvec 94-97%, attn 2-5% (GQA engaged
at 512: positions 510+, attn 3-5%; per-head at 128: 2%), norm 2-4%.

Three statements. (1) The gap is confirmed at the new ceiling: TGE
decode 4.6-5.6 t/s is ~15-17% of the current ~32.6 t/s wall while
llama's ~8-8.3 is ~25% -- TGE converts barely three-fifths of what
llama converts, and matvec owns 94%+ of every step, so the ALU gap the
kernel round aims at is real and current, not a stale-ratio artefact.
(2) Ratios sit in the §41 band (prefill ~50-56% both histories; decode
56-70% vs §41's 63-66% -- same band, load-shifted, not a trend).
(3) f16 secondaries run slower than f32 in both histories here,
contradicting the paired §29 +14% -- unpaired across time, correctly
unclaimed per the §36 precedent; no KV-format conclusion moves.

What this fires: revised-plan Step 1 (Q6_K single-token int8 dot,
differential-first, stash-built A/B, fidelity gate <= VNNI 1.6e-3,
ids parity) proceeds against this table, not the goal's single-channel
baselines. No code changed, no fidelity owed, suite 87/87 untouched.

## 44. Q6_K single-token int8 dot: slower at the kernel, reverted (2026-11-03)

Step 1 implementation (revised plan): new `q6k_int8_avx2` TU behind
bench `--kernel int8` (Q6_K-only by construction, everything else
delegates to the shipped path), differential-first
(`q6k_int8_dot_tracks_float_reference` vs the scalar oracle at 2% of
reference magnitude, plus Q4_K-delegation memcmp) -- green 88/88 on
first build, so the indexing and the 16-weight scale-group map were
right. Design: per-64 activation quant (max/127), weights kept int8
(6-bit exact), widen/mullo/madd with one horizontal reduction per
16-weight scale group and one float finalize per group -- the
reduction-minimal shape the analysis allowed (no bias-correction sums,
unlike the maddubs form). No generate wiring: the kernel had to show a
pulse first, and it showed the opposite.

| q6k down 3072x8192 x1, best of 5, same binary interleaved | float (shipped) | int8 (new) |
|---|---|---|
| R1 | 1.190 ms, 17.35 GB/s | 1.381 ms, 14.95 GB/s (+16%) |
| R2 | 1.259 ms, 16.40 GB/s | 1.504 ms, 13.73 GB/s (+19%) |
| R3 (wall read 60.72, session best) | 1.054 ms, 19.59 GB/s | 3.122 ms, 6.61 GB/s (+196%) |

Verdict: revert. The int path loses 3/3 pairs with consistent sign,
and R3 is the inversion-proof: at the fastest wall of the session the
float path posts its best number (19.59, the §30 18.71 band) while int8
posts its worst -- no load narrative can carry that. This is the Stage-2
mechanism prediction confirmed in silicon: one reduction per 16 weights
plus ~7 integer uops per lane does not beat the float unpack+FMA chain
it replaces; the tax moved domains, exactly as objected, and the extra
finalizes (16 per block vs one per row) dominate. Third integer-domain
loss on this engine (§12 twice, §28i, §34, now here -- 0 for 5).

Reverted in full (TU deleted, header/CMake/bench/test hunks
checked out, tree verified clean, suite back to 87/87). No fidelity
run: no numerics change ships (differential was pre-ship, on the new
binary only). No generate delta claimed: generate was never wired.
Next trigger for this shape, not attempted: none from the integer
domain -- the remaining lever per §41 is fewer weight passes
(speculation acceptance), not cheaper passes.

## 45. 2048-history head-to-head, f32 pair stands, matrix suspended (2026-11-03)

Round-3 Step 0, partial. Same file (Llama-3.2-3B Q4_K_M), same 6
threads, adjacent runs, both MinGW-GCC builds (llama cd8427d, TGE
d1a2e5c), `--prefetch`, llama `-r 3`:

| 2048-history | llama-bench cd8427d | TGE f32 (2050-tok prompt) | TGE/llama |
|---|---|---|---|
| prefill | 24.49 +- 4.39 | 11.5 | 47% |
| decode 63-64 tok | 4.85 +- 0.74 | 2.99 | 62% |

`phases` on the TGE run: decode matvec 80%, attn 19% (GQA at 2000+;
first history where attention is not noise -- 128/512 showed 0-7%),
norm 1%; prefill attn 15%. The 128/512 band (§43: prefill 52-56%,
decode 56-70%) holds at 2048: no long-context surprise in the
weight path, and the f32/i8 question stays open, not answered.

Suspended, not finished: during the f16 run the box collapsed (story
probe 0.81 t/s, matvec 100%, mem bandwidth healthy at 58.97 --
frequency/capacity collapse, not format). Two cooldowns (10+15 min)
did not restore it (story 4.07, then 0.90 at wall 61.75: bandwidth
fine, ALU ~10x slow). Cause identified, not fixed: a foreign `python`
(PID 19856, ~8000 CPU-s and climbing) is pinning the cores; power
scheme is Balanced, compounding clocks. Not killed -- not our process.
The collapsed f16 numbers (7.2/0.79) and probes (0.81/4.07/0.90) are
drift artifacts and stay out of the table. f16/i8 at 2048 plus the
parity check resume when the box is free; wall + story gate each
attempt (healthy = wall ~48-59 with story ~9-11, both at once --
bandwidth alone no longer suffices as a gate after the 61.75/0.90
split).

## 46. Bigram second-best fallback: tied everywhere, reverted (2026-11-03)

Revised-plan implementation (Stage 2 verified): new
`bigram_second_best_draft` in `apps/cli/main.cpp` as a fourth fallback
after exact spans, best bigram chain and gapped windows -- chains the
second-most-frequent successor per position (observed continuations
only, <2 distinct successors abstains), same 5-token contract, same
zero-alloc scan, same verifier gating with the d0 pre-check skipping
the weight pass on a miss. Fallback-superset preserved: every
pre-existing proposal byte-identical by construction. Stash-built
old/new, interleaved per prompt, Llama-3B Q4_K_M, `--threads 6`,
`--prefetch`, greedy `--speculative`:

| prompt, 12-15 tok | old (stash) | new (second-best) |
|---|---|---|
| story "Once upon a time" | 0 steps (correct abstention) | 0 steps, ids `11 304 ... 11 1070` identical |
| partial "John likes ..." | 3 steps, 5.00/step, ids identical | 3 steps, 5.00/step, ids identical |
| ambivalent "apples pears ... grapes ..." | 3 steps, 3.00/step, ids identical | 3 steps, 3.00/step, ids identical |

Verdict: revert. Tied on all three families with identical ids --
no family where the second-best chain fires and the first three miss
(the best chain or an exact span already covers every repetitive
history probed; the novel tail correctly abstains in both). Per the
§42a no-evidence rule a tie is a revert, not a keep: an extra fallback
that never fires is speculative surface, not a lever. The mechanism
window (best exists-but-misses AND gapped abstains) proved too narrow
to find by blind prompt-guessing, and each guess costs minutes on this
box -- the search itself is the expense the frugality rule forbids.

Honesty notes. (1) The box was collapsed throughout (wall read 42.23
vs the 48-59 healthy band; story probe 0.60-3.05 t/s vs 9-11; foreign
`python` PID 19856 still pinning at ~12.6k CPU-s, power Balanced) --
so no t/s ratio is claimed anywhere here: back-to-back identical
binaries swung 0.60 vs 2.64 t/s on the same story prompt, which is the
§28m rule in one pair. Acceptance/step and ids are deterministic in
history and survive the collapse; t/s does not. (2) The §45 f16/i8
matrix stays suspended (health gate failed both halves). (3) Thinking
guard is vacuous for a reverted drafter-only diff: non-spec paths never
call the drafter (gated on `speculate`/`speculate_sampled`), and every
spec run above was ids-identical -- no reasoning or creative output
existed to degrade. (4) No fidelity run: no numerics change ships
(differential-free, verifier/sampler/model untouched -- the §32/§39
precedent).

Reverted in full (`git checkout -- apps/cli/main.cpp`, stash binary
removed, tree verified clean); post-revert rebuild + `scripts/test.cmd`
green. Suite untouched throughout (drafter lives in main.cpp, covered
end-to-end per §28ad). Next trigger, not attempted: a prompt family
with a demonstrated best-misses/gapped-abstains split (acceptance
deficit on record first), or a unit surface for the drafter if
fallback-hunting resumes.

Stage-4 check on the reverted tree (this round's code delta is nil --
`git diff af51916 HEAD` is docs-only, +52 lines here): rebuild no-op,
`tge_tests` Passed, `aikit.py check aegean-tge` 4/4. Fidelity 31 pos
ctx 512: mean KL 1.467e-11, worst 5.531e-11, 30/30 -- exact match to
the §42 floor. Post-revert generate: story ids `11 304 ... 11 1070`
and partial 5.00/step ids identical to the Stage-3 pairs. Thinking
set (final binary, Llama-3B, collapsed box ~1.4-2.7 t/s so absolutes
unclaimed): R1 structured steps, R2 telephone-circle analogy with
map-back, R3 ordered everyday consequences, C1 8-item umbrella list
(temp 0.8, seed 1234), C2 weather-metaphor (same seed), C3 Emily story
continuation -- all fluent; equivalence to pre-round holds by
code-identity (no code delta exists to move them) plus the measured
spec-mode ids parity. No failures; nothing sent back.

## 47. Research-swarm round: 12 scouts, no kernel fires, logged (2026-11-03)

Twelve parallel scouts (L1-L6 literature, G1-G6 GitHub code), each
with receipts (source + section/file + numbers + mmap-first AVX2 fit).
Verification by sampling: llama.cpp PR #22345 fully confirmed
(merged by ggerganov 2026-04-26; `MUL_MAT q6_K f32 4096x14336` n=1
34.46 vs 26.91 GFLOPS +28%, n=8 128.53 vs 97.28 +32%, n=512 143.93
vs 101.07 +42%; pp512 63.15 vs 49.75 +27%; tg128 15.63 vs 15.53
parity). Its diff header confirms the live path is
`ggml/src/ggml-cpu/arch/x86/quants.c` (the bare-domain fetch 404s;
stale path, not fabrication). Remaining scout numbers are
receipts-with-source, not independent facts: nothing below lands
without a stash-built same-window A/B + fidelity gate, which is the
real verification.

Verdicts against TGE gates (mmap-first, unaligned-only, no
retrain, integer needs pre-proof + KL <= VNNI 1.6e-3):

- L1/G6 micro-unpack (MSB-xor, `hsum_float_8` tail, shuffle/LUT):
  NO-GO. MSB-xor saves byte-subs that exist only in the integer
  path, not TGE's float `write32` (mul-then-sub floats); `hsum`
  tail changes summation order (bit-identity gate fails by
  construction) for ~1-2 uops on a load-bound kernel -- needs full
  fidelity for unresolvable gain. Shuffle/LUT adds tables for the
  same uop count or needs MXFP4/FP8 quants TGE does not have.
- L2/G3 two-phase K-chunk (`kc=128-256`, `thread_local` fp32
  <=174 KB, dequant-once reuse-64x, MLAS 1D partition + complexity
  gate): PLAN-ONLY, no code this round. Sources are real
  (ik_llama PR #482 IQ 2.3-3.2x pp; SGLang #8750 INT8 4.37x GEMM;
  ZINC staged-vs-fused 8.1x DRAM-tax math) but none is AVX2
  Q4_K/Q6_K vs TGE `dot_f32`; single-phase interchange already
  reverted (+2/+11% harm, §39). Next: prototype behind bench
  flag with traffic math, differential-first, stash A/B.
- L3 attention-CPU: NO FIRE. All GPU-only or arithmetic-changing
  without KL (NoMAD PQ, sliding/local, FFD); GQA threshold stays
  256 (swept 1.05x median, §35); SlimAttention exact-CPU is the
  only hold, needs a paired 2048-ctx TGE measurement to fire.
- L4/G5 KV: NO-CODE. KIVI/GEAR CUDA-only, break `kv_half/q8`
  exclusivity, need new q2 CPU kernels; paging solves server
  fragmentation TGE does not have; prefix cache still has no
  caller (needs multi-turn `prefill` loop, §28q trigger stands).
- L5/G4 speculation: KEEP baseline + TRUNCATE-ONLY allowed.
  Spans 5..1 + bigram + gapped already cover LLMA/prompt-lookup
  (2-2.5x grounded, ~1.2x chat) and GOOSE heterogeneity without
  datastore/tree; recent-first/second-best stay reverted (ties);
  LogitSpec last-logit and ngram frequency-abstention are the
  only conditionals, both need acceptance-on-record first.
- L6/G1/G2 integer dots (`Q4_KxQ8_K` nibble-direct, Q6_K
  `unsignedDot-32*bsum` PR #22345, Q4_0/Q6_K AVX2): NO-GO under
  the Stage-2 integer gate. Same `maddubs` family as the 0-for-5
  (§12 x2, §28i, §34 KL 8e-3 + 2 flips, §44 -16-196% kernel);
  PR #22345 states no KL/top-1 and no vs-float-dequant A/B, and
  its tg128 is parity while TGE needs decode. Next trigger: a
  bench-flag prototype with pre-proof (reduction/uop analysis)
  + differential + KL <= 1.6e-3 + stash A/B win -- not opened.

Implementation this round: NONE (no code delta). Rationale per
§28ah frugality: every kernel candidate either breaks a hard
rule (repack/integer-without-price), changes numerics for
single-digit uops, or needs a prototype round of its own; the
drafter conditionals need acceptance deficits on record first.
What ships is the record above so no finding is re-researched
without its trigger. Tests 87/87 (`scripts/test.cmd` Passed
28.6s), `aikit.py check aegean-tge` 4/4 -- both re-run for this
docs-only change.

Stage-4 check on this round (code delta nil -- `git diff --stat`
is `docs/measurement-log.md +66`, no code): rebuild no-op
(`ninja: no work to do`), `ctest --preset release` Passed
(13.0s), `bench mem` wall read **57.76 GB/s** (healthiest window
this turn; foreign-process collapse of §§45-46 not present),
`aikit.py check aegean-tge` 4/4. Fidelity re-run at the same
token count (`--context 512 --tokens 31`, Llama-3B Q4_K_M):
corpus 30 positions, mean KL **1.467e-11**, worst 5.531e-11,
top-1 **30/30** -- exact match to the §42 floor, as required
with no numerics change. End-to-end re-run (`--prompt "Once
upon a time" --n 15 --threads 6 --prefetch --print-ids`):
promptids `12805 5304 264 892`, ids
`11 304 264 2678 14458 89777 304 279 20700 35231 315 279
47967 11 1070` -- prefix `11 304` and suffix `11 1070` match
the §46 story baseline exactly; sample share 0.1%, phases
decode matvec 98% (no structural surprise). Absolute t/s
(prefill 4.6, decode 2.72) unclaimed: single run, 4-token
prompt is fixed-cost dominated per the §36 precedent, and
cross-session absolutes are load narrative per §28m. Each
rejected finding traces: integer dots to PR #22345 (verified
merged, no KL/top-1 stated) and in-log §§12/28i/34/44;
tilings to §§31/32/33/39; drafter ties to §§42a/46; KV to
§§29/41/43 + §28q no-caller trigger. No failures; nothing
sent back to implementation. Triggers standing: K-chunked
two-phase prototype (bench-flag, differential-first),
LogitSpec/frequency-abstention acceptance-on-record, paired
SlimAttention measurement at 2048 ctx.

## 48. Hybrid bring-up [1/4]: Vulkan selector scaffolding, no offload (2026-11-03)

Backend bring-up first, numerics untouched: `TGE_ENABLE_VULKAN` +
`find_package(Vulkan QUIET)` (`TGE_HAS_VULKAN=0` without the SDK),
`include/tge/gpu/backend.hpp` + `src/gpu/noop.cpp` (loader probe via
pure OS calls, reports Unavailable, single CPU fallback) +
`src/gpu/vulkan_context.cpp` (real enumeration, compiled only with the
SDK) + `tests/test_gpu_fallback.cpp` (4 cases). Static thresholds pinned
by test: batch offload wish at tokens >= 8 (6-token verifier stays CPU),
GQA wish at span >= 256 with threads > 1; short-decode output budget
3584 + 607744 = 611328 bytes (hidden 896 + logits 151936 f32).

| Check | Result |
|---|---|
| `cmake --preset release` / `--build --preset release` | clean (Vulkan not found, `TGE_HAS_VULKAN=0`) |
| `tge_tests` | **94/94 green** (90 pre-existing incl. parallel sparse work + 4 new `gpu_*`) |
| `python aikit.py check aegean-tge` | **4/4 gates passed** |
| `tge info` | i5-14400F avx-vnni, 32 GiB installed / 6.6 avail (loaded box) |
| `vulkaninfo` | 1.4.344, RX 9060 XT discrete, heaps 15.93 host + 15.92 `DEVICE_LOCAL` GiB, INT8-dot accelerated |

No fidelity price owed (no numerics changed); no speedup claimed.
Next: `--gpu` flag + `Qwen2::use_gpu_` dispatch hook (needs the
in-flight sparse/K-chunk files to settle — shared `qwen2.*`/`main.cpp`
left untouched), then one offloaded op with measured transfer
accounting. Verdict: **kept (scaffolding)**.

## 49. K-chunk dot reorder: built, differential-green, unresolvable, reverted (2026-11-03)

Round-2 trigger from §47. Prototype: new TU
`dequant_batch_kchunk_avx2.cpp` (since removed) behind
`bench batch --kernel kchunk --kchunk-kc N` (default 256) —
dequantize once per row into worker scratch (unchanged), then
consume the row in K-chunks (per-token stack accumulators, one
write to out; no output re-touch, no `thread_local` in the pool
lambda, `tokens==1` verbatim shipped loop). R2-L1 scout receipts
(SGLang PR #8750 `loop_2d`/`BLOCK_N=64`/bounded per-thread
staging; ik_llama PR #482 dequant-once-vs-`Q8_0_R8` at N≥16/24,
32-row tiles, 2.28-3.15x PP-512 prefill-only) confirmed the
direction; kc sweep {128,256,512,1024,2048} fixed regardless.
Differential-first: new `dequant_batch_kchunk_matches_scalar_
reference` (cols {1024,3072,3328,8192} x tokens {1,7,64} x kc
sweep x Q4_K/Q6_K, near-tolerance 1e-4·cols+1e-3 per the shipped
discipline) green on first build — indexing and the 3328 tail
correct. Suite green with the prototype in (94 peer + new
differential, `ctest` Passed).

Same-binary interleaved A/B (old path untouched in the binary,
no stash needed), best-of-5, wall gate 51.25 read GB/s at start:

| q6k down 3072x8192 x64 | ms | q4k gate 8192x3072 x64 | ms |
|---|---|---|---|
| shipped | 30.739 | shipped | 18.487 |
| kc128 | 28.421 | kc128 | 22.021 (-19% lean-harm) |
| kc256 | 33.763 | kc256 | 19.035 |
| kc512 | 29.390 | kc512 | 17.597 |
| kc1024 | 24.850 | kc1024 | 34.530 |
| kc2048 | 23.559 | kc2048 | 39.475 |
| shipped re | 22.918 (-25% vs own earlier) | shipped re | 38.631 (2x vs own earlier) |

No kc wins ≥15% on both shapes; kc128 leans harmful on the
control; shipped swings 25-100% against itself. Inversion proof
(§37 tripwire): x1 q6k — shipped 21.025 ms vs kchunk 4.783 ms on
the verbatim-identical loop (4.4x on identical code). Wall
re-check at end: copy **6.59** (was 49.65), read 28.97 — box
collapsed mid-round; concurrent-worker contention observed
(peer Vulkan/sparse files landing in-tree, two stale `tge.exe`
locks cleared with a 120 s bounded wait, peer files untouched
throughout). Per the round gate and §§32/33/39 precedents:
**reverted in full** (TU deleted, 4-file checkout, post-revert
rebuild + `ctest` Passed 37.5s; peer files and §48 intact).
No fidelity run: bench-only, verifier/model/sampler untouched.
Retry trigger, not attempted: quiet box (wall ~48-59 AND story
~9-11 together, no concurrent worker) plus the same interleaved
pairs; kc default stays an open question inside that retry.

## 50. Static sparse subset (sink+stride+window, `--sparse`): kept opt-in (2026-11-03)

LSA Stage 3 candidate (a): content-blind kept-set geometry from LSA §3.2
(sink + selected long-range + window) with stride sampling instead of
scored buckets. `SparseConfig{enabled,sink=4,window=512,stride=4}` in
`qwen2.hpp`; subset gather inside `Qwen2::attention_head` (f32 KV only,
thread-local scratch, same kernel over compacted rows); precedence in
`Qwen2::attention()` before `uses_gqa` so long decode takes the per-head
sparse loop; exact dense path whenever span <= sink+window, KV is f16/i8,
or mode is Scalar. CLI: `generate`/`fidelity --sparse [--sparse-window N]
[--sparse-stride N]` (defaults 512/4); `bench continuous` stays dense.
Unit tests: index identity/short-span, sink/stride/window geometry at
1024/2048/8192, gather-vs-direct-subset reference (`test_attention.cpp`).

| Check | Dense | Sparse (sink 4, window 512, stride 4) |
|---|---|---|
| `tge_tests` | 90/90 green (3 new sparse cases) | same binary |
| fidelity Q8_0 ctx 1024, arch corpus | KL mean 9.248e-12, worst 2.418e-10; top-1 1023/1023 | KL mean 2.597e-02, worst 7.867; top-1 975/1023 |
| fidelity, first 600 toks | — | KL mean 4.002e-04, worst 3.865e-02; top-1 598/599 |
| generate ctx 512, 64 toks | decode 30.87 t/s | decode 23.06 t/s; ids identical 63/63 (exact fallback, span <= 516) |
| generate ctx 8192, 7001-tok prompt, 32 toks (run 1) | prefill 43.0 t/s (attn 54.0 s), decode 10.21 (attn 49%) | prefill 58.6 (attn 42.2 s), decode 9.32; ids 31/31 identical |
| same (run 2) | prefill 58.3 (attn 44.0 s), decode 11.70 | prefill 73.2 (attn 33.4 s), decode 16.63; ids 31/31 identical |
| `python aikit.py check aegean-tge` | 4/4 green | 4/4 green |

Verdict: **kept opt-in (default off)**. Prefill attn phase is consistently
lower (54.0→42.2, 44.0→33.4 s) and end-to-end prefill faster in both pairs,
but absolute tok/s moved ±35% between back-to-back dense runs: a parallel
worker was building in this repo throughout, so the box was contended and
no absolute speedup is claimed beyond the attn-phase direction. Decode is
noise (±attn share, matvec-bound). Fidelity prices the approximation
honestly: 48 flips/1023 on natural text at ctx 1024 (worse than VNNI's
12/497 at KL 1.6e-3) but identical ids on the repetitive 7k prompt.
Use for prefill-bound long prompts; natural-text long context stays dense
until scored selection (candidate b) is measured. One full-corpus sparse
fidelity run exited -1 with no output mid-stage, then passed cleanly on
rerun (193 s, exit 0): recorded as box flake, not a code path (the 600-tok
and full runs bracket it green). Mid-stage, `apps/cli/main.cpp` lost its
uncommitted sparse plumbing (parallel Vulkan bring-up reset the shared
file; `qwen2.*`/tests intact); re-applied, rebuilt 46/46, re-verified
90/90 + gates before commit. Rejections on the record (receipts in
goal Stages 1-3): distilled `W(idx)`+τ (§3.3, needs GPU distill + SGLang
maps, no transfer to Qwen2.5 GGUF CPU), N-LSA native (§3.4, needs
100B+12B pretrain), CuTe/H200 kernels + CP8/TP8 + 10M serving (no CUDA
path; `kHardMaxContext` 32768; mmap-first), tree/beam (single-stream
decode is flat per paper §4.1; revisit only on measured headroom).

## 50. Hybrid [1/4] restore after parallel commit (2026-11-03)

The parallel sparse commit (`b7eda07`) reverted `CMakeLists.txt` as a
side effect (dropped the K-chunk wiring and, with it, the Vulkan
scaffolding hunks). Re-applied the four bring-up hunks only
(`TGE_ENABLE_VULKAN`, `noop.cpp`, optional-find block, gpu test) onto
the pristine file; no other tree file touched.

| Check | Result |
|---|---|
| `cmake --preset release` | clean |
| `tge_tests` | **94/94 green** (90 incl. new sparse cases + 4 `gpu_*`) |
| `tge.exe` relink | **blocked: file locked** (parallel agent benchmarking; Windows holds running executables) — `tge_tests.exe` relinked fine, `libtge_core.a` rebuilt |
| full `aikit.py check` | deferred until the lock clears (prior 4/4 on equivalent code) |

No numerics changed; no fidelity price owed; no speedup claimed.
`tge.exe` binary on disk is their sparse build (21:38) until relink.
Verdict: **kept (scaffolding restored, pending relink)**.

Addendum (same session): lock cleared, `cmake --build --preset release`
fully green incl. `tge.exe`; `tge_tests` **94/94**; `python aikit.py
check aegean-tge` **4/4 passed**. One transient `test.cmd` FAIL observed
mid-window (parallel run holding `tge.exe` during the internal rebuild);
clean on immediate re-run. Verdict stands: **kept**.

## 51. Hybrid [2/4]: GPU control plumbing, no dispatch change (2026-11-03)

`--gpu auto|off|on` on `generate` + `fidelity` (default auto),
`Qwen2::set_use_gpu` + `wants_gpu_batch`/`wants_gpu_gqa` (backend
readiness excluded by construction), `--gpu on` without a ready backend
fails pre-load with the reason, `tge info` prints a GPU section, and the
dispatch itself is untouched: every mat still runs the proven CPU
kernel. Eligibility conjunctions from the revised plan are unit-tested
(tokens>=8, span>=256/threads>1, Q8_0|F32-only, clean flags, dense-only).

| Check | Result |
|---|---|
| `cmake --build --preset release` | clean (one transient `tge.exe` lock from a parallel bench run; green after 90 s) |
| `tge_tests` | **96/96 green** (94 + 2 `gpu_eligible_*`) |
| `tge.exe generate --gpu bogus` | exit 2, `takes auto\|off\|on` (parse-before-load) |
| `tge.exe generate/fidelity --gpu on` | exit 2 pre-load, `needs a ready GPU backend (vulkan-1 loader present (compile-time off))` |
| `tge.exe info` | `GPU/status: unavailable (CPU fallback)` + loader line |
| `python aikit.py check aegean-tge` | **4/4 passed** |

No numerics changed (same kernels execute); no fidelity price owed; no
speedup claimed. First real offload ([3/4]: output-region + prefill
batch + GQA shaders) is blocked on the Vulkan SDK (no `glslc`,
no headers, no `VULKAN_SDK` on this box; `winget search` inconclusive —
timed out). Verdict: **kept (plumbing)**.

## 52. SDK present: real backend builds, links, reports ready (2026-11-03)

The box now has LunarG SDK 1.4.363.0 (`C:\VulkanSDK\1.4.363.0`,
machine `VULKAN_SDK` set; `Bin\glslc.exe` + `Include\vulkan\vulkan.h` +
`Lib\vulkan-1.lib` all present). Reconfigured with the SDK visible:
`find_package` resolves headers + import lib, `TGE_HAS_VULKAN=1`, and
`src/gpu/vulkan_context.cpp` (fixed one const-pointer mistake in the
queue-family query: clamped non-const count) compiles and links under
MinGW against `vulkan-1.lib` — no link workaround needed.

| Check | Result |
|---|---|
| `cmake --build --preset release` | clean, `tge.exe` + `tge_tests.exe` relinked |
| `tge_tests` | **96/96 green** |
| `tge.exe info` | `GPU/status: ready`, `device: AMD Radeon RX 9060 XT` (was `unavailable (CPU fallback)`) |
| `python aikit.py check aegean-tge` | **4/4 passed** |

`glslc` is not on `PATH` yet (fresh shells need re-login or a PATH
edit); the CMake shader step in [3/4] must reference
`$(VULKAN_SDK)/Bin/glslc.exe` explicitly. No numerics changed; no
fidelity owed. Next: no-op compute dispatch (device/queue/fence +
`noop.comp` via `glslc`) to prove the submit path before any numerics.
Verdict: **kept (SDK bring-up)**.

## 53. First GPU dispatch: no-op round trip on RX 9060 XT (2026-11-03)

`src/gpu/shaders/noop.comp` (invocation 0 writes 1u) compiled by
`glslc -O` and embedded via `cmake/bin2h.cmake` (alignas(4) bytes, so
the binary stays single-file); `vulkan_context` grew a persistent
device/queue/pool plus `run_noop()` (storage buffer, descriptor,
pipeline, one dispatch, fence wait, exact readback); `tge info` prints
the round trip; `tge_tests` pins it (`gpu_noop_roundtrip`: Ready implies
ok + readback exactly 1, not-Ready implies clean no-op).

| Check | Result |
|---|---|
| `cmake --build --preset release` | clean (fixed one header-path mismatch: generated header now flat in the generated root) |
| `tge_tests` | **97/97 green** (96 + `gpu_noop_roundtrip`) |
| `tge.exe info` | `ready / AMD Radeon RX 9060 XT / roundtrip: ok (submit 2587.9 us, readback 1)` |
| `python aikit.py check aegean-tge` | **4/4 passed** |

Roofline note: 2.6 ms covers pipeline creation + submit + fence (pipeline
is built per call today); a persistent-pipeline design will cut the
steady-state submit toward the ~10-50 us budget in §3. Transfer on this
path is one 4-byte buffer — the accounting story starts with the first
real kernel. No numerics changed; no fidelity owed. Next: output-region
GEMV (Q8 dequant + dot, single sync) with its fidelity price.
Verdict: **kept (submit path proven)**.

## 54. First offload: decode output projection on GPU (2026-11-03)

`q8_gemv.comp` (1 thread/row, portable 32-bit-only GLSL: read_u32 for
unaligned blocks, integer fp16->fp32, per-block dot-then-scale matching
the host scalar kernel) + persistent device-local weights (upload once
per mmap pointer, +4 zero-pad for the tail word) + lazy pipeline +
per-call H2D/D2H. `Qwen2::output_logits()` tries the single-sync GPU
GEMV at both decode tails and falls back to the exact CPU matvec on any
failure; F32 weights stay CPU until their twin lands.

Two bugs caught by the gates, both fixed: the shader read the GGML
block backwards (quants-first; the host layout is scale-first per
`q8_gemv_scalar.cpp:22-27`) - the hybrid answered `!!!` while CPU
answered Paris, and the unit test had encoded the same wrong layout, so
it passed while the engine diverged. The fidelity run is the
load-bearing guard: the test now uses the true layout.

| Check | Result |
|---|---|
| `tge_tests` | **98/98 green** (`gpu_gemv_q8_matches_scalar_reference`: worst 1e-3, top-1 held) |
| `generate` 5+31 hybrid vs `--gpu off` (same box, parallel bench running: absolutes depressed, ratios comparable) | hybrid text **identical** ("Paris...French Alps...south of"); **7.17 vs 2.38 tok/s = 3.0x**; `gpu: output offload 795.1 ms, 18951168 B moved` (31 x 611328 B; first-token upload included, steady-state per-token NOT isolated - open for Stage 4) |
| `fidelity --tokens 512` (candidate on GPU path) | **KL mean 9.676e-12, worst 2.4e-10; max diff 1.9e-04; top-1 511/511** (f16 precedent 1.7e-6 - five orders tighter) |
| `python aikit.py check aegean-tge` | **4/4 passed** |

Roofline: 144 MB weights stream once (upload, ~13 ms worst case) then
0.61 MB/token vs 4.5-7 ms CPU saving closes as predicted; per-call
alloc + fence overhead is the next thing to measure, not assume.
No speedup claimed beyond the noisy 3x; controlled A/B is Stage 4.
Verdict: **kept (first offload priced and held)**.

## 55. Batch offload: prefill GEMM on GPU, numerics held, timing noisy (2026-11-03)

`q8_gemm.comp` (2D: 1 thread per token x row, same portable Q8 math) +
resident upload cache (one device buffer per pointer key, shared by both
kernels) + lazy second pipeline + `batch_matvec` hook behind
`wants_gpu_batch` (tokens >= 8, Q8_0, clean flags) with exact-CPU
fallback. Prefill `generate` prints its own `gpu:` transfer line.

| Check | Result |
|---|---|
| `tge_tests` | **99/99 green** (`gpu_gemm_q8_matches_scalar_reference`, worst 1e-3) |
| `fidelity --tokens 512` (batch + output both GPU) | **KL 9.676e-12, top-1 511/511** (identical to GEMV-only slice: batch adds no error) |
| generate text, 53-token prompt, hybrid vs `--gpu off` | **identical** ("...software spans from the") |
| `python aikit.py check aegean-tge` | **4/4 passed** |

Timing (NOT conclusive, recorded for the record): 8-token chunk hybrid
31.2 s (gpu 16.2 s / 18.2 MB) vs CPU 0.59 s; 64-token chunk hybrid 3.0 s
(gpu 1.8 s / 141 MB) vs a 53-token CPU run at 26 tok/s. The 8-token run
is 9x slower with 8x less compute than the 64-token run while CPU-side
attn moved 86 -> 26 ms between the same runs: the box (parallel bench
session + 79% RAM in use per `tge info`) is the dominant variable, not
the kernel. No speedup claimed; per-call fixed costs (descriptor/buffer
allocs per dispatch) and kernel fetch pattern are the two known
suspects, in that order - but suspects only until a quiet box says so.
Stage 4 requires: parallel session idle, 3x repeats, first-token upload
split from steady-state. Verdict: **kept (numerics priced, timing open)**.

## 56. Persistent dispatch resources: steady submit 0.2 ms (2026-11-03)

The `TGE_GPU_TRACE=1` split showed host-side churn dominating: first-call
io alloc + per-call descriptor pool/fence/cmd plus one weight upload per
matrix. Now persistent: one descriptor pool (reset per call), one command
buffer (reset + re-record), one fence (reset), grow-only mapped I/O
buffers, shared `run_q8_call` for both kernels, per-phase `alloc_ns` in
the result. Teardown releases everything before the device dies.

| Check | Result (quiet-ish box for the steady lines) |
|---|---|
| `tge_tests` | 99 pass incl. all 9 `gpu_*`; 1 failure in `dequant_batch_kq8_matches_float_oracle` (`tests/test_dequant.cpp:508`, worst 69945 vs tol 0.05) - the parallel session's in-flight K-quant kernel (their `kernels.hpp`/`test_dequant.cpp`/`dequant_batch_kq8_avx2.cpp` hunks); untouched by this change, CPU fallback path identical when `use_gpu` is off |
| trace, 8-token chunk | `alloc 0.02-0.03 ms` steady (was per-call pool+fence+cmd+2 allocs), `submit ~0.2 ms`, `d2h ~0.1 ms`; first-touch uploads 2-90 ms each (page faults + submit under load), pipeline cached |
| `fidelity --tokens 128` after refactor | **KL 1.2e-11, top-1 127/127** (no numeric drift from the refactor) |
| `python aikit.py check` | test gate FAILs only on the foreign kq8 case above (repro: `tge_tests`, same line); hygiene/guidelines/secrets pass |

Per-dispatch steady cost is now ~0.5 ms (alloc+submit+d2h+memcpy) vs ~9 ms
before, on a box that keeps getting re-contended (attn 7 -> 128 ms
between runs): absolute A/B claims stay gated on a quiet box (Stage 4).
Next: GQA attention offload (shader + host + unit test + fidelity -
all verifiable without a quiet box), then the controlled three-way.
Verdict: **kept (churn removed, foreign red documented)**.

## 51. Acceptance map: drafter saturated, LogitSpec stays unopened (2026-11-03)

Round-3 read-only measurement (no code; L5/G4 receipts consumed, not
re-scouted). Pre-registered families x modes, greedy + temp 0.1
(seed 1234) + repeat-penalty 1.1/window 4, Llama-3B, 34 runs:

| Family | Greedy | Sampled 0.1 | Penalty 1.1/w4 |
|---|---|---|---|
| F1 story (n=15) | 0 steps, ids `11 304 … 11 1070` both arms | 0.00/step (5 steps, divergent path) | — |
| F2 repetitive | 3.43/step (7 st, 24 acc) | 2.56/step (9/23) | 3.43/step, byte-identical |
| F3 partial | 2.71/step (7/19) | 1.36/step (14/19, ids identical) | 2.60/step, byte-identical |
| F4 triple | 2.50/step (6/15) | 1.00/step (15/15, ids identical) | **5.00/step**, byte-identical |
| F5 prose | 5.00/step (prompt-echo) | — | — |
| F6 code | 3.00/step (7/21) | 2.10/step (10/21) | — |
| F7 wrapper | 3.50/step (8/28) | 3.12/step (8/25, ids identical) | — |
| F8 doubled tail | 4.00/step (3/12) | — | — |

Greedy: every pair ids-identical. Sampled: F2 det-proven
(identical rerun, 9/23 both times) — cross-arm divergence at
single positions with reconvergence is RNG-consumption mechanics,
not a verifier bug; §28w "exact preservation" is distributional,
and the plan's stop-condition was corrected to determinism (not
cross-arm identity) before judging. F4-sampled is the floor at
exactly 1.00/step with identical ids — caught, not abstained.
F4-penalty hits ceiling: the penalty steers onto the repeated
triple and the drafter catches all of it (5.00/step). No family
shows abstain-or-miss with repetition present.

Corrections banked: `--temp`/`--repeat-penalty` (plan said
`--temperature`/`--penalty`, both hard-error); one `unknown
command` event from a peer relink racing exec (re-ran clean);
binary changed mid-block (peer's sparse landing — greedy block
on one binary, sampled block on the current one; per-binary
determinism holds, cross-binary absolutes unclaimed).
Verdict: **global close — LogitSpec last-logit and ngram
frequency-abstention stay unopened**; the spans+bigram+gapped
chain is saturated at 1.0-5.0/step everywhere repetition
exists. Next trigger: a prompt family with demonstrated
best-misses/gapped-abstains split, or acceptance < 1.0/step
with repetition present.

## 52. llama.cpp prefill-path teardown at cd8427d: integer dots on Q8_K activations (2026-11-03)

Round-4 research product (read-only; no TGE code). Two-scout swarm on
the pinned tree `cd8427d018c07eefab6c110f7f115c31ec4c6157`
(PR #29730, verified direct: commit touches only
`src/llama-context.cpp`). Sampling: dispatch table fetched and
confirmed verbatim (`Q4_K: vec_dot q4_K_q8_K, vec_dot_type Q8_K,
nrows 1`; `Q6_K` same shape; `Q8_K: from_float quantize_row_q8_K`;
`nrows 2` ARM-only); helpers (`mul_sum_us8_pairs_float`
maddubs→madd, `hsum_float_8`, `bytes_from_nibbles_32`,
`get_scale_shuffle_k4`, AVX2 `quantize_row_q8_0`) confirmed in the
real `arch/x86/quants.c` (4172 lines at pin); `assert(nrc == 1)`
pattern confirmed across dots. Scout line numbers for the K-dot
bodies (`quants.c:2102`, `:2490`) and the MUL_MAT driver
(`ggml-cpu.c:1174-1480`) are dual-scout-consistent but beyond
fetch-truncation depth — cited as scout receipts, not independently
re-read; the mechanism below rests on the triple-confirmed parts.

The mechanism (both scouts agree, receipts above): llama factorizes
the opposite way from TGE. Weights are never dequantized;
activations are quantized to Q8_K **once per token row**
(`ggml_compute_forward_mul_mat`, scalar `quantize_row_q8_K_ref`
even on AVX2, cost amortized over all `nr0` outputs), then
`64 x nr0` vec_dots per pass, each integer `maddubs`+`madd` with
one float scale per 256-block and the ±bias via precomputed bsums
(`dmin` side accumulator Q4_K; `q8sclsub << 5` Q6_K; nibbles stay
unsigned, never shifted by 8/32). Batch = one `MUL_MAT` per weight
(`llama-graph.cpp`), threads split N/M in `16x16` work-steal tiles
(`nchunk0/1`, `nth*4` rule — same shape as TGE bands), K stays
inside the dot, `nrc == 1` on x86.

Per-pass byte math, 64-wide (my derivation from the receipts):
gate 8192x3072 Q4_K — weights 13.5 MB DRAM both sides. TGE:
768 KB fp32 input L3-resident, re-read per row; dots float-FMA
over 24 KB L1-resident sets. llama: one 768 KB scalar quant pass
→ 224 KB Q8_K rows resident; dots integer-maddubs over ~3.6 KB
sets. Down 3072x8192 Q6_K — TGE dot set 64 KB spills 48 KB L1D
(§31); llama dot set ~9.5 KB resident. So llama holds two
compounding edges: ~4x dot ALU (32 int8 MACs vs 8 fp32 FMA per
instr) and L1 residency at wide cols. TGE's 139-208 vs ~300
GFLOPS prefill fits ALU-bound float dots exactly where int8
would be 2-4x.

Verdict: **bin (b) — conditional trigger ARMED, not fired.**
The admissible construction is now precise (per-row scalar Q8_K
activation quant into thread-local `wdata` + per-(row,token)
`q{4,6}_K_q8_K` maddubs dots, `nrc == 1`, unaligned-only,
mmap-safe, no repack — TGE's missing piece is the activation
quantizer + KxQ8 dispatch; its `q8_batch` covers Q8 weights
only). It reopens the integer lane despite 0-for-5 because it
specifies what those attempts lacked (per-superblock float
scale + bsums bias + per-row activation quant, production-used
upstream). Entry gates for a round-5 prototype, all required:
(1) binary re-pairing (clone still unlocatable — rebuild/
relocate and re-run §36 pairs first); (2) the prototype carries
its own price proof (differential → 3-corpus fidelity ≤ VNNI
1.6e-3, top-1 held; bench-flag only until the kernel pulses);
(3) quiet box + no concurrent worker. Excluded again with
receipt: tiled/repack/sgemm bypasses, VNNI-fused and ARM
`nrows=2` paths (break mmap-first). No PPL citation found for
Q4_K/Q6_K-x-Q8_K AVX2 — the TGE-measured fidelity in gate (2)
is the designated substitute.

Box watch (2026-11-03, wakeup re-check): wall read **61.00 GB/s** (best window yet; PID 19856 gone) but story probe **2.18 t/s** -- the §45 split (bandwidth healthy, ALU ~5x slow) with new pinners (`node` 37k + 17k CPU-s). Health gate needs both halves; still collapsed, no t/s claimed, §45 matrix stays suspended.

## 53. Binary re-paired at cd8427d: prefill gap confirms at 52% / decode at 55% (2026-11-03)

Round-5 infra round (no TGE code; the clone lives outside the repo).
Path, logged first per the §31 lesson:
`C:\Users\Yunus-Emre\AppData\Local\Temp\kilo\llama.cpp-cd8427d`
(shallow clone, detached HEAD at
`cd8427d018c07eefab6c110f7f115c31ec4c6157`, tree clean).
Procedure from `docs/build.md` at the pin (stock CPU build):
`cmake -B build`, then the §28al necessity re-proved itself —
`vendor/cpp-httplib` fails without `-D_WIN32_WINNT=0x0A00`
(`CreateFile2` undeclared), so
`cmake -B build -DCMAKE_CXX_FLAGS=-D_WIN32_WINNT=0x0A00`;
`cmake --build build --config Release --target llama-bench llama-cli`
(`-j 6`). The `.rsrc merge failure: multiple non-default manifests`
link notice recurs (§28al: harmless — both binaries run).
Versions confirm the pin:
`0.8.0-dev (build 1, commit cd8427d), GNU 16.2.0 for Windows AMD64`
(both tools). ggml 0.28.0, ggml commit cd8427d, CPU backend,
`-march=native`, OpenMP 5.2.

Same-window pairs, 6 threads (box loaded — absolutes depressed on
both sides, pairing is the point):
llama-bench `-p 128 -n 64 -r 3`: **pp128 22.16 ± 1.36**,
**tg64 5.20 ± 0.08**. TGE same window: 138-tok prefill
(tripled-F2 paragraph) **11.6 t/s** (matvec 95.5% of prefill,
attn 1.6%); story decode (F1) **2.88 t/s**, ids
`11 304 … 11 1070` both current-binary runs.
So: prefill **52%**, decode **55%** of pinned llama — the gap
confirms inside the standing bands (43-56% prefill, 56-70%
decode), now against a version-verified binary instead of the
lost §36 build. Scope correction: no Q8_0 GGUF exists on disk
(glob empty) — the §28al Q8_0 pair is unre-runnable; the Q4_K_M
pair above is the reference-model pair and the only one the
armed trigger needs.
Round-6 gate (1) **OPENS**: re-pairing done (path + version on
record). Remaining gates for the Q8_K-activation prototype:
own price proof (gate 2) + quiet box (gate 3). Binary note: the
TGE runs above used the peer-relined binary (`gpu: auto (ready,
CPU fallback armed)` vs the older loader-present line) — CPU
path confirmed armed, ids identical, no behavior delta.

## 57. Integer Q4_K/Q6_K x Q8_K prototype: differential-green,
2.2x pulse, model wiring written, verification parked on a
peer refactor (2026-11-03)

Round 6 Stage 3. New TU `src/cpu/avx2/dequant_batch_kq8_avx2.cpp`
(scalar per-row Q8_K activation quant + AVX2 `maddubs` dots,
transcribed from `quants.c` at cd8427d), `bench batch
--kernel kq8` hook, differential vs the shipped float oracle
plus an independent generic-scalar port of upstream's own dot
as triangulation. Note on numbering: parallel rounds duplicated
§§50-53 in this file; §57 is the first number free across both
committed and uncommitted sections.

Differential-first, all green on the final tree (101 cases,
0 failed): zero-input exact; constant inputs bit-identical;
random-input price bounded at 5% (measured ≤0.8%); generic
triangulation matches AVX2 to 1e-4 with generic==float on
constants (transcription verified, price isolated).
Two real bugs caught by the gates, both fixed: (1) a
hand-transcribed k4 shuffle table with dropped/duplicated rows
(proven by byte-compare against upstream: 252 vs 256 entries;
replaced with a computed mask of verified-identical bytes);
(2) repeated STALE-BINARY measurements from the link-lock
flake (`ld: Permission denied` with no live holder — AV scan
race; clears on retry) — intermediate "residual" numbers were
stale artifacts, and the final verdicts above are from a
verified-fresh link. Lesson recorded: gate every test run on a
confirmed build, never on a Select-Object tail.

Bench A/B, same binary, interleaved best-of-5 (wall read 52 GB/s
at the time): target q6k-down x64 shipped 23.689/23.095 ms vs
kq8 10.033/10.542 ms (**2.2-2.4x**, both rounds); control
q4k-gate x64 shipped 16.959/21.797 vs kq8 13.900/10.216
(+22% / +113%, same direction, gate passes — R1 ran on a
drifting box, R2 is the cleaner window); x1 decode-shape
10.5 ms vs shipped 9.2 ms (0.8x — quantizer unamortized, so
the model wiring below keeps decode on the float path by
construction).

Model wiring is WRITTEN but UNBUILT: `set_kq8()` on Qwen2,
`batch_matvec` routes Q4_K/Q6_K with tokens > 1 through
`gemv_dequant_batch_kq8_avx2` (decode untouched), generate and
fidelity accept `--kernel kq8` (candidate arm only in
fidelity). The tree does not compile right now for reasons
outside this round: the peer's mid-flight GEMV→GEMM rename
(`cpu::GemvArgs` → `cpu::GemmArgs` in `qwen2.cpp` batch F32
block) is inconsistent with HEAD's `GemmArgs` (no
rows/cols/matrix/ldm/vector members; `gemv_f32` still takes
`const GemvArgs&`) — repro: `cmake --build --preset release`
fails at `qwen2.cpp:934-941`. No peer file touched in
response. Pending, in order: tree compiles again → rebuild +
full suite → warm generate A/B → 3-corpus fidelity ≤ VNNI
1.6e-3 with top-1 held → then the default-switch decision.

## 58. kq8 verified end to end: 2.2x kernel, 1.83x prefill, fidelity at floor (2026-11-04)

Round-6 gate (2) and (3) closed on a green tree (peer fixed the
GemvArgs/GemmArgs rename; no peer file touched in return).
Rebuild + full suite green (108 cases, 0 failed - includes the
kq8 differential, the generic-scalar triangulation at 1e-3, and
zero-input exactness).

Bench A/B, same binary interleaved best-of-5, wall read
56.5 GB/s: target q6k-down x64 shipped 23.4/21.0 ms vs kq8
14.8/9.5 ms (**1.58x / 2.22x**); control q4k-gate x64 shipped
12.4/21.8 vs kq8 9.1/10.2 (**+37% / +113%**, same direction
both rounds - R1 ran while the box settled, R2 is the cleaner
window). x1 decode-shape: shipped 9.2 vs kq8 11.5 ms (0.8x -
quantizer unamortized, the reason model wiring keeps decode
on the float path by construction; `batch_matvec` routes kq8
only for tokens > 1).

Fidelity `--kernel kq8`, 31 pos ctx 512, all three corpora at
the floor: architecture.md 1.467e-11 / 30-30 top-1,
report-0.1.md 1.478e-11 / 30-30, sampling.cpp 4.308e-12 /
30-30 - about eight orders of magnitude inside the 1.6e-3
VNNI-class gate. The integer prefill path is fidelity-neutral
on real data at this token count.

End to end, same window: 138-tok warm prefill 27.0 -> 49.4 t/s
(**1.83x**, matvec 94% -> 91% of prefill, promptids
identical); story decode under `--kernel kq8` byte-identical
(`11 304 ... 11 1070`, tokens==1 guard holds, 6.40 t/s on a
healthier box than the loaded §51 runs).

Status: prototype fully verified (differential + bench pulse
+ fidelity + e2e). Remaining decision (not taken here):
default-switch for prefill batches vs keeping the opt-in flag
- needs a fresh quiet-window confirmation plus the 2048-ctx
ladder, as a follow-up round. Numbering note: §58 is the
first number free across committed and uncommitted sections
(parallel rounds duplicated §§50-53).


## 59. Round-7 checkpoint: load-proof items green, t/s parked for quiet (2026-11-04)

Box collapsed at read time (wall copy 15.7 / read 12.3 GB/s), so no
timed comparison was attempted - the following are deterministic and
load-independent by construction:

- Fidelity `--kernel kq8` at ctx 2048 (architecture.md, 31 pos):
  mean 1.467e-11, worst 5.531e-11, top-1 30/30 - identical to the
  512-ctx floor. The integer path is context-length independent, as
  constructed (per-superblock math never sees the span).
- Greedy story ids, default vs `--kernel kq8`: byte-identical
  (`11 304 ... 11 1070`, matches every baseline back to §46).
- Greedy repetitive F2 ids, default vs `--kernel kq8`: byte-identical
  (matches the round-3 F2 greedy baseline), so the drafter sees the
  same prefill output either way.

Still parked for a quiet box (wall read 48-59 AND story 9-11
together): paired A/B re-confirmation on the final tree, the
2048-ctx ladder t/s, then the flip-vs-flag decision with full
gates. Nothing in this checkpoint changes a file outside the log.
## 60. Round-7 window: end-to-end holds 1.82x, control shape unresolvable on collapsed box (2026-11-04)

Wall 48.2 at open, collapsed to 26 mid-window (foreign load); all
t/s below is same-window pairing only, absolutes unclaimed.
Target q6k-down x64: shipped 34.2 / kq8 12.8 (**2.68x**).
Control q4k-gate x64: 34.1 vs 34.1 (**tie**) then, after the
collapse, shipped 16.3 vs kq8 22.4 (**0.73x**). End-to-end
138-tok prefill in the same collapsed window: 12.5 -> 22.7 t/s
(**1.82x**, replicates round 6 exactly), matvec share down
94% -> 91%, kq8 ids `578`. Story decode under kq8:
`11 304 ... 11 1070`, unchanged.
Verdict so far: target + end-to-end win in every window (4 + 2
windows); control shape swings +113% to -37% with box state
and is unresolvable while collapsed. The flip stays parked:
ladder mechanics verified (prompt builder + promptids count;
full 2070-token ladder needs ~x120 repeats, not the x45 = 765
tried here), t/s ladder + quiet re-confirmation queued for the
armed wakeup. No code touched this window.

## 61. Default-switch: kq8 is now the prefill path, float kept as off-ramp (2026-11-04)

Round-7 verdict executed. `set_kq8` defaults true; `batch_matvec`
routes Q4_K/Q6_K with tokens > 1 through the integer path unless
`--kernel float` is passed (generate, fidelity; bench keeps
float default and selects kq8 explicitly). Scalar oracle and
tokens == 1 stay float by construction (scalar gate + tokens
guard, same `clean` idiom as the GPU/VNNI gates).
Healthy-window confirmation on the final tree: target
q6k-down x64 shipped 21.5 vs kq8 10.7 (**2.00x**); control
q4k-gate x64 shipped 15.8 vs kq8 10.3 (**+53%**); x1 decode
guard 1.5 vs 3.9 ms (kq8 0.38x unamortized - bypass justified).
2048-ctx ladder (2040-tok prompt): 23.6/25.0 -> 37.6 t/s
(**~1.6x**), attn absolute identical, ids `578` both arms.
Fidelity defaults now exercise kq8: architecture.md
1.467e-11 / 30-30 top-1 (floor), report-0.1.md 1.478e-11 /
30-30, sampling.cpp 4.308e-12 / 30-30, plus 2048-ctx
1.467e-11 / 30-30. Greedy ids identical on story, F2
repetitive, and the ladder prompt.
End to end, flipped default: 46-tok prefill 67.5 tok/s vs
`--kernel float` 40.4 tok/s (**1.67x**); story decode
unchanged (`11 304 ... 11 1070`, 10.04 t/s on a healthy box).
Control-shape caveat stands: one collapsed-box -37% exists
against three healthy-window wins (+22%, +113%, +53%);
verbatim from §60, the gate counts admissible windows only.
The wiring lives in shared files (qwen2, main) - commit
with the peer lane, not over it.

## 62. Post-flip head-to-head vs pinned llama: 71% prefill, 78% decode - goal not met (2026-11-04)

Same window, healthy box (wall 56.7, story 7-10 t/s), llama
rebuilt at cd8427d (`llama-bench -t 6`: pp128 **94.55 +- 7.59**,
tg64 **13.05 +- 0.39** - far above the round-5 22.16/5.20,
which were collapsed-box absolutes; same-window ratios from
round 5 stand). TGE flipped default: 138-tok prefill
**67.5 tok/s** (was 27.0 float: **1.67x**); 63-tok story
decode **10.19 tok/s** (`11 304 ... 47967`, matvec 97%).
Head-to-head: prefill 94.55 vs 67.5 (**TGE at 71%**, was 52%),
decode 13.05 vs 10.19 (**TGE at 78%**, was 55%).
The flip closed roughly half the gap on both paths. Decode
was never in kq8 scope (unpack-bound, drafter saturated) and
stays the next lane alongside long-ctx attention (12-20%
share after the matvec win). No code touched this window
beyond the flip already on record; box variance note: llama
absolutes swing 4x with box state, ratios only ever count
same-window.

## 63. Decode integer path: bench pulse real (2-2.8x), model fidelity fails - reverted, prefill stands (2026-11-04)

Round-8 outcome: NO-GO for decode, recorded once with the full
trail so it is never re-researched without new evidence.
New TU entry `gemv_dequant_kq8_row_avx2` (thread-local wdata,
no per-call heap alloc, no token loop, decode-pool
compatible) behind `bench batch --kernel kq8d` plus a tight
differential (zero exact, constant 1e-3, random 5%, generic
triangulation) - all green, 101+ cases passing.
Bench A/B, healthy wall, best-of-5 interleaved R1+R2:
gate 8192x3072 2.07x/1.46x, down 3072x8192 2.80x/2.10x,
vocab 128256x3072 1.77x. Gate (>=10% every shape) passes on
the kernel alone.
Model wiring (matvec tokens==1 branch, scalar-oracle gated)
then FAILED the gate that matters: fidelity default (prefill
kq8 + decode kq8) = mean KL 4.533e-11 -> 4.533e-3, worst
1.996e-2, top-1 28/30 (two flips) vs the 1.6e-3 VNNI-class
bar. Isolation proved it is the decode path and only it:
same binary with matvec forced float (batch kq8 kept) returns
to the floor 1.467e-11 / 30-30; `--kernel float`
(float everywhere) also floors. Verdict: matvec wiring
REVERTED (file back to pure float decode); TU row entry,
bench hook, and tests stay - bench-researchable, zero model
impact; `--kernel kq8` now means prefill-only, which is the
verified scope.
Mechanism unresolved and honestly logged as such: the row
entry is bit-faithful to upstream on every differential we
own (generic triangulation 1e-4, AVX2-vs-float inside price),
yet real-data decode diverges. Candidates: (a) activation
outliers pricing decode harder than prefill (per-256-block
amax crushes small values next to a 100x channel; random
test inputs cannot show this); (b) a transcription slip
invisible to random/constant inputs. What would reopen it:
proof of what llama actually runs for single-token decode
(the §52 teardown verified the batch path only - the decode
extrapolation was never sourced), or a differential on
outlier-bearing inputs with a priced gate. Box note: one
transient link-lock (`ld: Permission denied`, no live holder)
and several stale-binary runs during the round; every number
above is from a verified-fresh link.

## 64. IDEA-1568 ablation protocol added, no engine delta (2026-11-04)

Protocol-only change for deep-checked Approved IDEA-1568: added
`docs/ablation-1568.md` (4-family x86 batch-one triage, kill criteria,
back-to-back 20% rule). No kernel, dispatch, or model change; no
numbers claimed. Baseline: `python aikit.py check aegean-tge` 4/4
passed before and after (test 15.4s, hygiene, guidelines, secrets).
Live bench skipped per no-run constraint; existing log sections
19/50/71/75 are the baseline. Verdict: kept (docs).

## 65. BlockTable foundation header plus 6 unit tests, no engine wiring (2026-11-04)

Design-only port for IDEA-1517/1585 Wave 1 step 1:
`include/tge/models/block_table.hpp` (BlockTable: allocate/share/release
refcount, slot_for arithmetic, FNV-1a prefix hash, prefix cache) plus
`tests/test_block_table.cpp` (6 cases) wired in `CMakeLists.txt`.
`Qwen2` cache untouched; no behavior delta by construction.
Suite: 132/132 pass (6 new block_table OK).
`python aikit.py check aegean-tge` 4/4 passed.
Live bench skipped per no-run constraint; falsifier gates for the wiring
step remain: 8x32 >=58.3 t/s, identical ids, KL<=1e-9. Verdict: kept.

## 66. KV block layout helpers plus 3 unit tests, no engine wiring (2026-11-04)

Wave 1 step 2 for IDEA-1585: `include/tge/models/kv_block_layout.hpp`
(K column-major / V row-major reorder, block gather; exact round-trip,
fail-loud) plus 3 tests in `tests/test_block_table.cpp`.
Cache untouched; no behavior delta by construction.
Suite: 135/135 pass (3 new kv_layout OK).
`python aikit.py check aegean-tge` 4/4 passed.
Live bench skipped per no-run constraint. Verdict: kept.

## 67. GEMM clamp epilogue helpers plus 4 unit tests, kernel unwired (2026-11-04)

Wave 2 step 1 for IDEA-1539: `GemmArgs` gains `do_clamp/vmin/vmax`
(default off, behavior identical); new `gemm_epilogue.hpp` plus AVX2 TU
`src/cpu/avx2/gemm_epilogue_avx2.cpp` (min/max path plus scalar tail);
`tests/test_gemm_epilogue.cpp` (4 cases: defaults, bounds, AVX2==scalar
over 67 elems incl tail, wide-range identity). Microkernel store path
untouched; wiring gated on cubed bench within-run win.
Suite: 139/139 pass (4 new OK). `python aikit.py check aegean-tge`
4/4 passed. Verdict: kept.

## 68. Portable IO advice helpers plus 2 unit tests, no prefetch change (2026-11-04)

Wave 2 step 2 for IDEA-1586: `include/tge/memory/io_advice.hpp`
(page-aligned range math, bounded PrefetchBudget accounting; no OS calls,
gguf prefetch untouched) plus `tests/test_io_advice.cpp` (2 cases).
Suite: 141/141 pass (2 new OK). `python aikit.py check aegean-tge`
4/4 passed. Wiring (advise_range wrapper + bounded KV prefetcher) gated
on >=30% RSS cut at parity. Verdict: kept.

## 69. Decode default to physical cores (H3, kept: +40% story, ids identical)

`resolve_decode_threads()` preferred P-cores (6) on hybrid Windows; the
decode pools are unpinned since §28ab, so one worker per physical core
(10 here) beats one per P-core. Same-window interleaved A/B, Llama-3B
Q4_K_M, warm page cache:

| story decode, 15 tok | threads 6 | threads 10 unpinned |
|---|---|---|
| run 1 / 2 / 3 | 4.32 / 4.13 / 4.79 t/s | 4.97 / 6.41 / 6.72 t/s |
| best | 4.79 | 6.72 (**+40%**) |

| 241-tok prefill, threads 6 fixed | prefill-threads 6 | default (16) |
|---|---|---|
| best of 3 | 20.8 t/s | 37.8 t/s (**+82%**) |

3/3 pairs favor 10 on decode and 3/3 favor 16 on prefill; story and
reasoning ids byte-identical 6 vs 10 (placement only, no arithmetic, so
no fidelity owed). Post-change default-vs-explicit-6 re-check: 3.58 vs
2.88 and 4.13 vs 1.01 (second 6-arm hit a load spike). Caveats: box
loaded (absolutes 1-7 vs the 8.6-9.6 baseline), single session, so the
direction is 5/5 consistent but the magnitude will move on a quiet box.
Prefill needed no code (default was already all-logical). Tests green,
`src/core/dispatch.cpp` only (8+/3-). Verdict: kept as the default with
the per-machine caveat.

## 70. Shared-prefix skip in prefill_run (H1, kept: ~50% 2nd-turn prefill)

Spike first (throwaway worktree, env-guarded hook + `spike-h1` driver,
reverted by deleting the worktree): 241-token prompt, 120 shared
prefix tokens, Llama-3B Q4_K_M, threads 10, best-of-3 warm cache — hit
(reuse resident prefix rows, compute suffix only) vs miss (full
recompute): 19.18 vs 41.46 s (54%), 5.45 vs 10.37 s (47%), 18.80 vs
35.96 s (48%), final-logits memcmp identical 3/3. The ~50% matches the
prefix fraction (120/241) exactly: prefix compute eliminated, suffix
at full speed. Thinking set (reasoning + story ids) identical hook
on vs off.

Production (`670bcd9`, `src/models/qwen2.cpp` +40 lines, no other
files): always-on per-sequence last-call record in `prefill_run`;
on a generate-path call extending the same sequence/start with
identical leading tokens and a non-empty suffix, `consumed` starts at
the match length (guarded `count > recorded`, verifier `all_logits`
path excluded). Zero behavior change for single-prefill callers;
continuous slots and multi-turn loops benefit automatically. Fresh
build green, suite green, story + reasoning ids identical to every
prior run. Verdict: kept.

## 71. Six-deep greedy draft with five-deep fallback (2026-11-04)

The verifier runs a six-row batch but the greedy drafter only proposed
five, padding the last row with a zero token. The greedy path now drafts
six (`apps/cli/main.cpp`: `prompt_lookup_draft` takes `need`, greedy
passes 6, the whole batch carries drafts); the historical five-deep form
(exact spans, bigram chain, gapped windows) is the fallback inside the
same function, so a short history that cannot supply six continuations
behaves exactly as before. The sampled path keeps `need` 5: proposals
and verify rows byte-identical to before, accept loop untouched. The
penalized path penalizes all six rows including the bonus (row 5 was
unpenalized before, but unreachable with five drafts), so the bonus keeps
exact single-token penalty semantics. No numerics change anywhere the
model reads: drafter-only, verifier-gated, no fidelity run owed.

Llama-3.2-3B Q4_K_M, `--threads 6`, `--prefetch`, periodic 40-token
history, 15 timed tokens, interleaved best-of-3, old vs new binaries:

| round | base acceptance | new acceptance | base best | new best | ids |
|---|---|---|---|---|---|
| R1 no-penalty | 5.00/step 3/3 | 6.00/step 3/3 | 7.23 | 8.85 (+22%) | identical |
| R2 no-penalty (reversed) | 5.00/step 3/3 | 6.00/step 3/3 | 8.77 | 10.52 (+20%) | identical |
| R3 prod no-penalty | 5.00/step 3/3 | 6.00/step 3/3 | 5.93 | 5.60 (-6%, one collapse arm) | identical |
| penalty 1.1/w4 x2 | 5.00/step 2/2 | 6.00/step 2/2 | -- | -- | identical |

Acceptance is deterministic: 6.00 vs 5.00 in all 13 new-binary runs, 5.00
in all 13 base runs — +20% tokens per verify, +17.6% tokens per
verify+forward pair (20 vs 17 emitted per 3 steps). Wall-clock converts
when the box is quiet (R1/R2: +20-22%); R3 ran loaded and the pairs are
drift-dominated, reported as-is. Story abstains 0 steps on both with the
historical ids (`11 304 ... 11 1070`); reasoning + creative texts
identical; 11-token periodic history identical (1.00/step both). Tests
141/141 on the shipping binary. Verdict: kept.

## 2026-11-05 — SKIP-locked drafter + fused verify + cross-sequence CoW prefix

Three invented mechanisms landed together (invention book 2026-11-05:
32 mechanisms, steelman verdicts, temp-copy engine proofs before port;
the port is line-identical logic with production comments, no test
scaffolding). Tests 141/141 on the shipping binary.

SKIP-locked (`skip_locked_draft`, drafter-only, verifier-gated, d0
pre-check unchanged): period-1/period-2 lock proposes along the cycle.
Qwen2.5-0.5B Q8_0 periodic: 96.19 t/s at 6.00/step ceiling, ids
identical; story neutral (0 steps, identical ids). Llama-3.2-3B Q4_K_M:
period-1 lock 43-50 t/s at 5.25/step vs 5.7-9.9 nospec (interleaved AABB:
4.62x, 5.03x); periodic non-locked 5.12 -> 7.69 (1.50x, 2.12/step);
story neutral 10.55 vs 10.89 (0.97x), 0 steps, identical ids.

Fused verify steady state (generate loop only, no kernel change): after a
full-accept step the bonus KV is resident, so the next step re-verifies
the bonus + 5 history-drafts in ONE 6-row sweep and post-judges all rows
instead of a 1-row correction forward. Llama-3B lock: 30-32 -> 46-50
t/s (1.43x, 1.52x interleaved pairs, ids identical); 0.5B lock 133 ->
176 t/s (1.32x). Non-locked prompts never chain and behave exactly like
the baseline (0.5B periodic mean +3%, sign flips inside the band).

Cross-sequence CoW prefix (`prefill_run` + global snapshot, f32 only,
verifier excluded, >=16 common tokens): first start-0 prefill populates,
later sequences memcpy rows and skip recompute, byte-identical by
construction. 8x(120+32): 3.06-3.32x prefill-side on 0.5B (4 runs) and
3.06x on Llama-3B, FNV hash gates 7/7 per model vs cold recompute.
Verdict: all three kept; parked in the book (scale-cache harmful-shape,
dual-math inverted premise) stay out.

## 2026-11-06 — Verify-then-build session: five temporary spikes killed or amended, prices re-confirmed (all reverted, tree green)

Five temporary spikes (built, measured, fully reverted in-tree via
`git checkout --`, tests 141/141 after each revert; no commit, no
push) plus read-only re-confirmations on Llama-3.2-3B-Q4_K_M,
`--threads 6`, warm page cache. Reference health: wall read
~54-61 GB/s, story ids `11 304 … 11 1070` (~10 t/s quiet).

### S1. Trigram-chain drafter fallback — KILLED (dead code, reverted)
`trigram_chain_draft` (order-2 successor chaining) appended after the
gapped fallback in `prompt_lookup_draft`. Story spec: 0 steps, ids
identical; repetitive: 3 steps, 18 accepted (6.00/step), ids identical
to the pre-spike baseline. Whenever the trigram condition holds (pair
observed), the bigram condition holds and fires first — unreachable
except a 32-cap edge. Tie = revert per the no-evidence rule. Doubly
determined (triply with shipped SKIP-lock and the recorded trie kill).

### S2. IDEA-1325 pre-packed-B GEMM — KILLED at kernel level (reverted)
`gemm_packed_preb` (persistent-B cache, same panels/order →
max|reference-out| 7.629e-05, differential gate passes) behind
temporary `--kernel preb`. 1024^3 best-of-5: R1 packed 16T 11.98 ms
vs preb 11.87 (+0.9%), 1T +15% (reference itself drifted 102→121 ms
between processes); R2 reversed preb 16T 10.74 vs packed 14.01
(+30%), 1T -54%. Sign flips on both thread counts with identical
binaries — box state, not kernel (a skipped pack_b cannot cost -54%
on identical compute). Falsifier needs >=5% consistent: kill, no
model wiring. Retry trigger: quiet box + same interleaved pairs.

### S3. SlimAttention-style tiled prefill attention — KILLED (reverted)
Q-block(8) x K-block(128) tiling with per-(query,head) online softmax
in `attention_batch`, env-gated (`TGE_TILE_ATTN=1`), f32 dense path
only. 2043-token repetitive prefill: dense attn 12.83 s of 59.6 s vs
tiled 19.12 s of 62.4 s (attn +49% worse, prefill 0.95x). Scalar dots
+ `expf` lose to the AVX2 kernel by far more than the K/V traffic
saving. Ids byte-identical 31/31 (greedy-exact here); 27-tok ids
identical (path live, parity holds short). Reopen trigger:
vectorized tiled inner loop matching AVX2 compute rate.

### S4. Multi-prefix CoW table — REJECTED, not portable (reverted)
4-entry snapshot table (longest run >= 16) + temp `spike-mprefix`
driver (8 fresh seqs, 2x120 shared prefixes + 32 suffixes, FNV
determinism check). Two clean fast hits (~600 ms vs ~2250 ms
populates, deterministic hashes) then unexplained misses (one fell
through to the single snapshot, one full compute) plus
populate-on-hit slot waste (found and half-fixed mid-spike). A spike
that misbehaves inexplicably must not port. Single-prefix path
unaffected (story ids identical on the spike binary). Parked with a
precise trigger: clean re-spike (populate-only-on-miss, single
retired, untruncated stderr) that explains the misses first. Shipped
single-prefix CoW stands.

### S5. LUT nibble unpack — KILLED on uop arithmetic (no silicon)
Per 32 weights the shipped `write32` spends ~20+ uops; the nibble
split itself is 1-2. `pshufb` cannot remove the high-nibble shift
(it indexes low nibbles only) and replaces a 1-uop `and` with a
1-uop shuffle on the low path: max theoretical saving ~1 uop, ~5%,
vs the item's own >=15% kernel gate. Scout NO-GO stands with the
explicit count instead of a hunch.

### S6. 1517 stage-1 batch-side snapshot — construction proven, gate amended (reverted)
Contiguous-from-0 arrival tracking + end-of-`forward_batch`
snapshot + FNV MATCH gate vs a cold `prefill_run` snapshot for the
same 32-token prompt. Tracking fires (`[batch-snap] POPULATE slot=0
C=32`) but MATCH fails on Llama-3B by construction, not by bug:
`batch_matvec` forces float math (`float_only`, the
continuous-batch LEAK fix) while `prefill_run` takes the kq8
integer path on Q4_K — different arithmetic, so byte-identity was
never on the table (it would MATCH on Q8_0, the wrong model to
prove it on). Amendment, not kill: kind-tagged snapshots (reuse
the 1384 quantized-tier recipe) + KL pricing, never cross-path
sharing of rows between float and integer arithmetic. The epic's
headroom (up to ~44% redundant recompute at P=32) stands untouched
— it is about traffic, not arithmetic.

### Prices re-confirmed (no code this session)
KV f16 fidelity mean 3.1e-7 / top-1 30/30, i8 3.24e-4 / 30-30 at
ctx 512 and 2048 (floor 1.467e-11 throughout); story f16 9.60 vs
f32 9.97 t/s (overhead, no KV pressure). Sparse-2043 best-of-3
interleaved: prefill best 48.7 vs 48.1 (0.99x), decode best 6.41
vs 7.12 (+11.1%, round signs 0.86x/1.13x/1.09x — under the 15%
bar, flip-flopping); ids identical 6/6. KV-2048 repetitive:
i8 best-of-best +3.8% (neutral; round deltas drift-contaminated,
only best-of-best counts; does not overturn the §29 window).
Continuous triplet 8x32 best-of-1: P=8 20.29 → P=32 12.07 t/s
(crowd-out caution, shape only, absolutes unclaimed).
Cross-checked vs peer reports (1514 analytic, 1542 + recorded
trie kill, 1501 supersede with chunked-GQA + quantize-on-write
both shipping, kv-auto Llama validation, sparse-f16 recipe):
no conflicts; f16 numbers same order (1.658e-6 cited there vs
3.1e-7 here on Llama-3B).

Commands run (selection; every spike followed
build → measure → `git checkout --` → rebuild → `tge_tests`
141/141 → `git status` clean): `bench mem`, story/repetitive
`generate` (+`--speculative`, `--kv f16|i8`, `--sparse`,
`--print-ids`), `bench batch` (q4k gate / q6k down x64),
`bench continuous` (8x32/4 slots, `--prompt-tokens` 1/8/32),
`fidelity` (arch corpus, ctx 512/2048, 31 tok, f32/f16/i8),
2043-token id-list prefill pairs (dense/sparse/i8/tiled),
temp `spike-mprefix` (8-seq A/B + FNV) and `spike-batchsnap`
(MATCH gate) drivers on spike binaries, `git log/status/diff`
throughout. Verdicts: S1 kill, S2 kill, S3 kill, S4 reject
(not portable), S5 kill, S6 amend-and-proceed (epic-owned).

## 2026-11-08 - Integration: shipped stack (q8 tail, 4 GQA chunks, block prefill; kq8d opt-in)

Shipped defaults. Before/after are same-window pairs; full evidence in the cycle reports.

| Change | Before | After | Verdict |
|---|---|---|---|
| GQA q8 value vector bound (full 128-dim head) | span-4096 kernel best 0.621-0.833 ms/call; attention/matvec 0.482-0.556 | 0.335-0.562 ms/call; attention/matvec 0.421-0.459; decode 1.059x in the matched pair | kept |
| GQA decode chunks 2 -> 4 at span >= 1024 | attention/matvec 0.482 / 0.479; decode 2.01 / 1.99 t/s | 0.421 / 0.459; decode 2.10 / 2.49 t/s (1.045x / 1.251x) | kept |
| Query-block prefill attention, f32 KV | 8k prefill 682.7 s / 600.9 s; attention 471.9 s / 406.9 s | 475.0 s / 501.7 s; attention 251.2 s / 249.2 s (1.88x / 1.63x); ids identical | kept |
| Query-block prefill attention, f16 KV | 8k prefill 544.2 s / 654.6 s; attention 295.7 s / 386.3 s | 401.6 s / 430.9 s; attention 200.8 s / 211.4 s (1.47x / 1.83x); ids identical | kept |
| Query-block prefill attention, i8 KV | 8k attention 283.5 s / 220.2 s | 276.9 s / 205.6 s (1.02x / 1.07x) | rejected, per-token path kept |
| Integrated defaults vs HEAD, 4k prompt / 63 tokens | 2.62 t/s (HEAD binary) | 2.95 t/s (1.126x); ids identical at 4k/63 and 1024/32 | kept |

Opt-in decode scopes (`--kq8d`, default off):

| Scope | 399-pos KL / top-1 | 2,399-pos KL / top-1 | Speed | Verdict |
|---|---|---|---|---|
| head | 1.874e-4 / 393/399 | 1.915e-4 / 98.83% | head matvec 3.3-3.6x; decode 1.29x at 4k / 63 tok | kept opt-in |
| head,down | 9.711e-4 / 390/399 | 7.874e-4 / 97.67% (f16 97.79%) | decode 1.29x at 4k; down +10-36% solo | kept opt-in |
| head,down,attn | 1.576e-3 / 1,167/1,199 (97.33%) | - | - | rejected: top-1 below the 97.6% bar |

`--kq8d` is documented in `tge generate --help`, `tge fidelity --help`, the README and ADR-009. The integrated flags reproduce the spike numbers exactly on the 399-position corpus (off 1.020e-11/399, head 1.874e-4/393, head+down 9.711e-4/390).

Tests 146/146 (new block-attention differential), `aikit.py check aegean-tge` 4/4.

## 2026-11-08 - Prefill chunk default: 128 rows up to 4096 prompt tokens (kept)

Before: every prompt was processed in 64-token chunks. After: 128-token chunks when the prompt is <= 4096 tokens, 64 above. Same block-attention kernel, same accumulation order per chunk.

| prompt | 64-token chunks | 128-token chunks | ratio | ids |
|---|---|---|---|---|
| 2048 (pair 1) | 30.2 t/s | 30.1 t/s | 0.997x | identical |
| 2048 (pair 2) | 23.8 t/s | 32.5 t/s | 1.366x | identical |
| 2048 (pair 3) | 13.6 t/s | 26.8 t/s | 1.971x | identical |
| 2048 (pair 4) | 21.9 t/s | 24.0 t/s | 1.096x | identical |
| 4096 (pair 1) | 23.1 t/s | 27.2 t/s | 1.177x | identical |
| 4096 (pair 2) | 20.0 t/s | 29.8 t/s | 1.490x | identical |
| 8180 (pair 1) | 16.5 t/s | 14.9 t/s | 0.903x | identical |
| 8180 (pair 2) | 17.8 t/s | 16.6 t/s | 0.934x | identical |

Verdict: kept, span-aware. 128 is faster in 3/4 pairs at 2048 (median ~1.23x) and 2/2 at 4096, but 64 wins both 8192 pairs by 7-11%, so long prompts keep the old chunk. Buffers are sized for 128 rows (logits_batch and scores), which costs about +58 MB at context 4096 and +83 MB at 8192; the default run at 2048 measured 26.7 t/s with repeat ids identical. Tests 146/146, 4/4 gates.

## 2026-11-08 - f16 block attention: widen each K/V row once per key (kept)

Before: `attention_block_head_avx2` converted every f16 K/V row from half to float inside the query loop. A 128-query chunk therefore widened each row 128 times. After: `widen_row` converts the row once into a 256-float aligned stack buffer (heap fallback above 256) and both passes reuse it. The arithmetic order and the float values are unchanged, so output is bit-identical.

| prompt, f16 KV | pairs | attention ratio (median) | prefill ratio (median) | ids |
|---|---:|---:|---:|---|
| 2048 tokens | 6 | 1.034, 1.030, 1.034, 1.075, 1.142, 1.178 -> **1.054** | 0.891, 1.022, 1.026, 1.110, 1.170, 1.321 -> **1.068** | identical 6/6 |
| 4096 tokens | 4 | 0.828, 1.376, 1.250, 0.995 -> **1.123** | 0.855, 1.367, 1.290, 0.927 -> **1.109** | identical 4/4 |

5/6 pairs at 2048 and 3/4 at 4096 favor the hoist; the median is the claim. Fidelity at 399 positions is unchanged: f16 mean KL 3.252e-07, worst 2.709e-05, top-1 399/399; f32 stays 1.020e-11 / 399/399. The block-attention differential test passes.

Also measured and reverted this cycle: adaptive speculative depth/cooldown (8 pairs, median forward ratio 1.006), 256-row prefill chunks (2048 0.946x/0.976x; 4096 1.086x/0.996x), and a Q4_K scale-metadata cache (6-thread 1.16-1.37x, default 16-thread 0.945-1.037x). Details: `reports/idea-cycle-negative-results-2026-11-08.md`. Tests 146/146, `aikit.py check aegean-tge` 4/4.

## 2026-11-08 - Block prefill attention: 8-query value-accumulation tile (kept)

Before: the value pass of `attention_block_head_avx2` held every query's accumulator in `args.out`, so a 128-query chunk kept 128 x head_dim floats live (64 KiB at head_dim 128, 32 KiB at 64) — past L1. Each key iteration read and wrote the full set. After: the pass runs in 8-query tiles whose accumulators (8 x head_dim, 2-4 KiB) live in an aligned stack buffer and are copied to the output when the tile finishes. Key order per query and the FMA order are unchanged, so the result is bit-identical. The trade is re-streaming each value row once per tile instead of once per key.

| prompt / KV | pairs | attention ratio (median) | prefill ratio (median) | ids |
|---|---:|---|---:|---|
| 2049 / f16 | 3 | 1.690, 1.986, 1.319 -> **1.690** | 1.304, 1.240, 0.785 | identical 3/3 |
| 2049 / f32 | 3 | 1.327, 1.576, 1.645 -> **1.576** | 1.118, 1.050, 1.156 -> **1.118** | identical 3/3 |
| 4081 / f16 | 2 | 2.409, 1.774 -> **2.09** | 1.546, 1.128 | identical 2/2 |
| 4081 / f32 | 2 | 1.679, 2.333 -> **2.006** | 1.180, 1.443 | identical 2/2 |

Attention is faster in 10/10 pairs; the matvec phase is untouched and drifted 0.71-1.34x across the same runs, so the total-prefill ratios move with the window and the attention phase is the attributable number. Median total prefill is about 1.14x at 2049 and 1.31x at 4081. Fidelity is exactly the pre-change values: f32 mean KL 1.020e-11, f16 3.252e-07, top-1 399/399 both; the block-attention differential test passes. A temporary `TGE_GQA_CHUNKS` decode override left in `src/models/qwen2.cpp` by a prior session was dropped before the commit (inert, no report or ledger trace). Tests 146/146, `aikit.py check aegean-tge` 4/4. Report: `reports/idea-invented-value-tile-keep.md`.

## 2026-11-08 - 16-query value tile vs the shipped 8-query tile (rejected)

The kept tile-8 report recorded a 16-query tile as the follow-up: still L1-resident at head_dim 128 (8 KiB) and half the value-row re-streams per 128-query chunk. Tile-8 and tile-16 binaries (same source except `kValueTile`) were run in alternating pairs on Llama-3.2-3B-Q4_K_M, `--threads 6 --gpu off`, n=2. Ratios are tile-8 time / tile-16 time, so >1 favors tile-16.

| prompt / KV | pairs | attention ratio (tile16 vs tile8) | ids |
|---|---:|---|---|
| 2049 / f16 | 4 | 1.056, 0.927, 0.997, 0.845 -> **0.962** | identical 4/4 |
| 4081 / f16 | 3 | 1.022, 0.890, 0.836 -> **0.890** | identical 3/3 |

Median 0.927 across seven pairs: the 16-query tile is slower. The predicted row-traffic saving does not pay for the larger accumulator footprint. In the 2049 pairs the untouched matvec phase favored the tile-16 arm in 3/4 (1.051-1.109), so the attention shortfall is not a uniform window bias. Reverted to tile 8; tests 146/146. Details: `reports/idea-cycle-negative-results-2026-11-08.md` §4.

## 2026-11-08 - 128-row chunks above 4096 retested after the value tile (inconclusive, 64 kept)

The 8192-token chunk rule (64 rows above 4096) was chosen before the value tile, when a 128-query chunk's 64 KiB value-accumulator set lost 7-11%. The tile cut that set to 4 KiB, so 128 was retested. Shipped binary vs a same-source 128-everywhere binary, alternating pairs at 8161 tokens, f16 KV, Llama-3.2-3B-Q4_K_M, `--threads 6 --gpu off`, n=2. Ratios are 128/64; <1 favors 128.

| pair | total 64 → 128 (s) | ratio | matvec 64 → 128 (s) | attention 64 → 128 (s) | ids |
|---:|---|---:|---|---|---|
| 1 | 308.8 → 335.1 | 1.085 | 170.4 → 184.5 | 121.3 → 131.7 | identical |
| 2 | 482.1 → 401.3 | 0.832 | 288.9 → 222.3 | 173.3 → 159.2 | identical |
| 3 | 421.5 → 313.4 | 0.743 | 257.8 → 185.8 | 144.2 → 109.6 | identical |
| 4 | 259.8 → 302.5 | 1.164 | 152.1 → 174.1 | 92.1 → 111.0 | identical |

Median 0.958, geometric mean 0.940, but the spread is inside the drift: the identical 64-row arm's matvec phase ranged 152.1-288.9 s (1.90x) across the session, and the phase ratios disagree within pairs. The isolated kernel is the reliable datum: `bench batch --format q4k --kernel kq8 --repeat 10` gives 3.462 ms / 348.9 GFLOPS at 64 tokens vs 7.199 ms / 335.6 GFLOPS at 128 on 3072x3072 (64 is 3.8% better per token), and 9.255 vs 18.559 ms (equal per token) on 8192x3072. Per-token matvec throughput is flat-to-worse at 128, so the chunk count is not the lever. Kept the 64-row cap; ids identical 4/4, tests 146/146 after the revert. Details: `reports/idea-cycle-negative-results-2026-11-08.md` §5.
