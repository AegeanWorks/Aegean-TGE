# Measurement Log

Every change to this engine is recorded here with the number that justified it
(project guideline G-002). A change with no entry is not finished.

The point is not ceremony. Three of the optimisations below looked obviously
correct on paper and measured worse, and two more were only rejected because a
second run disagreed with the first. Without the numbers, the plausible story
survives.

## How to add an entry

1. Take a baseline: `.\scripts\measure.ps1 > before.txt`.
2. Make the change.
3. Take the result: `.\scripts\measure.ps1 > after.txt`.
4. If the delta is small, repeat both two or three times — this machine swings
   up to 20 percent run to run.
5. Add a row below with both numbers and the verdict, then commit the row with
   the change itself.

Record the numbers that were measured, not the numbers that were expected. For a
rejected change, record what it lost by; that is the part future work needs.

## Log

Machine: Intel i5-14400F (6P + 4E), 16 GiB single DIMM, Windows 11.
All figures are medians or best-of-N on this machine; n/a means the change was
functional rather than a performance claim.

| Date | Change | Metric | Before | After | Verdict |
|---|---|---|---|---|---|
| 2026-11-01 | Packed A and B panels, 144/256/256 blocking, work-stealing pool | f32 GEMM 2048³ | 18.2 GFLOPS | 246.5 GFLOPS | **kept** |
| 2026-11-01 | Pad packed A stride from 6 to 8 floats | f32 GEMM 1024³ | 62.0 GFLOPS | 52.7 GFLOPS | reverted |
| 2026-11-01 | Software prefetch in GEMV, distance 512 | GEMV 4096², 16 threads | 32.1 GB/s | 28.4 GB/s | reverted |
| 2026-11-01 | `_MM_HINT_NTA` on that prefetch | GEMV 4096², 16 threads | 28.4 GB/s | 29.8 GB/s | reverted (inside noise) |
| 2026-11-01 | Dedicated threaded GEMV decode kernel | GEMV 4096² | 18.4 GB/s (1 thread) | 35.4 GB/s (16 threads) | **kept** |
| 2026-11-01 | Decode thread policy: all logical to P-cores | decode, 0.5B Q8_0 | 21.8 t/s (16 threads) | 32.5 t/s (6 threads) | **kept** |
| 2026-11-01 | Per-matrix thread sizing for decode | decode, 0.5B Q8_0 | 32.8 t/s | 27.5 t/s | reverted |
| 2026-11-01 | Batched prefill kernel (weight row reused across tokens) | prefill, 510 tokens | 11 t/s | 170.9 t/s | **kept** |
| 2026-11-01 | Widen prefill token group from 4 to 6 | prefill, 510 tokens | 170.9 t/s | 191.6 t/s | **kept** |
| 2026-11-01 | Align the 64-token prompt chunk to the six-token group | prefill, 510 tokens | 191.6 t/s | 185.4 t/s | reverted |
| 2026-11-01 | Integer inner loop for prefill (`maddubs`, quantized activations) | prefill, 510 tokens | 186.0 t/s | 150.0 t/s | reverted |
| 2026-11-01 | AVX-VNNI `dpbusd` inner loop | prefill kernel, ffn 4864x896 | 267.9 GFLOPS | 284.0 GFLOPS | reverted (see note) |
| 2026-11-01 | AVX-VNNI `dpbusd` inner loop | prefill, 510 tokens end to end | 186.0 t/s | 150.9 t/s | reverted |
| 2026-11-01 | Skip large metadata arrays when loading the model body | load, 0.5B Q8_0, id prompt | 65 ms | **24 ms** | **kept** |
| 2026-11-01 | Same change, text prompt (tokenizer path) | load + tokenizer | 125 ms | 127 ms | neutral |

### What a prepared sidecar would and would not buy

Asked whether model load could be sped up by writing derived files at first load.
Measuring where the time actually goes answered it:

- GGUF header and tensor-table parsing is **~2 ms**. Caching it in a manifest
  would save nothing worth counting.
- The cost is the **vocabulary**: 151,936 token strings plus 151,387 merge
  rules. Materialising them costs more than mapping 639 MiB of weights, because
  the weights are mapped lazily and the strings are not.
- KV-cache allocation adds ~23 ms per 4096 tokens of context, scaling linearly:
  load was 40.8 / 59.0 / 83.7 ms at context 512 / 4096 / 8192.

So the useful artefact is not a manifest of the weights but a **cached
vocabulary**: the ~100 ms text path is almost entirely 300k string allocations
plus two hash maps. A binary sidecar holding the pieces and a merged rank table
would cut most of it. That is the next thing worth building here, and it is worth
it only for repeated CLI invocations — a server builds the vocabulary once.

Dequantising K-quant weights into a prepared form is the other candidate, and it
is a trap: Q4_K_M at 638 MiB becomes roughly 2.5 GiB as float, which this machine
with 2 GB free cannot hold. The 20 ms saved would be paid for in swap.

### Ablation: what actually caused the GEMM speedup

The first entry bundles three changes into one 18.2 -> 246.5 GFLOPS figure, which
is not one cause. `bench gemm --kernel unpacked|unblocked|packed` separates them.
GFLOPS, best of 2-3:

| 2048³ | 1 thread | 16 threads |
|---|---|---|
| unpacked (no packing, no blocking) | 18.80 | n/a (control is single-threaded) |
| + packing, no blocking | 43.16 | 33.14 |
| + blocking (shipped) | 27.23 | **180.37** |

| 512³ | 1 thread | 16 threads |
|---|---|---|
| unpacked | 72.75 | n/a |
| + packing, no blocking | 33.59 | 24.36 |
| + blocking (shipped) | 51.36 | **122.51** |

What this says:

- **The pool is the largest single factor**: 27.23 -> 180.37 at 2048³, 6.6x.
- **Blocking is what lets the pool work.** Without blocking, adding threads makes
  the kernel *slower* (43.16 -> 33.14): the packed panel spans the whole matrix,
  so every worker re-reads it from DRAM. With blocking it scales 6.6x. Blocking
  contributes almost nothing single-threaded — 43.16 -> 27.23 is a *regression* —
  and only pays at 16 threads.
- **Packing alone is a large-shape win and a small-shape loss**: 18.80 -> 43.16 at
  2048³, but 72.75 -> 33.59 at 512³. Blocking recovers most of the small-shape
  loss (33.59 -> 51.36) without giving up the large-shape gain.

So the honest attribution is: the pool multiplies, blocking is the precondition
for that multiplier, and packing trades small-matrix performance for large. The
single "246.5 GFLOPS" number was 1 thread to 16 threads *and* three code changes
at once; it overstates any one of them by a wide margin.

Caveats on this table: the unpacked control ignores its thread count, so it has
no 16-thread figure; and these were taken while the machine had 3.8 GB free, so
absolute values sit below earlier runs. Ratios are what matter here.

### Notes

**Four of the six rejections looked correct on paper.** Padding the A panel
removes cache-line splits; prefetching hides DRAM latency; the integer inner loop
has fewer instructions; and the VNNI variant genuinely is faster per shape. Each
lost for a reason that only the measurement exposed — extra panel traffic, a
hardware prefetcher that was already doing the job, `psignb` contending for one
port, and a per-call quantization cost that small matrices cannot amortize.

**VNNI is the instructive one.** It is 6 percent faster on the ffn projection and
13 percent faster on the output projection, but 2.4x slower on a 896x896
projection, so end to end it loses. It stays behind `bench batch --kernel vnni`
rather than in dispatch: enabling it for the output projection alone would be
worth roughly 11 percent, but it changes the logits and breaks token parity with
llama.cpp, which is a trade to make deliberately.

**Two entries above are the same idea measured twice.** The `maddubs` version and
the VNNI version both replace the widening with an integer dot; the second was
the better-built attempt and still lost end to end. Recorded separately because
the second one isolates the integration cost, which is the actionable finding.
