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
