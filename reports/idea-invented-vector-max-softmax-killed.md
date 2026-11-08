# INVENTED: vectorize the block-prefill softmax max scan — KILLED (sub-0.5% share, no resolvable gain)

- **Idea**: The block kernel's softmax starts with a scalar `std::max` reduction over each query's score row. The compiler leaves it scalar (two max instructions in the whole kernel). Replace it with four AVX2 max accumulators plus a horizontal max. Max selection is exact, so the resulting softmax scores and ids are bit-identical.
- **Cycle date**: 2026-11-08
- **Verdict**: **KILLED on share before the measurement matters.** Per query with span `S` and head width `D`, the max pass costs `S` max ops while the QK and value passes cost `2 * S * D` FMAs; at `D=128` the max scan is at most **~0.4%** of attention compute. The isolated kernel bench shows no consistent gain: best-of ratios 0.880 / 1.186 / 0.773 / 1.037 across four cells, with per-process medians straddling 1 and spreads of 10-30 ms. Ids are identical and no arithmetic changed.
- **Tree state**: spike reverted from `src/cpu/avx2/attention_avx2.cpp` and `tests/test_attention.cpp`; branch `idea/vector-max-softmax` deleted; `git status` clean; nothing committed except this report.

## 1. Mechanism

Current block-kernel softmax per query:

```text
float maximum = -INFINITY;
for (index = 0; index < span; ++index) maximum = std::max(maximum, scores[index]);
```

The spike replaces it with four `__m256` accumulators (32 values per iteration), an 8-wide remainder loop, a scalar tail, and a horizontal max through `_mm_max_ps` and two shuffles. The maximum is exactly the same value for finite scores, so `exp(score - max)`, the sum, the normalized weights and the output are bit-identical. QK scores are finite (dot products of finite activations), so the NaN-ordering difference between `std::max` and `vmaxps` cannot occur.

## 2. Falsifier

- **KEEP** if the isolated block kernel is >=2% faster at span 8192 with identical ids (the only honest way to price a sub-percent slice).
- **KILL** if the gain is below 2% or the bench cannot separate it.

Measured: no consistent gain; kill.

## 3. Measured numbers

Kernel bench (128 queries, `D=128`, best-of-12 calls per process, four alternating processes per cell; ratio = baseline / spike, >1 means the spike is faster):

| KV | span | baseline best (ms) | spike best (ms) | ratio (best) | baseline median | spike median |
|---|---:|---:|---:|---:|---:|---:|
| f32 | 4096 | 10.010 | 11.379 | 0.880 | 11.546 | 13.308 |
| f16 | 4096 | 9.598 | 8.090 | 1.186 | 11.236 | 11.458 |
| f32 | 8192 | 19.261 | 24.931 | 0.773 | 27.961 | 28.419 |
| f16 | 8192 | 16.618 | 16.023 | 1.037 | 20.460 | 30.095 |

The sign flips between f32 and f16 in both spans and the medians move by up to 50%; the box drift is far larger than the predicted <0.5% effect. Nothing here supports a keep.

## 4. Numerics and behavior

- Block-vs-per-head differential test passes; full suite **163/163** with the temporary bench.
- Thinking set (fixed + creative, f32): prompts A ("The quick brown fox jumps over the lazy dog. Explain why.", n=32) and B ("Write a haiku about circuits dreaming.", n=24) have **identical ids** on both binaries — expected, because the maximum is exact.
- Fidelity is not owed: the change selects the same maximum and leaves every arithmetic operation unchanged.

## 5. Full reimplementation recipe (to rebuild the spike)

`src/cpu/avx2/attention_avx2.cpp`, `attention_block_head_avx2`, softmax loop:

1. Replace the scalar max loop with four `_mm256_max_ps` accumulators initialized to `-INFINITY`; consume 32 scores per iteration, then 8, then a scalar tail.
2. Fold with `_mm256_max_ps(_mm256_max_ps(max0, max1), _mm256_max_ps(max2, max3))`, split to two `__m128`, and reduce with `_mm_max_ps` plus two shuffles (`_MM_SHUFFLE(1,0,3,2)`, then `_MM_SHUFFLE(2,3,0,1)`); read the result with `_mm_cvtss_f32`.
3. Keep the existing exp/sum/normalize loop unchanged.
4. Optional measurement: the temporary `TGE_ATTN_BENCH=<span>` test (128 queries, `D=128`, f32 or `TGE_ATTN_BENCH_KV=f16`) prints best-of-12 milliseconds per process; compare four alternating processes per cell.

## 6. Reopen triggers

- A profile showing the max scan above ~5% of attention (would need a much smaller head width or a softmax form that scans scores more than once).
- The real long-context lever is score-array traffic, not the max arithmetic: a single-pass online-softmax block (flash-style over the full 128-query chunk and its 64 KiB accumulator) or narrower score storage. Both are separate ideas with their own numerical/fidelity pricing.
- Do not re-spike the scalar max on this schedule: the FMA ratio bounds the gain below 0.5%.
