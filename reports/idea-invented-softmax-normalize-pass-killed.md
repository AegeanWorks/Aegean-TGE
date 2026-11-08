# INVENTED: drop the softmax normalization pass and scale each query's accumulator once — KILLED (ids/fidelity identical, no repeatable >3% gain)

- **Idea**: The block-prefill softmax runs three passes over each query's score row: max scan, exp + sum (in place), then a third pass multiplying every score by `1/sum`. The value pass then FMA-weights each value by the normalized score. Instead, keep the exp values unnormalized and scale each query's finished accumulator by `1/sum` during the final tile copy. At span `S` and head width `D`, this removes `S` reads + `S` writes + `S` multiplies per query and adds `D` multiplies.
- **Cycle date**: 2026-11-08
- **Verdict**: **KILLED.** Numerically free (ids and fidelity byte-identical) but no repeatable gain: end-to-end attention is +7% raw at 2049 but +2.7% after matvec-drift normalization, and -1.9% at 4081; the isolated kernel bench shows 0.975-1.072 across four cells with up to 25% process spread. The stated >=3% falsifier is not met.
- **Tree state**: spike reverted from `src/cpu/avx2/attention_avx2.cpp` and `tests/test_attention.cpp`; branch `idea/softmax-scale-output` deleted; `git status` clean; nothing committed except this report.

## 1. Mechanism

Current `attention_block_head_avx2` softmax:

```text
for each query:
    pass A: max over span
    pass B: exp(score - max) in place, sum
    pass C: score *= 1/sum          <-- removed
```

Value pass reads the normalized scores. The spike keeps pass A/B, stores `inverse_by_token[token] = 1/sum`, and replaces the third pass with a scaled final copy:

```text
for each tile token:
    output[i] = tile_acc[i] * inverse_by_token[token]   (AVX2 8-wide)
```

`inverse_by_token` is a 128-float aligned stack buffer with a heap fallback for `tokens > 128` (production chunks are <=128). The accumulator FMA order, key order, and per-query max/exp are unchanged; only the final scaling moves.

Predicted gain: pass C is one of three score-row passes. With `S` scores and `D` accumulator values per query, it removes about `3*S` float operations and adds `D`. If the softmax phase is 10-20% of attention, the attention cap is ~3-7%; end-to-end prefill cap ~1%.

## 2. Falsifier

- **KEEP** if 4081-token f16 attention is >=3% faster and the fixed thinking set ids are identical, with fidelity unchanged.
- **KILL** if the gain is below 3% or the ids move.

Measured: ~2.7% at 2049 after drift normalization, -1.9% at 4081, kernel bench 0.975-1.072; ids identical. Kill.

## 3. Measured numbers

Model `Llama-3.2-3B-Instruct-Q4_K_M.gguf`, `--threads 6 --gpu off`, n=2, same-source baseline vs spike binaries, alternating order. Ratios are baseline / spike, so >1 means the spike is faster.

### 2049 tokens, f16 KV

| pair | attention baseline (ms) | attention spike (ms) | ratio | matvec ratio | drift-adjusted |
|---:|---:|---:|---:|---:|---:|
| 1 | 6,871.9 | 8,289.5 | 0.829 | 0.855 | 0.970 |
| 2 | 7,455.8 | 6,969.7 | 1.070 | 0.978 | 1.094 |
| 3 | 6,039.2 | 5,622.2 | 1.074 | 1.060 | 1.013 |
| 4 | 6,218.6 | 5,624.9 | 1.106 | 1.062 | 1.041 |

Raw median **1.072**; matvec-normalized median **1.027**. The first pair's whole window favored the baseline (matvec 0.855), so the raw number overstates the gain.

### 4081 tokens, f16 KV

| pair | attention baseline (ms) | attention spike (ms) | ratio | matvec ratio | drift-adjusted |
|---:|---:|---:|---:|---:|---:|
| 1 | 27,911.3 | 50,364.4 | 0.554 | 0.597 | 0.928 |
| 2 | 39,893.0 | 35,920.7 | 1.110 | 0.970 | 1.144 |
| 3 | 34,725.0 | 26,963.5 | 1.288 | 1.312 | 0.981 |

Raw median 1.110, drift-adjusted median **0.981**. The box moved up to 68% on the untouched matvec phase inside a pair; the normalized ratios are the only usable signal and they straddle 1.

### Isolated block-kernel bench (128 queries, one head, `D=128`)

Temporary `TGE_ATTN_BENCH` test, best-of-12 calls per process, 3-6 alternating processes per cell:

| KV | span | baseline best (ms) | spike best (ms) | ratio |
|---|---:|---:|---:|---:|
| f32 | 4096 | 11.491 | 11.788 | 0.975 |
| f16 | 4096 | 11.955 | 11.747 | 1.018 |
| f32 | 8192 | 21.857 | 20.395 | 1.072 |
| f16 | 8192 | 24.419 | 23.729 | 1.029 |

An earlier f16/4096 pair read 13.526 vs 8.716 (1.372); six alternating reruns did not reproduce it (spike range 11.747-13.845). Per-process values within one cell ranged 11.955-19.522 for the same binary, so the cell medians, not the minima, are the honest read: f16/4096 ~12.5 vs ~12.8 (neutral), f32/8192 ~26.5 vs ~25.2 (~5%, inside the spread). No cell clears 3% repeatably.

### Numerics and behavior

- `tge fidelity --kv f16`, 399 positions: mean KL **3.252e-07**, worst 2.709e-05, top-1 **399/399** — identical to master.
- Fixed + creative thinking set: Prompt A ("The quick brown fox jumps over the lazy dog. Explain why.", n=32, ids `320 2028 ... 16053`) and Prompt B ("Write a haiku about circuits dreaming.", n=24, ids `4815 34 ... 9290 25`) are **byte-identical** on both binaries.
- Full suite 162/162 after adapting the block test to normalize both score rows before comparing (the spike intentionally leaves the block scores unnormalized).

## 4. Why it failed

- Pass C is a small slice. The value pass does `S * D` FMA pairs per query; pass C does `S` reads + writes + multiplies. Their ratio is about `3/D` within the softmax's share of attention, so a 1-2% attention effect is the realistic ceiling on this schedule.
- The added final scale (`D` multiplies per query) offsets part of it, and the scaled copy replaces a `memcpy` with an AVX2 multiply/store, so the copy is no longer a pure move.
- The box's process-to-process spread (up to 25% in the isolated kernel bench, 68% in one end-to-end pair) is larger than the effect, so any "win" would need far more pairs than a cycle can justify.

Safe but useless: keep the three-pass form. It is more obviously correct with the existing score buffer and costs no measurable time.

## 5. Full reimplementation recipe (to rebuild this exact spike from `master`)

Touch two files.

1. `src/cpu/avx2/attention_avx2.cpp`, `attention_block_head_avx2`:
   - Before the softmax loop, declare `constexpr int kStackTokens = 128; alignas(32) float stack_inverse[kStackTokens]; std::vector<float> heap_inverse; float* inverse_by_token = stack_inverse; if (tokens > kStackTokens) { heap_inverse.resize(tokens); inverse_by_token = heap_inverse.data(); }`.
   - Replace the final `const float inverse = 1.0F / total; for (...) scores[index] *= inverse;` with `inverse_by_token[token] = 1.0F / total;` and delete the loop.
   - In the tile copy at the end of pass 2, replace the `std::memcpy(output, tile_acc + ..., head_dim * sizeof(float))` with an AVX2 loop: `const float inverse = inverse_by_token[token]; const __m256 vinverse = _mm256_set1_ps(inverse); for (index = 0; index + 8 <= head_dim; index += 8) _mm256_storeu_ps(output + index, _mm256_mul_ps(_mm256_loadu_ps(accumulator + index), vinverse)); for (; index < head_dim; ++index) output[index] = accumulator[index] * inverse;`
2. `tests/test_attention.cpp`, `attention_block_matches_head_path`: the block scores are now unnormalized; sum them per `(token, head)` and compare `blk_scores / blk_sum` against `ref_scores / ref_sum` (tolerance 1e-4). Add the temporary bench test described below if a kernel number is wanted.

Optional kernel bench (spike-only): a `TGE_ATTN_BENCH=<span>` test builds 128 random queries and `span` K/V rows (`D=128`, f32 or `TGE_ATTN_BENCH_KV=f16`), sets `start = span - 128`, calls `attention_block_head_avx2(args, 0)` in 3 rounds x 4 calls and prints the best milliseconds as `[attn-bench] <kv> span=<span> tokens=128 best=<ms> ms`.

## 6. Reopen triggers

- Only if a profile shows the score normalization pass is a meaningful share (>=10% of attention) — e.g., a schedule where scores live in a slower tier or the value pass becomes much cheaper.
- The real lever in this family is a single-pass online softmax (flash-style), which merges the max and exp scans but changes arithmetic more; price it as its own idea with the same fidelity/ids gates.
- Do not reopen for the current schedule on this box: the effect is below the measurement floor.
