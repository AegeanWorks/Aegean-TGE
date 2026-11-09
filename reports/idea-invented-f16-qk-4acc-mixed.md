# INVENTED: four-accumulator f16 QK dot in block prefill attention — MIXED (kernel +2.2–4.9%, end-to-end ~1%, ids/fidelity unchanged)

- **Idea**: The block-prefill QK pass widens each f16 K row to float and dots it against the query. The f16 branch uses two FMA accumulators over a 16-value stride, while the f32 `dot_avx2` already uses four over 32. A 32-wide, four-accumulator f16 loop should hide more FMA latency for the 128-wide head.
- **Cycle date**: 2026-11-09
- **Verdict**: **MIXED.** The kernel gains are real but small and at the box's noise edge: best 1.042x at 4096 and 1.022x at 8192, median 1.049x / 1.040x across four alternating processes. End-to-end at 4081 tokens is ~1% total prefill (normalized attention ~1.05x). Ids are identical on three f16 cases and fidelity is unchanged (3.252e-07, 399/399), so the reorder is safe; the gain is below a solid keep bar.
- **Tree state**: spike reverted from `src/cpu/avx2/attention_avx2.cpp` and `tests/test_attention.cpp`; branch `idea/f16-qk-4acc` deleted; `git status` clean; master rebuilt 165/165; nothing committed except this report.

## 1. Mechanism

Per (key, query) in `attention_block_head_avx2`, f16 path before:

```text
acc0, acc1 = 0
for index step 16: acc0 += q[i..i+8]·k[i..i+8]; acc1 += q[i+8..i+16]·k[i+8..i+16]
remainder step 8 into acc0
score = horizontal_sum(acc0 + acc1)
```

Spike:

```text
acc0..acc3 = 0
for index step 32: four FMAs, one per 8-value chunk
remainder step 16 into acc0/acc1, step 8 into acc0
score = horizontal_sum((acc0+acc1) + (acc2+acc3))
```

The K row is already widened once per key; only the reduction tree changes. The f32 path (`dot_avx2`, four accumulators) is untouched. Because the summation grouping changes, the scores differ at rounding level; the f16 KV quantization (3.25e-7 KL) already dominates, and the measured fidelity is unchanged.

## 2. Falsifier

- **KEEP** if the f16 block kernel is >=3% faster at both 4096 and 8192 spans, ids are identical, and fidelity is unchanged.
- **MIXED** if the gain is only in the medians or below 3% in one cell.
- **KILL** if slower or ids/fidelity move.

Measured: medians 4–5% both spans, best 1.022x at 8192; ids/fidelity unchanged. MIXED.

## 3. Measured numbers

### Isolated block kernel (128 queries, head_dim 128, f16; 4 alternating processes, best-of-12 per process)

| span | baseline best | spike best | best ratio | baseline median | spike median | median ratio |
|---:|---:|---:|---:|---:|---:|---:|
| 4096 | 10.091 ms | 9.684 ms | **1.042x** | 12.287 ms | 11.717 ms | **1.049x** |
| 8192 | 23.017 ms | 22.524 ms | **1.022x** | 26.767 ms | 25.737 ms | **1.040x** |

Suite 166/166 (the temporary bench is inert without its env var; the block-vs-head differential passes at its 1e-4 tolerance).

### End-to-end 4081-token prefill, f16, two alternating pairs

| pair | arm | prefill total (ms) | matvec (ms) | attention (ms) | attn/matvec |
|---:|---|---:|---:|---:|---:|
| 1 | baseline | 139,305.9 | 97,606.2 | 31,344.6 | 0.3211 |
| 1 | spike | 162,414.3 | 115,663.7 | 34,308.6 | 0.2966 |
| 2 | spike | 128,106.7 | 91,357.1 | 27,021.4 | 0.2958 |
| 2 | baseline | 140,467.5 | 100,035.5 | 29,837.9 | 0.2983 |

The spike arm in pair 1 ran in an 18.5% slower window (matvec 115.7 vs 97.6 s), so its raw attention is slower; normalized by the untouched matvec, attention is 1.083x (pair 1) and 1.008x (pair 2), median ~1.046x. Total prefill normalized is ~1.017x / 1.001x — about 1% at this length, consistent with an attention phase that is ~22–30% of prefill.

### Numerics

- `tge fidelity --kv f16`, 399 positions: mean KL **3.252e-07**, worst 2.709e-05, top-1 **399/399** — identical to the pre-change values.
- f16 generated ids identical on prompt A (n=32), prompt B (n=24), and a 64-token repetitive prompt (n=64), base vs spike binaries.

## 4. Why it is not a keep

- The kernel lead is only 2.2% in the best 8192 cell; the median ~4% is within this box's ±20% variance and the raw end-to-end pairs are drift-dominated.
- The normalized end-to-end effect is ~1% total prefill at 4081; at 3B decode (where attention is a smaller share) it would be smaller still.
- The change is safe (ids/fidelity unchanged, 166 tests), so it is a good candidate to re-open on a quiet box rather than a reject.

## 5. Full reimplementation recipe

1. `src/cpu/avx2/attention_avx2.cpp`, `attention_block_head_avx2`, the `args.half` branch of the QK pass: replace the 16-wide two-accumulator loop with a 32-wide four-accumulator loop, then 16-wide (`acc0/acc1`), 8-wide (`acc0`), and the scalar tail.
2. Reduce with `horizontal_sum((acc0+acc1) + (acc2+acc3))`; multiply by `args.scale`.
3. Leave the f32 `dot_avx2` path and the widened row buffer unchanged.
4. Bench: a temporary `TGE_ATTN_BENCH=<span>` test (`TGE_ATTN_BENCH_KV=f16`) with 128 queries, `head_dim=128`, `start = span - 128`; four alternating processes per cell, best-of-12 per process.
5. Gates: `tge_tests`, `tge fidelity --kv f16` at 399 positions, and f16 ids on prompts A/B plus a repetitive 64-token case.

## 6. Reopen trigger

- A quiet box (identical-arm matvec spread <10%) with at least six alternating pairs at 4096 and 8192; adopt only if attention is >=3% faster in every pair and total prefill >=1%, with ids unchanged.
- Do not integrate on this run: the effect is inside the measurement floor of the current box.
