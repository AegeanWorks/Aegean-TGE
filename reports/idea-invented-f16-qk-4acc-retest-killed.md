# INVENTED retest: four-accumulator f16 QK dot — KILLED on a quiet box

- **Cycle**: 17
- **Date**: 2026-11-09
- **Family**: block-prefill attention kernel
- **Prior**: `reports/idea-invented-f16-qk-4acc-mixed.md` (MIXED; reopen trigger:
  quiet box, six alternating pairs at 4096 and 8192, adopt only if attention is
  >=3% faster in every pair and total prefill >=1%)
- **Verdict**: **KILL.** The reopen trigger was examined and failed. Not integrated.
- **Tree state**: spike and temporary bench reverted, branch
  `idea/f16-qk-4acc-retest` deleted, this report is the only change.

## Mechanism (retested)

In `attention_block_head_avx2`, the f16 QK pass widens one K row per key and dots
it against every query. Baseline uses two FMA accumulators over 16-value strides;
the spike uses four over 32, then 16/8/scalar remainders, and reduces with
`horizontal_sum((acc0+acc1) + (acc2+acc3))`. Only the reduction grouping changes;
the f32 path, the widened row buffer and the softmax/V passes are untouched.

## Why the retest was worth one cycle

The prior run was MIXED, not a reject: medians were 4-5% but the best 8192 cell
was only 2.2% and the box had ±20% drift. The box is now roughly twice as fast as
that session (0.5B decode 69 t/s vs 40 t/s earlier in the day), so the kernel floor
is lower and the prior "inside box noise" explanation could be checked directly.

## Falsifier

Adopt only if the f16 block kernel is **>=3% faster in every alternating pair** at
both 4096 and 8192, with the block-vs-head differential green (ids/fidelity
unchanged in the prior run). Anything less stays killed; a regression at 8192 is
disqualifying because long-context prefill is the only place the block path pays.

## Measured — quiet-box kernel, 4 alternating processes per arm per span

Each process is one `tge_tests` run with `TGE_ATTN_BENCH=<span>
TGE_ATTN_BENCH_KV=f16`; the temporary bench runs the full block head (QK + softmax
+ V) over 128 queries, head_dim 128, `start = span - 128`, best-of-12 within the
process. Best and median across four processes:

| span | baseline best | spike best | best ratio | baseline median | spike median | median ratio |
|---:|---:|---:|---:|---:|---:|---:|
| 4096 | 8.055 ms | 7.933 ms | **1.015x** | 8.546 ms | 8.146 ms | **1.049x** |
| 8192 | 16.253 ms | 16.166 ms | **1.005x** | 16.509 ms | 17.472 ms | **0.945x** |

Per-pair ratios (plain/spike in each alternating pair):

| span | pair 1 | pair 2 | pair 3 | pair 4 |
|---:|---:|---:|---:|---:|
| 4096 | 1.057x | 1.003x | 1.077x | 1.008x |
| 8192 | 0.980x | 1.026x | 0.905x | 0.930x |

Within-arm spread: 4096 base 6.9%, spike 4.2%; 8192 base 2.1%, spike **12.8%**
(16.166-18.233 ms). The 8192 spike is slower in two of four pairs, including the
slowest sample of the whole cell.

## Why it fails

- At 4096 the best gain is 1.5%, below the 3% bar, and only two of four pairs clear
  3%. The median 4.9% is the same "median good, best weak" pattern as the prior
  MIXED run, which is what the reopen trigger was written to reject.
- At 8192 the extra accumulators are not just neutral: the spike is slower in half
  the pairs (up to 10%) and its spread is 12.8%. The block kernel at long spans is
  bound by the per-key row walk and the softmax/V passes, not by QK FMA latency, so
  the four-accumulator loop buys nothing and adds register pressure.
- End-to-end was not spent: at 4081 tokens attention is ~22-30% of prefill, so a
  1.5% kernel best is <=0.5% total prefill, and the 8192 regression is the
  long-context risk. The kernel falsifier already fails.

## Correctness note

The spike measured **168/168** (167 + the env-gated bench test); the existing
`attention_block_matches_head_path` differential at 1e-4 passed with the reordered
sum. The change is a summation regrouping, and the prior run measured fidelity
**3.252e-07** mean KL, 399/399 top-1, unchanged. Fidelity was not rerun because the
performance falsifier failed first. After the revert the suite is back to 167/167
and `aikit.py check aegean-tge` is 4/4.

## Full reimplementation recipe (for the record)

1. `src/cpu/avx2/attention_avx2.cpp`, `attention_block_head_avx2`, the
   `if (args.half)` branch of the QK loop (around the `row_f32` dot):
   four `__m256` accumulators; 32-wide main loop (one FMA per 8-value chunk into
   acc0..acc3), then 16-wide into acc0/acc1, then 8-wide into acc0, then the scalar
   tail for `head_dim % 8`.
2. Reduce with `horizontal_sum(_mm256_add_ps(_mm256_add_ps(acc0, acc1),
   _mm256_add_ps(acc2, acc3)))`, then `*= args.scale`.
3. Temporary bench: an env-gated test in `tests/test_attention.cpp` reading
   `TGE_ATTN_BENCH` (span) and `TGE_ATTN_BENCH_KV=f16`, allocating 128 queries,
   `span` half K/V rows of head_dim 128, `start = span - 128`, one head, and timing
   12 calls of `attention_block_head_avx2`; print the best.
4. Build baseline and spike binaries by swapping the single attention file, run
   four alternating processes per arm per span, best-of-12 per process.
5. Remove the bench and the kernel change before the report commit.

## Disposition

Close the f16 QK 4-accumulator line. Do not reopen it without a CPU or compiler
change; on this AVX2 box the block path is not FMA-latency-bound. The block head's
cost is the key-row walk plus softmax/V, so the next attention idea should target
the row walk or the softmax/V share, not the QK reduction tree.

Nothing was pushed; the workspace remains 15+ reports and integrations ahead of
`origin/master` pending the human push gate.
