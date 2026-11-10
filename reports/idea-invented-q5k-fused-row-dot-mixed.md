# INVENTED: fused Q5_K row dot — MIXED (kernel 1.649x, end-to-end flat on the only Q5_K model)

- **Cycle**: 21
- **Date**: 2026-11-09
- **Family**: quantized decode matvec / K-quant
- **Verdict**: **MIXED.** The kernel win is real and bit-identical, but it does not
  translate on the only Q5_K model available. Not integrated.
- **Tree after the cycle**: spike and temporary bench reverted, branch
  `idea/q5k-fused-dot` deleted, this report is the only change

## Mechanism

Same fused-register treatment as the Q4_K/Q6_K keeps. Q5_K decode dequantizes a
whole row into a thread-local float scratch and then `dot_f32` reads it back.
The spike adds `fma32_q5k` (same nibble+high-bit decode and
multiply-then-subtract values as `dequantize_group_q5_k`, FMA'd into the
accumulator that `dot_f32` would use at those lanes) and `dot_q5k_row_avx2`.
The result is **bit-identical**: the same 32-value group order, the same
mul-then-sub, the same four-accumulator reduction. Prefill (`tokens > 1`) keeps
the dequant-once-and-reuse row path.

## Kernel measurement

In-process, alternating old/new, 6 threads, Q5_K 8192 x 8192 (46 MB), single
token, best of 40 alternating rounds:

| base `gemv_dequant_batch_avx2` | fused `gemv_q5k_fused_avx2` | ratio |
|---:|---:|---:|
| 2661.7 us | 1614.3 us | **1.649x** |

The differential test (`dequant_q5k_fused_row_matches_batch`) compares exact
`==` against the batch kernel; fidelity on the real model is identical (below).

## End to end — 0.5B Q5_K_M, the only Q5_K model available

Greedy, story prompt, n=48-64, `--kv f32 --gpu off`, alternating pairs:

| threads | base best t/s | fused best t/s | best gain | pair ratios |
|---:|---:|---:|---:|---|
| 6 | 63.42 | 63.68 | 1.004x | 0.979, 1.004, 0.951, 1.011 |
| 2 | 25.99 | 25.19 | 0.969x | 0.694, 0.995 |
| 1 | 13.76 | 13.66 | 0.993x | 0.894, 0.993 |

Ids are identical. This is flat, not a win. At 6 threads the 0.49 GB model runs
at ~31 GB/s, which is the DRAM roof; at 1 thread it is at the single-core roof.
The fused kernel removes instructions, but the decode is waiting on memory, so
nothing changes. The Q4_K/Q6_K keeps showed end-to-end gains on the 3B because
that file decodes at ~13 GB/s, below the roof; no Q5_K file of that size exists
here.

## Fidelity

`tge fidelity --model 0.5B-Q5_K_M --corpus docs/architecture.md --context 512
--tokens 200 --threads 1 --gpu off` (199 positions), base and fused:

| binary | mean KL | worst KL | top-1 |
|---|---:|---:|---:|
| master | 1.034e-11 | 1.068e-10 | 199/199 |
| fused | 1.034e-11 | 1.068e-10 | 199/199 |

Identical, as the bit-identity argument requires.

## Why MIXED, not KEEP

The kernel is exact and strictly less work, but the loop's keep bar is an
end-to-end gain; the only Q5_K model is bandwidth-saturated and shows none at
6, 2 or 1 threads. This is the f16 QK 4-accumulator precedent: a real kernel
effect that does not clear the end-to-end bar stays out of the tree.

Reopen trigger: a Q5_K model >=1.5B, or a loaded window where achieved decode
bandwidth is below the DRAM roof (like the 3B Q4_K_M at ~13 GB/s), or a
per-shape sweep at the model's actual small matrices if a larger file appears.
The recipe is ready and safe to retest; it is bit-identical and decode-only.

## Tests

169/169 with the spike (existing suite + exact differential + env-gated bench);
167/167 after the revert; `aikit.py check aegean-tge` 4/4.

## Anchors (spike, reverted)

- `src/cpu/avx2/dequant_avx2.cpp` — `fma32_q5k`, `dot_q5k_row_avx2`.
- `src/cpu/avx2/dequant_batch_avx2.cpp` — `gemv_q5k_fused_avx2`.
- `src/models/qwen2.cpp:990` — Q5_K single-token dispatch after the kq8d branch.
- `include/tge/cpu/kernels.hpp:188` — declarations.
- `tests/test_dequant.cpp` — differential test and bench.

## Full reimplementation recipe

1. Add `fma32_q5k(quants, high, nibble_shift, bit, scale, minimum, input,
   acc0..acc3)`: copy `dequantize_group_q5_k`'s nibble extraction and high-bit
   `cmpeq`/mask/or, compute `q * scale - minimum` (mul then sub), then FMA each
   8-lane group into acc0..acc3.
2. Add `dot_q5k_row_avx2`: per 176-byte block read d/dmin, scales (6-bit unpack),
   high at +16, low at +48; per 64-value chunk call the group at `in+chunk`
   (nibble shift 0, high bit `bit`) and `in+chunk+32` (shift 4, bit `bit+1`);
   advance low by 32, scales by 2, bit by 2; reduce exactly as `dot_f32`.
3. Add `gemv_q5k_fused_avx2` with the band skeleton; require
   `kind == Q5_K && tokens == 1`, else fall back.
4. Dispatch Q5_K in `Qwen2::matvec` when `!scalar_kernels_ && avx2`.
5. Tests: exact equality against the batch kernel; env-gated bench.

Pitfall: the high-bit row is stationary across the four chunks; only the bit
index advances. Keep the dequant multiply-then-subtract and the four-accumulator
order.

## Disposition

MIXED. Keep the recipe for a larger Q5_K file; nothing was pushed. 28 commits
remain ahead of `origin/master`.
