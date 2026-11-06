# INVENTED: Vectorized Activation Quantization — KILLED at the probe gate (conversion share ceilings at ~5%, realistic port ~3-4% < bar; no spike built)

- **Idea**: The kq8 integer path quantizes every activation row with a scalar loop (`quantize_row_q8k`: abs-max scan, mantissa-trick round, bsums) once per (projection, chunk). Vectorize it with AVX2 (max-ps scan, cvtps_epi32 pack, integer bsums) and collect the saving across every kq8 projection.
- **Cycle date**: 2026-11-06
- **Verdict**: KILLED analytic at the probe gate — no spike built, per the dead-idea rule. A temp in-tree microbench (times the REAL function, hardened against dead-code elimination with a full-checksum sink) measures 4.3-4.4 µs/row at 3072 cols and 11.6 µs/row at 8192. Ceiling math on the reference model (Llama-3B, 28 layers, 7 kq8 projections, 64-token chunks, pp512): conversion ≈ 540 ms of a ~10 s prefill ≈ 5.4% upper bound. A realistic 2-3x vectorization ships 2.7-3.6% — under the 5% bar. The bar applies to the shipped improvement, not the measured cost, and the shipped improvement cannot clear it.
- **Tree state at close**: `git status` clean, no diff, stash unchanged (only the pre-existing `leftover peer work shelved 2026-11-04`). Spike branch `spike-qquant-probe` created (zero commits) and deleted; all three probe files reverted with empty diff verified. `master` ahead of `origin/master` by 112 after this report. Nothing pushed (human gate).

## 1. Mechanism (as proposed, NOT built)

AVX2 `quantize_row_q8k`: horizontal-max scan, `cvtps_epi32` pack (replacing the mantissa trick), integer bsums; same floats out (bit-identical), same layout. Killed before implementation — the probe numbers below close the gate first.

## 2. Predicted gain — bounded below the bar before writing the kernel

Required: shipped ≥5%. Best case: conversion eliminated entirely (÷∞) = ~5.4% — at the bar, not above it, and unreachable (vectorization is 2-3x, not ∞). Realistic port: 2.7-3.6%. The falsifier fails on arithmetic, not just measurement.

## 3. Falsifier outcome — kill at the ANALYZE gate

Stated: PROCEED (build the vectorized spike) only if the conversion share ≥5% with a port ceiling above the bar. Measured share ceilings at ~5.4% (all-7-projections, every-chunk-full — the max), port ceiling ~3-4%. Kill: no vectorized spike, per the rule against building what the numbers already fail.

## 4. File anchors (current `master`, read but untouched except the reverted probe)

- `src/cpu/avx2/dequant_batch_kq8_avx2.cpp:62-101` (scalar quantizer — the cost being priced; comment already says "scalar, matching upstream cost").
- `src/cpu/avx2/dequant_batch_kq8_avx2.cpp:365-379` (per-token-row call loop — the ×tokens multiplier).
- Grounds: kq8 program (integer-dot wins, conversion unexamined until now), batchcvtph + kq8scratch kills (same conversion-cost family).

## 5. What was done (probe-gated kill — spike branch held only the instrument)

1. `git checkout -b spike-qquant-probe` (zero commits, deleted at close). Baseline: build 54/54, `tge_tests` 141/141.
2. ANALYZE: confirmed the per-(token,row) call structure (not once-per-call as first sketched — the loop at :375 makes conversion tokens× worse than the naive read, which is exactly why the probe was needed before designing).
3. Built a temp in-tree probe (`spike-qquant`; 200 rows × best-of-5): 3.9/10.4 µs weak-sink → **4.3-4.4/11.6 µs hardened** (the weak sink understated ~10%; the hardened full-checksum numbers are the ones that count — a metrology note for future probes: never sink two points of a mapped buffer).
4. Ceiling math (§2) → kill. No vectorized spike built.
5. `aikit.py check aegean-tge` → 2/4 (test + secrets pass; hygiene + guidelines fail pre-existing, same pair as every prior cycle).
6. Reverted all three files, `git checkout master`, branch deleted, rebuilt master 54/54, tests 141/141.
7. No fidelity run (nothing numeric built or changed — probe reads only); no thinking set (diagnostic instrument only, normsplit/attnsplit precedent); no scratch files (stdout only, not kept).

## 6. Measured numbers (temp `spike-qquant`, best-of-5, warm)

| shape | weak sink | hardened (full checksum) |
|---|---|---|
| cols=3072 (attn/gate/up rows) | 3.9 µs/row | **4.3–4.4 µs/row** |
| cols=8192 (ffn-down rows) | 10.4 µs/row | **11.6 µs/row** |

Ceiling: 28 layers × (6×64×4.35µs + 64×11.6µs) × 8 chunks ≈ 540 ms / ~10 s Llama pp512 ≈ **5.4% (max)**; realistic port (2-3x) ships **2.7-3.6%**. 0.5B scales the same way (~4%).

## 7. Full reimplementation recipe (do NOT build — rebuild the probe only to re-verify the kill)

To reproduce the negative result: re-apply the §5 probe, rerun, recompute §6. Expect same class (4-12 µs/row). Revival trigger: a prefill shape where activation quant demonstrably exceeds 10% of wall (then a 2x vectorization clears the bar with margin) — no such shape on record; the kq8 path's own matvec dominates everywhere measured.

## 8. Verdict and why

**KILLED at the probe gate: the conversion is real but sub-shippable.** This is the loop working as designed — a 20-minute probe answered what a week-long kernel port would have, and the answer (ceiling ~5%, realistic ~3%) fails the bar before any vector code exists. The kq8 conversion-cost family is now fully mapped (batchcvtph: decode ~0%; kq8scratch: allocator ~0%; this: quantizer ~3-4% ceiling). Next-cycle candidates: the 1517 stage-2 skip harness remains the only live structural front; the human-gated port shelf stands.

## 9. Exact commands run and their results

- `git checkout -b spike-qquant-probe` → branch (zero commits, deleted at close).
- Baseline: build 54/54; tests 141/141.
- Probe (3 files: TU bench fn + header decl + `spike-qquant` driver); rebuild → green; tests 141/141; `spike-qquant` ×2 → §6 table.
- `aikit.py check` → 2/4 (test + secrets pass; pre-existing fails).
- Revert 3 files → clean; `git checkout master`; branch deleted; rebuild 54/54; tests 141/141; report committed; no push.

(End of file.)
