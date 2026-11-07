# IDEA-FUSION: Residual-Add + RMSNorm Fusion — KILLED analytic (mechanism caps at ~1%)

- **Idea**: Fuse the scalar residual-add loop (`x += residual`) into the following `rms_norm` pass in the batch/prefill path, eliminating one DRAM round-trip of the hidden vector per norm site. New invented direction (not in dossiers); checked against measurement-log first — no prior fusion work except single-token dot fusion (§28i, killed on no-win + alignment fault; different mechanism).
- **Cycle date**: 2026-11-07
- **Verdict**: KILLED without a spike — the mechanism cannot move the bottleneck. No code written, tree untouched.
- **Tree state at close**: clean, no diff, stash untouched. Nothing to revert. Nothing pushed.

## 1. Mechanism as designed (not built)

Fuse `for axis: x[axis] += residual[axis]` + `rms_norm(h, x, w)` into one pass (add while accumulating squares, then normalize+write), at both batch sites (`forward_batch` ~1648ff, `prefill_run` chunk loop ~2227ff). Bit-identical by construction (same op order per element). Predicted gain assumed memory-bound norms.

## 2. Falsifier — fails on arithmetic, no spike warranted

Phase breakdown on current tree (0.5B-Q8_0, 2048 ctx, 6 threads): prefill matvec 4379 ms (59%), attn 1392 (16%), **norm 1915 (22%)**. Elementwise traffic accounting per token per layer (dim 896, ffn 4864, 2 norms + rope + silu + adds ≈ 93 KB) totals ≈4.6 GB over the run — ≈77 ms at streaming bandwidth, yet the phase measures 1915 ms (25x). The phase is therefore compute-bound, not memory-bound: silu alone is 239M exp+div evaluations (2048×24×4864), and at ~25-40 cycles each for AVX2 exp + `_mm256_div_ps` that is ~6-10B cycles ≈ the measured 1.9 s. The exp is already an AVX2 approximation (`exp_avx2`, shared with softmax); the division is full-precision.

Fusion removes memory passes only. Saving one streaming pass per norm site (~700 MB total, ~12 ms best case) against a 1915 ms compute-bound phase caps the win at ~1% of prefill — below the 3% falsifier bar on any box, before noise. Per the loop rule (falsifier fails on today's tree → kill on the record, no spike for dead ideas), no spike was built.

## 3. What would actually move it (prescribed, not attempted)

The lever is exp/div cost, not passes: RCP+Newton for the division and/or a cheaper exp with a priced fidelity gate (current floor 9.6e-3? verify against f16 tier 1.7e-6 and VNNI 1.6e-3 before choosing the tier). Even a perfect 3x on exp+div is only ~10% of prefill — file under “measured small upside, priced numerics” and attempt only on a quiet box with best-of-5.

## 4. Exact commands run (all read-only; no code touched)

- Phase breakdowns: `tge generate --tokens "<2048 ids>"` on 0.5B-Q8_0 (norm 22% of prefill, stable across runs).
- `git grep` for residual-add + norm call sites (2 pairs: forward_batch ~1648/1657, prefill chunk ~2227/2237); `rms_norm`/`silu_gated` AVX2 sources read to confirm vectorized exp + full-precision div.
- No build, no tests affected, no fidelity owed (nothing changed); `git status` clean.

(End of file.)
