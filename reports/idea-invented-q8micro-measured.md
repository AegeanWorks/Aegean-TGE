# INVENTED (i8-geometry follow-up): q8 Dequant Microbench — MEASURED (1.6-2.2x compute tax, smooth across head_dim; arithmetic branch closed, structural branch narrowed)

- **Idea**: The geometry puzzle needs a mechanism. First hypothesis: the dequant tax varies wildly with head_dim (a pathological shape would explain one model losing while others win). Microbench the exact q8 idioms (widen-then-FMA dots, widen-mul-fmadd V update) vs the f32 counterparts per head_dim (64/96/128) on synthetic L1-hot data, best-of-3.
- **Cycle date**: 2026-11-06
- **Verdict**: MEASURED. Dequant compute is 1.6x (dots, dim 64) to 2.2x (dots, dim 128), V-update 1.4-1.7x — smooth and monotonic, NO pathology at any tested dim. That closes the arithmetic branch: no head_dim makes dequant disproportionately expensive, so the cross-model variance is NOT in the per-element arithmetic. Combined with end-to-end i8 wins on per-head (which require traffic ≥ ~50% of attention), the differentiator must be loop structure/parallelism (grouped online-softmax overhead, group-size dependent) — narrowed, not solved.
- **Tree state at close**: `git status` clean, no diff, stash unchanged (only the pre-existing `leftover peer work shelved 2026-11-04`). Spike branch `spike-q8micro` created (zero commits) and deleted; all three spike files reverted with empty diff verified. `master` ahead of `origin/master` by 116 after this report. Nothing pushed (human gate).

## 1. Mechanism (as built, reverted)

Temp `spike-q8micro` subcommand + 4 exported probers in `attention_avx2.cpp` (declared in `kernels.hpp`): `dot_avx2` vs `q8_dot_avx2` (widen-then-FMA) and f32 vs q8 V-update loops (mirroring the kernel idioms verbatim), 200k iters each, best-of-3, ns/row.

## 2. Predicted gain — diagnostic, no gain claimed

Falsifier: EITHER a head_dim with a discontinuous tax jump (mechanism FOUND — proceed to a targeted fix) OR smooth scaling everywhere (mechanism is NOT arithmetic — close that branch). Measured smooth → close it.

## 3. Falsifier outcome — arithmetic branch closed by measurement

| head_dim | dot f32 | dot q8 | ratio | vsum f32 | vsum q8 | ratio |
|---|---|---|---|---|---|---|
| 64 (0.5B) | 1.9 ns | 3.5 ns | 1.84x | 1.6 ns | 2.2 ns | 1.38x |
| 96 (Phi) | 2.3 ns | 4.9 ns | 2.13x | 1.9 ns | 3.1 ns | 1.63x |
| 128 (Llama) | 2.9 ns | 6.4 ns | 2.21x | 2.5 ns | 4.2 ns | 1.68x |

Monotonic, smooth, no cliff — a 96-wide path costs exactly what linear scaling predicts. The puzzle's signs do not live here.

## 4. File anchors (master = pre-spike; spike reverted)

- `src/cpu/avx2/attention_avx2.cpp` (`widen_i8`, `q8_dot_avx2`, the grouped q8 V arms — the idioms probed).
- `include/tge/cpu/kernels.hpp` (temp decl site).
- Grounds: the geometry round-robin (signs to explain), int8qk kill (dequant family), attnsplit (sub-phase shares).

## 5. What was done (temporary spike, fully reverted)

1. `git checkout -b spike-q8micro` (zero commits, deleted at close). Baseline: build 54/54, `tge_tests` 141/141.
2. ANALYZE: kernel idioms isolated (no grouped-path rewrite needed for the probe — the question is per-element cost, which the idioms answer directly).
3. Built the probe (3 files); rebuilt, tests green; ran best-of-3 per dim (§3 table).
4. `aikit.py check` 2/4 (test + secrets pass; pre-existing hygiene + guidelines fail).
5. Reverted all three files, master rebuilt 54/54, tests 141/141. No thinking set / fidelity (diagnostic instrument on synthetic data; no model numerics touched).

## 6. Measured numbers — see §3 table. Derived constraint (not a measurement): end-to-end per-head i8 wins (+13-24%) REQUIRE traffic ≥ ~50% of f32 attention (else the 1.6-2.2x compute tax would dominate) — consistent with effective (non-roof) DRAM bandwidth, and a checkable prediction for any future traffic measurement.

## 7. Full reimplementation recipe (diagnostic — rebuild only to re-verify)

Re-add the §1 probers; expect the §3 ratios (smooth, monotonic, no jumps). Revival trigger for the ARITHMETIC branch: a discontinuous jump at some width (would name the exact shape to avoid). The remaining STRUCTURAL branch needs grouped-path instrumentation (online-softmax overhead per group size) — named follow-up, not this cycle.

## 8. Verdict and why

**MEASURED: dequant arithmetic is smooth — the puzzle is structural, not arithmetic.** This closes one of the two candidate explanations with numbers and sharpens the other into a concrete next probe (grouped online-softmax overhead vs group size). Combined with the interaction cycle (per-head i8 wins everywhere), the actionable guidance stands unchanged while the mechanism hunt proceeds one branch at a time. Next-cycle candidates: the grouped-path structural probe; otherwise the 1517 stage-2 harness remains the only live structural front, and the human-gated port shelf stands.

## 9. Exact commands run and their results

- `git checkout -b spike-q8micro` → branch (zero commits, deleted at close).
- Baseline: build → 54/54; `tge_tests` → 141/141.
- Spike (§1); rebuild → 54/54; `tge_tests` → 141/141; `spike-q8micro` → §3 table.
- `aikit.py check` → 2/4 (test + secrets pass; pre-existing fails).
- Revert 3 files → clean; `git checkout master`; branch deleted; rebuild 54/54; tests 141/141; report committed; no push.

(End of file.)
