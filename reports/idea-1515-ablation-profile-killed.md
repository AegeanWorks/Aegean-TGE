# IDEA-1515 STAGE-1: Per-Layer i8 Sensitivity Profile — KILLED (flat where it matters; joint effect irreducible)

- **Idea ID**: IDEA-1515 (dossier `scout-filing-2026-aegean-tge.md`: MultiAlloc-style budget-aware mixed-precision KV tier; falsifier: fidelity KL worse than flat i8 at matched bytes, or decode regresses vs `--kv i8`)
- **Cycle date**: 2026-11-06
- **Verdict**: KILLED stage-1 — the profile this idea's policy needs does not exist: every layer's marginal i8 error is ~zero (KL ≤ 3.1e-8, top-1 86/86 on all 24 layers) while the joint flat-i8 effect (KL 4.6e-2, 10 flips) is ~1500x the largest marginal. Nothing to promote, nothing (shippable) to demote: sub-8-bit tiers are dead per KIVI stage-2a (KL 0.13). Spike fully reverted; only this report committed.
- **Tree state at close**: `git status` clean (no diff), spike branch created and deleted with nothing committed, stash unchanged (pre-existing peer work only), `master` ahead of `origin/master` by 21 after this report. Nothing pushed (human gate).

## 1. Mechanism (stage-1 slice)

MultiAlloc ports need a per-layer sensitivity profile to split a fixed KV
budget. This stage measures it directly: an env-gated `kv_ablate_` mode that
quantizes exactly one layer's KV to i8 (native engine recipe, rest f32) and
prices it with the standard fidelity harness, plus flat f32/i8 references.
Kill bar (stated): flat profile — max layer KL ≤ 2x median — means no skewed
sensitivity to exploit. Proceed bar: ≥ 4x skew with exploitable absolutes.

## 2. Predicted gain — none (diagnostic)

Stage-1 buys the go/no-go for a policy stage-2 (static per-layer bit table +
mixed read/store + decode bench). A skewed profile would name promotable
layers (keep fp32/f16) and demotable ones; a flat one ends the idea.

## 3. Falsifier (from the dossier, staged)

Full falsifier needs matched-bytes policy evaluation (stage-2). Stage-1 gate:
kill if layer marginals are all ~zero relative to the joint effect (no
skew worth a policy), or if sub-8-bit demotion is the only lever (already
dead). Both trip: max marginal 3.07e-8 vs joint 4.581e-2.

## 4. File anchors (current `master`, untouched except by the reverted spike)

- `src/models/qwen2.cpp` — store sites (prefill/decode/verify `else` chains),
  read sites (`attention_head`, `attention_gqa` q8 branches), sparse gates,
  `Qwen2::load` alloc, `reset_cache`.
- Spike (reverted): `kv_ablate_` member + `kv_layer_is_q8(layer)` helper +
  `TGE_KV_ABLATE` env parse with range/single-seq fail-loud guards, q8-array
  allocation alongside the f32 mirror, 3 store-site + 2 read-site condition
  swaps, sparse-gate ablation exclusion. No CLI surface (env only).

## 5. What was done (temporary spike, fully reverted)

1. `git checkout -b spike-1515-ablate` (no commit ever made on it).
2. hpp (4 lines) + cpp (~65 lines, mostly the load/reset branches).
3. Build clean; `tge_tests` 141/141 (harness needs no new tests: fidelity IS
   the instrument; routing correctness is proven by the numbers differing
   per layer).
4. Flat f32 + flat i8 references, then all 24 layers × fidelity (86-pos
   corpus, ctx 512, 1 thread, `--gpu off`), plus out-of-range L=24 (clean
   fail-loud error, guard verified end-to-end).
5. Reverted both files, verified empty diff, deleted branch. Final gates on
   master: build clean, `ctest` pass, `aikit.py check` 2/4 (test + secrets
   pass; hygiene + guidelines fail pre-existing).

## 6. Measured numbers (Qwen2.5-0.5B-Q8_0, 86 teacher-forced positions)

Flat references: f32 KL 2.121e-11 (86/86); flat i8 KL **4.581e-2**, worst
7.645e-1, top-1 **76/86**.

| L | mean KL | top-1 | L | mean KL | top-1 |
|---|---|---|---|---|---|
| 0 | 2.75e-11 | 86/86 | 12 | 5.43e-09 | 86/86 |
| 1 | 1.89e-09 | 86/86 | 13 | 2.66e-09 | 86/86 |
| 2 | 3.03e-11 | 86/86 | 14 | 2.17e-11 | 86/86 |
| 3 | 2.19e-11 | 86/86 | 15 | 5.17e-11 | 86/86 |
| 4 | 3.07e-08 | 86/86 | 16 | 4.38e-09 | 86/86 |
| 5 | 2.86e-09 | 86/86 | 17 | 2.02e-11 | 86/86 |
| 6 | 1.32e-09 | 86/86 | 18 | 8.77e-11 | 86/86 |
| 7 | 2.79e-09 | 86/86 | 19 | 8.76e-10 | 86/86 |
| 8 | 3.16e-10 | 86/86 | 20 | 1.73e-10 | 86/86 |
| 9 | 1.98e-11 | 86/86 | 21 | 1.63e-09 | 86/86 |
| 10 | 2.69e-09 | 86/86 | 22 | 2.12e-08 | 86/86 |
| 11 | 1.12e-08 | 86/86 | 23 | 1.21e-09 | 86/86 |

Reading: top-1 perfect on all 24; every marginal KL ≤ 3.1e-8 — six orders of
magnitude below the flat-i8 joint effect. The max/median ratio (~19x) is
numerically above the naive 4x proceed bar, but it is a ratio of two zeros:
the largest marginal (3.07e-8) is 0.00007% of the joint error. Promoting L4
to fp32 would save 3e-8 of 4.6e-2 — unmeasurable. The i8 error is a JOINT
depth-compounding effect (sum of marginals ≈ 7e-8 vs joint 4.6e-2:
super-additive by ~600,000x), irreducible by per-layer bit allocation among
shipped tiers. L=24 correctly refused (`layer out of range`).

## 7. Full reimplementation recipe (kill-grade — rebuild only to re-verify)

1. hpp: `int kv_ablate_ = -1;` + `bool kv_layer_is_q8(int layer) const;`
   (private; true when `kv_q8_ || (kv_ablate_ >= 0 && layer == kv_ablate_)`).
2. cpp: load parses `TGE_KV_ABLATE` (range + single-seq fail-loud), allocates
   q8 arrays alongside the f32 mirror in ablation mode; `reset_cache` fills
   both; 3 store-site conditions → `kv_layer_is_q8(index)`; 2 read-site
   conditions → `kv_layer_is_q8(layer)`; sparse gates gain
   `&& kv_ablate_ < 0`.
3. Gate: flat f32/i8 references + 24 × ablated fidelity runs — expect the
   table in §6 (all 86/86, all KL ≤ 3.1e-8) and the L=24 refusal.

## 8. Verdict and why

**KILLED without a policy ever being designed.** A mixed-precision policy
needs skewed layer sensitivity; the measurement shows the opposite structure:
zero everywhere marginally, everything jointly. Neither lever moves it —
promotion saves 3e-8 (noise), demotion needs sub-8-bit tiers that KIVI
stage-2a killed at KL 0.13. The falsifier lesson is general: screen absolute
marginals before ratios — a 19x ratio of two zeros is not headroom. Do not
revisit without (a) a sub-8-bit tier that prices inside budget (reopens
demotion), or (b) evidence that longer contexts concentrate error in specific
layers (reopens promotion) — neither exists today.

## 9. Exact commands run and their results

- `git checkout -b spike-1515-ablate` → new branch.
- Probe (hpp + cpp, ~70 lines, no test-file changes needed).
- `cmake --build --preset release` → 54/54; `tge_tests` → 141/141.
- Flat f32 (KL 2.1e-11) + flat i8 (KL 4.6e-2, 76/86) references.
- 24 × `TGE_KV_ABLATE=L fidelity` → §6 table (all 86/86); L=24 → clean error.
- `git checkout -- <2 files>` → diff empty; `git checkout master`;
  `git branch -D spike-1515-ablate`.
- Final: build clean, `ctest` pass, `aikit.py check` 2/4 (pre-existing fails).
- No push.
