# INVENTED: i8 KV Diverse-Text Size Ladder — CONDITIONAL keep, boundary above 1.5B (monotonic in size, format-independent)

- **Idea**: Invented follow-up to the diverse-corpus i8 verdict: is the degradation model-size or weight-format dependent? Ladder: 0.5B-Q8_0, 0.5B-Q4_K_M (format control), 1.5B-Q4_K_M, Llama-3B-Q4_K_M — same 431-position diverse corpus, same harness, `--kv i8`.
- **Cycle date**: 2026-11-06
- **Verdict**: CONDITIONAL KEEP, sharpened: i8 holds (≥98.6% top-1) on ≥3B-class models for diverse long-form; degrades below it (0.5B ~93%, 1.5B ~97%). The ladder is monotonic in SIZE and format-independent (0.5B fails equally in Q8_0 and Q4_K_M weights), so the driver is redundancy, not weight quant. Stated falsifier tripped as designed: 1.5B top-1 96.7% < 98.6% bar → boundary is above 1.5B.
- **Tree state at close**: `git status` clean apart from PRE-EXISTING, UNCOMMITTED peer spike work (verified byte-identical to the saved baseline before and after — untouched). No branch created, no code touched (existing binary only), nothing to revert. `master` ahead of `origin/master` by 45 after this report (reports only). Nothing pushed (human gate).

## 1. Mechanism

Same fidelity harness (full-scalar oracle vs production AVX2), fixed diverse
corpus (431 positions, ctx 512, 1 thread, `--gpu off`), vary model only.
Falsifier (stated): 1.5B top-1 < 98.6% or KL > 5e-3 → boundary above 1.5B;
1.5B clean (≥99%, KL ≤ 5e-4) → boundary at/below 1.5B.

## 2. Predicted gain — none (diagnostic)

A size ladder for the conditional keep-grade: where does the i8 tier hold on
diverse long-form, and is the driver size or weight format?

## 3. Falsifier (stated) — trips on the top-1 arm, cleanly

1.5B: 417/431 = 96.7% < 98.6% bar (14 flips); KL 4.09e-3 < 5e-3 bar (passes).
The bar splits — top-1 is the load-bearing metric for greedy serving, so the
trip stands: boundary above 1.5B.

## 4. File anchors (current `master`, untouched — no code this cycle)

- Corpus: temp `fidelity_diverse.txt` (10 mixed topics, ~12 KB, not committed).
- Harness: `apps/cli/main.cpp` (`run_fidelity`); KV: `store_kv_q8`.
- Prior rungs: `idea-invented-i8-longctx-confirmed.md` (repetitive),
  `idea-invented-diverse-i8-flagged.md` (0.5B-Q8_0 diverse, locator).

## 5. What was done (pure measurement — no spike, no branch)

1. Build + `ctest` green (gate).
2. f32 sanity + i8 verdicts already on record (0.5B-Q8_0, Llama-3B); this
   cycle adds: 1.5B-Q4_K_M f32 (KL 7.2e-12, 431/431, sanity) + i8, and the
   0.5B-Q4_K_M i8 format control.
3. `aikit.py check` 2/4 (test + secrets pass; hygiene + guidelines fail
   pre-existing).
4. No thinking-set diff owed: fidelity verdicts only (no generation changes).
   No fidelity-threshold re-litigation beyond the stated bar.

## 6. Measured numbers (diverse 431-pos corpus, ctx 512, `--kv i8` unless noted)

| model | KL mean | worst KL | top-1 | flips |
|---|---|---|---|---|
| 0.5B Q8_0 f32 (sanity) | 9.6e-12 | 1.4e-10 | 431/431 | 0 |
| 0.5B Q8_0 i8 | 2.224e-2 | 0.69 | 406/431 (94.2%) | 25 |
| 0.5B-Q4_K_M i8 | 2.585e-2 | 0.46 | 398/431 (92.3%) | 33 |
| 1.5B-Q4_K_M f32 (sanity) | 7.2e-12 | 1.1e-10 | 431/431 | 0 |
| 1.5B-Q4_K_M i8 | 4.090e-3 | 0.20 | 417/431 (96.7%) | 14 |
| Llama-3B-Q4_K_M i8 (prior) | 1.21e-4 | 1.0e-3 | 428/431 (99.3%) | 3 |

Monotonic in size (KL: 2.6e-2 → 4.1e-3 → 1.2e-4; flips 33/25 → 14 → 3);
format-independent at fixed size (0.5B Q8_0 vs Q4_K_M both fail ~equally).
Driver: redundancy, not weight quant.

## 7. Full reimplementation recipe (verification protocol)

1. Corpus as §4 (temp file, mixed topics, ~500 tokens).
2. `fidelity --model <0.5B-Q8_0|0.5B-Q4_K|1.5B-Q4_K|Llama-3B-Q4_K_M>
   --corpus <file> --context 512 --tokens 600 --threads 1 --gpu off [--kv i8]`;
   expect the §6 row for the matching cell within box noise (±20% KL).
3. Rule for tier choice: diverse long-form + <3B params → f32/f16 KV (or
   accept ~94–97% top-1); ≥3B → i8 stands.

## 8. Verdict and why

**CONDITIONAL KEEP, now with a size-calibrated condition.** The diverse-corpus
challenge resolves into a clean size ladder, not a tier failure: i8 is safe
where redundancy is ample (≥3B: 99.3%, KL 1e-4) and degrades where it is not
(sub-3B: 93–97%). The usage rule from the prior report sharpens accordingly
(model-size, not just content-type). Flag the flip-locator follow-up as
closed for i8 (mechanism understood: difficulty-driven error, redundancy-gated).

## 9. Exact commands run and their results

- Peer-baseline diff re-verified IDENTICAL before starting (untouched).
- `cmake --build --preset release` → 54/54; `ctest` → pass.
- 1.5B f32 diverse: KL 7.2e-12, 431/431 (222 s).
- 1.5B i8 diverse: KL 4.090e-3, worst 0.20, 417/431 (239 s).
- 0.5B-Q4_K_M i8 diverse: KL 2.585e-2, worst 0.46, 398/431.
- `aikit.py check` → 2/4 (pre-existing fails). No branch, no edits, no push.
