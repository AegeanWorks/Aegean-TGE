# IDEA-1497: Folded-Bias Compacted KV Cache — KILLED WITHOUT SPIKE (no far-context carrier; both readings land on dead precedent)

- **Idea ID**: IDEA-1497 (dossier `kilo-pass2-2026-11-03.md`: arXiv:2610.36520 §6 — foldable attention bias β∈[-3,3] as k' = k + β√d·u, closed-form u per head fitted on context queries, own query path on unmodified kernels, error bounded 3/2·D_V·|q·u−1|; falsifier: KL worse than dropping-β baseline on 12/12 settings, greedy match <180/200 LongBench-style, or >4.5pt accuracy drop on far-context queries)
- **Cycle date**: 2026-11-06
- **Verdict**: KILLED on the record, no spike built (dead-idea rule: every falsifier arm needs far-context/LongBench eval that does not exist here, and both plausible readings of the mechanism land on already-measured dead precedent). No branch created, no code touched, nothing to revert.
- **Tree state at close**: `git status` clean (no diff), stash unchanged (pre-existing peer work only), `master` ahead of `origin/master` by 35 after this report. Nothing pushed (human gate).

## 1. Mechanism (as dossiered, both readings)

Fold a fitted per-head bias into cached keys (k' = k + β√d·u, u closed-form
from context queries) with a compensating query-path term, keeping standard
kernels and cache geometry. Reading A: the fold absorbs outlier directions so
K quantizes better (KV-quantization aid). Reading B: content-aware attention
surgery whose value appears at far context (>4K past context, LongBench).

## 2. Predicted gain — and why neither reading survives contact

- Carrier: all three falsifier arms demand far-context/LongBench-style eval
  (12 settings, 180/200 greedy matches, >4.5pt far-context deltas). No such
  harness, corpus, or calibration-prompt set exists in-tree; max measured
  contexts are single-digit k on short corpora. Small-context proxying would
  not falsify fairly (the claim is intrinsically far-context), so no honest
  spike fits one cycle.
- Reading A dies with the sub-8-bit tiers it would aid: int2 KL 0.13
  (`1fc2869`), int4 KL 4.6e-3 (`6b0c84b`) vs the 1e-3 bar; symmetric variants
  worse (RHT probe, `0b4097e`). A bias fold cannot resurrect formats that miss
  by 4–128x on tensor error the fold itself would have to remove first.
- Reading B dies with the content-aware attention family: 17–20% attention
  share caps end-to-end wins ~11% pre-scoring (`idea-1514`, `idea-1341`),
  and any skipped-or-shifted content must clear KL 1e-9 against the static
  subset's 2.6e-2 precedent.

## 3. Falsifier (from the dossier) — unstageable here, which is the kill

All three arms (12-setting KL, 180/200 LongBench greedy, far-context
accuracy) require eval machinery that does not exist on this box. An idea
whose every falsifier arm is unstageable cannot be measured into the
library; recording the carrier gap is the honest output.

## 4. File anchors (current `master`, read but untouched)

- `src/models/qwen2.cpp` + `src/cpu/avx2/attention_avx2.cpp` (where a fold
  would land; untouched).
- `reports/idea-1516-kivi-stage2a-killed.md`, `idea-1385-int4kv-killed.md`,
  `idea-1330-rht-probe-killed.md` (KV-quant precedent killing reading A).
- `reports/idea-1514-antidiagonal-sparse-killed-analytic.md`,
  `idea-1341-chunk-indexed-killed-analytic.md` (scored-family precedent
  killing reading B).

## 5. What was done (ANALYZE only — no spike per the dead-idea rule)

1. Re-read the dossier against the current tree under both readings.
2. Verified the carrier gap (no far-context/LongBench eval surface).
3. Gates for the record: build clean, `ctest` pass, `aikit.py check` 2/4
   (test + secrets pass; hygiene + guidelines fail pre-existing).
4. No thinking-set diff owed: same binary, zero behavior change.

## 6. Measured numbers (inventory — the experiment is unstageable)

| check | result |
|---|---|
| far-context/LongBench eval surface | none (max measured: single-digit k, short corpora) |
| sub-8-bit tiers the fold would aid | dead (KL 0.13 / 4.6e-3 vs 1e-3 bar) |
| content-aware attention headroom | capped ~11% pre-scoring, KL-barred |

## 7. Reimplementation recipe (deliberately NOT recommended — kill-grade)

For the record: per-head u fitting on context queries + folded-K store path
+ q·u-compensated query path + a LongBench-style far-context harness with
the dossier's three gates. Do not build any of it: the harness is the tall
pole, and both mechanism readings are already answered against. Revisit only
with (a) a far-context eval harness on this box AND (b) a live sub-8-bit tier
(for reading A) or renegotiated fidelity bar (for reading B).

## 8. Verdict and why

**KILLED without spike.** An intrinsically far-context idea with no
far-context carrier, whose two plausible mechanisms each independently fail
on measured precedent. A small-context proxy spike would be dishonest
(pass proves nothing about >4K; the dossier's own gates say so).

## 9. Exact commands run and their results

- `cmake --build --preset release` → 54/54; `ctest` → pass.
- `aikit.py check` → 2/4 (pre-existing fails). No branch, no edits, no push.
