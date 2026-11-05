# IDEA-1341: Chunk-Indexed KV Selection with QShare Prefill Reuse — KILLED WITHOUT SPIKE (scored-sparsity family dead; QShare dies with selection)

- **Idea ID**: IDEA-1341 (dossier `paper-2611-03200.md` §2: C=16 chunk/page selection with per-group mean-key summaries, one shared index query per KV group, adjacent-prefill-query selection reuse/QShare; 11.7x prefill / 13.0x decode core-attention speedups at 1M tokens; dossier's own gate cut off at "Gate on end-to-end tokens" — stated operationally below)
- **Cycle date**: 2026-11-06
- **Verdict**: KILLED on the record, no spike built (dead-idea rule: same falsifier arms already tripped for the scored-sparsity family in the 1514 cycle; QShare presumes the selection it reuses). No branch created, no code touched, nothing to revert.
- **Tree state at close**: `git status` clean (no diff), stash unchanged (pre-existing peer work only), `master` ahead of `origin/master` by 28 after this report. Nothing pushed (human gate).

## 1. Mechanism (as dossiered)

Prefill groups KV into fixed C=16 chunks summarized by mean keys; one index
query per KV group is shared across the group; adjacent prefill queries reuse
selections (QShare); decode reads only selected chunks. Claimed value:
prefill-side win plus decode memory-traffic reduction, no retraining.

## 2. Predicted gain — and why it does not transfer (same arithmetic as 1514)

The paper's 11–13x are core-attention microbenchmarks at 1M context.
End-to-end on today's tree follows Amdahl on the measured shares (2081-token
Q8_0 run, same binary family, 1514 cycle): prefill attn 17%, decode attn 20%.
Optimistic model (keep 30%, pay ~15% scoring): end-to-end ≈ 89–90% ⇒ ~11%
cap before scoring eats in — and scoring here is HEAVIER than antidiagonal
sums (mean-key construction per chunk + per-group index query + selection
compaction). The dossier itself names both risks ("scoring/selection overhead
can erase the win end-to-end"; "QShare reuse assumes adjacent-query locality
that greedy tge decode may not exhibit"; "mean-key summaries may not transfer"
with 2pp+ RULER gaps disqualifying).

## 3. Falsifier (stated operationally; dossier gate truncated)

Kill if (a) attention share caps the end-to-end win below 5% after scoring
cost at measurable contexts (trips: 17–20% share → ~11% cap pre-scoring, and
the scoring here exceeds the antidiagonal proxy cost), or (b) ≤30%-density
content selection cannot hold KL ≤ 1e-9 on natural text (trips: shipped static
sparse at ~63% density already prices KL 2.6e-2 with 48/1023 flips — a
scored 30% subset skips more and holds less). QShare falls with selection:
reusing a selection that fails the KL bar only reuses the failure faster.

## 4. File anchors (current `master`, read but untouched)

- `src/cpu/avx2/attention_avx2.cpp`, `src/models/qwen2.cpp` (`Qwen2::attention*`
  group/head paths) — where scoring + skip would land; no selection machinery
  exists (content-blind static subset only, `--sparse`, default off).
- `reports/idea-1514-antidiagonal-sparse-killed-analytic.md` — the family kill
  with today's-tree phase measurements this verdict rests on.
- `docs/measurement-log.md` §50 — static-sparse fidelity precedent (KL
  2.597e-02 at ~63% density).

## 5. What was done (ANALYZE only — no spike per the dead-idea rule)

1. Re-read the dossier mechanism + its own listed weaknesses against the
   current tree; confirmed this is the scored-sparsity family with a heavier
   proxy (mean keys + index queries) and a reuse policy on top.
2. Compared against the 1514 verdict: every number it needs (attention share,
   KL precedent, scoring-cost lower bound) already measured against it, with
   strictly worse constants here.
3. Gates for the record: build clean, `ctest` pass, `aikit.py check` 2/4
   (test + secrets pass; hygiene + guidelines fail pre-existing).
4. No thinking-set diff owed: same binary, zero behavior change.

## 6. Measured numbers (cited, not re-run — same tree as the 1514 cycle)

2081-token Q8_0, CPU-only, 10 threads: prefill attn 1546.1/8906.75 ms (17%),
decode attn 44.9/233.47 ms (20%). Static-sparse fidelity at ~63% density:
KL mean 2.597e-02, top-1 975/1023. Paper's 13.5x-class microbenchmarks do not
transfer to 2k end-to-end (GQA 14/2, head_dim 128, matvec-bound) — same
conclusion, heavier proxy, plus a reuse policy with no verified locality.

## 7. Reimplementation recipe (deliberately NOT recommended — kill-grade)

For the record: per-chunk mean-key summaries beside the KV store, per-group
index-query scoring, top-k chunk compaction into the existing sparse-gather
path, adjacent-prefill-query selection cache. Expected outcome per §2–§3: KL
arm fails first (worse than antidiagonal: mean keys discard more signal),
then the beat-auto arm on Amdahl headroom. Do not build.

## 8. Verdict and why

**KILLED without spike.** The dossier's own weaknesses section already
contains this verdict: scoring overhead erases the win end-to-end, mean-key
summaries risk >2pp quality gaps, and QShare's locality is unverified on
greedy decode. Today's tree quantifies all three (17–20% caps, 2.6e-2 at
higher density, no locality evidence). Revisit only with (a) measured
contexts where attention dominates matvec (tens of k, not 2k), or (b) a
renegotiated fidelity bar for sparse attention — neither holds on this box.

## 9. Exact commands run and their results

- `cmake --build --preset release` → 54/54; `ctest` → pass.
- `aikit.py check` → 2/4 (pre-existing fails). No branch, no edits, no push.
