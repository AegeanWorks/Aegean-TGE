# IDEA-1452: No-Gradient Chunk-Indexed Sparse KV Retrofit (CISA) — SUPERSEDED BY 1341 (same paper, same C=16 mechanism, same analytic kill)

- **Idea ID**: IDEA-1452 (dossier `arxiv-sweep-aegean-tge.md`: arXiv:2611.03200 §§2–4 — no-gradient dense retrofit via fixed-budget chunk selection (C=16 chunk means, per-KV-group indexer, QShare): MRCR/LongBench-v2/GraphWalks within 1.5pp of dense with 8–16x fewer KV reads, 2.29x fixed-output throughput on 1M-token prompts)
- **Cycle date**: 2026-11-06
- **Verdict**: SUPERSEDED by IDEA-1341 — same source paper (2611.03200), same mechanism (C=16 chunk/page selection with per-group mean-key summaries, shared index query per KV group, adjacent-query selection reuse/QShare, decode reads selected chunks only), and 1341's analytic kill transfers in full with strictly worse constants here (heavier proxy than the antidiagonal family it was judged against). No independent spike (dead-idea rule: re-spiking 1452 would re-argue the identical arithmetic on the identical mechanism). No branch, no code touched, nothing to revert.
- **Tree state at close**: `git status` clean (no diff, no untracked), stash unchanged (only the pre-existing `leftover peer work shelved 2026-11-04`, untouched), `master` ahead of `origin/master` by 58 before this report. Nothing pushed (human gate).

## 1. Mechanism (as dossiered) — identical to 1341's

Prefill groups KV into fixed C=16 chunks summarized by mean keys; one index query per KV group shared across the group; adjacent prefill queries reuse selections (QShare); decode reads only selected chunks. No gradients, no full-KV scan. Must-be-true: mmap-first CPU path implements chunk-mean scoring + page-ID gather + QShare reuse with vectorised dots and prefetch as a new KV-read attention variant.

## 2. Predicted gain — already capped, against 1452

Paper-side 11–13x are core-attention microbenchmarks at 1M tokens. End-to-end on today's tree follows Amdahl on measured shares (2081-token Q8_0: prefill attn 17%, decode attn 20% — 1341 §6): ~11% cap before scoring, and scoring here is heavier than the antidiagonal proxy (mean-key construction per chunk + per-group index query + selection compaction). Since 1341's verdict, the 1451 spike measured the scoring-cost problem directly on this engine (norm pass + gather ≈ 2x attention for +3 flips at matched budget) — the chunk-mean proxy reads every key row the subset was meant to skip, same traffic arithmetic, larger constants. QShare reuses a selection that fails the KL bar only faster (1341 §3: shipped static sparse at ~63% density already prices KL 2.6e-2 with 48/1023 flips — a scored 30% subset skips more and holds less).

## 3. Falsifier (from the dossier) — resolved per arm without building

"CPU prototype shows the representative scan (N/C dots) dominates" — resolved by inheritance with reinforcement: 1341's share-cap argument plus 1451's measured 2x scoring cost — the N/C scan plus mean construction plus compaction cannot fit under the ~11% cap. "Realization needs vLLM/FlashAttention tile M=128 + B300 NVFP4 for any speedup" — zero carriers (no FlashAttention/vLLM-tile/B300 paths; compounds the 1519 MXFP4 kill). "Quality collapses on the CPU KV-f16/GQA path — read-reduction only with no capacity/mmap win (retains full KV, QShare prefill-only, ≥1M GraphWalks −4.62pp)" — the dossier's own §6 note concedes all three: full KV retained (no mmap win), QShare prefill-only (decode is the bound half at 20% attn), and the 1M-scale GQA-path gap. No 1M-context harness exists here (max timed history 2050 — 500x below the claim's regime), so the scale arm is untestable-here and recorded as such, not as a pass.

## 4. File anchors (current `master`, read but untouched)

- `src/` + `include/` grep for `chunk-mean|chunk_mean|page-ID|page_id|QShare|indexer|CISA` → zero matches (the 1451 scored proxy was fully reverted; no selection foothold remains).
- `reports/idea-1341-chunk-indexed-killed-analytic.md` §§2–3/6 (the transferred share-cap + KL-precedent kill), `reports/idea-1451-scored-sparse-killed.md` §6 (the measured scoring-cost reinforcement), `reports/idea-1514-antidiagonal-sparse-killed-analytic.md` (the family kill both rest on).
- Registry: 1452 status Superseded; 1341 status killed-analytic — the chain this report closes.

## 5. What was done (ANALYZE only — inheritance, not duplication)

1. Read the dossier body against the 1341 report mechanism-for-mechanism; confirmed same paper, same C=16 constants, same QShare, same decode-reads-selected shape.
2. Carrier greps (§4): CISA machinery zero in `src/` + `include/`.
3. Gates for the record on the unmodified tree: `cmake --build --preset release` clean (54/54); `ctest --preset release` pass; `tge_tests.exe` 141/141; `python aikit.py check aegean-tge` → secrets pass, hygiene + guidelines fail pre-existing, test gate flaked once at 2.8 s (the §56/63 race signature — fourth occurrence, no code behind it) with the immediate direct `scripts\test.cmd` rerun passing. ctest + 141/141 green in both attempts.
4. No fidelity run owed (no new arithmetic); no bench A/B owed (the identical arithmetic exists on record twice over); no thinking-set diff owed (same binary).

## 6. Measured numbers — none new (inherited twice over, not re-run)

Applicable numbers live in 1341 §6 (17%/20% shares, KL 2.597e-02 at ~63% density) and 1451 §6 (+64–114% scoring cost, 978 vs 975/1023). Re-running either under a third idea number would manufacture activity, not evidence. Gate record this cycle: build 54/54, ctest pass, 141/141, aikit 1 + 2 pre-existing fails + 1 flaked-then-green test gate (§5.3).

## 7. Full reimplementation recipe (deliberately NOT recommended — revival starts at 1341/1451 triggers, never here)

Do not revive 1452 independently: any revival IS the scored-sparsity revival (sub-linear scorer clearing both falsifier arms at once — 1451 §8 — plus a 1M-context harness for the scale arm, which does not exist here). To re-verify this supersede: confirm the registry chain (1452 → 1341), re-run the §4 greps (still zero), and confirm the 1341/1451 triggers remain uncleared.

## 8. Verdict and why

**SUPERSEDED BY 1341, kill inherited with reinforcement from 1451.** Two filings, one paper, one mechanism, two concordant kills (analytic share-cap + measured scoring cost). The loop's honest output is the link, so no future cycle re-argues chunk-indexed sparsity without clearing 1451 §8 first.

## 9. Exact commands run and their results

- Dossier read (§28–34) + 1341-report cross-check: same paper/mechanism/constants confirmed.
- Carrier greps (§4): CISA machinery zero (code).
- `cmake --build --preset release` → 54/54; `ctest --preset release` → pass; `tge_tests.exe` → 141/141.
- `python aikit.py check aegean-tge` → test gate flaked once (2.8 s race), direct `scripts\test.cmd` rerun → pass; secrets pass; hygiene + guidelines fail pre-existing. No branch, no edits, no bench, no fidelity, no push.

(End of file.)
