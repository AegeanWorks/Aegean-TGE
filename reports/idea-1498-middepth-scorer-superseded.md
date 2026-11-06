# IDEA-1498: Single Mid-Depth Scorer for Shared KV Eviction Sets — SUPERSEDED BY 1451 (same paper, same mechanism, same measured kill)

- **Idea ID**: IDEA-1498 (dossier `kilo-pass2-2026-11-03.md`: arXiv:2611.02230 §1 contribution #2 — one mid-depth layer (0.4–0.75 depth) scores every cache plus one shared token set for all heads, no extra forward pass; +18–43pp over per-head SnapKV at 5–15% budgets on 0.5B–14B; calibration prompts with known answers via survival model S^l; shared set transfers to 32K financial-filing extraction +8/12pp)
- **Cycle date**: 2026-11-06
- **Verdict**: SUPERSEDED by IDEA-1451 — same source paper, same mechanism (one calibrated mid-depth scorer + one shared token set), and 1451's measured speed-arm kill transfers in full: scored selection costs ~2x prefill attention for +3 flips at a matched kept budget (idea-1451 report §6: +64–114% attn paired 3/3, 978 vs 975/1023). No independent spike built (dead-idea rule: re-spiking 1498 would re-tread the identical measurement on the identical mechanism). No branch, no code touched, nothing to revert.
- **Tree state at close**: `git status` clean (no diff, no untracked), stash unchanged (only the pre-existing `leftover peer work shelved 2026-11-04`, untouched), `master` ahead of `origin/master` by 57 before this report. Nothing pushed (human gate).

## 1. Mechanism (as dossiered) — identical to 1451's

Replace per-layer own-scoring with one calibration-prompt-selected mid-depth scorer and one shared token set for all heads/caches, in `src/cpu/avx2/attention_avx2.cpp`, no extra forward pass. Must-be-true: scorer picked via calibration prompts (survival model S^l); shared set transfers to 32K financial extraction and behaves like RULER needles (not open-book QA) on coding long-context.

## 2. Predicted gain — already priced, against 1498

Paper-side +18–43pp over per-head SnapKV at 5–15% budgets. Against this engine the load-bearing sub-claim (scored middle beats blind stride at a matched budget) measured: +3 flips quality edge (978 vs 975/1023, KL −7.5%) at ~2x attention cost — the scoring pass re-reads every key row the subset was meant to skip (traffic arithmetic, 1451 §6). 1498's heavier machinery (calibration + cross-layer capture + same-order scoring traffic, minus cross-layer amortization) inherits that ~2x burden of proof and shows nothing to pay it with.

## 3. Falsifier (from the dossier) — resolved per arm without building

"Mid-depth pick at or below own-layer default on ≤14B at binding 5–15% budgets" — resolved by inheritance: the only scored-vs-static comparison this engine can run (1451's matched-budget proxy, a stronger per-position baseline than a unified set) wins 3 flips at 2x cost — the pick cannot clear a cost gate at any budget. "No transfer to coding long-context" / "LongBench open-book QA shows no beat (as paper admits)" / "32K financial-filing +8/12pp" — untestable here, recorded as such: no LongBench harness, no coding-long-context corpus, max measured history 2050 (32K extraction is 16x beyond anything timed); the paper's own open-book admission stands unrebutted. "Accuracy drops more than 4.5pt on far-context queries" (companion arm in the dossier) — no far-context harness exists to run it against.

## 4. File anchors (current `master`, read but untouched)

- `src/` + `include/` grep for `mid-depth|middepth|mid_depth|calibrat|survival|scorer` → zero matches (the 1451 proxy was fully reverted; no foothold for the machinery remains).
- Dossier target file `src/cpu/avx2/attention_avx2.cpp` — contains no scoring/selection hooks (attention selection lives in `src/models/qwen2.cpp` gather sites: static sparse only).
- `reports/idea-1451-scored-sparse-killed.md` §6 (the transferred measurement), §8 (the burden-of-proof trigger this report inherits).
- Registry: 1498 status Superseded; 1451 status killed-by-spike — the chain this report closes.

## 5. What was done (ANALYZE only — inheritance, not duplication)

1. Read the dossier body against the 1451 report line by line; confirmed same paper, same mechanism, same target file, compatible falsifiers (§§2–3).
2. Carrier greps (§4): mid-depth machinery zero in `src/` + `include/`.
3. Gates for the record on the unmodified tree: `cmake --build --preset release` clean (54/54); `ctest --preset release` pass; `tge_tests.exe` 141/141; `python aikit.py check aegean-tge` → secrets pass, hygiene + guidelines fail pre-existing, test gate flaked once at 2.5 s (the §56/63 link-lock/AV race signature — third occurrence, no code behind it) with the immediate direct `scripts\test.cmd` rerun rebuilding everything and passing. ctest + 141/141 green in both attempts.
4. No fidelity run owed (no new arithmetic); no bench A/B owed (the identical measurement exists on record); no thinking-set diff owed (same binary).

## 6. Measured numbers — none new (inherited, not repeated)

The applicable numbers live in `idea-1451-scored-sparse-killed.md` §6 and are not re-run: fidelity static 2.597e-02/975 vs scored 2.401e-02/978; prefill attn +64–114% scored (3/3 same sign); short-span ids identical all arms. Re-measuring them under a different idea number would manufacture activity, not evidence. Gate record this cycle: build 54/54, ctest pass, 141/141, aikit 1 + 2 pre-existing fails + 1 flaked-then-green test gate (§5.3).

## 7. Full reimplementation recipe (deliberately NOT recommended — revival starts at 1451's triggers, never here)

Do not revive 1498 independently: any revival IS the 1451 revival (sub-linear scorer with chunk-mean amortization or reused quantities, clearing both falsifier arms at once — 1451 §8). To re-verify this supersede: confirm the registry chain (1498 → 1451), re-run the §4 greps (still zero), and confirm no LongBench/coding-longctx/32K harness has appeared that could run 1498's untestable arms — until all three hold, there is nothing to build under this number.

## 8. Verdict and why

**SUPERSEDED BY 1451, kill inherited.** Two filings, one mechanism, one measurement. The paper-side extras (calibration prompts, survival model, 32K transfer) have no carrier and no harness here, and the paper concedes the open-book arm itself. The loop's honest output is the link, so no future cycle re-spikes scored selection without clearing 1451 §8 first.

## 9. Exact commands run and their results

- Dossier read (§95–101) + 1451-report cross-check: same paper/mechanism/target.
- Carrier greps (§4): mid-depth machinery zero (code).
- `cmake --build --preset release` → 54/54; `ctest --preset release` → pass; `tge_tests.exe` → 141/141.
- `python aikit.py check aegean-tge` → test gate flaked once (2.5 s race), direct `scripts\test.cmd` rerun → pass; secrets pass; hygiene + guidelines fail pre-existing. No branch, no edits, no bench, no fidelity, no push.

(End of file.)
