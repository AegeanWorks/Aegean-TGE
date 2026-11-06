# IDEA-1499: Static-Split Draft KV Cache with Periodic Refresh — KILLED WITHOUT SPIKE (RIFT twin of the dead 1526; no draft model, no 32K harness)

- **Idea ID**: IDEA-1499 (dossier `kilo-pass2-2026-11-03.md`: arXiv:2610.31098 RIFT — static split ρ=0.5 (half sliding-window, half SnapKV-importance) with adaptive/periodic refresh every Δ=128 during decode; Qwen3.5-4B drafting for 122B retains >98% full-cache acceptance at 32K-in/4K-out with 8K budget; inference-only, no retraining)
- **Cycle date**: 2026-11-06
- **Verdict**: KILLED on the record, no spike built (dead-idea rule: RIFT twin of the already-killed IDEA-1526 — same paper, same draft-model-KV mechanism, same missing carrier, same measured no-decay). No branch created, no code touched, nothing to revert.
- **Tree state at close**: `git status` clean apart from PRE-EXISTING, UNCOMMITTED peer spike work (verified UNCHANGED this cycle by normalized full-diff compare after a false alarm from a UTF-16 snapshot artifact — see §9; peer files untouched). `master` ahead of `origin/master` by 46 after this report (reports only). Nothing pushed (human gate).

## 1. Mechanism (as dossiered)

Split a draft model's KV budget statically (half sliding-window, half
SnapKV-importance) with adaptive/periodic refresh every 128 decoded tokens,
in the speculative draft path; acceptance length ≥95% of full-KV on unseen
items; no target retraining, no KD.

## 2. Predicted gain — and why every arm fails here

- No draft model to split a budget across (retrieval drafters are KV-free:
  SKIP-lock, prompt_lookup spans/bigrams/gapped, bigram stats — measured).
- No 32K/4K harness (max meaningful contexts single-digit k, no
  long-output acceptance rig beyond best-effort runs).
- No staleness to refresh away: measured long-output acceptance HOLDS
  (3.45/step at 47 tokens → 3.48/step at 127, deterministic 2/2, 1526
  report). A refresh schedule for non-existent decay is a no-op.
- Static vs dynamic split changes nothing about the above: both need the
  same missing carrier.

## 3. Falsifier (from the dossier) — unstageable on all arms

RIFT-stat vs SnapKV (4.07) / SWA at 32K/4K with 8K budget, or refresh
overhead >30% of draft decode: no draft KV, no 32K harness, nothing to
refresh. The falsifier needs an engine this tree is not.

## 4. File anchors (current `master`, read but untouched)

- `reports/idea-1526-draft-kv-refresh-killed-analytic.md` — the twin kill
  (same paper, same missing carrier, measured no-decay 3.45→3.48).
- `apps/cli/main.cpp` (retrieval drafters only — no draft-model KV);
  `src/models/qwen2.cpp` (only the target model's KV).

## 5. What was done (ANALYZE only — no spike per the dead-idea rule)

1. Read the dossier section; confirmed it is RIFT's static-split twin of
   the killed dynamic-refresh variant (identical preconditions).
2. Verified no new carrier vs 1526 (grep-clean draft-KV absence stands;
   no 32K harness appeared).
3. Gates for the record: build clean, `ctest` pass, `tge_tests` 141/141,
   `aikit.py check` 2/4 (test + secrets pass; hygiene + guidelines fail
   pre-existing).
4. No thinking-set diff owed: same binary, zero behavior change.

## 6. Measured numbers (cited, not re-run — same tree as the 1526 cycle)

Draft acceptance 3.45/step (47 tok) → 3.48/step (127 tok), deterministic;
i8 long-context holds 99.3% on Llama-3B. Nothing about a static split
changes these numbers — the mechanism they would feed on does not exist.

## 7. Reimplementation recipe (deliberately NOT recommended — kill-grade)

Would require inventing a draft model with KV, then splitting + refreshing
it on a 32K harness — i.e. precisely the unbuilt half of 1526. Do not build.
Revisit only with a draft model AND measured long-output decay; neither
exists.

## 8. Verdict and why

**KILLED without spike.** A twin of a dead idea with the same missing
carrier, the same contradicted decay premise, and a falsifier that needs a
32K engine. Filing separately from 1526 only because the register lists it
separately; analytically they are one kill.

## 9. Exact commands run and their results

- Dossier read + carrier re-check (zero draft-KV surface, as in 1526).
- `cmake --build --preset release` → 54/54; `ctest` → pass; 141/141.
- `aikit.py check` → 2/4 (pre-existing fails). No branch, no edits.
- Peer-dirt byte compare: initial raw compare FALSE-ALARMED (UTF-16 snapshot
  artifact doubled line counts: 621 vs 311); normalized full-diff compare =
  IDENTICAL (311 lines, 0 diff lines). Peer work untouched throughout.
- No push.
