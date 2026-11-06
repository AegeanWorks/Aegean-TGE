# IDEA-1499: Static-Split Draft KV Cache with Periodic Refresh — KILLED WITHOUT SPIKE (RIFT twin of the dead 1526; no draft model, no 32K harness)

- **Idea ID**: IDEA-1499 (dossier `kilo-pass2-2026-11-03.md`: arXiv:2610.31098 RIFT static split ρ=0.5 — half sliding-window, half SnapKV-importance — with adaptive/periodic refresh every Δ=128 during decode; Qwen3.5-4B drafting for 122B retains >98% full-cache acceptance at 32K-in/4K-out on 8K budget; inference-only, no retraining)
- **Cycle date**: 2026-11-06
- **Verdict**: KILLED on the record, no spike built (dead-idea rule: same paper, same draft-model-KV mechanism as the killed IDEA-1526 — there is no draft model to split a budget across, no 32K/4K evaluation to run the falsifier on, and the measured long-output behavior runs opposite to the premise). No branch created, no code touched, nothing to revert.
- **Tree state at close**: `git status` shows my report commit plus PRE-EXISTING, UNCOMMITTED peer spike work (`run_spike_mprefix`, +134/+128, no active process, board silent) — untouched again. Branch never created. `master` ahead of `origin/master` by 44 after this report (reports only). Nothing pushed (human gate).

## 1. Mechanism (as dossiered)

Static 50/50 split of a draft model's KV budget (sliding window + SnapKV-importance halves) with periodic refresh every 128 decoded tokens, in the speculative draft path; acceptance length at ≥95% of full-KV on unseen items.

## 2. Predicted gain — and why every precondition fails here

- No draft model: the engine's drafters are KV-free token-id retrieval
  (SKIP-lock, prompt spans, bigram, gapped, all exact-match or count-based;
  verified no draft-KV state anywhere in the 1526 cycle). A budget split
  presumes a budget; there is none.
- No 32K/4K harness: the falsifier demands 32K-in/4K-out acceptance-length
  measurement at 8K budget; max meaningful contexts here are single-digit k
  with no long-output acceptance rig.
- No staleness to refresh away: measured long-output acceptance holds flat
  (3.45/step at 47 tokens → 3.48/step at 127, deterministic 2/2, 1526
  report). A refresh schedule for non-existent decay is a no-op by
  construction.

## 3. Falsifier (from the dossier) — unstageable on all arms

RIFT-stat vs SnapKV AL (4.07) / SWA (5.23) on 32K/4K at 8K budget, or refresh
overhead >30% of draft decode: no draft model, no 32K harness, no refresh
target. The static/dynamic distinction the idea offers over 1526 changes
nothing — both need the same missing carrier.

## 4. File anchors (current `master`, read but untouched)

- `reports/idea-1526-draft-kv-refresh-killed-analytic.md` — the twin kill
  (same paper, same missing carrier, measured no-decay).
- `apps/cli/main.cpp` (retrieval drafters — no KV), `src/models/qwen2.cpp`
  (only the target model's KV caches exist).

## 5. What was done (ANALYZE only — no spike per the dead-idea rule)

1. Read the dossier section; identified it as RIFT's static-split twin of
   the already-killed dynamic-refresh variant.
2. Confirmed no new carrier vs 1526 (same missing draft model, same missing
   long harness, same measured no-decay).
3. Gates for the record: build clean, `ctest` pass, `aikit.py check` 2/4
   (test + secrets pass; hygiene + guidelines fail pre-existing).
4. No thinking-set diff owed: same binary, zero behavior change.

## 6. Measured numbers (cited, not re-run — same tree)

Draft acceptance 3.45/step (47 tok) → 3.48/step (127 tok); i8 tier holds
99.3–100% where priced. Nothing in the static-split direction is unmeasured
that 1526 left open.

## 7. Reimplementation recipe (deliberately NOT recommended — kill-grade)

Would require inventing a draft model with KV, then splitting and refreshing
it — i.e. the architecture this tree deliberately avoids (retrieval drafts
exist precisely to carry zero draft state). Do not build. Revisit only with
a draft MODEL plus a measured long-output decay — neither exists.

## 8. Verdict and why

**KILLED without spike.** A twin of a dead idea with the same missing
carrier, the same contradicted premise, and a falsifier that needs a 32K
harness this box does not run. Filing separately from 1526 only because the
register lists it separately.

## 9. Exact commands run and their results

- Dossier read + carrier check (no new surface vs 1526).
- `cmake --build --preset release` → 54/54; `ctest` → pass.
- `aikit.py check` → 2/4 (pre-existing fails). No branch, no edits
  (peer spike untouched), no push.
