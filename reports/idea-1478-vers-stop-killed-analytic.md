# IDEA-1478: Doubt-Marker-Gated Early Termination (VERS, Harness-Only) — KILLED WITHOUT SPIKE (accuracy-gated falsifier, zero accuracy carriers)

- **Idea ID**: IDEA-1478 (dossier `arxiv-sweep-2026-11-01-03.md`: arXiv:2611.03040 §§3.2–3.4/4–6 — one fixed inference-time direction at paragraph breaks saves many reasoning tokens at small cost on easy tasks, zero deploy-time weight changes; explicitly NOT the weight/activation machinery — harness-side only: paragraph-boundary stop policy with doubt-marker gating)
- **Cycle date**: 2026-11-06
- **Verdict**: KILLED on the record, no spike built (dead-idea rule: every falsifier arm is accuracy-gated and the tree has zero accuracy carriers — no task eval, no ground-truth sets, no grading; the one arm a spike could run (detection overhead vs tokens saved) is meaningless without the accuracy it must not lose). No branch, no code touched, nothing to revert.
- **Tree state at close**: `git status` clean (no diff, no untracked), stash unchanged (only the pre-existing `leftover peer work shelved 2026-11-04`, untouched), `master` ahead of `origin/master` by 62 before this report. Nothing pushed (human gate).

## 1. Mechanism (as dossiered)

Harness-side stop policy: at paragraph boundaries, gate early termination on doubt markers — stop the reasoning trace where the model signals doubt, saving tokens on easy pinned sets. No weight/activation changes (the per-family training variant is explicitly excluded); the CLI/eval side owns the policy.

## 2. Predicted gain — and why it cannot be priced here

Paper-side: large token savings at small accuracy cost on easy math. Against this tree the price cannot be rung up: token savings without accuracy are trivially manufacturable (stopping early always saves tokens; the question is whether answers stay right), and the accuracy half has no instrument — `tge fidelity` prices numerics vs a scalar oracle (KL), not task correctness; `generate` prints text/ids with nothing to grade them against; `tests/` holds unit tests only (the sole accuracy-adjacent string is a sampling-NaN regression comment). Building the policy without the grader would produce a token-saved number that answers none of the falsifier's arms.

## 3. Falsifier (from the dossier) — unconstructible, all three arms

"Accuracy drops beyond the 2pp gate on the pinned set" — no pinned set exists (no ground-truth QA/math files in repo, no eval command). "Measured t/s unchanged/worse (detection overhead eats savings on CPU)" — the only runnable arm, and it proves nothing alone: overhead-vs-savings without accuracy is a speedometer without a destination. "Hard-problem slice repeats the reported several-point regression" — no hard-problem slice exists either; the paper's own transfer-with-limits warning (§§5–6) stands unrebutted and untestable.

## 4. File anchors (current `master`, read but untouched)

- `tests/` grep for `accuracy|ground-truth|eval_set|benchmark-task|grade|score_answer` → zero (one sampling-NaN comment only).
- `src/` grep for `doubt|early-termin|stop_policy|stop-sequence|paragraph` → zero (EOS handling lives in the sampler/tokenizer path; no policy hooks).
- CLI surface: `generate|fidelity|bench|plan` — grading appears in none; fidelity's KL is a numerics price, not a task score.
- Dossier's own exclusion (§236–237: "explicitly NOT the weight/activation machinery... Harness-side only") — the harness it assumes does not exist here.

## 5. What was done (ANALYZE only — no spike per the dead-idea rule)

1. Read the dossier body against the current tree; confirmed harness-only scope with an accuracy-gated falsifier.
2. Carrier greps (§4): accuracy eval zero; stop-policy machinery zero.
3. Gates for the record on the unmodified tree: `cmake --build --preset release` clean (54/54); `ctest --preset release` pass; `tge_tests.exe` 141/141; `python aikit.py check aegean-tge` → secrets pass, hygiene + guidelines fail pre-existing, test gate flaked once at ~2.3 s (the §56/63 race signature — seventh occurrence, no code behind it) with the immediate direct `scripts\test.cmd` rerun rebuilding everything and passing. ctest + 141/141 green in both attempts.
4. No fidelity run owed (no numerics changed); no bench A/B owed (no policy to pair — and a t/s-only pairing would answer nothing); no thinking-set diff owed (same binary).

## 6. Measured numbers — none owed (gate without an instrument)

No benchmark belongs to this kill: the falsifier demands accuracy deltas and the tree cannot produce one. Gate record: build 54/54, ctest pass, 141/141, aikit 1 + 2 pre-existing fails + 1 flaked-then-green test gate (§5.3).

## 7. Full reimplementation recipe (deliberately NOT recommended — revival is a harness project, never a policy tweak)

Do not implement the stop policy first: without a grader it can only demonstrate token deletion, not savings. Revive only as a harness program, in order: (1) a pinned ground-truth set (easy math with answers + a hard slice, versioned outside the repo weights); (2) an eval runner (generate at fixed seed/threads + grader + tokens + t/s ledger); (3) only then the paragraph-boundary doubt policy, gated on the dossier's own falsifier (≤2pp accuracy, t/s up net of detection, no hard-slice regression). To re-verify this kill: re-run the §4 greps (still zero) and confirm no eval runner with ground truth has appeared.

## 8. Verdict and why

**KILLED without spike.** An accuracy-gated policy idea with no accuracy instrument, whose only runnable arm (overhead timing) cannot distinguish savings from deletion. The paper's transfer limits stand as the last word until a grader exists to challenge them.

## 9. Exact commands run and their results

- Dossier read + carrier greps (§4: accuracy eval zero; stop-policy zero).
- `cmake --build --preset release` → 54/54; `ctest --preset release` → pass; `tge_tests.exe` → 141/141.
- `python aikit.py check aegean-tge` → test gate flaked once (~2.3 s race), direct `scripts\test.cmd` rerun → pass; secrets pass; hygiene + guidelines fail pre-existing. No branch, no edits, no bench, no fidelity, no push.

(End of file.)
