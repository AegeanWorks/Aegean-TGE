# INVENTED: Sampler Scan Vectorization — KILLED (scans are ~1% of sampling; the 5% bar is unreachable even at zero cost)

- **Idea**: Invented, from the §28t open item ("vectorizing the scans is open"): vectorize the scalar full-vocabulary scans in the sampler — the totals (reductions) in `categorical_sample`/`speculative_accept` and the serial acceptance/rejection walks — which the dossier note says cost ~4 full-vocab scans per accepted step on the spec-low-temp path where sampling is ~48% of the step.
- **Cycle date**: 2026-11-06
- **Verdict**: KILLED without an implementation spike (dead-idea rule: the falsifier's bar is arithmetically unreachable on measured numbers). A temporary-instrumentation round measured the actual split on the box: spec temp-0.1 repetitive (4 steps, 4.25/step, 20 out) — sampling 152.41 ms (79.8% of decode) splits into totals 1.49 ms + walks 0.01 ms + everything-else ~151 ms. The scans are 1.0% of sampling; even ZERO-COST scans (the theoretical ceiling, no vectorization can beat it) move end-to-end by 0.8% — 6x short of the stated 5% bar. Plain temp-0.1 confirms: scans 1.60 ms of 11.25 ms sampling (14%), which is 0.7% of decode. No implementation spike was built because no implementation can clear a bar the ceiling misses by 6x. Nothing to revert beyond the instrumentation itself (fully reverted, verified clean).
- **Tree state at close**: `git status` clean (no diff, no untracked), stash unchanged (only the pre-existing `leftover peer work shelved 2026-11-04`, untouched). Spike branch created (zero commits) and deleted; all spike files reverted with empty diff verified. `master` ahead of `origin/master` by 85 after this report. Nothing pushed (human gate).

## 1. Mechanism (as analyzed, not built)

Candidate: AVX2 horizontal-sum reductions for the totals + kept-as-is serial walks (prefix-dependent, early-exit on peaked temp-0.1 distributions). Killed before building: the instrumented split shows totals 1.49 ms and walks 0.01 ms per 20-token spec run — vectorizing totals to zero still leaves 151 ms unaccounted for, so the mechanism addresses ~1% of the problem it was proposed for.

## 2. Predicted gain — refuted by measurement, not by reasoning alone

Predicted by the dossier note ("single digits"): at most single-digit percent of sampling. Measured ceiling (zero-cost scans): 1.0% of sampling on spec, 14% of sampling but 0.7% of decode on plain. The 5% end-to-end bar from the falsifier is unreachable in both regimes — not close, not borderline.

## 3. Falsifier outcome — kill bar holds with wide margin

Kill if scans < 5% of sampling on the spec shape (equivalently: even free scans can't move end-to-end ≥5%): measured 1.0% — fails by 5x on the share and 6x on the end-to-end ceiling. Gate had a built-in escalation (implement + price if scans ≥5%): not triggered, correctly.

## 4. File anchors (master = pre-spike; spike reverted, nothing to show)

- `src/models/sampling.cpp` (`categorical_sample` totals + walk, `speculative_accept` totals + remainder/walk — the four instrumented loops, reverted).
- `include/tge/models/sampling.hpp` (`sampler_scan_ns` accessor — reverted).
- `apps/cli/main.cpp` (`scan :` print in the timers block — reverted).
- Grounds: doc §28t (the open item + its own single-digits warning), §1568 protocol (triage gate: speculative stays decode-side, which this round re-confirms — the gap is not a regression, just a small ceiling).

## 5. What was done (temporary instrumentation, fully reverted — no behavior change, by construction the ids/exp paths are untouched)

1. `git checkout -b spike-sampler-scans` (zero commits ever made on it; deleted at close).
2. Spike (3 files, counters only — no arithmetic touched): two thread-local accumulators + accessor + 4 timed regions + 1 print line. Deliberately NO behavior change, so no differential test was owed (nothing to be differentially different); the gates below are the check.
3. `cmake --build --preset release` clean (54/54); `ctest --preset release` pass; `tge_tests.exe` 141/141 (identical count — instrumentation adds no tests and changes no paths).
4. Box health: wall 59.20→54.45 bracketing; aikit 2/4 (test PASS, no flake this round; hygiene + guidelines fail pre-existing).
5. Gap reproduction + split (§6): plain temp-0.1 story + spec temp-0.1 repetitive, same seed 7.
6. Thinking set on the instrumented binary (counters don't touch numerics): story ids byte-identical to every baseline (`11 1052 …` family); creative text fluent.
7. Reverted all 3 files (`git checkout --`, empty diff verified), `git checkout master`, `git branch -D spike-sampler-scans`; final gates on reverted master: build clean, `ctest` pass, `tge_tests` 141/141; aikit 2/4 (test PASS).

## 6. Measured numbers (0.5B-Q8_0, fox ×7 prompt, 6 threads, gpu off, temp 0.1, seed 7)

| run | decode | forward / sample | scan totals | scan walks |
|---|---|---|---|---|
| plain, 20 tok | 230.09 ms (82.58 t/s) | 218.86 / 11.25 ms (4.9%) | 1.49 ms | 0.11 ms |
| spec, 4 steps / 17 acc / 20 out | 191.04 ms (99.45 t/s) | 183.01 / 152.41 ms (79.8%) | 1.49 ms | 0.01 ms |

Totals are identical across runs (same ~30 full-vocab reductions); walks vanish on temp-0.1 (peaked distributions exit after 1–2 tokens — correctly NOT a vectorization target). Residual sampling (~151 ms spec): vectorized exp softmaxes + accept/reject logic — a different program, explicitly not this cycle (named below).

## 7. Full reimplementation recipe (do NOT build this — the recipe is the negative result; build only the named follow-up)

Do not vectorize the scans: the ceiling math (§6: ≤0.8% end-to-end even free) forbids it. To re-verify the kill: re-apply §5.2, re-run the §6 pair, confirm scans ≤2% of sampling on spec. Revival trigger ONLY (named, not built): a sampler cost profile where scan-class work (totals/walks, not exp/eval) exceeds 5% of sampling end-to-end on a shipped path — never observed; the 1568 triage gate watches for it. If that trigger ever fires, the recipe is: AVX2 horizontal-sum totals (share the norm/attention hsum idiom), keep walks scalar, chi-square gate per §28t precedent, same-seed paired A/B.

## 8. Verdict and why

**KILLED on the falsifier's bar with 6x margin, by measurement rather than reasoning.** The open item from §28t is now closed data instead of open speculation: the scans were never the cost — totals are 1.49 ms of 152 ms sampling, walks are noise on peaked distributions, and the remaining ~151 ms lives in already-vectorized exps plus accept/reject logic. The loop's honest output is the split plus the ceiling proof; a vectorization spike would have spent a cycle to learn what two counters learned in one run.

## 9. Exact commands run and their results

- `git checkout -b spike-sampler-scans` → branch (zero commits, deleted at close).
- 3-file instrumentation (§5.2; no behavior change, no new tests).
- `cmake --build --preset release` → 54/54; `ctest --preset release` → pass; `tge_tests.exe` → 141/141 (unchanged count).
- Wall 59.20→54.45; `aikit.py check` → 2/4 (test PASS, no flake — the 2.9 s flake of the 1568 round did not recur; pre-existing hygiene + guidelines fails).
- Plain + spec pair (§6 table); thinking set (story ids identical; creative text fluent).
- `git checkout -- <3 files>` → clean; `git checkout master`; `git branch -D`; final master gates green; report committed; no push.

(End of file.)
