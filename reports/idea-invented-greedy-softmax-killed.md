# INVENTED: Greedy (temp<=0) Softmax Skip — KILLED analytic (fast path already ships; residual copy-skip ceilings ~0.2%)

- **Idea**: Skip the full-vocabulary softmax on the greedy decode path (temp≤0): argmax is invariant under strictly-monotone softmax, so copy + penalty + linear scan suffices — no exp, no normalize, no divide.
- **Cycle date**: 2026-11-06
- **Verdict**: KILLED analytic — the hypothesized gap does not exist on today's tree. `sample_token` (`src/models/sampling.cpp:198-207`) already takes exactly the proposed fast path for temp≤0 (copy + penalty + scan, softmax only on the temp>0 arms). The only residual delta, skipping the scores copy when penalty is also off (argmax straight on logits, as the drafter d0 pre-check already does), moves ~1.2 MB of cache-hot traffic per token ≈ 25 µs against ~13 ms/token ≈ 0.2% — under the bar by 25x. No spike per the dead-idea rule; no branch, no edits, nothing to revert.
- **Tree state at close**: `git status` clean, no diff, stash unchanged (only the pre-existing `leftover peer work shelved 2026-11-04`). `master` ahead of `origin/master` by 95 after this report. Nothing pushed (human gate).

## 1. Mechanism (as hypothesized, not built)

Guard the greedy single-token draw: temp≤0 → copy logits to scores, apply repeat penalty, linear argmax scan; skip `softmax_probabilities` entirely. Bit-identical ids by monotonicity (softmax preserves order; argmax sees the same ranking with or without it).

## 2. Predicted gain — moot (already realized), residual ~0.2%

The full softmax skip the idea proposes is the shipped behavior — its gain is already in every greedy number on record. The only unshipped micro-delta (skip the copy too when `no_penalty`, argmax on logits directly): ~1.2 MB R+W of cache-hot traffic ≈ 25 µs/token vs ~13 ms/token decode ≈ 0.2% « 5% bar.

## 3. Falsifier outcome — fails on today's tree, kill

Stated falsifier: kill if the greedy path already skips softmax (no redundant work exists to remove). `sampling.cpp:198-207` does exactly the copy+penalty+scan; the drafter d0 pre-checks (`main.cpp:877`, `:888`, `:908`) likewise note "argmax needs no softmax (monotone)". The revival trigger that would reopen this: a regression that routes temp≤0 through `softmax_probabilities` — re-grep `sample_token` before any future sampler work (the 1568 triage watch covers it).

## 4. File anchors (current `master`, read but untouched)

- `src/models/sampling.cpp:183-207` (`sample_token`: copy + penalty always; temp≤0 returns the scan at :198–207; softmax only on temp>0 arms :222+, :247+).
- `apps/cli/main.cpp:678-684` (`draw` → `sample_token`), `:877/:888/:908` (drafter argmax pre-checks, softmax-free by comment).
- Grounds: the sampler program's closed reports (scans killed, timers fixed) — this was the last unexamined corner of the sampler cost map.

## 5. What was done (ANALYZE only — no spike per the dead-idea rule)

1. Re-read the hypothesized gap against the current tree: traced `draw()` → `sample_token` and found the fast path already shipped.
2. Checked the drafter pre-checks for the same pattern (present, commented).
3. Ceiled the residual copy-skip (~0.2%, §2).
4. Gates on the unmodified tree: `cmake --build --preset release` 54/54; `tge_tests.exe` 141/141; `aikit.py check` 2/4 (test + secrets pass; hygiene + guidelines fail pre-existing, same pair as every prior cycle).
5. No fidelity run owed (no arithmetic exists to price); no bench A/B owed (no second path exists to pair); no thinking-set diff owed (same binary, zero behavior change).

## 6. Measured numbers (ceilings — no new benchmark belongs to an analytic kill)

| quantity | value | source |
|---|---|---|
| greedy softmax cost on today's tree | 0 (fast path ships) | `sampling.cpp:198-207` |
| residual copy-skip ceiling | ~0.2% (≈25 µs vs ~13 ms/token) | 1.2 MB hot traffic vs 0.5B 10T decode rate |
| loop bar | 5% | unmet by 25x |

## 7. Full reimplementation recipe (deliberately NOT recommended — nothing to build)

There is no change to reimplement: the fast path ships. To re-verify the kill: re-read `sample_token`'s temp≤0 arm and confirm no `softmax_probabilities` call precedes the scan; run one greedy story generate and confirm the `sample` timer stays in the ~2% class (copy+penalty+scan cost, not exp cost). Revisit ONLY if the §3 trigger fires.

## 8. Verdict and why

**KILLED analytic: the idea is already the code.** The sampler cost map is now complete with no unexamined corners: scans killed, exp/totals/walks split, double-count fixed with a ship recipe, and the greedy arm proven softmax-free. Next-cycle candidate: the S2 pre-packed-B retry trigger (verify caller-less on today's tree → likely the same analytic outcome via 1325).

## 9. Exact commands run and their results

- Code re-read (`sample_token`, `draw`, drafter pre-checks) — gap absent, see §4.
- `cmake --build --preset release` → 54/54; `tge_tests.exe` → 141/141; `aikit.py check aegean-tge` → 2/4 (test + secrets pass; pre-existing hygiene + guidelines fails).
- No branch created, no edits made, no benchmarks beyond the gate suite, report committed, no push.

(End of file.)
