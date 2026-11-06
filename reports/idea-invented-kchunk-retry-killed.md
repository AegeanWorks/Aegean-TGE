# INVENTED (trigger §49): K-Chunk Dot Reorder Retry — KILLED analytic (primary best-of-5 already fails the falsifier on clean data; code absent)

- **Idea**: The standing §49 retry trigger: rebuild the reverted K-chunk dot reorder prototype (`dequant_batch_kchunk_avx2.cpp`, TU deleted) and re-run it on a quiet box with the same interleaved pairs, since the round's re-runs collapsed with the box (shipped re-run −25%/2x vs itself; x1 q6k 4.4x artifact on a verbatim-identical loop).
- **Cycle date**: 2026-11-06
- **Verdict**: KILLED analytic — the retry is unwarranted, not merely unattempted. The PRIMARY best-of-5 ran pre-collapse behind a 51.25 GB/s wall gate on clean data, and it already fails §49's own falsifier (a kc default ≥15% on BOTH shapes): kc1024 wins down (+19%) but collapses gate (−87%); kc512 is +5%/+4%; kc128 harms gate (−19%). The collapse muddied only the re-runs, and the 4.4x artifact cuts both ways — a round unstable enough to show 4.4x on identical code cannot lend its favorable outliers to a rebuild case. Rebuilding a deleted TU to re-fail a failed falsifier is not a cycle. No spike per the dead-idea rule; no branch, no edits, nothing to revert.
- **Tree state at close**: `git status` clean, no diff, stash unchanged (only the pre-existing `leftover peer work shelved 2026-11-04`). `master` ahead of `origin/master` by 100 after this report. Nothing pushed (human gate).

## 1. Mechanism (as built in §49, now absent)

Dequantize once per row into worker scratch (unchanged), consume the row in K-chunks (per-token stack accumulators, one write to `out`; no output re-touch, no `thread_local` in the pool lambda, `tokens==1` verbatim shipped loop), behind `bench batch --kernel kchunk --kchunk-kc N` (default 256). Reorder-only → KL ~0 by construction (differential `dequant_batch_kchunk_matches_scalar_reference` was green first build).

## 2. Predicted gain — already measured, already short

§49 primary (same-binary interleave, best-of-5, wall-gated): no kc clears 15% on both shapes. Best per-shape numbers belong to different kc values in opposite directions — a tuning fork, not a setting.

## 3. Falsifier outcome — kill (primary fails it; collapse is not the binding constraint)

Stated: kill the retry if the clean-data primary already fails "kc default ≥15% on both shapes". It does (§2). The standing trigger assumed the collapse hid a win; the numbers show the win was never there to hide — the collapse only destroyed the re-runs. Carrier check today: `kchunk` zero matches in code (TU + flag + test all reverted); a retry = rebuild-from-description, and the description's own data says don't. Revival trigger ONLY: a from-scratch derivation (not a re-run) showing why one kc must dominate both shapes — i.e., new analysis, not a quieter box. My kq8hoist cycle independently closes the neighboring reorder family at ~0% on this tree.

## 4. File anchors (current `master`, read but untouched)

- `docs/measurement-log.md:2946-2993` (§49: mechanism, differential, primary table, collapse, trigger).
- `src/cpu/avx2/dequant_batch_kq8_avx2.cpp` (the shipped loop kchunk would re-tile — untouched).
- Grounds: `idea-1569-inside-gemm-killed-analytic.md` (tuning-fork precedent: per-shape winners, no global setting), kq8hoist kill (reorder family ~0%).

## 5. What was done (ANALYZE only — no spike per the dead-idea rule)

1. Re-read §49 against today's tree: confirmed the TU/flag/test are fully absent (carrier grep: `kchunk` zero code matches); confirmed the primary ran wall-gated pre-collapse.
2. Recomputed the falsifier against the primary table (§2) — fails on clean data.
3. Gates on the unmodified tree: `cmake --build --preset release` 54/54; `tge_tests.exe` 141/141; `aikit.py check` 2/4 (test + secrets pass; hygiene + guidelines fail pre-existing, same pair as every prior cycle).
4. No fidelity run owed (no numerics exist to price — nothing built); no bench A/B owed (rebuilding a bench-only prototype to re-measure it is the thing being killed); no thinking-set diff owed (same binary, zero behavior change).

## 6. Measured numbers (cited — no new benchmark belongs to an analytic kill)

§49 primary best-of-5, same-binary interleave (ms):

| kernel | q6k down 3072×8192×64 | q4k gate 8192×3072×64 |
|---|---|---|
| shipped | 30.739 | 18.487 |
| kc128 | 28.421 (+8%) | 22.021 (−19%) |
| kc256 | 33.763 | 19.035 |
| kc512 | 29.390 | 17.597 (+5%) |
| kc1024 | 24.850 (+19%) | 34.530 (−87%) |
| kc2048 | 23.559 | 39.475 |

No row clears +15% on both columns. Collapse artifacts (shipped re −25%/2x; 4.4x identical-loop) postdate this table and affect only its re-runs.

## 7. Full reimplementation recipe (deliberately NOT recommended — rebuild only to re-verify the kill)

To reproduce the negative result (do not ship): re-apply §49's construction from the log description (new TU, `--kernel kchunk --kchunk-kc`, differential-first vs scalar, same-binary interleave best-of-5 on the two shapes with a wall gate). Expected: the same tuning fork (per-shape winners, no global kc), i.e. another kill. Revisit ONLY on the §3 trigger (a derivation, not a quieter box).

## 8. Verdict and why

**KILLED analytic: the §49 retry trigger is retired.** The collapse narrative gave the idea more credit than its clean data earns: the primary already fails the falsifier, the artifact record shows instability in both directions, and the code is gone. With S2 (preb) and now K-chunk both retired on the record, the bench-kernel retry shelf is empty — every remaining trigger either lives in the 1517 epic or behind the human port gate. Next-cycle candidates: the 1517 stage-2 skip harness remains the only live structural front.

## 9. Exact commands run and their results

- Carrier grep (`kchunk`, `kchunk-kc`, `kc128`) — zero code matches (log-only); §49 re-read (§2 table re-verified against the falsifier).
- `cmake --build --preset release` → 54/54; `tge_tests.exe` → 141/141; `aikit.py check aegean-tge` → 2/4 (test + secrets pass; pre-existing hygiene + guidelines fails).
- No branch created, no edits made, no benchmarks beyond the gate suite, report committed, no push.

(End of file.)
