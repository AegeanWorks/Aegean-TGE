# INVENTED: Sampler Timer-Overlap Fix Validation — SHIP RECIPE (identity 225.88 vs 226.02, trajectories + ids byte-identical)

- **Idea**: Validate the one-line-class fix named by three sampler rounds: in `run_generate`'s sampled-spec path, the sample window [st0,st1] contains the per-draft verify forward that is also booked to `fwd_seconds`. Scope the window by subtracting the verify span so the printed `sample` figure is exclusive and `forward + sample ≈ decode wall` again.
- **Cycle date**: 2026-11-06
- **Verdict**: SHIP RECIPE — the fix is behavior-neutral (timers only) and the accounting identity closes on the fixed binary. Porting (3-line diff, §7) is a human-gated commit, not this cycle: zero spike code survives.
- **Tree state at close**: `git status` clean, no diff, stash unchanged (only the pre-existing `leftover peer work shelved 2026-11-04`). Spike branch `spike-timers-fix` created (zero commits) and deleted; the one file reverted with empty diff verified. `master` ahead of `origin/master` by 93 after this report. Nothing pushed (human gate).

## 1. Mechanism (as spiked — 3 lines in `apps/cli/main.cpp`, reverted)

1. Declare `double verify_span_seconds = 0.0;` before the sampled-spec accept block (fresh zero each decode step).
2. After the verify forward books `fwd_seconds`, record the same span into `verify_span_seconds`.
3. Book the sample window as `(st1 - st0) - verify_span_seconds` instead of `(st1 - st0)`.

Greedy paths untouched — re-verified on today's tree that their verify calls (the :785/:927 sites) sit outside every `samp_seconds` window (the only accumulations are the sampled block and the plain s0/s1 path).

## 2. Predicted gain — accounting, landed exactly

No t/s change (timers only). Deliverable: the printed `sample` figure drops from double-counted (~75% of decode on spec shapes) to exclusive (~4–6%), and `forward + sample` returns to within timer granularity of the wall.

## 3. Falsifier outcome — all arms pass, fix ships on the record

1. **Identity** (spike binary, 0.5B-Q8_0, spec temp-0.1 seed-7, fox, n=20, 6 threads): forward 216.59 + sample 9.29 = 225.88 vs wall 226.02 → gap **0.14 ms** (≤0.5 ms required). Sample share now 4.1%.
2. **Equivalence** (same prompt, master binary): trajectory 5 steps / 7 accepted (1.40/step) / 12 emitted on BOTH; ids byte-identical 20/20. Master prints the double-count live (forward 231.95 + sample 82.44 = 314.39 > wall) — the bug, reproduced side-by-side.
3. **Thinking set**: story + creative ids byte-identical spike-vs-master.
4. No test breakage: `tge_tests` 141/141 on spike and reverted master.

## 4. File anchors (master = pre-spike; spike reverted)

- `apps/cli/main.cpp` (`run_generate`: st0 :1041, verify v0..v1 :1070–1074, st1/samp booking :1115–1116; greedy verify sites :785/:927 outside all samp windows; timers print :1169; spec print :1188–1194).
- Grounds: `idea-invented-sampler-split-measured.md` (overlap proof), `idea-invented-sampler-timers-measured.md` (identity with instrument: 129.45 vs 129.57), `idea-invented-sampler-scans-killed.md` (scans kill stands).

## 5. What was done (temporary spike, fully reverted)

1. `git checkout -b spike-timers-fix` (zero commits, deleted at close). Baseline on branch: build 54/54, `tge_tests` 141/141.
2. Re-read the window/verify/greedy sites on the current tree (ANALYZE — overlap confirmed byte-for-byte with the recorded analysis).
3. Applied the 3-line fix (§1). Rebuilt 54/54, `tge_tests` 141/141.
4. Spec gate run (spike): §6 table + full stdout saved; identity 0.14 ms.
5. Thinking set on spike (story + creative, `--print-ids`, saved); re-ran all three prompts on reverted master: trajectory + ids identical.
6. `aikit.py check aegean-tge` → 2/4 (test PASS; hygiene + guidelines fail pre-existing, same pair as every prior cycle).
7. Reverted the file (`git checkout --`, empty diff), `git checkout master`, branch deleted, rebuilt master 54/54, `tge_tests` 141/141, `git status` clean, stash untouched, scratch outputs deleted.

## 6. Measured numbers (Qwen2.5-0.5B-Q8_0, `--gpu off`, 6 threads, warm page cache)

| run (fox, spec, temp-0.1, seed 7, n=20) | forward | sample | fwd+samp vs wall | spec trajectory | ids |
|---|---|---|---|---|---|
| spike (fixed) | 216.59 ms | 9.29 ms (4.1%) | 225.88 vs 226.02 (Δ 0.14) | 5 steps / 7 acc / 12 out | 20/20 |
| master (buggy) | 231.95 ms | 82.44 ms (34.2%) | 314.39 > wall (double-count live) | identical | identical |

No fidelity run owed: zero arithmetic change (two doubles and a subtraction in the CLI timer path; the engine untouched). No bench A/B owed: no throughput delta exists to pair (timer-only by construction, confirmed by identical trajectories).

## 7. Full reimplementation recipe (keep-grade fix — safe to ship as a human-gated commit)

1. `apps/cli/main.cpp`, sampled-spec block: add `double verify_span_seconds = 0.0;` before the accept `if`; after the verify `fwd_seconds += ...` line add the same-span assignment; change the `samp_seconds +=` line to subtract it (§1 verbatim — 3 lines, no other file).
2. Do NOT touch the greedy paths (verified clean) or the timers print (its semantics become correct automatically).
3. Gate the port: build 54/54, `tge_tests` 141/141, spec gate run reproduces §6 within run noise (identity ≤0.5 ms, trajectory 5/7/12, ids match master exactly), story + creative ids byte-identical vs pre-port binary, `aikit.py check`.
4. After the port lands, re-price the §28t/§1568 sampler-share claims against corrected timers (ledger correction, not code — expected true share ~4–6% on spec shapes, ~2% plain).

## 8. Verdict and why

**SHIP RECIPE: the double-counting bug the sampler program diagnosed is fixed by 3 timer-only lines with the identity closed (0.14 ms) and behavior proven byte-identical.** This retires the last named sampler follow-up that required code: the program is now complete end to end (scans killed, overlap explained, identity closed with instrument, fix validated, re-pricing scoped as ledger work). Next-cycle candidates: the live fronts are the 1517 epic's stage-2 skip harness and the human-gated ports (mprefix-table, sparse no-gather, batchsnap kind tags, this fix).

## 9. Exact commands run and their results

- `git checkout -b spike-timers-fix` → branch (zero commits, deleted at close).
- Baseline: `cmake --build --preset release` → 54/54; `tge_tests.exe` → 141/141.
- 3-line fix (§1); rebuild → 54/54; `tge_tests` → 141/141.
- `generate --model <0.5B-Q8_0> --threads 6 --gpu off --speculative --temp 0.1 --seed 7 --prompt <fox> --n 20 --print-ids` (spike) → §6 row (identity 0.14 ms).
- Thinking set (spike, story + creative, saved); same three prompts on reverted master → trajectory + all ids identical.
- `python aikit.py check aegean-tge` → 2/4 (test PASS; pre-existing hygiene + guidelines fails).
- `git checkout -- apps/cli/main.cpp` → clean; `git checkout master`; `git branch -D spike-timers-fix`; rebuild 54/54; tests 141/141; scratch deleted; report committed; no push.

(End of file.)
