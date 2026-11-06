# INVENTED: Sampler Timer Overlap Fix (Measurement) — MEASURED (identity holds to 0.1 ms; true sampling 6.1% of spec decode, not 73.6%)

- **Idea**: Invented follow-up, queued by the split round (which left ~92 ms of "sampling" unattributed): add a dedicated verify-in-sample counter at the one verify call inside the sampled-spec sample window, and report sample-exclusive time, testing the double-counting hypothesis directly.
- **Cycle date**: 2026-11-06
- **Verdict**: MEASURED — hypothesis confirmed exactly. Spec temp-0.1 repetitive (4 steps, 17 accepted, 20 out): forward 121.60 ms + sample 99.46 ms reported, but verify-in-sample 87.51 ms sits in both; sample-exclusive = 7.85 ms (6.1% of decode), and forward + sample-exclusive = 129.45 ms vs 129.57 ms wall — the accounting identity holds to 0.1 ms. The "sampling 79.8% / 73.6% of spec decode" headline that three cycles of reports quoted was a timer artifact all along; true sampler work is exp ~5.6-6.0 + totals ~1.5-1.7 + walks ~0 + residual logic ~0.3 ms. No behavior was changed (timers only); nothing to keep or kill beyond the measurement itself. Named follow-ups (not this cycle): (1) apply the one-line window scoping permanently (human-gated commit, not a spike); (2) re-price the §28t/§1568 sampler-share claims against corrected timers.
- **Tree state at close**: `git status` clean (no diff, no untracked), stash unchanged (only the pre-existing `leftover peer work shelved 2026-11-04`, untouched). Spike branch created (zero commits) and deleted; all 3 spike files reverted with empty diff verified (plus a grep confirming zero `SPIKE`/`verify_in_sample` remnants). `master` ahead of `origin/master` by 87 after this report. Nothing pushed (human gate).

## 1. Mechanism (as spiked)

One `double verify_in_sample_seconds` accumulator beside `fwd_seconds`/`samp_seconds` in `run_generate`; +1 line at the single in-window verify call (line ~1082); one extra print line showing verify-in-sample and sample-exclusive with its decode share. Greedy-spec verify sites (lines ~785, ~927) verified OUTSIDE any samp window (their paths never touch `samp_seconds` — checked by grep), so only the sampled-spec call needed the counter. Deliberately minimal: no timer refactor, no behavior change.

## 2. Predicted gain — diagnostic, and it landed exactly

Predicted: forward + sample_excl ≈ decode wall (accounting identity restored), and the true sampling share lands near the instrumented sum (~7 ms). Measured: 121.60 + 7.85 = 129.45 vs 129.57 wall — 0.12 ms apart (timer granularity + print rounding). The identity holds, which retires the double-counting hypothesis as the explanation AND as a question.

## 3. Falsifier outcome — both arms pass as stated

(1) Identity: 129.45 vs 129.57 (≤0.5 ms gap required; actual gap 0.12 ms). (2) Re-priced shares: spec sampling 6.1% of decode (not 73.6%), plain greedy sampling 1.9% (verify counter reads 0.00, as constructed — no verify calls on that path). The 5%-bar conversation from the scans round is now moot in the informative direction: true sampling is 6% on spec shapes, of which ~76% is already-vectorized exp.

## 4. File anchors (master = pre-spike; spike reverted)

- `apps/cli/main.cpp` (counter decl, single accumulate site, print block — all reverted; grep-verified clean).
- `src/models/sampling.cpp`, `include/tge/models/sampling.hpp` (untouched this cycle — the scan timers live here only in the reverted prior spike).
- Grounds: sampler-split (exp 5.56–5.99 + totals 1.49–1.67 + walks 0.01 vs 99.46 reported), sampler-scans kill (bar math unchanged — the true 6.1% only strengthens it).

## 5. What was done (temporary instrumentation, fully reverted)

1. `git checkout -b spike-sampler-timers` (zero commits ever made on it; deleted at close).
2. Spike (1 file, 3 edits): counter decl, one accumulate line at the unique in-window verify call, one print block. Verified the greedy verify sites sit outside all samp windows (grep for `samp_seconds +=` shows only the sampled block and the plain path).
3. `cmake --build --preset release` clean (54/54); `ctest --preset release` pass; `tge_tests.exe` 141/141 (unchanged count — timers only, no behavior).
4. Box health: not separately re-gated (same window class as the split round; box variance note stands).
5. Spec temp-0.1, seed 7, fox ×7, 20 tokens: §6 table. aikit 2/4 (test PASS; hygiene + guidelines fail pre-existing).
6. Thinking set on instrumented binary (timers don't touch numerics): story ids byte-identical to baseline; creative text fluent.
7. Reverted the 1 file (`git checkout --`, empty diff + `grep -R SPIKE` clean); `git checkout master`; `git branch -D`; final gates on reverted master: build clean, `ctest` pass, `tge_tests` 141/141.

## 6. Measured numbers (0.5B-Q8_0, fox ×7, temp 0.1, seed 7, 6 threads, gpu off)

| run | decode | forward | sample (reported) | verify-in-sample | sample-excl | identity check |
|---|---|---|---|---|---|---|
| spec, 4 steps / 17 acc / 20 out | 129.57 ms (146.6 tok/s) | 121.60 ms | 99.46 ms (74.6%) | 87.51 ms | 7.85 ms (6.1%) | 121.60+7.85=129.45 ≈ 129.57 ✓ |
| plain greedy, story | 127.25 ms | 124.49 ms | 2.41 ms (1.9%) | 0.00 ms | 2.41 ms (1.9%) | 124.49+2.41=126.90 ≈ 127.25 ✓ |

Identity holds both directions (spec: 0.12 ms gap; greedy: 0.35 ms gap — both inside timer granularity + phase overlap). True sampling decomposition (spec): exp ~5.6–6.0 + totals ~1.5–1.7 + walks ~0.01 + residual accept logic ~0.3 ms.

## 7. Full reimplementation recipe (diagnostic instrument + one-line fix recipe — see §8 for which to ship)

Rebuild the diagnostic: §5.2 verbatim; gate: build 54/54, 141/141 tests; spec run reproduces §6 within run noise (verify-in-sample ≈ reported-minus-true gap; identity within 0.5 ms). To FIX (not this cycle — needs human review): narrow the sample window so verify forwards are excluded — i.e., split `samp_seconds += (st1 - st0)` into pre-verify and post-verify segments, or equivalently report `sample_excl = samp - verify_in_sample` in place of the current figure; then re-run the §6 pair and require forward + sample_excl ≤ decode wall. Do NOT also "fix" the greedy path (verified clean). Do not touch numerics.

## 8. Verdict and why

**MEASURED: the accounting identity closes the double-counting question (129.45 vs 129.57), the re-priced sampling share is 6.1% on spec shapes (1.9% plain), and the sampler program is now complete end to end.** Three cycles, one arc: scans are ~1% of reported sampling and untouchable (kill); reported sampling itself was ~13x overstated by double-counted verify forwards (this round); true sampling is three-quarters vectorized exp with a small scan tail. No future sampler work is justified on these numbers without a new workload where the fixed timer says otherwise — which the §1568 triage watch now covers with a correct instrument.

## 9. Exact commands run and their results

- `git checkout -b spike-sampler-timers` → branch (zero commits, deleted at close).
- 1-file spike (§5.2; greedy-path verify sites verified outside all samp windows by grep first).
- `cmake --build --preset release` → 54/54; `ctest --preset release` → pass; `tge_tests.exe` → 141/141 (unchanged).
- `aikit.py check` → 2/4 (test PASS; pre-existing hygiene + guidelines fails).
- Spec pair: forward 121.60 + verify-in-sample 87.51 + sample-excl 7.85 vs wall 129.57 (§6); plain: story ids match, creative fluent.
- `git checkout -- <file>` → clean (+grep verified zero remnants); `git checkout master`; `git branch -D`; final master gates green; report committed; no push.

(End of file.)
