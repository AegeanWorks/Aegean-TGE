# INVENTED: Sampler Exp-vs-Logic Split — MEASURED (exp path unaccounted: only ~7ms of 99ms "sampling" is scans+exp; verify forwards are double-counted)

- **Idea**: Invented follow-up, queued by the sampler-scans kill (which left ~92% of reported sampling unattributed): instrument `softmax_probabilities` (exp path) alongside the totals/walks counters to split sampling into (a) vectorized exp softmaxes, (b) scalar reductions, (c) serial walks, (d) everything else (accept/reject logic). Same temporary-instrumentation pattern, same falsifier shape: attribute ≥90% of sampling to named buckets or report the gap honestly.
- **Cycle date**: 2026-11-06
- **Verdict**: MEASURED — with a bigger finding than the split itself. On the spec temp-0.1 repetitive shape (4 steps, 17 accepts, 20 emitted): exp 5.99 ms + totals 1.67 ms + walks 0.01 ms = 7.67 ms of 99.46 ms reported "sampling". The remaining ~92 ms is NOT sampler work at all: the `samp_seconds` window in `generate` starts before the accept loop and ends after it, while the per-draft verify forwards (`verify_draft_logits` → `prefill_run`, 6-token batches) run INSIDE that window and are ALSO booked to `fwd_seconds`. Forward + sample (124.65 + 99.46 = 224.1 ms) exceeds the decode wall (133.23 ms) — impossible without double counting, and code inspection confirms the overlap (verify call at main.cpp:1071 sits inside the [st0, st1] sample window while adding to fwd). So: (1) the true sampler cost on this shape is ~7-8 ms, not ~99 ms; (2) the "sampling 79.8%" headline is a timer-accounting artifact; (3) exp is ~78% of TRUE sampling cost (5.99/7.67), totals ~22%, walks ~0%. The scans verdict from last cycle stands (and strengthens: 1.68/133.2 = 1.3% of wall time). Named follow-ups (not this cycle): fix the timer overlap (one-line scope change, needs its own gate); re-price "sampling share" claims after the fix.
- **Tree state at close**: `git status` clean (no diff, no untracked), stash unchanged (only the pre-existing `leftover peer work shelved 2026-11-04`, untouched). Spike branch created (zero commits) and deleted; all 4 spike files reverted with empty diff verified. `master` ahead of `origin/master` by 85 after this report. Nothing pushed (human gate).

## 1. Mechanism (as spiked)

Same three files as the scans round plus one counter: thread-local `g_sampler_exp_ns`, timed wrapper around the AVX2/scalar dispatch in `softmax_probabilities`, `sampler_scan_ns(exp, totals, walks)` accessor, and the `scan :` print widened to three columns. No numerics touched (timers only).

## 2. Predicted gain — diagnostic, and it found a bookkeeping bug instead of a hotspot

Predicted: attribute ≥90% of sampling to (a)/(b)/(c)/(d). Measured: (a)+(b)+(c) = 7.7% — the attribution FAILS, which is itself the result: the residual is double-counted verify-forward time, not unattributed sampler work. The falsifier's spirit (explain sampling) is satisfied by explaining why it can't be explained from inside the sampler.

## 3. Falsifier outcome — attribution fails honestly, and the failure is the finding

No bucket reaches 90%: exp 6%, totals 2%, walks ~0% of reported sampling. Rather than manufacturing attribution, the cycle traced the gap to its source: verify forwards (~92 ms across 4 steps: four 6-token prefill_runs at ~23 ms each — consistent with the 0.5B 70-token prefill rate of ~290 tok/s) sit inside BOTH timers. Corroboration is threefold and independent: (i) arithmetic (124.65 + 99.46 = 224.1 > 133.23 wall — impossible without overlap); (ii) code inspection (§4: verify call inside the sample window, fwd-booked); (iii) the instrumented sum (7.06–7.67 ms) matches the expected true sampling cost (24 softmaxes × ~0.25 ms + ~20 accepts × ~0.05 ms ≈ 7 ms).

## 4. File anchors (master = pre-spike; spike reverted)

- `apps/cli/main.cpp` (`st0`…`st1` sample window vs `v0`…`v1` fwd window inside it — the overlap; `timers` print).
- `src/models/sampling.cpp` (timed `softmax_probabilities` + totals/walks — reverted).
- `include/tge/models/sampling.hpp` (3-arg accessor — reverted).
- Grounds: sampler-scans kill (scans ~1% of a then-unexplained total); §28t (the 171 ms figure inherits the same overlap); 1568 triage (no verdict changes — the triage never depended on the split).

## 5. What was done (temporary instrumentation, fully reverted)

1. `git checkout -b spike-sampler-split2` (zero commits ever made on it; deleted at close — after one wrong-name delete attempt that failed safe, corrected immediately).
2. Spike (4 files): same scan timers as last round + exp wrapper + widened print + accessor.
3. `cmake --build --preset release` clean (54/54); `ctest --preset release` pass; `tge_tests.exe` 141/141 (unchanged count — instrumentation adds no tests).
4. Box health: not separately re-gated (same window class as the scans round, ~2 h later; spec trajectory identical: 4 steps / 17 acc / 20 out both runs).
5. Spec temp-0.1 repetitive ×2 (identical trajectory both runs): §6 table.
6. Plain temp-0.1 sanity (not the gate; confirms no behavioral drift): story ids byte-identical to baseline; creative text fluent.
7. Reverted all 4 files (`git checkout --`, empty diff verified + grep-confirmed zero SPIKE markers); `git checkout master`; `git branch -D spike-sampler-split2` (after one wrong-name attempt); final gates on reverted master: build clean, `ctest` pass, `tge_tests` 141/141; aikit 2/4 (test PASS, no flake — second consecutive clean run; hygiene + guidelines fail pre-existing).

## 6. Measured numbers (0.5B-Q8_0, fox ×7 prompt, n=20, temp 0.1, seed 7, --speculative, 6 threads)

| run | decode (t/s) | forward | reported sample | exp | totals | walks | accounted |
|---|---|---|---|---|---|---|---|
| spec R1 (scans round) | 191.04 (99.45) | 183.01 | 152.41 (79.8%) | — | 1.49 | 0.01 | — |
| spec R2 (this round) | 133.23 (142.61) | 124.65 | 99.46 (74.7%) | 5.99 | 1.67 | 0.01 | 7.67 (7.7%) |
| plain temp-0.1 | 134.02 (82.08) | 127.25 | 6.76 (5.0%) | 3.42 | 0.90 | 0.01 | 4.33 (64%) |

(Run-to-run absolute drift is box load; the accounting identity is load-independent.) Residual ("everything else"): plain ~2.4 ms (draw overhead, penalty checks — plausible); spec ~92 ms (verify forwards, double-counted — proven by (i)–(iii) above).

## 7. Full reimplementation recipe (diagnostic instrument — rebuild to re-verify, then STOP; the timer fix is separate)

Rebuild: §5.2 (3 counters + wrapper + print). Gate: build 54/54, 141/141 tests, spec run reproduces §6 within run noise (exp+totals+walks ≈ 6–8 ms while reported sample stays ~100 ms). To re-verify, confirm the forward+sample > wall inequality persists on master (it must — nothing changed). NEXT (named, not built): (1) fix the timer overlap (scope `st0`…`st1` to exclude verify forwards, or subtract the fwd-booked verify spans; one-line-class change, needs its own gate run showing forward+sample ≤ decode wall); (2) after the fix, re-price the "sampling share" headline that §§28s/28t and the 1568 round quoted — expect it to fall from ~75% to ~5-10% on this shape. To re-verify THIS report: re-run the §6 pair; the accounting identity (exp+totals+walks ≪ reported sample, forward+sample > wall) must reproduce.

## 8. Verdict and why

**MEASURED: the scans are 1.5% of sampling, the exps ~6%, and the remaining ~92% is double-counted verify-forward time — a timer bug, not a hotspot.** This closes the sampler program with unusual completeness: not only is there nothing to vectorize (scans round), there was never as much sampling as the dashboards claimed (this round). The loop's honest output is the corrected ledger plus two named, scoped follow-ups neither of which is this cycle.

## 9. Exact commands run and their results

- `git checkout -b spike-sampler-split2` → branch (zero commits, deleted at close).
- 4-file instrumentation (§5.2; one duplicate-block compile error caught pre-build).
- `cmake --build --preset release` → 54/54; `ctest --preset release` → pass; `tge_tests.exe` → 141/141 (unchanged).
- Spec temp-0.1 ×2 (identical trajectory: 4 steps, 17 acc, 20 out both runs): §6 table.
- `aikit.py check` → 2/4 (test PASS, no flake; pre-existing hygiene + guidelines fails).
- Thinking set on spike binary (identical paths; timers don't touch numerics — verified by identical story ids): story ids byte-identical; creative text fluent.
- `git checkout -- <4 files>` → clean (grep-verified zero SPIKE markers); `git checkout master`; `git branch -D` (after one wrong-name attempt); final master gates green; report committed; no push.

(End of file.)
