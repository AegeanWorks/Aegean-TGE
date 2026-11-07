# THREAD SWEEP + QUEUE STATE — NO CHANGE (thermal drift, needs cool-box ABBA)

- **Date**: 2026-11-07. Box heat-soaked after a full day of benchmarking (runs slow down run-over-run regardless of settings).
- **Verdict**: NO CHANGE. No code touched (`git status` clean, no diff, stash untouched). Nothing pushed (human gate).

## 1. Prefill thread-count probe (measurement only, no code)

Hypothesis from an earlier best-of-2 (cool box): 4 threads beat 6-10 on 0.5B 2048-tok prefill (295 vs ~240 t/s). Interleaved ABBA retest just now (same shape, n=8), run order 4,6,6,4:

| run | threads | prefill t/s | decode t/s |
|---|---|---|---|
| 1 | 4 | 293.8 | 55.1 |
| 2 | 6 | 292.8 | 65.7 |
| 3 | 6 | 291.3 | 61.9 |
| 4 | 4 | 255.7 | 51.0 |

Run 1 (coolest box) is fastest; everything decays after — thermal drift, not threads. 4T mean 274.8 vs 6T mean 292.1 reverses the earlier signal. Unresolvable on a hot box: INCONCLUSIVE. Do NOT change the prefill thread default on this evidence. Revisit only with a cool-box ABBA ×3 (n=8, 0.5B 2048, order-balanced); if 4T wins every pair by ≥10%, then implement the default change under the standard protocol.

## 2. Registry tagging (this cycle's kept books)

- IDEA-1714 → Integrated (null guards ported, `0301bbf`, contract test green).
- IDEA-1517 → Integrated (S1 populate `5010573` + S2 skip `b8016a9`; staged recipe complete).
- Registry now: 26 Rejected, 16 Superseded, 4 Integrated (1394, 1568, 1714, 1517), 1 Approved (IDEA-1516, mixed probe/kill — needs a read before any action). No invented ideas in the registry (reports are their record).

## 3. Queue state

Port shelf nearly empty. Remaining timing-sensitive work (hoist re-verify, thread sweep, any new kernel A/B) is parked until a cool quiet box — measuring now would mint false verdicts in either direction, as today's pairs demonstrate (identical binaries differ ±17% run-to-run). Functional/robustness shelf: 1516 read-through, then new invented directions.

## 4. Exact commands run and their results

- `tge generate --model 0.5B-Q8_0 --tokens "<2048 ids>" --n 8 --threads {4,6,6,4}` → §1 table (box decayed across the session).
- Registry retag via script (history + review_log appended, evidence links); `git status` clean (registry is gitignored local state, nothing to commit).
- No build (no code change); report committed; no push.

(End of file.)
