# INVENTED (nogather follow-up): Hoist Head-Invariant Sparse Index Build — KILLED (off/on pairs +9%/−0%/−4%: build cost is noise, dots+gather dominate)

- **Idea**: The no-gather report's named follow-up: `build_sparse_index(span, config)` depends only on (span, sink, window, stride) — identical for every head of a query — but `attention_head` rebuilds it per (position, layer, head): 14 redundant builds per query position on 0.5B (24 on Llama). Cache per thread on (span, sink, window, stride); first head builds, the rest hit. Same list → bit-identical by construction.
- **Cycle date**: 2026-11-06
- **Verdict**: KILLED on the stated falsifier (≥10% prefill-attn win at identical ids). Interleaved off/on pairs: +9.5% / −0.3% / −3.8% — no consistent direction, all inside box drift (matvec crept +20% across the window). The index build (one small-vector alloc + ~1k push_backs) is noise next to per-head dots over ~1k kept rows + the gather memcpys. Attribution complete: build ≪ gather < dots.
- **Tree state at close**: `git status` clean, no diff, stash unchanged (only the pre-existing `leftover peer work shelved 2026-11-04`). Spike branch `spike-indexhoist` created (zero commits) and deleted; the one spike file reverted with empty diff verified; scratch deleted. `master` ahead of `origin/master` by 103 after this report. Nothing pushed (human gate).

## 1. Mechanism (as spiked, reverted)

`src/models/qwen2.cpp` only, env-gated (`TGE_INDEX_HOIST=1`): thread-local `hoist_kept` + (span, sink, window, stride) key; on miss rebuild via `build_sparse_index`, else reuse; a `kept_fresh` local preserves today's rebuild path when the env is off (the A/B control runs in the same binary). No header change, no kernel change, no-gather untouched (orthogonal by design — measured on the shipped gather path).

## 2. Predicted gain — not realized

Expected ~10%+ of the sparse attn phase (14x build redundancy eliminated). Measured ~0%: the build is a small alloc + linear fill, dwarfed by the per-head dot work it feeds.

## 3. Falsifier outcome — kill

Stated: PROCEED only with ≥10% prefill-attn win at identical ids. Pairs (off→on): 2558.6→2801.5, 2864.1→2856.7, 2960.8→2847.9 — best pair −3.8%, worst +9.5%, i.e. noise around zero with the box drifting underneath. Ids identical 6/6 (as constructed). No fidelity run (no numerics change — same list values; timers/precedent pattern for accounting-only changes).

## 4. File anchors (master = pre-spike; spike reverted)

- `src/models/qwen2.cpp` (`attention_head` sparse block — the per-head build site; `build_sparse_index` at :143).
- Grounds: no-gather report §7 (the named follow-up + its falsifier shape), sparse phasesplit (gather 22:1 over build — the ranking this confirms from the other end).

## 5. What was done (temporary spike, fully reverted)

1. `git checkout -b spike-indexhoist` (zero commits, deleted at close). Baseline: build 54/54, `tge_tests` 141/141.
2. ANALYZE: build inputs confirmed head-invariant (span + config only); key chosen as (span, sink, window, stride); lifetime checked (borrow never escapes the block — the nogather lesson applied pre-emptively).
3. Applied §1 (env-gated, control preserved). Rebuilt 54/54, `tge_tests` 141/141.
4. Measured: 0.5B fox×200 --sparse, off/on interleaved ×3 (§6 table; ids 8/8 identical all 6). Thinking set on spike (saved).
5. `aikit.py check aegean-tge` → 2/4 (test + secrets PASS; hygiene + guidelines fail pre-existing, same pair as every prior cycle).
6. Reverted the file (empty diff), `git checkout master`, branch deleted, rebuilt master 54/54, `tge_tests` 141/141, master thinking set: story + creative ids byte-identical vs spike.
7. Scratch deleted; `git status` clean, stash untouched.

## 6. Measured numbers (0.5B-Q8_0 fox×200 = 2600 tok, 10 threads, --context 4096, --sparse, --n 8; prefill attn ms)

| round | hoist off | hoist on | delta |
|---|---|---|---|
| 1 | 2558.6 | 2801.5 | +9.5% |
| 2 | 2864.1 | 2856.7 | −0.3% |
| 3 | 2960.8 | 2847.9 | −3.8% |

Prefill wall t/s off/on: 232.7/217.8, 209.6/204.8, 198.5/205.8 (drift-dominated, same story). Box drift noted (matvec 6059→7253 across rounds); pairs adjacent. No consistent direction → ~0%.

## 7. Full reimplementation recipe (kill-grade — rebuild only to re-verify)

Re-apply §1; gate on build 54/54, `tge_tests` 141/141, and the §6 interleave reproducing ~0% with identical ids. No fidelity owed (same values by construction). Revival trigger: a shape where per-position build dominates dots (requires heads ≫ kept-rows with tiny spans — contradicts the sparse span gate; effectively never).

## 8. Verdict and why

**KILLED: the index build is not the bottleneck — the dots and the gather are.** This closes the sparse-prefill cost map end to end: dots ~halved by the subset (the win), gather ~85% of the remaining overhead (removed by the no-gather recipe, pending the human gate), build ~noise (this round). The nogather-prefill port recipe stands unchanged (its numbers never included a build share worth hoisting). Next-cycle candidates: the 1517 stage-2 skip harness and the human-gated port shelf (now six deep: mprefix-table, sparse decode no-gather, sparse prefill no-gather, batchsnap kind tags, timers fix, detect memoize).

## 9. Exact commands run and their results

- `git checkout -b spike-indexhoist` → branch (zero commits, deleted at close).
- Baseline: `cmake --build --preset release` → 54/54; `tge_tests.exe` → 141/141.
- Spike (§1, env-gated); rebuild → 54/54; `tge_tests` → 141/141.
- 0.5B fox×200 `--sparse` off/on interleave ×3 → §6 table (ids 8/8 identical all 6).
- Thinking set (spike, saved); `aikit.py check` → 2/4 (test + secrets pass; pre-existing fails).
- Revert → clean; `git checkout master`; branch deleted; rebuild 54/54; tests 141/141; master thinking set → ids identical; scratch deleted; report committed; no push.

(End of file.)
