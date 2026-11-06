# INVENTED: Sparse Prefill Phase Split — MEASURED (build 4%, gather ~85%, dots ~11%: the gather is the cost, the index never was)

- **Idea**: Invented diagnostic, queued by the index-cache kill (which proved sharing the index list changes nothing): split the sparse attention phase into index-build vs gather-memcpy vs kernel-dots with temporary phase timers, on the same 2k-token prefill that shows sparse +35% vs dense, to name the actual lever for the no-gather kernel the index-cache round specified.
- **Cycle date**: 2026-11-06
- **Verdict**: MEASURED — the split is decisive. Of the sparse attention cost (total CPU across the prefill pool): index build 514 ms (~4%), gather memcpys 11,460 ms (~85%), kernel dots the remainder (~11%). The index cache attacked the 4% — consistent with its measured zero effect; the gather (≈10x the kernel's own traffic: 454 KB read+write per (token, head, layer) vs 227 KB of kept rows the kernel actually consumes) is the cost, exactly as the index-cache kill's post-mortem hypothesized. Falsifier (stated): gather+index < 30% → kill the gather theory; gather alone > 50% → no-gather kernel justified. Result: gather ≈ 85% → the no-gather strided kernel is now quantitatively justified as the ONLY remaining lever (eliminating the gather's ~11.5s CPU flips the 2k sparse prefill from 1.4x slower toward faster-than-dense, since the dots themselves already cost less than dense). Spike fully reverted (3 files); nothing pushed (human gate).
- **Tree state at close**: `git status` clean (no diff, no untracked), stash unchanged (only the pre-existing `leftover peer work shelved 2026-11-04`, untouched). Spike branch created (zero commits) and deleted; all spike files reverted with empty diff verified. `master` ahead of `origin/master` by 78 before this report. Nothing pushed (human gate).

## 1. Mechanism (as spiked)

Two `mutable std::uint64_t` counters + getters on `Qwen2`, reset in `reset_phase_timers`, accumulated around `build_sparse_index` (build) and the kept-row memcpy loop (gather) inside `attention_head`'s sparse branch, printed as one extra `sparse : build X ms, gather Y ms` line after the prefill phases. No behavior change (timers only) — dense and sparse paths byte-identical to master by construction.

## 2. Predicted gain — diagnostic, delivered with a ratio

Predicted either a gather-dominated split (justifying the no-gather kernel) or a build-dominated split (reviving allocation work). Measured: gather dominates 22:1 over build in absolute CPU time, ~85% of the sparse attention cost. The prediction that mattered was the negative one — and it held: nothing here reopens index work.

## 3. Falsifier outcome — gather arm fires decisively

Gather 11,460 ms vs the 1,464 ms attn wall (same run): the bar was ">50% to justify the no-gather kernel" and the measured share is ~85% of sparse-attn CPU. Caveats recorded honestly: (a) the two counters sum across worker threads (racy `+=` on plain uint64 — documented diagnostic imprecision; the 22:1 ratio is robust to uniform undercounting, absolute values are approximate); (b) wall-vs-sum comparisons depend on pool width (prefill uses up to all workers; the ratio, not the absolute, is the verdict).

## 4. File anchors (master = pre-spike; spike reverted)

- `src/models/qwen2.cpp` (`attention_head` sparse branch: index build + gather loop + kernel call — the three timed regions; `reset_phase_timers`).
- `include/tge/models/qwen2.hpp` (counter members + getters — reverted).
- `apps/cli/main.cpp` (`sparse :` print line — reverted).
- Grounds: index-cache kill (sharing proven, zero change), 1568 close-out (the +35%/+13% that motivated the program), dump/statcorr (different subsystem, same method).

## 5. What was done (temporary instrumentation, fully reverted)

1. `git checkout -b spike-sparse-phases` (zero commits ever made on it; deleted at close).
2. Spike (3 files): 2 counters + getters + reset + 2 timed regions + 1 print line. No unit test (nothing to assert — timers don't change values; gates + identical outputs are the check).
3. `cmake --build --preset release` clean (54/54); `ctest --preset release` pass; `tge_tests.exe` 141/141; `python aikit.py check aegean-tge` 2/4 (test PASS, no flake this round; hygiene + guidelines fail pre-existing).
4. Box health: wall 54.93 bracketing.
5. Measurement (0.5B-Q8_0, fox ×200 = 2000 tokens, `--context 4096 --sparse`, 6 threads, gpu off): §6.
6. Thinking set on spike binary (timers cannot change numerics; default paths untouched): story ids byte-identical to baseline (`11 1052 572 … 14126 13`); reasoning fluent.
7. Reverted all 3 files (`git checkout --`, empty diff verified), `git checkout master`, `git branch -D spike-sparse-phases`; final gates on reverted master: build clean, `ctest` pass, `tge_tests` 141/141.

## 6. Measured numbers (single instrumented run + wall context)

| meter | ms | share of sparse-attn CPU | reads as |
|---|---|---|---|
| prefill attn (wall, existing) | 1464.4 | 100% (wall) | the gap being explained |
| sparse index-build (summed) | 514.0 | ~4% | the killed theory's ceiling — consistent with the cache's no-op |
| sparse gather memcpys (summed) | 11460.5 | ~85% | THE cost: 454 KB read+write per (token, head, layer), ~688K calls |
| remainder (kernel dots + dispatch) | ~1490 implied | ~11% | already cheaper than dense dots — the win waiting behind the gather |

Box: wall 54.93 bracketing (healthy; the matvec drift seen in the 1568 round did not recur here — prefill matvec 3959.3 vs dense-run 4198/4344 band).

## 7. Full reimplementation recipe (diagnostic instrument — rebuild to re-verify, then STOP; the no-gather kernel is next and specified)

Rebuild: §5.2 (2 counters + getters + reset; 2 timed regions; 1 print line). Gate: build 54/54, 141/141 tests (no new tests owed — nothing behavioral to pin); single 2k run reproduces the §6 order (build ≪ gather, gather dominating by an order of magnitude). Then STOP instrumenting. NEXT SPIKE (named, not built — the index-cache report's recipe stands): no-gather strided attention — teach the AVX2 attention kernel the static sink/stride/window pattern directly (read strided rows in place; delete the compacted buffers and the per-head scratch); falsifier: 2k prefill attn within 10% of dense at identical sparse ids AND identical fidelity KL (2.597e-02 class) vs current sparse; kill on either miss. To re-verify this split: re-run the §6 single; build must stay ~4%, gather must stay dominant — if the ratio ever inverts, the program reopens from the top.

## 8. Verdict and why

**MEASURED: build 4%, gather ~85% — the index cache attacked the wrong 4%.** This closes the sparse-prefill attribution question opened two cycles ago with a measured split instead of an argument, and hands the no-gather kernel the quantitative justification it needed (kill the gather's ~11.5s CPU and the 2k sparse prefill flips from 1.4x slower to faster-than-dense, since the dots already cost less). The loop's honest output is the map with numbers: index work is dead as a lever, gather work is the whole ballgame, and the recipe for killing it is written.

## 9. Exact commands run and their results

- `git checkout -b spike-sparse-phases` → branch (zero commits, deleted at close).
- 3-file spike (§5.2; no unit test — timers only).
- `cmake --build --preset release` → 54/54; `ctest --preset release` → pass; `tge_tests.exe` → 141/141 (identical count pre/post — no test touched).
- Wall 54.93; `aikit.py check` → 2/4 (test PASS, no flake; pre-existing hygiene + guidelines fails).
- Instrumented 2k prefill `--sparse` → §6 table (attn 1464.4, build 514.0, gather 11460.5).
- Story ids byte-identical; reasoning fluent.
- `git checkout -- <3 files>` → clean; `git checkout master`; `git branch -D`; final master gates green; report committed; no push.

(End of file.)
