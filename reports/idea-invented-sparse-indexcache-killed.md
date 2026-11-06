# INVENTED: Sparse Shared Index Cache — KILLED (160K+ cache hits, zero timing change; the cost is the gather, not the build)

- **Idea**: Invented, queued by the 1568 close-out (sparse prefill attn +35% vs dense at 2k on 0.5B-Q8_0: 1024→1403 ms; +13% on Llama-3B: 7731→8750). Mechanism: `build_sparse_index` heap-allocates a fresh vector per (token, layer, head) call — ~672K allocations per 2k-token prefill — while the index itself depends only on the span (content-blind static stride), so 13 of every 14 builds are byte-identical duplicates across a token's heads. Spike: `build_sparse_index_into` (in-place fill, bit-identical contract) + one thread-local cached buffer per worker keyed on span (13/14 hits by construction, zero allocs after warmup), no other changes.
- **Cycle date**: 2026-11-06
- **Verdict**: KILLED on the stated falsifier (pre-fill attn must improve ≥10% vs un-cached sparse: measured 1420 vs master band 1403–1456 ms, i.e. 0% — identical within run noise). The no-op is itself the finding: per-call heap allocation is NOT where sparse prefill time goes. The remaining cost is structural — per-head gather memcpys into thread-local scratch plus per-head kernels (vs dense batched GEMM), plus the forced per-head dispatch (the GQA grouped kernels stay dense by design). Identity arm holds: spike sparse ids == master sparse ids byte-identical (`576 3974 13876` on the 2k prompt), and the new unit test pins into==by-value exactly (spans 0/1/64/515/516/517/1024/2048/8192, enabled+disabled, refill-after-dirty) — the spike changed allocation strategy only, provably nothing else. Named next: a no-gather strided kernel (read the strided pattern in-place instead of materializing compacted buffers) or shared cross-head gather — the "one index list per query shared across the group" future-work note points the same way. Spike fully reverted (3 files + test); nothing pushed (human gate).
- **Tree state at close**: `git status` clean (no diff, no untracked), stash unchanged (only the pre-existing `leftover peer work shelved 2026-11-04`, untouched). Spike branch created (zero commits) and deleted; all spike files reverted with empty diff verified. `master` ahead of `origin/master` by 77 before this report. Nothing pushed (human gate).

## 1. Mechanism (as spiked)

`build_sparse_index_into(span, config, vec)` (exact same arithmetic as `build_sparse_index`, which now delegates to it); `attention_head`'s sparse branch keeps one `thread_local` index buffer + span tag and skips the rebuild on tag match (13/14 same-span hits across a token's heads; buffer storage reused across spans, zero allocs after warmup). Gather, kernels, dispatch, GQA-vs-per-head routing: all untouched, deliberately — the spike isolates exactly the index-build cost.

## 2. Predicted gain — not realized, and the non-result is load-bearing

Predicted: eliminating ~672K heap allocations + ~640K redundant O(span) fills per 2k prefill recovers most of the 35% attn gap. Measured: nothing (1420 vs 1403/1428/1456 master band). The estimate was wrong in the only way that matters: small-vector heap traffic at this scale costs ~0.1 s against a 1.4 s attn phase dominated by gather memcpys into thread-local scratch (kept rows × head_dim × 2, per head per token, L3-resident) plus per-head kernel launches. Arithmetic that should have preceded the spike is now on the record for the next one.

## 3. Falsifier outcome — kill arm fires cleanly; identity arm holds exactly

Kill bar was ≥10% attn improvement: measured +0% (1420 inside the 1403–1456 master band). Identity bar was byte-identical sparse ids at matched settings: `576 3974 13876` both binaries on the 2000-token fox prompt (three tokens, all engaged past the 516 threshold) — plus the unit test's exact-equality across 9 spans × 2 configs. The spike was behavior-preserving by construction AND by measurement; it was simply aimed at the wrong cost.

## 4. File anchors (master = pre-spike; spike reverted)

- `src/models/qwen2.cpp` (`build_sparse_index_into` + `build_sparse_index` delegating wrapper + cached thread-local in `attention_head` — reverted).
- `include/tge/models/qwen2.hpp` (`build_sparse_index_into` decl — reverted).
- `tests/test_attention.cpp` (`sparse_index_into_matches_by_value` — green, reverted with the spike).
- Grounds: 1568 close-out (the +35%/+13% that motivated the spike), 1451/1452/1498 (scored family — orthogonal; this spike touches only the static subset's plumbing).

## 5. What was done (temporary spike, fully reverted)

1. `git checkout -b spike-sparse-indexcache` (zero commits ever made on it; deleted at close).
2. Spike (3 files + 1 test): into-builder + delegating wrapper; cached thread-local in the hot path (declared `mutable`-free: plain function-locals, no struct changes); engagement probe (temporary stderr counters, removed before measuring — verified 160K+ hits: the cache engaged massively).
3. Mid-spike process note: no dead ends this round — the one edit that needed a second pass was a mangled function body from two overlapping edits (rewritten cleanly before building; the tree-diff discipline caught it).
4. `cmake --build --preset release` clean (54/54); `ctest --preset release` pass; `tge_tests.exe` 142/142 (141 + 1 new).
5. Box health: wall 54.93 bracketing; aikit 2/4 (test PASS, no flake this cycle — the 2.9 s flake of the 1568 round did not recur; hygiene + guidelines fail pre-existing).
6. Engagement proof (temporary counters, since removed): MISS lines on new spans, 16× "10000 hits" lines — sharing works as designed.
7. Falsifier measurement: spike sparse 2k prefill (293.3 dense / 273.4 sparse; attn 1062 dense / 1420 spike-sparse) vs master sparse via stash dance (272.0, attn 1456.3; ids byte-identical `576 3974 13876` both).
8. Thinking set on spike binary: story short-prompt ids identical to the dense baseline (dense fallback, span < 516); reasoning answer fluent.
9. Reverted all 3 files (`git checkout --`, empty diff verified), `git checkout master`, `git branch -D spike-sparse-indexcache`; final gates on reverted master: build clean, `ctest` pass, `tge_tests` 141/141; aikit re-verified post-revert (141 green, test PASS).

## 6. Measured numbers (0.5B-Q8_0, fox ×200 = 2000 tokens, ctx 4096, 6 threads, gpu off)

| run | prefill | prefill attn | decode | ids (3 tok) |
|---|---|---|---|---|
| dense (spike binary) | 293.3 t/s (6819 ms) | 1062 ms | 67.18 t/s | — (dense baseline) |
| sparse, uncached (master, stash-built) | 272.0 t/s (7352 ms) | 1456.3 ms | 68.42 t/s | 576 3974 13876 |
| sparse, cached index (spike) | 273.4 t/s (7316 ms) | 1420.1 ms | 65.81 t/s | 576 3974 13876 |

Reading: the cache changes nothing (attn within master band, ids identical) — the 13/14 duplicate builds were never the cost. The sparse-vs-dense attn gap (+35%) survives intact and is now localized to gather+per-head dispatch, not allocation.

## 7. Full reimplementation recipe (kill-grade — rebuild to re-verify, then STOP; the no-gather kernel is next)

Rebuild: §5.2 (into-builder + delegating wrapper; cached thread-local + span tag in the hot path; engagement counter first, remove after proof; equivalence unit test). Gate: build 54/54, 142/142 tests (equivalence must be exact, not tolerant); engagement proof (hits ≫ misses on a 2k run); fidelity-scale proof: 2k prefill sparse attn within noise of un-cached sparse (the EXPECTED result — this spike's purpose is the negative); ids byte-identical master-vs-spike. Do NOT extend (bigger caches, fancier sharing — the sharing already proved complete and free). NEXT SPIKE (named, not built): no-gather strided attention — teach the AVX2 attention kernel the static stride/window/sink pattern directly (read strided rows in place, no compacted copies, no per-head scratch); falsifier: 2k prefill attn within 10% of dense at identical sparse ids AND identical fidelity KL (2.597e-02 class) vs current sparse; kill on either miss. To re-verify this kill: re-run the §6 pair — cached must still tie uncached, and dense must still lead both on prefill.

## 8. Verdict and why

**KILLED on the stated bar, with the cost map redrawn.** The index cache works exactly as designed (160K+ hits, zero behavior change, zero allocs) and buys exactly nothing — which retires the "allocation storm" theory of sparse-prefill cost with evidence instead of assertion, and points the program at the one remaining suspect class: per-head gather traffic + forced per-head dispatch (the GQA grouping the sparse path bypasses by design). Until a no-gather kernel exists, the 1568 triage language stands verbatim (opt-in, decode/long-context/large-model gated) — and now carries a measured reason for the prefill half.

## 9. Exact commands run and their results

- `git checkout -b spike-sparse-indexcache` → branch (zero commits, deleted at close).
- 3-file spike + 1 test (§5.2; one mangled-function rewrite pre-build; engagement counters added then removed).
- `cmake --build --preset release` → 54/54; `ctest --preset release` → pass; `tge_tests.exe` → 142/142 on spike, 141/141 after revert.
- Wall 54.93; `aikit.py check` → 2/4 (test PASS, no flake; pre-existing hygiene + guidelines fails).
- Engagement: 160K+ hits logged; 2k prefill spike-sparse (273.4 t/s, attn 1420) vs master-sparse via stash dance (272.0 t/s, attn 1456, ids identical `576 3974 13876`).
- Story ids (short) == dense baseline; reasoning fluent.
- `git checkout -- <3 files>` → clean; `git checkout master`; `git branch -D`; final master gates green; report committed; no push.

(End of file.)
