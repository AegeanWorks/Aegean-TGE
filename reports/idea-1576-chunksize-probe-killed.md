# IDEA-1576 STAGE-1: Prefill Chunk-Size Sensitivity Probe (64 → 256) — KILLED (chunk size is not the lever)

- **Idea ID**: IDEA-1576 (dossier `goal-sweep-round2-2026-11-03.md`: single-pass unchunked CPU prefill with static continuous KV tensor). Staged: the static-tensor half describes the current fixed-slot caches (ships); this stage probes the unchunked half via chunk-size sensitivity (64 vs 256 rows).
- **Cycle date**: 2026-11-06
- **Verdict**: KILLED stage-1 — no repeatable >5% prefill gain (best-of-3 +3.8% with overlapping ranges, inside box noise). Chunk growth does not fix the actual traffic pattern (activations re-read per output row regardless). Spike fully reverted; only this report committed.
- **Tree state at close**: `git status` clean (no diff), spike branch created and deleted with nothing committed (one-line change), stash unchanged (pre-existing peer work only), `master` ahead of `origin/master` by 29 after this report. Nothing pushed (human gate).

## 1. Mechanism (stage-1 slice)

`kPrefillChunk` (64) bounds prefill batch rows so activations live in cache;
each chunk re-streams weight matrices from DRAM. Quadrupling to 256 should cut
weight re-reads ~4x (33→8 passes over a 2081-token prompt) if weight streaming
dominates prefill. One-constexpr spike; load-time scratch (`batch_x_`,
`scores_`, prefill rows) follows automatically.

## 2. Predicted gain (with the math)

Weight traffic per 3072-wide Q4_K row-group ≈ 4.7 MB per pass; 33 → 8 passes
saves ~117 MB DRAM reads per 2081-token prefill (~25 ms at ~5 GB/s) against a
~9 s prefill — small but the matvec share (59%) makes even single-digit
percent plausible. Falsifier (stated): kill if no repeatable >5% prefill t/s
gain best-of-3 at 2081 tokens, or any ids divergence.

## 3. Falsifier (stated for stage-1)

<5% best-of-3 prefill gain → kill (unchunked prefill has no headroom worth the
L3/cache-residency risk the 64-row design guards). Any ids divergence → kill.

## 4. File anchors (current `master`, read but untouched except the reverted constant)

- `src/models/qwen2.cpp:33` (`kPrefillChunk`), `:492–500` (load-time sizing),
  `:1774` (`capacity`), `:1869+` (chunk loop honoring `consumed`).
- Comment at `:28-33`: the batch kernel re-reads activations per output row.

## 5. What was done (temporary spike, fully reverted)

1. `git checkout -b spike-1576-chunk`; one-line change 64 → 256.
2. Build clean; `ctest` pass; 141/141.
3. 2081-token Q8_0 generate prefill best-of-3 (spike), saved binary, stashed,
   rebuilt baseline (64), best-of-3. Saved spike binary reused for parity.
4. Thinking-set parity: ~110-token prompt + 16 decoded ids byte-identical
   256-chunk vs 64-chunk (promptids + ids) — compute-order-only confirmed, no
   numerics gate owed.
5. Reverted the constant, verified empty diff, deleted branch. Final gates on
   master: build clean, `ctest` pass, `aikit.py check` 2/4 (test + secrets
   pass; hygiene + guidelines fail pre-existing).

## 6. Measured numbers (Qwen2.5-0.5B-Q8_0, 2081 tokens, 10T, `--gpu off`, warm page cache)

| | run 1 | run 2 | run 3 | best |
|---|---|---|---|---|
| spike (256) | 275.5 t/s | 274.1 | 253.3 | **275.5** |
| baseline (64) | 265.5 t/s | 245.3 | 246.2 | **265.5 (+3.8%)** |

Ranges overlap (spike min 253.3 < baseline max 265.5); delta inside documented
box variance. Medians differ more (274.1 vs 246.2) but the overlap forbids a
repeatability claim — exactly what best-of-3 with a 5% bar is for. The batch
kernel's per-row activation re-reads (code comment) dominate regardless of
chunk count: fewer weight passes cannot fix activation traffic.

## 7. Full reimplementation recipe (kill-grade — rebuild only to re-verify)

1. `src/models/qwen2.cpp:33`: `kPrefillChunk` 64 → 256 (single line; scratch
   sizing follows automatically, no other edit).
2. Rebuild; run the §6 protocol (2081-token prompt, `--n 1`, 10T, `--gpu
   off`, best-of-3 both binaries via stash) — expect +0–4% inside noise.
3. Ids parity: ~110-token prompt + `--n 16 --print-ids` both binaries —
   expect byte-identical.

## 8. Verdict and why

**KILLED stage-1.** Chunk size moves prefill by noise (±band), because the
dominant traffic (per-row activation re-reads) is chunk-count invariant.
Unchunked prefill as a direction is closed by this measurement plus the
standing cache-residency design note — unless a future batch kernel reuses
activations across rows (a different idea: kernel fusion, not chunk sizing).
The static-tensor half needs no work (fixed-slot caches already are static
continuous tensors).

## 9. Exact commands run and their results

- `git checkout -b spike-1576-chunk` → new branch.
- One-line constant change → build 54/54 → 141/141.
- Spike prefill best-of-3 (275.5/274.1/253.3); saved binary; stash; rebuild.
- Baseline prefill best-of-3 (265.5/245.3/246.2).
- Ids parity pair (byte-identical).
- `git stash pop`; `git checkout -- src/models/qwen2.cpp` → diff empty;
  `git checkout master`; `git branch -D spike-1576-chunk`.
- Final: build clean, `ctest` pass, `aikit.py check` 2/4 (pre-existing fails).
- No push.
