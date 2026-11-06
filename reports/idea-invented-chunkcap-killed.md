# INVENTED: L2-Cap GQA Chunk Sizing for Long-Context Decode — KILLED (parallelism knob, already saturated; L2 never binds)

- **Idea**: Invented ("wait less" direction): cap GQA decode chunks by L2 working-set size instead of the shipped parallelism formula (`min(16, ceil(threads*2/kv_heads))`), on the hypothesis that long-span chunks spill cache.
- **Cycle date**: 2026-11-06
- **Verdict**: KILLED — chunk count is a parallelism knob, not a cache knob, and the shipped formula already saturates it. N=1 halves throughput (parallelism starvation: 2 tasks on 10 threads); N=4 runs 16% under baseline; N=16 matches baseline. Working sets are L2-resident at every setting (184 KB–1.8 MB vs MBs of cache), so an L2 cap would never bind at measurable spans. Ids byte-identical across N=1/16/default (merge is numerically transparent here — no fidelity owed beyond that). Spike fully reverted; only this report committed.
- **Tree state at close**: `git status` clean (no diff), spike branch created and deleted with nothing committed (2 small edits), stash unchanged (pre-existing peer work only), `master` ahead of `origin/master` by 38 after this report. Nothing pushed (human gate).

## 1. Mechanism

Env-gated `TGE_GQA_CHUNKS` override of the chunk count in `attention_gqa`,
clamped to `[1, kMaxGqaChunks]` (the group scratch is sized for 16 — verified
the hard way: N=32 overflows it and crashes, recorded here so nobody
re-discovers it). Sweep N ∈ {1, 4, 16} vs shipped formula (~10 chunks) at
3.6k-token KV, 0.5B Q8_0, 10 threads.

## 2. Predicted gain — none (diagnostic)

Falsifier (stated): kill if no chunk setting beats the shipped formula by
>5% best-of-3, or any ids divergence. Pass bar would have required a
cache-driven win at some N.

## 3. File anchors (current `master`, untouched except by the reverted spike)

- `src/models/qwen2.cpp:1294-1298` (chunk formula), `:57-58`
  (`kMaxGqaChunks = 16` scratch bound), `merge_gqa_chunks` (exact merge).

## 4. What was done (temporary spike, fully reverted)

1. `git checkout -b spike-chunkcap` (no commit ever made on it).
2. 15-line env override + `<cstdlib>` include.
3. Build clean; `ctest` pass; 141/141.
4. Crashed N=32 run → diagnosed scratch bound → clamped to 16 (kept in the
   recipe as a guardrail).
5. 3.6k-token prompt, decode-16 best-of-3 at N=1/4/16/default; ids parity
   triple (N=1, N=16, default).
6. Reverted both hunks, verified empty diff, deleted branch. Final gates on
   master: build clean, `ctest` pass, `aikit.py check` 2/4 (test + secrets
   pass; hygiene + guidelines fail pre-existing).

## 5. Measured numbers (Qwen2.5-0.5B-Q8_0, 3641-token KV, 10T, `--gpu off`, decode-15 t/s)

| chunks | best-of-3 | verdict vs baseline 60.83 |
|---|---|---|
| 1 | 30.49 / 29.57 / 30.13 | **−50%** (2 tasks, 8 threads idle) |
| 4 | 50.41 / 51.15 / 50.80 | −16% |
| 16 | 59.50 / 59.98 / 60.36 | −1% (plateau) |
| shipped (~10) | 60.83 / 59.88 / 58.63 | — |

Ids `576 3974 ... 11416` byte-identical across N=1/16/default. No setting beats
the formula; deviations only hurt (parallelism starvation) or tie (plateau).
Working sets at every tested count are L1/L2-resident, so no cache-capacity
mechanism exists for an L2-cap to exploit at these spans.

## 6. Full reimplementation recipe (kill-grade — rebuild only to re-verify)

1. `attention_gqa`: resolve `chunks` from the shipped formula, then override
   from `TGE_GQA_CHUNKS` if in `[1, kMaxGqaChunks]`; add `<cstdlib>`.
2. Gate: 3.6k-token prompt + `--n 16` decode best-of-3 at N ∈ {1,4,16} plus
   unset, 10T, `--gpu off` — expect the §5 table (N=1 halves, N=16 ties).
3. Never exceed `kMaxGqaChunks` without resizing the group scratch
   (`group_maxima/sums/acc` sized `kv_heads * kMaxGqaChunks * group`).

## 7. Verdict and why

**KILLED by measurement.** The knob was misidentified: chunk count governs
thread occupancy, not cache residency, and the shipped formula already sits
on the plateau (anything coarser starves threads; finer ties within noise and
cannot exceed the scratch bound). An L2-cap policy would be dead code — the
cap never binds where the formula already saturates. Revisit only at spans
where the formula's chunks exceed cache (tens of k tokens with wider heads),
with the scratch bound raised first.

## 8. Exact commands run and their results

- `git checkout -b spike-chunkcap` → new branch.
- 2-hunk spike → build 54/54 → 141/141.
- Crashed N=32 probe → clamp fix → rebuild.
- Baseline/N=4/N=1/N=16 best-of-3s (§5); ids parity triple (identical).
- `git checkout -- src/models/qwen2.cpp` → diff empty;
  `git checkout master`; `git branch -D spike-chunkcap`.
- Final: build clean, `ctest` pass, `aikit.py check` 2/4 (pre-existing fails).
- No push.
