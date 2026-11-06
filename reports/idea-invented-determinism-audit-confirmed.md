# INVENTED: Thread-Count Determinism Audit — CONFIRMED (14/14 identical ids, 0.5B + Llama-3B)

- **Idea**: Invented verification ("wait less" adjunct): greedy generate must be bit-identical across thread counts (band splits change FP summation order) and repeat runs — otherwise every A/B in the library is confounded. Stated falsifier: any ids divergence across {1, 6, 10} threads → CONDITIONAL (report nondeterminism scope); all identical → CONFIRMED.
- **Cycle date**: 2026-11-06
- **Verdict**: CONFIRMED — 14/14 runs byte-identical: 0.5B-Q8_0 fox 6/6 (1T×2, 6T×2, 10T×2), 0.5B story 3/3, Llama-3B-Q4_K_M fox 3/3, Llama-3B haiku 2/2 (1T vs 10T). Run-to-run determinism holds too (no stochastic path in greedy generate). This bounds the earlier near-tie concern (1387 trie cycle): thread-count flips live only in speculative verify batches, not in the base engine — band-split summation order never flips an argmax here.
- **Tree state at close**: `git status` clean apart from PRE-EXISTING, UNCOMMITTED peer spike work (`run_spike_mprefix`, verified byte-identical to the patch saved two cycles ago — untouched again). No branch created, no code touched by this cycle, nothing of mine to revert. `master` ahead of `origin/master` by 42 after this report (reports only). Nothing pushed (human gate).

## 1. Mechanism

Fixed prompts × {1, 6, 10} threads × {0.5B-Q8_0, Llama-3B-Q4_K_M} ×
fixed/creative prompts, `--print-ids`, diff the id sequences. Greedy,
no sampling, no speculation: any divergence would implicate band-split
FP reduction order.

## 2. Predicted gain — none (diagnostic)

Pass bar: 100% ids identity. Fail bar: any divergence → scope it
(which configs, which positions) and flag the library's A/B comparability.

## 3. Falsifier (stated) — passes cleanly

Zero divergences in 14 runs. The engine is exactly reproducible across
thread counts on both models.

## 4. File anchors (current `master`, untouched — measurement only)

- `apps/cli/main.cpp` (`command_generate`, `--print-ids`).
- Threading: `core::ThreadPool`, banded kernels (`gemv_*`, `attention_gqa`
  chunking) — deterministic band assignment per thread count.

## 5. What was done (pure measurement — no spike, no branch)

1. Build + `ctest` green (141/141) before benchmarking.
2. 14 generate runs as §1 (box heavily loaded today — t/s depressed
   machine-wide, e.g. Llama-3B fox 1.4–5.5 t/s vs ~11 historical; ids
   unaffected, and t/s is not a verdict input here).
3. `aikit.py check` 2/4 (test + secrets pass; hygiene + guidelines fail
   pre-existing).
4. No thinking-set diff owed beyond the runs themselves (fixed + creative
   prompts ARE the thinking set here; all identical).

## 6. Measured numbers

| model | prompt | threads | ids |
|---|---|---|---|
| 0.5B Q8_0 | fox, n=32 | 1,1,6,6,10,10 | identical `576 ... 1854 13` (6/6) |
| 0.5B Q8_0 | story, n=24 | 1,6,10 | identical (3/3) |
| Llama-3B Q4_K_M | fox, n=16 | 1,6,10 | identical `320 ... 16 25` (3/3) |
| Llama-3B Q4_K_M | haiku, n=16 | 1,10 | identical (2/2) |

## 7. Full reimplementation recipe (verification protocol)

Fixed prompts (fox + one creative), `--n` as above, `--threads ∈ {1,6,10}`,
`--gpu off --print-ids`, diff id lines. Expect byte identity everywhere.
If a future kernel change breaks this, bisect by thread count (1T isolates
band-split effects) before touching anything else.

## 8. Verdict and why

**CONFIRMED.** The base engine is exactly reproducible across thread counts
on both reference workloads. A/B comparability across the whole library
holds; the only known nondeterminism source remains speculative-verify
near-ties (different code path, documented in the 1387 report).

## 9. Exact commands run and their results

- Peer-dirt byte-compare vs saved patch → IDENTICAL (untouched).
- `cmake --build --preset release` → 54/54; `ctest` → pass.
- 14 generate runs (§6 — all ids identical).
- `aikit.py check` → 2/4 (pre-existing fails). No branch, no edits by me,
  peer spike left exactly as found, no push.
