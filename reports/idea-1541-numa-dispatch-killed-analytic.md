# IDEA-1541: NUMA-Aware Chunked Work-Stealing Decode Dispatch — KILLED WITHOUT SPIKE (precondition absent)

- **Idea ID**: IDEA-1541 (dossier `kilo-pass3-2026-11-03.md`: "NUMA-aware chunked work-stealing decode dispatch"; register notes: work-stealing ships, no NUMA-aware chunking guard)
- **Cycle date**: 2026-11-06
- **Verdict**: KILLED on the record, no spike built (dead-idea rule: the mechanism's required precondition is absent on this tree+box). No branch created, no code touched, nothing to revert.
- **Tree state at close**: `git status` clean (only `master` ahead of `origin/master`, now by 11 with the three report commits), stash untouched (`stash@{0}` leftover peer work), nothing pushed (human gate).

## 1. Mechanism (as dossiered)

Deal decode row-bands NUMA-locally (first-touch / node-pinned arenas per socket)
and let the work-stealing pool steal across nodes only on imbalance, so decode
matvec reads stay on the local memory controller instead of crossing the
interconnect. Target files would have been `src/core/thread_pool.cpp` (pool +
`pin_worker_to_p_core`), `src/core/cpu.cpp` (topology), `src/core/dispatch.cpp`
(decode thread count), and the banded kernels (`q8_gemv_avx2.cpp`,
`gemv_avx2.cpp`, `q8_batch_avx2.cpp`).

## 2. Predicted gain (with the math) — and why it is zero here

On a 2-socket box with cross-node bandwidth/latency penalty p and a fraction f
of remote reads, localizing bands saves ≈ f·p of matvec time (decode is
~70–96% matvec). On a single-node box f = 0 by construction, so predicted gain
= **0.0%**, before any implementation cost (topology queries, node masks,
first-touch allocator changes, extra dispatch branches).

## 3. Falsifier (stated for this idea)

Kill if the engine's own topology report shows a single NUMA node (no remote
traffic exists to remove), or if same-run A/B of node-pinned vs unpinned decode
fails to exceed 5% — the first arm already trips, so no A/B is owed.

## 4. File anchors (current `master`, read but untouched)

- `src/core/cpu.cpp` (lines ~147–170): counts `RelationNumaNode`, defaults
  `numa_nodes` to 1; `src/core/thread_pool.cpp` (~line 82): only processor group
  0 pinnable with a plain affinity mask — no NUMA API surface anywhere.
- `src/core/dispatch.cpp` (~line 80): H3 decode default (unpinned, physical
  count) — the standing placement policy this idea would have refined.
- Kernel banding: `gemv_q8_0_avx2()` / `gemv_avx2()` four-bands-per-thread
  contiguous ranges (hardware-prefetcher/TLB-friendly by comment).
- `tge info` CPU section prints `NUMA nodes : <topology.numa_nodes>`.

## 5. What was done (ANALYZE only — no spike per the dead-idea rule)

1. Re-read the dossier concept + register history against the current tree
   (files above): confirmed the gap statement is accurate (no NUMA-aware
   chunking guard exists) but the *precondition* is missing, not the code.
2. Measured the precondition on today's box with the project's own instrument:
   `tge.exe info` → **NUMA nodes: 1**; `Win32_ComputerSystem` →
   NumberOfProcessors **1**, 16 logical; CPU i5-14400F, 10P/16L (6P+4E).
   One socket, one node, one memory controller (single DIMM per the machine
   table) — f = 0, predicted gain exactly zero.
3. Gates for the record: `cmake --build --preset release` clean, `ctest`
   pass, `tge_tests.exe` 141/141; `aikit.py check` 2/4 (test + secrets pass;
   hygiene + guidelines fail pre-existing and unrelated — C++-only repo, README
   drift untouched by this cycle).
4. No thinking-set diff owed: same binary throughout, zero behavior change.

## 6. Measured numbers

| check | result |
|---|---|
| `tge info` NUMA nodes | **1** |
| sockets (`Win32_ComputerSystem.NumberOfProcessors`) | **1** |
| logical processors | 16 (6P+4E, 10 physical) |
| cross-node decode traffic available to remove | 0 bytes/s by construction |
| projected end-to-end gain | 0.0% (f = 0) |

No benchmark A/B run: pinning experiments on one node compare today's banding
against itself. Prior H3 numbers (P-core vs all-physical placement, +40% story)
already cover the placement dimension that exists here.

## 7. Reimplementation recipe (only if a multi-node box ever appears — NOT recommended now)

1. `src/core/cpu.cpp`: extend `CpuTopology` with per-node logical masks
   (one `KAFFINITY`/group mask per `RelationNumaNode`, group-aware beyond 64
   logicals); expose `numa_nodes` + masks via `tge::cpu::detect()`.
2. `src/core/thread_pool.cpp`: first-touch pool arenas per node
   (`VirtualAllocExNuma` or portable fallback), worker→node home assignment,
   steal-local-first with cross-node steal only past an imbalance watermark.
3. Kernels: node-sized super-bands (multiples of today's bands) homed per node;
   keep the contiguous-range discipline inside each node.
4. Gate: 2-socket box, `tge info` ≥2 nodes, same-run decode A/B pinned-home vs
   unpinned >5% best-of-3 + ids identical. Until such a box exists, this stays
   killed.

## 8. Verdict and why

**KILLED without spike.** The idea is correctly scoped but inapplicable: one
NUMA node means no remote reads, no interconnect penalty, and no headroom of
any size — the falsifier's first arm trips at the precondition. Revisit only on
a measured multi-node machine; on this box any NUMA-aware dispatch change would
be dead code around today's banding.

## 9. Exact commands run and their results

- `tge.exe info` → NUMA nodes: 1; 10 physical / 16 logical (6P+4E); kernels avx2-fma.
- `Get-CimInstance Win32_ComputerSystem` → NumberOfProcessors 1, 16 logical.
- `cmake --build --preset release` → 54/54; `ctest --preset release` → pass.
- `python ../../aikit.py check aegean-tge` → 2/4 (test, secrets pass; hygiene, guidelines fail pre-existing).
- No spike branch, no edits, no push.
