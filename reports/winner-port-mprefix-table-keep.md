# WINNER PORT: Multi-Prefix CoW Table — LANDED, keep-grade (1.72x prefill-side, identical ids)

- **Source recipe**: `reports/idea-invented-mprefix-table-proceed.md` (S4 parked trigger retired there).
- **Port commit**: `dec08a2` (`perf(aegean-tge): multi-prefix CoW table replaces first-wins single snapshot`), 1 file (`src/models/qwen2.cpp`, +158/−84), temp measure driver reverted before commit. Nothing pushed (human gate).
- **Verdict**: WINNER AGAIN on remeasure — kept in tree as the default prefill path. memcpy-only, zero arithmetic change.

## 1. What landed

4-slot `MultiPrefix` table (tokens ≤256, per-layer K/V rows, geometry tags, LRU `last_use`, `valid`) beside the retired single snapshot (struct + HIT + POPULATE + load-clear all replaced). HIT: longest leading-token run across valid same-geometry slots, iff best ≥ 16 AND prompt longer than the run; memcpy K/V into the sequence slot, bump LRU, log `[cow-mp] HIT ... slot=N`; else log `[cow-mp] MISS seq best slot`. Populate-only-on-miss: re-scan; best ≥ 16 means the prompt already hit (no populate, no slot waste); true miss with ≥ 16 prompt tokens fills an invalid slot else the LRU slot (`[cow-mp] EVICT` on replacement — zero occurred in all runs). Guards unchanged from shipped CoW: f32-only, `start_position==0`, `sequence>=0`, `all_logits==nullptr`. The `[cow-mp]` lines stay in the port as the attribution instrument.

## 2. Remeasure (this tree; baseline = pre-port binary + same temp driver, both reverted after)

0.5B-Q4_K_M, 8×(120+32) two-prefix cohort, threads 6, interleaved best-of-3:

| run | single total | table total | pattern |
|---|---|---|---|
| 1 | 2936.7 ms | 1750.4 ms | 2 POP + 6 HIT, 0 EVICT |
| 2 | 2963.2 ms | 1707.1 ms | same |
| 3 | 2929.6 ms | 1850.2 ms | same |

Best-of-3: **2929.6 → 1707.1 ms (1.72x prefill-side)**. Every MISS a first-touch populate; per-seq ids byte-identical all 6 runs (`198 198 198 220 6565 406 220 82`). Llama-3.2-3B-Q4_K_M one pair: **16.06 → 9.65 s (1.66x)**, ids identical (`198 3215 8 198 198 220 315 198`), 2 POP + 6 HIT, zero EVICT.

## 3. Gates

Build 54/54, `tge_tests` 141/141 (port + final tree). Fidelity f32 floor on port binary: 9.551e-12, 1023/1023 (path arithmetic untouched). Thinking set (0.5B-Q8_0): story + haiku ids byte-identical to pre-port. `aikit.py check aegean-tge` 2/4 (test + secrets pass; hygiene + guidelines fail pre-existing). Ship threshold from recipe (≥6/8 HIT with zero unexplained MISS) exceeded: 6/6 eligible hits, all misses first-touch populates.

## 4. Why it stays

Repeat win in the predicted shape on two models with full attribution and zero id movement. The single snapshot is retired (not gated) per the recipe — dual paths caused the S4 confusion, and the table covers the single's use case identically (slot 0 behaves as the old snapshot).

## 5. Notes for the next ports

- 1517-S1 batch populate (validated spike) should target this table's shape when ported (populate from `forward_batch` stores), not a separate snapshot.
- Next winner in queue: no-gather sparse decode / kv-auto / 1517 stage-2 skip — same treatment: baseline binary saved first, implement, remeasure, keep on repeat win.

(End of file.)
