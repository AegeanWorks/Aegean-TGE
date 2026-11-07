# WINNER PORT: 1517-S1 Batch Populate — LANDED, keep-grade (MATCH, overhead never-negative)

- **Source recipe**: `reports/idea-1517-stage1-populate-spike-measured.md` (spike validated against the retired single snapshot; this port targets the multi-prefix table landed in `dec08a2`).
- **Port commit**: `5010573` (`perf(aegean-tge): populate multi-prefix table from forward_batch stores`), 1 file (`src/models/qwen2.cpp`, +136), temp MATCH driver reverted before commit. Nothing pushed (human gate).
- **Verdict**: WINNER AGAIN on remeasure — kept in tree as the default batch path. memcpy-only, zero arithmetic change.

## 1. What landed

At the end of `forward_batch` (all layers' K/V rows resident): detect a sequence whose batch rows are exactly positions `[0..P)` with `P>=16` (batch-order contiguity + position-0 presence + full-coverage check); build the ordered token list; re-scan the table (populate-only-on-miss — a best run ≥ 16 means the prompt already hit, no populate); else fill an invalid slot first else the LRU slot (`[cow-mp] EVICT` on replacement) and memcpy the resident rows, logging `[cow-mp] POPULATE ... (batch)` with the slot. Guards: f32 KV only. No env gate — the validated spike ran gated, the port runs by default because identity and overhead are proven below. Batch capacity is 64 rows, so the scan is trivial.

## 2. Remeasure (this tree; baseline = pre-port binary, same temp driver both)

MATCH (0.5B-Q8_0): `[cow-mp] POPULATE C=32 slot=0 (batch)` → `[cow-mp] HIT seq=1 C=32 slot=0` → final-logits memcmp **MATCH** vs cold prefill (control without batch path: no batch POPULATE, MATCH trivially — proving the HIT consumed batch bytes).

Overhead A/B, continuous 8×32 P=32, Q8_0 0.5B, threads 6, interleaved best-of-3:

| pair | baseline | port | delta |
|---|---|---|---|
| 1 | 113.49 t/s | 118.30 t/s | +4.3% |
| 2 | 86.63 t/s | 87.76 t/s | +1.3% |
| 3 | 87.94 t/s | 90.48 t/s | +2.9% |

Best +4.3%, median +2.9%; port never slower in any pair; all transcripts identical. (Absolute drift across pairs is box breathing, absorbed by interleaving.)

## 3. Gates

Build 54/54, `tge_tests` 141/141 (port + final tree). Fidelity f32 floor on port: 9.551e-12, 1023/1023. Thinking set (0.5B-Q8_0): story + haiku ids byte-identical to pre-port. `aikit.py check aegean-tge` 2/4 (test + secrets pass; hygiene + guidelines fail pre-existing). Ship threshold (MATCH + overhead <5% + identical ids) passed on all arms.

## 4. Why it stays

Stage-1 is validated twice (spike + port) with a complete recipe: the batch snapshot is consumed by the shipped HIT path with bit-identical logits and costs nothing measurable. This retires the only structural doubt about batch-vs-prefill byte equality on the live table shape. The remaining work is the stage-2 skip with its own falsifier — named, not built.

## 5. Next winner in queue

No-gather sparse decode / kv-auto / 1517 stage-2 skip — same treatment: baseline binary saved first, implement, remeasure, keep on repeat win.

(End of file.)
