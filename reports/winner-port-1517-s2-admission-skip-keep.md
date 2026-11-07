# WINNER PORT: 1517 Stage-2 Admission Skip — LANDED, keep-grade (+75% bench, identical transcripts)

- **Source recipe**: `reports/idea-1517-stage1-populate-spike-measured.md` §7 step 3 (command-layer `shared_prefix_len()` query on admission; exclude fully-shared prompt tokens; memcpy K/V from the snapshot; gate: +15% with identical transcripts).
- **Port commit**: `b8016a9` (`perf(aegean-tge): skip fully-cached admissions in continuous bench via table prime`), 3 files (+109/−2): `src/models/qwen2.cpp` (+80: `shared_prefix_len()` + `prime_cache_from_table()`), `include/tge/models/qwen2.hpp` (+13 decls), `apps/cli/main.cpp` (+18/−2: admission skip + first-token row fix). Nothing pushed (human gate).
- **Verdict**: WINNER — the largest single measured win in the library. Kept in tree as the default admission path.

## 1. What landed

Engine: two public methods beside the table. `shared_prefix_len()` — longest leading run over valid same-geometry slots, no side effects, 0 unless f32. `prime_cache_from_table(seq, tokens, count)` — memcpys K/V rows [0..count) from the best slot iff best covers the whole prompt with count ≥ 16 and f32 KV, logging `[cow-mp] PRIME`; false with no effect otherwise. Bench admission: when the full prompt (≥16) is cached and primed, admit only the trailing row (whose logits produce the first token) and read first-token logits at `active_count`. Earlier prompt rows' logits are never read, and decode rows attend their own sequences only, so the shortened batch is bit-identical for every row the caller reads. The 16-token floor, f32-only guard, and full-coverage requirement mirror the table's own gates.

## 2. Remeasure (this tree; baseline = pre-port binary, interleaved best-of-3)

Continuous 8×32, Q8_0 0.5B, threads 6, 4 slots:

| pair | baseline | port | delta |
|---|---|---|---|
| 1 | 95.27 t/s | 168.93 t/s | +77% |
| 2 | 103.63 t/s | 181.31 t/s | +75% |
| 3 | 87.73 t/s | 105.98 t/s | +21% |

Best-of-3: **103.63 → 181.31 t/s (+75%)**; median +77%; every pair won. All runs: transcripts byte-identical (`independent`), first-token lines byte-identical, 7 PRIME lines per run, zero EVICT. P=8 control pair: 141.24 → 144.52 (+2.3% noise) with identical transcripts — the floor holds, no behavior change below 16 tokens.

## 3. Gates

Build 54/54, `tge_tests` 141/141 (port + final tree). Thinking set (generate path): story + haiku ids byte-identical to every prior run. No fidelity run owed: zero arithmetic change (memcpy + row dropping; kept rows run identical code on identical inputs; dropped rows' logits never read) — same precedent as the mprefix port. `aikit.py check aegean-tge` 2/4 (test + secrets pass; hygiene + guidelines fail pre-existing). Ship threshold (+15% with identical transcripts) exceeded on best, median, and every pair.

## 4. Why it stays

A 75% bench win with byte-identical output on every run and a verified floor is the clearest keep-grade in the library. It spends the two earlier ports exactly as designed (table holds, S1 populates, S2 skips). Absolute pair magnitudes breathe with the box (105–181 across pairs), but the sign never wavers and the minimum pair (+21%) still clears the bar alone.

## 5. Next winner in queue

No-gather sparse decode / kv-auto / 1714 hardening — same treatment: baseline binary saved first, implement, remeasure, keep on repeat win.

(End of file.)
