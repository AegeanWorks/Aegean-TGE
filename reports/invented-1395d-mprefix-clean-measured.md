# IDEA-1395d / S4-clean: multi-prefix table re-spike (populate-only-on-miss, single retired) — MEASURED: S4 misses explained, 6/6 hits, byte-identical

- **Idea ID**: S4 clean re-spike (parked trigger from `docs/measurement-log.md` §S4: clean re-spike with populate-only-on-miss, single retired, untruncated stderr that explains the misses first).
- **Cycle date**: 2026-11-07
- **Verdict**: MEASURED — the S4 misses are explained as spike bugs, not structural: populate-on-hit slot churn evicted prefixes before reuse (→ full compute), and running alongside the non-retired single caused fall-throughs. Clean re-spike: 8/8 sequences resolve correctly (2 populates + 6 hits, longest-run wins, zero unexplained), per-seq FNV-1a identical to cold recompute on all 8, default-path story ids unchanged. Keep-grade: proceed to KL-pricing/next (S6-style amendment or integration proposal).
- **Tree state at close**: `master`, `git status` clean, spike branch created and deleted with nothing committed on it, stash untouched (pre-existing peer stash 2026-11-04 only). Final gates on master: build clean, `tge_tests` 141/141. Nothing pushed (human gate).

## 1. Mechanism (as spiked)

4-entry f32 snapshot table (`PrefixEntry`, file-static in `qwen2.cpp`). Lookup: longest common run across valid same-geometry entries, hit needs run ≥ 16 and prompt longer. Populate: ONLY on lookup miss (best < 16) into first-invalid else round-robin slot, capped at 256 rows. Single-prefix path retired while `TGE_MPREFIX=1` (both its hit and populate blocks gated off) so every HIT/MISS is the table's. Untruncated stderr per sequence: LOOKUP (count/best/entry/runner-up/valid-mask), HIT (C/entry), POPULATE (C/slot/best/runner), MISS with reason (`hit-consumed` vs `short-prompt`).

## 2. Falsifier (from the parked trigger)

Misses reproduce without explanation on the clean re-spike → structural, re-kill. Did not trip: all 8 sequences either populated (2) or hit (6); the only MISS lines are the populate-side's own correct no-op logs on hit-consumed sequences.

## 3. File anchors (current `master`, untouched after revert)

- `src/models/qwen2.cpp`: SharedPrefix struct ~:45-59, `mprefix_best()` helper + table (spike-only), load clear ~:609-640, table lookup inserted before single-hit ~:1825, single-hit gate `&& !g_mprefix_on`, table populate + single-populate gate ~:2010-2130, `reset_cache` untouched (table is cross-sequence by design, cleared per load).
- `tests/test_mprefix_spike.cpp` (temp, reverted): 8-seq driver (prefix A ids 100-219 ×4, B 5000-5119 ×4, 32-token suffixes 20000+), `Qwen2::load(..., 2048, false,false,false, 8)` + `prefill()` per seq, FNV-1a over returned logits, table-vs-cold compare. Registered in `CMakeLists.txt` `tge_tests` list (reverted).
- Prior art: `docs/measurement-log.md` §S4 (misses), §S6/1517 (snapshot context).

## 4. What was done (temporary spike, fully reverted)

1. `git checkout -b spike-1395d-mprefix`; 5 edits (table+helper+load/env/clear, lookup, 2 single gates, populate) + temp test + 1 CMakeLists line. Two self-inflicted detours fixed: driver first used `forward_batch` (no table on that path; capacity throw) — switched to public `prefill()`; env leak on throw poisoned later tests' stderr — wrapped body in try/catch with unset+rethrow.
2. `cmake --build --preset release` clean; `tge_tests` **142/142** (141 + driver).
3. Driver evidence (0.5B, f32, 8×152 prefill): seq0 POPULATE slot0 (A); seq1-3 HIT C=120 entry0; seq4 best=0 → POPULATE slot1 (B); seq5-7 HIT C=120 entry1. Runner-up 0 throughout (prefixes disjoint by construction). FNV table-vs-cold equal all 8 seqs.
4. Thinking-set gate: fox-loop story ids on spike binary (env unset) `578 39935 35308 13` ×4 — identical on clean master after revert (default path provably untouched: table code unreachable when env off, single gates short-circuit on `!g_mprefix_on`).
5. `python aikit.py check aegean-tge` → 2/4 (test + secrets pass; hygiene + guidelines fail pre-existing on master, same record as Cycles 1-4).
6. Reverted (`git checkout -- .`, deleted temp test file, `git checkout master`, `git branch -D`), rebuilt master clean, `tge_tests` 141/141, `git status` clean.

## 5. Measured numbers

| seq | prompt | table decision | evidence |
|---|---|---|---|
| 0 | A+ suf0 | POPULATE slot 0 (C=152) | best=0 (empty table) |
| 1-3 | A+ suf1-3 | HIT C=120 entry 0 | longest run wins |
| 4 | B+ suf4 | POPULATE slot 1 | best=0 (B unseen) |
| 5-7 | B+ suf5-7 | HIT C=120 entry 1 | longest run wins |

FNV-1a table vs cold: 8/8 equal (e.g. seq0 2677932439339287919, seq6 9074218077129842766). Timing (0.5B, not a benchmark): full driver (16 prefills + 2 loads) inside suite run, no timeout. Saved tokens: 720 (6×120) of 1216 prompt tokens ≈ 59% prefill-side recompute avoided on this pattern.

## 6. Verdict

Keep-grade, proceed to pricing: misses explained (S4's populate-on-hit churn + non-retired single), table path byte-identical, default path untouched. Next: S6-style KL pricing + kind-tagging (f16/i8 tiers explicitly excluded here — table is f32-only like the single), or an integration proposal (4-entry static table + populate-only-on-miss + retired single behind a runtime flag, not env).

## 7. Reimplementation recipe

1. `qwen2.cpp`: `PrefixEntry` + `g_mprefix[4]` + `g_mprefix_next/hits/saved/on` + `mprefix_best()` per §3; load: clear table + read `TGE_MPREFIX==1`; gate both single blocks with `!g_mprefix_on`; insert lookup block before single-hit and miss-only populate block before single-populate (§4 pattern, memcpy loops mirror single's with per-entry `cached` stride).
2. Temp driver `tests/test_mprefix_spike.cpp` per §4 (Qwen2.5-0.5B, 8 seqs, `prefill()`, FNV over returned vocab row, table-vs-cold TGE_CHECK, try/catch env cleanup) + one CMakeLists line; revert both after measuring.
3. Gate: suite green (142/142 with driver), FNV 8/8 equal, fox ids spike-vs-master identical, aikit 2/4 pre-existing.

## 8. Citation verification

No external citations (invented re-spike of a logged session). Internal: `measurement-log.md` §S4 trigger quoted verbatim in §2; single-prefix behavior comments at `qwen2.cpp:35-44` match the retired code; driver design mirrors the S4 temp-driver description (8 seqs, 2×120+32).

## Commands run and their results

- `git checkout -b spike-1395d-mprefix` → switched; 5 spike edits + temp test + CMakeLists line; `cmake --build --preset release` → clean; `tge_tests` → 142/142 (driver FNV 8/8 equal).
- `tge generate` fox story (spike binary, env unset) → `578 39935...` ×4; same on reverted master → identical.
- `python aikit.py check aegean-tge` → 2/4, pre-existing fails.
- Revert (`checkout -- .`, `rm` temp test, `checkout master`, `branch -D`) → master, clean (verified); rebuild + `tge_tests` 141/141.
