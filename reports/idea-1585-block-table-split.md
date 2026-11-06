# IDEA-1585: Block-Table Slot Mapping + KV Block Layout + Tiled Scheduler — SPLIT (headers ship, scheduler ships, wiring specified; 58.3 bar re-grounded at 2x)

- **Idea ID**: IDEA-1585 (dossier `scout-2026-11-goal-vllmpaged.md`: vLLM `block_table.cpp` slot mapping + `cpu_attn_impl.hpp` reshape_and_cache K-column-major/V-row-major per block + `cpu_attn_vec.hpp` AttentionScheduler L2-tiled KV-split with online-softmax ReductionWorkItemGroup; claim: logical→(block,offset) mapping + transposed block layout + tiled scheduler upgrades fixed-4-slot continuous batching to true block-paged mixed batches; layout bit-neutral, scheduler overhead pool-handoff scale)
- **Cycle date**: 2026-11-06
- **Verdict**: SPLIT with a re-grounded bar. (a) Slot mapping (`BlockTable` allocate/share/release + FNV-1a prefix hash + `slot_for`): ships as headers + tests (§§65–66), engine wiring open and specified below — tracked under the 1517 staged recipe, not built here. (b) K-column-major/V-row-major block layout (`kv_block_layout` gather + round-trip): same — design surface ships, Qwen2 cache untouched by design. (c) L2-tiled KV-split scheduler with online softmax: SUPERSEDED — ships natively as chunked GQA (`attention_batch` per-chunk online-softmax states + `merge_gqa_chunks`) + 64-row prefill activation blocking + L2-sized GEMM blocking. (d) Falsifier bar `bench continuous` 8x32 ≥58.3: RE-GROUNDED — today's tree does 101–113 on the bar's own shape (0.5B-Q8_0), nearly 2x; the bar is obsolete and replaced below. First Llama-3B 8x32 reference recorded (19.2–19.9, tight). No spike (headers done, scheduler ships, wiring is a staged engine change with its own gates — building a fragment here would fork the 1517 track). No branch, no code touched, nothing to revert.
- **Tree state at close**: `git status` clean (no diff, no untracked), stash unchanged (only the pre-existing `leftover peer work shelved 2026-11-04`, untouched), `master` ahead of `origin/master` by 55 before this report. Nothing pushed (human gate).

## 1. Mechanism (as dossiered, per third)

Slot mapping: logical position → (block, offset) via refcounted block tables with prefix-hash sharing, enabling paging/eviction and mixed prefill-decode batches beyond fixed slots. Block layout: K written column-major, V row-major per block (BlockSizeAlignment=32) so sequential QK/PV sweeps stay vector-friendly — bit-neutral reorder. Tiled scheduler: L2-derived tile sizes, GQA head-group packing, KV split across threads with online-softmax reduction. Distinct within the paged theme (dossier: complements 1384 compression+sharing, 1517 hash-partitioned prefill, 1553 prefetch policy — not a dupe).

## 2. Predicted gain — and the per-third disposition

| third | disposition | grounds |
|---|---|---|
| slot mapping | SHIPS as headers; wiring specified (§8), 1517-tracked | `BlockTable` + tests, zero engine calls (by design — §§65–66) |
| block layout | SHIPS as helpers; same wiring | `kv_block_layout` gather + exact round-trip, engine cache untouched |
| tiled scheduler + online softmax | SUPERSEDED (ships natively) | `attention_batch` chunking + `merge_gqa_chunks` + 64-row blocking + 144/256/256 L2 GEMM tiles |
| >32k context unlock | CAPPED by design | `kHardMaxContext = 32768` (exactly 32k, not more — deliberate) |
| 8x32 ≥58.3 bar | RE-GROUNDED at ~2x | §6: 101–113 on the bar's own shape |

## 3. Falsifier (from the dossier) — resolved per arm without building

"Same-run 2048-decode drops >5% vs GQA baseline" — moot: no second scheduler exists to A/B (the scheduler third IS the GQA path). "Fidelity KL vs scalar >1e-9" — moot: layout is a bit-neutral reorder (helpers exact round-trip 135/135) and the scheduler prices at 5e-12 (GQA differential floor). "`bench continuous` 8x32 aggregate <58.3" — PASSES with ~2x margin on re-grounding (§6): worst run 101.4 > 58.3. The remaining genuinely-open falsifier is the future wiring round's: engine-wired paged KV must hold the REPLACEMENT bar (§8) with identical ids — specified, not run (no wiring exists to run it against).

## 4. File anchors (current `master`, read but untouched)

- `include/tge/models/block_table.hpp` (allocate/share/release refcount, `slot_for` logical→(block,offset), FNV-1a `prefix_hash`), `include/tge/models/kv_block_layout.hpp` (K-column-major/V-row-major gather, exact round-trip), `tests/test_block_table.cpp` (135/135 per §65 — headers + tests, zero engine calls).
- `src/models/qwen2.cpp:26` (`kHardMaxContext = 32768`), `:1492` ("sequence outside the slots" — fixed-slot scheme the mapping would generalize), `:440-476` (flat per-slot `k_cache_/v_cache_` family — the layout the block gather would replace), `:1294-1403` (`attention_batch`: KV-head × chunk tasks, per-chunk online-softmax states, `merge_gqa_chunks` — the shipped KV-split tiled scheduler).
- `src/cpu/avx2/gemm_avx2.cpp:26-29` (kMc/kKc/kNc 144/256/256 — L2-sized tiles), `src/models/qwen2.cpp:28-29` (64-row prefill activation blocking).
- `apps/cli/main.cpp` (`bench continuous`: `--requests/--tokens/--sequences/--context/--prompt-tokens/--kv`; defaults 8/32/4 = the bar's shape).
- Precedents: `idea-1517-stage1-headroom-proceed.md` (staged paged recipe — the wiring's track), `idea-1553-paged-prefetch-split.md` (sibling third, different mechanism).

## 5. What was done (ANALYZE + bar grounding — no spike per the staged-track rule)

1. Read the dossier body against the current tree; split per third (§2), keeping the dossier's complements-not-dupes map intact (1384/1517/1553 cross-referenced, none re-judged).
2. Carrier inventory (§4): headers ship unwired by design; scheduler ships natively; fixed slots + flat cache + 32k cap confirmed in-engine.
3. Gates for the record: `cmake --build --preset release` clean (54/54); `ctest --preset release` pass; `tge_tests.exe` 141/141; `python aikit.py check aegean-tge` 2/4 (test + secrets pass; hygiene + guidelines fail pre-existing).
4. Grounded the bar (§6): wall-gated `bench continuous` 8x32 best-of-3 on 0.5B-Q8_0 (the bar's shape) + first Llama-3B 8x32 reference. No fidelity run owed (no new numerics; bit-neutral + priced floors); no thinking-set diff owed (same binary).
5. No branch, no edits; tree verified clean + stash unchanged at close.

## 6. Measured numbers (wall-gated window: 58.58 → 56.22 read GB/s, healthy throughout)

| run | aggregate decode |
|---|---|
| 0.5B-Q8_0 8x32 R1/R2/R3 | 101.39 / 113.43 / 108.62 t/s (best 113.4) |
| Llama-3B-Q4_K_M 8x32 R1/R2/R3 | 19.21 / 19.21 / 19.92 t/s (best 19.9, ±2%) |
| §25 bar (pinned era, same shape) | 58.28 t/s |

The bar is obsolete: worst run today (101.4) clears it by 74%. The margin decomposes to shipped work since §25: unpinned decode pools (§28ab), physical-core default (§69), CoW prefix (§70/CoW), SKIP-locked fast paths — i.e., the scheduler the dossier wants to port has been outbuilt by the native one. Replacement bar for the wiring round (§8): 0.5B-Q8_0 8x32 worst-of-window ≥100 AND Llama-3B 8x32 ≥19 (tight), both wall-gated, identical ids.

## 7. Full reimplementation recipe (the wiring round — specified, NOT built; next staged step after 1517)

1. Slot allocation: route `forward_batch` sequence admission through `BlockTable` (acquire on new sequence, share on prefix-hash hit, release on finish) replacing the fixed-slot index; keep flat caches initially — pure slot-mapping diff, gate on identical ids + replacement bar (§6).
2. Block layout: repoint the K/V store + gather sites (`:1542-1553, :1705-1716, :1922-1933` f32/f16/i8 triples + attention gather) at `kv_block_layout` K-column-major/V-row-major blocks with `slot_for` indexing; gate on the exact round-trip tests first, then fidelity ≤5e-12 vs scalar + identical long-context ids (sparse + CoW interplay explicit).
3. Mixed batches: extend `bench continuous` admission to prefill-decode mixing over the block pool (the dossier's 0.7 half); gate on the replacement bar with mixed shapes, not just 8x32.
4. Never: torch-bound code (algorithm port only, per dossier clean-room note), BlockSizeAlignment=32 literalism (native 64-B/16-wide blocking already L2-fitted — re-derive, don't transcribe), >32k context (capped by `kHardMaxContext` design decision — lift only with its own harness round). To re-verify this split: re-run the §4 inventory (headers still unwired, scheduler still native) and the §6 pair (bar still cleared).

## 8. Verdict and why

**SPLIT: headers ship, scheduler ships, bar re-grounded at 2x, wiring specified for its own staged round.** The dossier's implementable surface is already 80% in-tree in native form; the remaining 20% (engine-wired paged KV + mixed batches) is a multi-file engine change that belongs to the 1517 staged track with the replacement bar in §6 — building a fragment here would fork that track for no measurable gain. The loop's honest output is the map plus current numbers, so the wiring round starts from grounds, not guesses.

## 9. Exact commands run and their results

- Carrier inventory (§4 greps + reads): headers present/unwired; scheduler native (chunking + online-softmax merge + 64-row blocking + L2 tiles); fixed slots + flat cache + 32k cap in-engine.
- `cmake --build --preset release` → 54/54; `ctest --preset release` → pass; `tge_tests.exe` → 141/141.
- `python aikit.py check aegean-tge` → 2/4 (pre-existing hygiene + guidelines fails).
- `bench continuous --requests 8 --tokens 32 --threads 6`: 0.5B-Q8_0 → 101.39/113.43/108.62; Llama-3B → 19.21/19.21/19.92 (walls 58.58 → 56.22 bracketing).
- No branch, no edits, no fidelity (moot arms), no push.

(End of file.)
