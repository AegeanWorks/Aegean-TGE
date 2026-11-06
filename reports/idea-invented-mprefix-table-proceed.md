# INVENTED: Multi-Prefix CoW Clean Re-Spike — PROCEED (S4 trigger retired: 2 populates + 6 hits, zero unexplained, ids identical)

- **Idea**: Invented follow-up executing the exact parked trigger from the verify-then-build session (S4): clean re-spike of the multi-prefix snapshot table with populate-only-on-miss, the single snapshot retired (within the spike run), and untruncated stderr so every MISS is attributable. The shipped `g_shared_prefix` is first-prefix-wins (`tokens.empty()` gate), so a second distinct shared prefix can never populate and falls to full recompute — the gap, reproduced on today's tree before the fix.
- **Cycle date**: 2026-11-06
- **Verdict**: PROCEED — the table mechanism is validated on two models with a complete port recipe (§7). Porting (table as default, single retired, env gate removed) is a named human-gated follow-up, not this cycle: zero spike code survives.
- **Tree state at close**: `git status` clean, no diff, stash unchanged (only the pre-existing `leftover peer work shelved 2026-11-04`). Spike branch `spike-mprefix-clean` created (zero commits) and deleted; both spike files reverted with empty diff verified. `master` ahead of `origin/master` by 90 after this report. Nothing pushed (human gate).

## 1. Mechanism (as spiked)

`src/models/qwen2.cpp` only (default path untouched; table active solely under `TGE_MPREFIX_TABLE=1`):

- New `MultiPrefix` table (`g_mprefix_table[4]`: tokens ≤256, k/v rows, layers, kv_dim, LRU `last_use`, `valid`), plus `g_mprefix_hits` / `g_mprefix_populates` counters and an env reader caching `TGE_MPREFIX_TABLE==1`.
- HIT block: longest leading-token run across valid same-geometry slots; HIT iff best ≥ 16 AND prompt longer than the run (mirrors the single-snapshot floor). memcpy per-layer K/V rows into the sequence slot, skip recompute, bump LRU, log `[cow-mp] HIT ...`; else log `[cow-mp] MISS seq best slot`.
- POPULATE block: re-scan the table; a best run ≥ 16 means this prompt already HIT above, so **no populate** (the fix for the prior spike's populate-on-hit slot waste). On a true miss with ≥ 16 prompt tokens, fill an invalid slot else the LRU slot, logging `EVICT` on replacement (zero occurred), then `[cow-mp] POPULATE ...`.
- Existing single-snapshot HIT/POPULATE conditions each gained `&& !mprefix_table_enabled()`; the load-time clear block also resets the table. Temp driver: `spike-mprefix` subcommand in `apps/cli/main.cpp` (public `prefill` API only) — 8 sequences, seqs 0–3 share 120-token prefix A, seqs 4–7 share distinct 120-token prefix B, each with a 32-token unique suffix; prints per-seq prefill ms + argmax id.

## 2. Predicted gain — landed

Two prefixes in one process should cost 2 full prefills + 6 suffix-only hits instead of 8 full prefills. Measured best-of-3 (interleaved, §6): 0.5B cohort 4201.2 → 2256.1 ms (**1.86x prefill-side**); second-prefix cohort alone 3033.5 → 1126.5 ms (**2.69x**); Llama-3B cohort 27.9 → 16.3 s (**1.71x**), second-prefix cohort 20.4 → 4.1 s (**4.93x**).

## 3. Falsifier outcome — all three arms pass

1. **Identity**: per-seq argmax ids byte-identical single-vs-table on every run, both models (0.5B: `8/198/334/13/220/13/13/1870` ×8 runs; Llama-3B: `27/198/567/27/220/220/220/220` ×2 runs).
2. **Hit rate**: 6/8 HIT + 2/8 POPULATE on all four table runs (3× 0.5B + 1× Llama-3B); per-seq hit ~140–170 ms vs full ~640–720 ms on 0.5B.
3. **Attribution**: every MISS maps to a first-touch populate (seq0 empty table, seq4 first B touch); zero EVICT lines; zero MISS on a previously populated prefix — the S4 "unexplained miss" failure mode did **not** reproduce. Verdict: not-killed, proceed-grade.

## 4. File anchors (master = pre-spike; spike reverted)

- `src/models/qwen2.cpp` (≈L45–55 struct/table, ≈L605–615 clear, ≈L1824–1867 single HIT gated, ≈L2000–2033 single POPULATE gated, new table HIT + populate-only-on-miss blocks appended after each).
- `apps/cli/main.cpp` (temp `command_spike_mprefix` + `spike-mprefix` dispatch with `--table` — reverted).
- Grounds: verify-then-build S4 entry in `docs/measurement-log.md` (the parked trigger); shipped single-prefix recipe (`idea-1384-qcow-split.md` §-anchors).

## 5. What was done (all temporary, fully reverted)

1. Committed the prior session's pending `docs/measurement-log.md` entry docs-only (tree was dirty on entry; now `3f3a84a`), so this cycle started clean.
2. `git checkout -b spike-mprefix-clean` (zero commits, deleted at close).
3. Spike edits (§1: 4 qwen2.cpp regions + temp driver). `cmake --build --preset release` clean (54/54); `tge_tests.exe` 141/141 on the spike binary.
4. Box health: `bench mem` read 60.27 GB/s (reference band 54–61).
5. Gap reproduction on today's logic (single mode): seq0 POPULATE A + seqs1–3 HIT; seqs4–7 full 152-token compute, no populate — first-wins confirmed empirically.
6. A/B: 1 warm + 3 measured interleaved pairs (single/table) on Qwen2.5-0.5B-Q4_K_M, 6 threads, warm page cache; 1 pair on Llama-3.2-3B-Q4_K_M. Full stderr captured to files, never truncated.
7. Thinking set on spike binary (0.5B-Q8_0): story + creative `--print-ids` saved; re-ran on reverted master binary: **ids byte-identical on both prompts** (only timing lines differ).
8. `aikit.py check aegean-tge` → 2/4 (test PASS; hygiene + guidelines fail pre-existing, same as prior cycles).
9. Reverted both files (`git checkout --`, empty diff), `git checkout master`, `git branch -D`, rebuilt master clean (54/54), `tge_tests` 141/141, `git status` clean, stash untouched, scratch outputs deleted.

## 6. Measured numbers

Qwen2.5-0.5B-Q4_K_M, 8×(120+32), `--threads 6`, best-of-3 interleaved (ms/seq):

| run | s0 | s1 | s2 | s3 | s4 | s5 | s6 | s7 | total |
|---|---|---|---|---|---|---|---|---|---|
| single best (s2) | 700.0 | 162.2 | 152.5 | 153.0 | 652.8 | 1093.7* | 653.4 | 633.6 | **4201.2** |
| table best (t4) | 702.7 | 148.6 | 141.6 | 136.7 | 633.8 | 170.0 | 142.5 | 180.2 | **2256.1** |

\* box-noise outlier (+52%); best-of-best per loop rule absorbs it — the other single B-cohort seqs sit 633–656 ms. Table runs: t2 2256.8, t3 2491.0 — pattern identical (2 POPULATE + 6 HIT, zero EVICT) all 3×.

Llama-3.2-3B-Q4_K_M, one pair (s):

| run | s0 | s1–s3 (hits) | s4 | s5–s7 | total |
|---|---|---|---|---|---|
| single | 4395.7 | 1032.7/1032.6/1028.2 | 4889.5 | 4986.5/5230.5/5285.8 | 27881.5 |
| table | 4807.9 | 1178.3/1114.5/1094.0 | 5113.4 | 1058.1/862.9/1105.5 | 16334.6 |

No fidelity run owed: zero arithmetic change (memcpy of same-token/same-position rows); determinism gates (§3.1 + thinking set) substitute, as precedented.

## 7. Full reimplementation recipe (for the human-gated port — NOT this cycle)

1. In `src/models/qwen2.cpp`: add the `MultiPrefix` table (§1) beside `SharedPrefix`; gate the two single-snapshot blocks on `!table_active` (or delete them if the single is retired at port time — recommend retire: dual paths caused the S4 confusion).
2. Add the table HIT scan (longest run, floor 16, prompt-longer-than-run) with per-(layer,pos) memcpy using the existing `(seq,layer,pos)` dst / `(layer,cached,pos)` src index math; add populate-only-on-miss (re-scan; skip when best ≥ 16; invalid-slot-first else LRU; log EVICT).
3. Extend the load-time clear to the table. Keep f32-only + `start_position==0` + `sequence>=0` + `all_logits==nullptr` guards (the S6 kind-tagging question — float-vs-integer snapshot sharing — stays out: this table is f32-prefill only, same as shipped).
4. Keep the `[cow-mp]` log lines in the port (they are the attribution instrument; cost is one integer scan + fprintf per prefill).
5. Gate the port: rebuild 54/54, `tge_tests` 141/141, rerun the §6 8-seq pair on 0.5B (expect 2 POPULATE + 6 HIT, ids identical to pre-port binary), one Llama-3B pair, story/creative id-identity, `aikit.py check`. Ship threshold: ≥6/8 HIT with zero unexplained MISS, else revert the port.

## 8. Verdict and why

**PROCEED: the S4 parked trigger is retired with a passing measurement.** Populate-only-on-miss + single-retired + untruncated logging produced deterministic 2-populate/6-hit runs on two models, ids identical everywhere, zero evictions, zero unexplained misses — the exact failure modes that killed S4 did not reproduce. The port recipe (§7) is the follow-up; it needs human review (default-path change + ADR-worthy thresholds), so it is named, not built. Next-cycle candidates (untouched by this round): S6 kind-tagged snapshots (epic-owned), S2 pre-packed-B retry on a quiet box, S3 tiled-attention with a vectorized inner loop.

## 9. Exact commands run and their results

- `git add docs/measurement-log.md` + `commit "docs(aegean-tge): verify-then-build session..."` → `3f3a84a` (prior session's entry preserved; tree clean).
- `git checkout -b spike-mprefix-clean` → branch (zero commits, deleted at close).
- Spike edits (§1); `cmake --build --preset release` → 54/54; `tge_tests.exe` → 141/141.
- `tge.exe bench mem` → read 60.27 GB/s (healthy).
- `tge.exe spike-mprefix --model <0.5B-Q4_K_M> --threads 6` (×4) + `--table` (×4), interleaved, stderr to files → §6 table; log grep: MISS only first-touch, zero EVICT.
- `tge.exe spike-mprefix` ± `--table` on `<Llama-3.2-3B-Q4_K_M>` → §6 Llama table; identical log pattern, ids identical.
- Thinking set (0.5B-Q8_0, spike binary): story ids `576 3974 ... 1854 13`, creative ids `758 264 ... 23607 11`; re-run on reverted master: **both byte-identical**.
- `python aikit.py check aegean-tge` → 2/4 (test PASS; hygiene + guidelines pre-existing fails).
- `git checkout -- <2 files>` → clean; `git checkout master`; `git branch -D spike-mprefix-clean`; rebuild 54/54; `tge_tests` 141/141; `git status` clean; stash list unchanged; scratch files deleted; report committed; no push.

(End of file.)
