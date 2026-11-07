# IDEA-1517 STAGE-1: Populate-from-Batch Snapshot — MEASURED, PROCEED (bytes identical, overhead ~0%)

- **Idea ID**: IDEA-1517 (dossier `scout-filing-2026-aegean-tge.md`: vLLM-style paged KV + token-hash prefix lookup + chunked prefill; falsifier: on the 8x32 bench-continuous load, aggregate <58.3 t/s baseline or any free-running id divergence). This cycle builds **stage 1** of the staged recipe from `idea-1517-stage1-headroom-proceed.md`: track per-sequence stored prefix in `forward_batch`'s store loop and snapshot a full prompt [0..P) for the staged skip. No skip yet (stage-2); this stage proves the bytes match a cold `prefill_run` snapshot with zero overhead.
- **Cycle date**: 2026-11-07
- **Verdict**: PROCEED (stage-1 gate passed on all arms). The batch snapshot is byte-usable by the shipped CoW HIT path (final-logits memcmp MATCH), populate overhead is unmeasurable inside box noise (median -1.2%, best +1.8%, kill bar was >5% regression), transcripts identical, fidelity and thinking set unchanged. Porting (populate as default, no env gate) plus the stage-2 skip is the named human-gated follow-up, not this cycle: zero spike code survives.
- **Tree state at close**: `git status` clean, no diff, stash unchanged (only the pre-existing `leftover peer work shelved 2026-11-04`). Spike branch `spike-1517-s1-batch-populate` created (zero commits) and deleted. Both spike files reverted with empty diff verified; rebuilt master 54/54, tests 141/141. `master` ahead of `origin/master` by 143 after this report. Nothing pushed (human gate).

## 1. Mechanism (as spiked, reverted)

`src/models/qwen2.cpp` only (default path untouched; active solely under `TGE_BATCH_PREFIX=1`):

- After `forward_batch`'s layer loop (all layers' K/V rows resident in `k_cache_`/`v_cache_`), scan the batch for a sequence whose rows are exactly positions `[0..P)` with `P>=16` (batch-order contiguity check plus a position-0 presence guard plus a full-coverage `ordered[]` fill check). On a hit with an empty snapshot and f32 KV only, copy tokens plus per-layer K/V rows into `g_shared_prefix` with the exact `prefill_run` populate memcpy shape (cap 256), and log `[batch-prefix] POPULATE`.
- Temp driver `spike-batchprefix` in `apps/cli/main.cpp` (public API only): model A loads (clears snapshot) → `forward_batch` 32 prompt rows (id 128, positions 0..31, seq 0) → `prefill` 40 tokens (32 shared + 8 new) on seq 1 → record final logits; model B loads fresh (clears snapshot) → cold `prefill` same 40 tokens → memcmp final logits → print MATCH/MISMATCH. The seq-1 prefill is the shipped CoW HIT path consuming the batch snapshot, so MATCH proves byte-identity end to end.

## 2. Predicted gain — stage-1 carries no win by design

Stage-1 adds one scan per `forward_batch` call plus a single populate memcpy per process (snapshot goes non-empty after the first admission). Predicted: overhead ~0% (one `getenv` per batch call, one O(count^2) scan with count<=64, one 32x24x128-float memcpy ≈ 0.4 MiB), correctness FNV/MATCH gate. The win it unlocks is stage-2's skip (17-44% of wall at P=8..32 per the headroom report). Stated falsifier: kill if (a) batch-vs-cold logits MISMATCH, (b) continuous overhead >5% best-of-3, or (c) any ids diverge.

## 3. Falsifier outcome — all arms pass

1. **Identity**: `[batch-prefix] POPULATE seq=0 C=32` then `[cow-prefix] HIT seq=1 C=32` then `MATCH (vocab 151936)`. Control (env unset): no batch POPULATE, no HIT, both models cold-POPULATE C=40, still MATCH — proving the HIT came from the batch snapshot.
2. **Overhead**: 8x32 P=32 continuous, interleaved best-of-3, Q8_0 0.5B, threads 6 (§6 table). Median -1.2%, best +1.8% — sign flips between pairs, textbook box noise, far under the 5% kill bar.
3. **No collateral**: all transcripts identical (`independent`); fidelity 9.6e-12 / 1023-1023; story + haiku ids byte-identical spike-vs-master.

## 4. File anchors (master = pre-spike; spike reverted)

- `src/models/qwen2.cpp` (≈L45-55 `SharedPrefix` + cap 256; ≈L609-615 load-time clear; :1468-1661 `forward_batch` store loop — zero snapshot refs, the gap; :1819-2033 `prefill_run` HIT + populate — the reused shape). Spike added: `<cstdlib>`, `batch_prefix_populate_enabled()`, the populate block after the layer loop — all reverted.
- `apps/cli/main.cpp` (`command_continuous_bench` admits prompts via `forward_batch` only; temp `command_spike_batchprefix` + dispatch — reverted).
- Grounds: `reports/idea-1517-stage1-headroom-proceed.md` (staged recipe + re-based bars), `reports/idea-1384-qcow-split.md` (CoW byte-identity precedent), `reports/idea-invented-mprefix-table-proceed.md` (populate-only-on-miss + HIT-vs-cold pattern).

## 5. What was done (temporary spike, fully reverted — with one collision)

1. `git checkout -b spike-1517-s1-batch-populate` (zero commits, deleted at close).
2. First attempt stood down mid-cycle: a second live spike (`spike-1395-tail`, TGE_KV_TAIL) plus a fresh master commit shared the tree ("never two spikes alive at once"). My hunks reverted, HEAD back on master, snapshots + resume note parked outside the repo, 60m retry wakeup armed, board notified. Retry found the tree clean (peer 1395 cycle closed with 5 reports, fully reverted, branch gone) and resumed here — this report covers the resumed run only.
3. Spike edits (§1: qwen2.cpp 3 hunks + main.cpp driver + dispatch). `cmake --build --preset release` clean (54/54); `tge_tests.exe` 141/141 on the spike binary.
4. Box health: `bench mem` read 58.89 GB/s (best band of the week; copy 53.62).
5. MATCH check (§6.1) on 0.5B-Q8_0 + control; continuous A/B 3 interleaved pairs (§6.2); `fidelity` on spike binary (§6.3); thinking set spike-vs-master (§6.4); `aikit.py check aegean-tge` → 2/4 (test + secrets pass; hygiene + guidelines fail pre-existing, same as every prior cycle).
6. Reverted both files (`git checkout --`, empty diff), `git checkout master`, `git branch -D`, rebuilt master clean (54/54), `tge_tests` 141/141, `git status` clean, stash untouched, scratch outputs (outside repo) kept as the resume trail.

## 6. Measured numbers

### 6.1 Snapshot cross-check (0.5B-Q8_0, threads 6)

| run | batch POPULATE | CoW HIT | logits vs cold |
|---|---|---|---|
| `TGE_BATCH_PREFIX=1` | `seq=0 C=32 layers=24 kv_dim=128` | `seq=1 C=32 hits=1` | **MATCH** (151936 floats memcmp) |
| control (env unset) | — (both models cold-POPULATE C=40) | — | MATCH (trivially) |

### 6.2 Continuous overhead A/B (8 req x 32 tok, 4 slots, P=32, threads 6, best-of-3 interleaved)

| pair | env off (baseline path) | env on (populate) | delta |
|---|---|---|---|
| 1 | 108.38 t/s | 115.50 t/s | +6.6% |
| 2 | 113.49 t/s | 107.13 t/s | −5.6% |
| 3 | 99.67 t/s | 100.45 t/s | +0.8% |

Best: 113.49 vs 115.50 (+1.8%); medians: 108.38 vs 107.13 (−1.2%). Signs flip pair to pair while absolutes drift (108→99 across pairs) — the §28m load signature, not a code effect. All six runs: transcripts identical (`independent`). Verdict on the overhead arm: **no measurable cost, no measurable gain** — exactly the stage-1 prediction.

### 6.3 Fidelity + thinking set (spike binary)

`fidelity --model 0.5B-Q8_0 --corpus docs/architecture.md`: 1023 positions, mean KL 9.551e-12, worst 3.341e-10, max |diff| 1.936e-04, top-1 1023/1023 — the full-program floor (cf. §28z ~1e-11), unanimous top-1. Thinking set (0.5B-Q8_0): story `Once upon a time` ids `11 1052 572 ... 304 264` and haiku ids `11 21080 389 ... 16876 45 474` byte-identical spike-vs-master (rebuilt master, diff empty).

## 7. Full reimplementation recipe (keep-grade port + stage-2, NOT this cycle)

1. Port stage-1: in `forward_batch`, after the layer loop, add the populate block (§1 verbatim) but unconditional (drop the env gate) guarded on f32-only + empty snapshot + exact-`[0..P)` run with `P>=16` + full-coverage check; keep the `[batch-prefix] POPULATE` log line (it is the attribution instrument).
2. Gate the port: rebuild 54/54, `tge_tests` 141/141, rerun the §6.1 MATCH (expect batch-POPULATE + HIT + MATCH) and the §6.2 pair (expect median within noise + identical transcripts), fidelity ≤1e-11 with 1023/1023, thinking ids identical. Ship threshold: MATCH + overhead <5% + identical ids, else revert the port.
3. Stage-2 (skip, next cycle, separate falsifier): command-layer `shared_prefix_len()` query on admission reusing the snapshot; exclude fully-shared prompt tokens from the batch (keep one trailing row for first-token logits); memcpy K/V from snapshot into the slot (reuse the CoW HIT body). Gate: full 8x32 A/B vs the re-based bars (P=8 >217, P=32 >143, +15% with identical transcripts) + fidelity-neutral (memcpy only).
4. Explicitly NOT included: paged allocation/eviction, BlockTable wiring (1517/1585 tracks own it), quantized snapshots in the batch path (f32 only, matching shipped CoW), any change to the dense kernels or the Scalar oracle.

## 8. Verdict and why

**PROCEED: stage-1 is validated with a complete port recipe.** The batch snapshot is not "similar" — it is consumed by the shipped HIT path and produces bit-identical logits, and carrying it costs nothing measurable. This retires the only structural doubt the headroom report left open (whether batch stores equal prefill stores byte-for-byte: they do). The remaining work is the stage-2 skip with its own falsifier, named not built. Next-cycle candidates: the stage-2 skip A/B; the human-gated port shelf keeps the grouped-hoist and mprefix recipes ahead of this one.

## 9. Exact commands run and their results

- `git checkout -b spike-1517-s1-batch-populate` → branch (zero commits, deleted at close; first attempt stood down on peer collision, resumed clean).
- Baseline clean HEAD 1bd093f: `cmake --build --preset release` → 54/54; `tge_tests.exe` → 141/141.
- Spike edits (§1); rebuild → 54/54; `tge_tests.exe` → 141/141.
- `tge bench mem` → copy 53.62 / read 58.89 GB/s (healthy).
- `TGE_BATCH_PREFIX=1 tge spike-batchprefix --model <0.5B-Q8_0>` → POPULATE + HIT + MATCH, exit 0; control (env unset) → two cold POPULATEs + MATCH, exit 0.
- `tge bench continuous --requests 8 --tokens 32 --threads 6 --sequences 4 --prompt-tokens 32` x3 pairs off/on → §6.2 table, all transcripts identical.
- `tge fidelity --model <0.5B-Q8_0> --corpus docs/architecture.md` (spike) → KL 9.551e-12, 1023/1023.
- Thinking set (spike, 0.5B-Q8_0): story + haiku ids saved; re-run on reverted master → both byte-identical.
- `python aikit.py check aegean-tge` → 2/4 (test + secrets pass; hygiene + guidelines fail pre-existing).
- `git checkout -- <2 files>` → clean; `git checkout master`; `git branch -D spike-1517-s1-batch-populate`; rebuild master 54/54; `tge_tests` 141/141; `git status` clean; stash unchanged; report committed; no push.

(End of file.)
