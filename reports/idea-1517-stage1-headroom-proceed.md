# IDEA-1517 STAGE-1: Continuous-Batch Prefix Sharing — Headroom Measured, Falsifier Re-based, EPIC PROCEED with Staged Recipe

- **Idea ID**: IDEA-1517 (dossier `scout-filing-2026-aegean-tge.md`: vLLM-style paged KV + token-hash prefix lookup + chunked prefill; falsifier: on the 8x32 bench-continuous load, aggregate <58.3 t/s baseline or any free-running id divergence)
- **Cycle date**: 2026-11-06
- **Verdict**: PROCEED as a staged epic (no single-cycle spike fits — see §8). This stage-1 delivers exactly what gates the epic: (a) the dossier's 58.3 bar is STALE (today's tree does 3.2x it with no sharing at all — re-based below), (b) the sharing headroom is quantified large (redundant shared-prefix prefill is ~17% of wall at P=8 and ~44% at P=32), (c) a 3-stage recipe with per-stage falsifiers. No code touched (measurement + analysis only); nothing to revert.
- **Tree state at close**: `git status` clean apart from PRE-EXISTING, UNCOMMITTED peer spike (byte-identical to the saved normalized baseline at cycle start and end — untouched). No branch created, no code written. `master` ahead of `origin/master` by 48 after this report (reports only). Nothing pushed (human gate).

## 1. Mechanism (staged epic: continuous-batch shared-prefix skip)

All 8 requests in the falsifier workload share the identical prompt (transcripts
byte-identical today). After the first admission prefills it, every later
admission recomputes the same K/V through `forward_batch` (which has NO CoW
consult — verified: the CoW hit/populate lives only in `prefill_run`, and the
mixed-batch path never consults the snapshot). The staged fix: populate the
snapshot from batched stores → per-admission shared-prefix query → skip
shared rows from the batch (first-token logits via one trailing row) →
verify ids + beat the re-based bar.

## 2. Predicted gain (with the math — measured workload shape)

8 requests × 32 tokens, 4 slots, 0.5B-Q8_0, threads 6, identical seeds
(transcripts identical ⇒ 100% of prompt tokens shareable):
- P=8: prefill 64 tok (shared redundant: 56) + decode 248 tok.
- P=32: prefill 256 tok (shared redundant: 224) + decode 248 tok.
At measured rates (prefill ~233 t/s, wall from §6): redundant recompute ≈
17% of wall at P=8, up to ~44% at P=32. That is the prize — not a sliver
(the zero-copy half of 1384 died at 2%; this is the recompute itself).

## 3. Falsifier (stated for stage-1; dossier bar updated)

Kill the epic now if shared-prefix recompute is <10% of wall even at P=32
(no headroom worth a multi-cycle build). It is ~44% → PROCEED. The dossier's
"aggregate <58.3 t/s" arm is simultaneously RETIRED as stale: today's tree
clears it 2–3x with zero sharing (188.9 at P=8, 124.0 at P=32, best-of-3).
The epic's real bar (set here): beat the re-based baselines by >15% with
identical transcripts, or kill at integration.

## 4. File anchors (current `master`, read but untouched)

- `apps/cli/main.cpp` (`command_continuous_bench`, batch assembly) — where
  skip decisions must land; `bench continuous --prompt-tokens` workload.
- `src/models/qwen2.cpp` (`forward_batch` store loop — no CoW consult;
  `prefill_run` CoW hit/populate — the pattern to port).
- `reports/idea-1384-qcow-split.md` (single-sequence CoW + quantized
  snapshot recipe this epic reuses).

## 5. What was done (measurement + analysis — no spike: epic exceeds one cycle)

1. Re-read 1517 against the tree; confirmed the continuous path has no CoW.
2. Re-ran the falsifier workload fresh (re-baselining the stale 58.3 bar).
3. Quantified the sharing headroom from the workload shape + measured rates.
4. Gates: build clean, `ctest` pass, `aikit.py check` 2/4 (test + secrets
   pass; hygiene + guidelines fail pre-existing).
5. No thinking-set diff owed: same binary throughout, zero behavior change.

## 6. Measured numbers (Qwen2.5-0.5B-Q8_0, 8 req × 32 tok, 4 slots, threads 6)

| prompt | best-of-3 aggregate | transcripts | old bar | re-based bar (epic must beat) |
|---|---|---|---|---|
| P=8 | 160.1 / 174.8 / **188.9 t/s** | identical | 58.3 | >217 (+15%) |
| P=32 | 119.7 / 124.0 / 95.1 | identical | n/a | >143 (+15%) |

Box was loaded (P=32 run 3 dipped to 95.1); best-of-3 per the gate rule.
Identical transcripts confirm 100% prefix overlap — the sharing premise is
measured, not assumed.

## 7. Full reimplementation recipe (keep-grade epic, STAGED with gates)

- **Stage 1 (populate-from-batch, falsifiable alone)**: track per-sequence
  contiguous stored prefix in `forward_batch`'s store loop; when a full
  prompt [0..P) lands, snapshot it (reuse CoW populate shape). Gate: unit
  test that snapshot bytes equal a cold `prefill_run` snapshot for the same
  prompt (FNV MATCH, like the 1384 gates). Kill if MATCH fails.
- **Stage 2 (skip)**: command-layer `shared_prefix_len()` query on admission;
  exclude fully-shared prompt tokens from the batch (keep one trailing row
  for first-token logits); memcpy K/V from snapshot into the slot (reuse the
  CoW hit body). Gate: identical transcripts vs unshared baseline AND
  prefill-side win on the replay harness (expect ~4-5x per the 1384 numbers).
- **Stage 3 (integrate + verify)**: full 8x32 `bench continuous` A/B vs the
  §6 re-based bars (+15% with identical transcripts) + fidelity-neutral
  (no numerics change — memcpy only) + 141/141.
- Explicitly NOT included: paged allocation/eviction, BlockTable wiring,
  quantized snapshots in the batch path (1384's recipe covers the latter
  for `prefill_run`; port only after stage 3 lands).

## 8. Verdict and why

**PROCEED as a staged epic.** The two questions that gate it are now answered
with numbers: the old bar is stale (cleared 3.2x with no sharing — it cannot
judge anything), and the headroom is large (up to ~44% of wall is redundant
recompute on identical-seed loads; diverse real-world prompts share less,
which is why the bar is set at +15%, not at the headroom ceiling). Single-cycle
spikes were considered and rejected on the record: skipping requires
populate-tracking + batch-restructuring + verification together (the falsifier
only reads the end-to-end number), which does not fit one falsifiable cycle.
The next useful step is stage 1, gated as above — not more analysis.

## 9. Exact commands run and their results

- Re-read 1517 + verified `forward_batch` has no CoW consult (grep).
- `cmake --build --preset release` → 54/54; `ctest` → pass.
- `bench continuous` P=8 best-of-3 → §6 row; P=32 best-of-3 → §6 row.
- `aikit.py check` → 2/4 (pre-existing fails). No branch, no edits
  (peer spike untouched, byte-verified before and after), no push.
