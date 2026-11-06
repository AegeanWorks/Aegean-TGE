# INVENTED: S6-Amendment Stage-1 Re-spike — Kind-Tagged Batch Snapshot + Same-Kind FNV Gate — MEASURED (Q8_0 MATCH 3/3, Q4_K refused-by-construction, kind refined to whole-model)

- **Idea**: Invented follow-up executing the S6 amendment from the verify-then-build session: re-spike the IDEA-1517 stage-1 construction (track contiguous-from-0 arrivals in `forward_batch` + end-of-call snapshot + FNV MATCH gate vs a cold `prefill_run` snapshot of the same 32-token prompt), but with kind-tagged snapshots so the cross-path comparison the S6 spike attempted can never false-share. Sharing is allowed only when tags match.
- **Cycle date**: 2026-11-06
- **Verdict**: MEASURED — the construction is proven and the amendment is refined with a new finding (§8): the kind is a whole-model property (24 kq8-eligible projections on 0.5B-Q4_K_M), not a per-projection one, because FFN arithmetic leaks into next-layer K/V rows through the residual. Q8_0: MATCH 3/3 byte-identical. Q4_K: tags FLOAT/KQ8, sharing refused by construction, cross-kind price quantified. Nothing shippable this cycle (diagnostic gate for the epic); full rebuild recipe in §7.
- **Tree state at close**: `git status` clean, no diff, stash unchanged (only the pre-existing `leftover peer work shelved 2026-11-04`). Spike branch `spike-batchsnap2` created (zero commits) and deleted; both spike files reverted with empty diff verified. `master` ahead of `origin/master` by 91 after this report. Nothing pushed (human gate).

## 1. Mechanism (as spiked)

`src/models/qwen2.cpp` only (all spike code under `TGE_BATCHSNAP2=1`; default path untouched) plus a temp `spike-batchsnap2` driver in `apps/cli/main.cpp` (reverted):

- File-local `BatchSnap` (tokens, k/v rows, layers, kv_dim, `BsnapKind::{Float,Kq8}`, valid) + `batchsnap_enabled()` env reader + byte-wise FNV-1a helper.
- `forward_batch`: per-call locals track contiguous-from-0 arrivals on layer 0 only (`bsnap_seq/expect/track_ok`); at end of call, a single-sequence call with stores exactly covering [0..count) snapshots the slot rows with kind FLOAT (the batch path forces `float_only`, so the tag is unconditional) and prints `[batchsnap] POPULATE kind=FLOAT ... fnv=...`; anything else prints `[batchsnap] SKIP ...` (zero occurrences all cycle).
- `prefill_run` end: the COMPARE gate. Prefill kind mirrors the `batch_matvec` kq8 dispatch, scanned over **all** layers × all 7 K/V-and-residual-relevant projections (wq/wk/wv/wo/w_gate/w_up/w_down): KQ8 iff `kq8_ && !scalar` and any maps to Q4_K/Q6_K. Same geometry + same tokens + kind match → elementwise max|Δ|/mean|Δ| over all K/V rows; verdict word MATCH / MISMATCH-SAMEKIND / REFUSED-CROSSKIND.
- Driver: one `forward_batch` (32-token prompt, positions 0..31, slot 0) then one cold `prefill` (same 32 tokens, slot 1); engine prints do the judging.

## 2. Predicted gain — diagnostic, landed as specified

Stage 1 saves nothing end-to-end (tracking + snapshot only, no skip) — the deliverable is the gate verdict, and the prize stays epic-owned (~44% redundant recompute at P=32 on identical-seed loads). Expected: Q8_0 MATCH (S6's own prediction), Q4_K explained refusal. Both held.

## 3. Falsifier outcome — all arms pass

1. **Q8_0 same-kind MATCH** (0.5B-Q8_0, 3 runs): tags 0/0, `kq8elig=0`, `wq=wk=wv=8`, same_tokens=1, FNV `c301c1ea7d3f7e61` all three runs, maxabs=0.0, per-layer lmax 0/0/0/0 → MATCH 3/3. Kill arm (a) not triggered.
2. **Q4_K kind refusal** (0.5B-Q4_K_M, 2 final runs + 3 diagnostic): tags FLOAT/KQ8 (`kq8elig=24`), same prefill FNV `3c1c52c9a35b72db` all five runs → REFUSED-CROSSKIND 2/2, deterministic. Kill arm (b) not triggered — tags genuinely differ.
3. **Tracking**: POPULATE fired on all 8 runs, zero SKIP lines → kill arm (c) not triggered.
4. Driver argmax id=72 on every run both models (determinism anchor).

## 4. File anchors (master = pre-spike; spike reverted)

- `src/models/qwen2.cpp` (spike block after the `SharedPrefix` struct; tracking locals + layer-0 hook in `forward_batch`'s store loop; end-of-call POPULATE before `return logits_batch_`; COMPARE gate after the prefill chunk loop, before the CoW populate).
- `apps/cli/main.cpp` (temp `command_spike_batchsnap2` + dispatch — reverted).
- Grounds: verify-then-build S6 entry + `idea-1517-stage1-headroom-proceed.md` (epic, re-based bars, 3-stage recipe); dispatch truth `batch_matvec` kq8 condition (the `!float_only && tokens>1 && Q4_K|Q6_K` conjunction).

## 5. What was done (all temporary, fully reverted)

1. `git checkout -b spike-batchsnap2` (zero commits, deleted at close). Baseline on branch: build 54/54, `tge_tests` 141/141.
2. Spike edits (§1). First build caught my own brace slip; repaired, rebuilt clean.
3. Box health: `bench mem` read 54.67 GB/s (reference band 54–61).
4. Q8_0 run → MATCH (fnv `c301...`). Q4_K run → surprise: kind FLOAT/FLOAT yet maxabs=0.388 — tag checked only layer-0 wq/wk/wv while the divergence signature (layer-0 exact, layers 1–3 lmax 0.11/0.13/0.15 compounding) pointed at residual-carried FFN arithmetic.
5. Rebuilt with per-layer diagnostics → confirmed; then with the whole-model tag (all layers × 7 projections) → `kq8elig=24`, REFUSED-CROSSKIND.
6. Repeats: Q8_0 MATCH 3/3 (identical FNV), Q4_K REFUSED 2/2 (identical FNV) — best-of gate is determinism here (FNV is load-independent), not timing.
7. Thinking set on spike binary (0.5B-Q8_0, no env → default path): story + creative ids saved; re-ran on reverted master: **both byte-identical**.
8. `aikit.py check aegean-tge` → 2/4 (test PASS; hygiene + guidelines fail pre-existing, same as prior cycles).
9. Reverted both files, `git checkout master`, branch deleted, rebuilt master 54/54, `tge_tests` 141/141, `git status` clean, stash untouched, scratch outputs deleted.

## 6. Measured numbers (Qwen2.5-0.5B, `--threads 6`, 32-token prompt)

| run | batch FNV | prefill FNV | kinds | kq8elig | maxabs | meanabs | verdict |
|---|---|---|---|---|---|---|---|
| Q8_0 ×3 | `c301c1ea7d3f7e61` | `c301c1ea7d3f7e61` | 0/0 | 0 | 0.0 | 0.0 | MATCH 3/3 |
| Q4_K diag | `d03302a0aae00dd` | `3c1c52c9a35b72db` | 0/0* | — | 3.878e-01 | 1.662e-02 | MISMATCH-SAMEKIND (tag wrong — layer-0-only) |
| Q4_K ×2 final | `d03302a0aae00dd` | `3c1c52c9a35b72db` | 0/1 | 24 | n/a (refused) | n/a | REFUSED-CROSSKIND 2/2 |

\* interim tag bug, kept on the record: it produced the price measurement, not a verdict. Q4_K per-layer lmax (layers 0–3): 0.0/1.1e-01/1.3e-01/1.5e-01 — exact at layer 0 (embeddings+norms shared), compounding deeper (FFN arithmetic in the residual).

No fidelity run owed: zero arithmetic change (reads + prints + FNV only); determinism gates (FNV identity + thinking-set identity) substitute, as precedented. No bench A/B owed: diagnostic with no second path to pair; no timing claimed.

## 7. Full reimplementation recipe (diagnostic instrument — rebuild to re-verify, then STOP)

1. Re-apply §1 (5 regions in `qwen2.cpp` + temp driver in `main.cpp`).
2. Gate: build 54/54, `tge_tests` 141/141; Q8_0 run must print POPULATE then COMPARE ... MATCH with FNV `c301c1ea7d3f7e61`; Q4_K run must print `kq8elig=24` and REFUSED-CROSSKIND with prefill FNV `3c1c52c9a35b72db`.
3. To re-verify THIS report: reproduce the §6 table exactly (FNVs are load-independent — any deviation is a real regression, not box variance).
4. Port guidance (epic-owned, NOT this cycle): the stage-1 snapshot must carry the whole-model kind (max over layers×projections, not layer 0); the stage-2 skip may consult only kind-equal snapshots; the kl-pricing alternative from S6 is now unnecessary for the float path (byte-identity proven) and remains the only option if anyone ever wants cross-kind sharing (price on record: max 0.39/mean 0.017 — far above any 1e-5-class bar, so don't).

## 8. Verdict and why

**MEASURED: the S6 amendment works as amended, with one refinement.** Same-kind snapshots are byte-identical across the batch/prefill path boundary (Q8_0 MATCH 3/3), and the kind tag correctly refuses the Q4_K cross-path share that killed S6 — while naming exactly why (24 kq8-eligible projections, FFN arithmetic in the residual; layer-0-exact/compounding-deeper signature). The epic's stage-1 gate is now a proven instrument with a rebuild recipe. Next-cycle candidates: 1517 stage-2 skip harness (epic-owned), S2 pre-packed-B retry on a quiet box, S3 vectorized tiled attention.

## 9. Exact commands run and their results

- `git checkout -b spike-batchsnap2` → branch (zero commits, deleted at close).
- Spike edits (§1; one brace slip caught at build, repaired).
- `cmake --build --preset release` → 54/54 (final); `tge_tests.exe` → 141/141 (spike + master).
- `bench mem` → read 54.67 GB/s (healthy).
- `spike-batchsnap2 --model <0.5B-Q8_0>` ×3 → MATCH, FNV `c301...` ×3; `--model <0.5B-Q4_K_M>` ×5 (2 interim-tag, 1 diag, 2 final) → REFUSED-CROSSKIND ×2 final, stable FNVs.
- Thinking set (spike vs reverted master, 0.5B-Q8_0): story ids + creative ids **both byte-identical**.
- `aikit.py check` → 2/4 (test PASS; pre-existing hygiene + guidelines fails).
- `git checkout -- <2 files>` → clean; `git checkout master`; `git branch -D`; rebuild 54/54; tests 141/141; scratch deleted; report committed; no push.

(End of file.)
