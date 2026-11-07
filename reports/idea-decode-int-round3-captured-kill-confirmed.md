# IDEA-DECODE-INT ROUND 3: Captured Real-Activation Differential — KILL CONFIRMED (rel ≤ 28.6%, p99 10%; mechanism found)

- **Idea**: Execute the reopen clause from the round-2 kill (`idea-decode-int-round2-killed.md` §8): a differential on captured REAL activations. Temp dual-compute hook in `Qwen2::matvec` (env-gated `TGE_KQ8D_CAP=1`): production float path kept, kq8d row kernel run alongside on the live Q4_K decode rows, per-(rows,cols) worst relative error + activation amax logged. All temp code reverted.
- **Cycle date**: 2026-11-07
- **Verdict**: KILL CONFIRMED (third independent ground). The kernel is exact on synthetic inputs (round 2: rel ≤ 0.006 incl. single-spike outliers) but diverges up to 28.6% per row on real activations (p99 10%). The 1013/1023 fidelity flips are explained, not anomalous. Nothing to keep; hook reverted with empty diff verified.
- **Tree state at close**: clean, no diff, stash untouched (pre-existing entry only). No branch (spike developed in-tree, reverted via `git checkout --`). `master` ahead of `origin/master` by 169 after this report. Nothing pushed (human gate).

## 1. Measured numbers (0.5B-Q4_K_M, story prompt, n=16, live decode rows)

360 captured (rows,cols) samples, single shape (896, 4864) — gate/up projections (the only decode shapes with cols%256==0; QKVO/output fall back by kind guard):

- amax (activation outlier magnitude): up to 36.2 — real outliers present.
- rel (row error / row peak): max 0.286, p99 0.100.
- Same harness on synthetic single-spike outliers (round 2): rel ≤ 0.006.

## 2. Mechanism (the finding that reconciles every prior measurement)

It is not the max, it is the distribution. A single 100x channel stretches ONE 256-block (round 2: clean); real activations are broadly heavy-tailed, so EVERY block's amax scale is large and the bulk small values drown in all blocks (round 3: dirty). Secondary consistency notes: (a) the prefill-kq8 floor numbers compared tiers sharing the same quantizer (kernel agreement, not float fidelity); (b) the round-2 max|diff| 0.66 and the 10 flips follow directly from 10%-class row errors at argmax-sensitive positions; (c) decode shapes with cols%256!=0 never reach the kernel (kind/cols fallback) — the kill is confined to shapes the kernel actually serves.

## 3. What remains genuinely open (prescribed, not attempted)

Outlier-aware activation quantization as a NEW idea, falsified by THIS harness (no new infrastructure needed): per-32-block activation scales (Q8_0 granularity) or fp32 outlier channels (LLM.int8()-style), bar = capture rel p99 < 0.02 AND fidelity ≤ 1.6e-3 with top-1 held AND the bench pulse re-verified. Anything coarser than per-32 on real activations is now measured dead twice (synthetic is not evidence — this round proves the distribution matters). The capture hook recipe (§7) is the falsifier harness.

## 4. What was done (temporary, fully reverted)

1. Temp dual-compute hook in `Qwen2::matvec` + `<cstdlib>`-free env read (getenv already available) — 40 lines, reverted.
2. Build 54/54, tests 145/145 green with hook compiled in.
3. `TGE_KQ8D_CAP=1 generate` (0.5B-Q4_K_M, story, n=16) → §1 aggregation (360 samples).
4. `git checkout -- src/models/qwen2.cpp` → empty diff verified; rebuild 54/54; tests 145/145.
5. `aikit.py check aegean-tge` → 2/4 (test + secrets pass; hygiene + guidelines fail pre-existing).
6. Report committed; no push.

## 5. Exact commands run and their results

- Spike hook added; build 54/54; `tge_tests` 145/145.
- Capture run → 360 samples, 1 shape; python aggregation → §1.
- Revert → empty diff; rebuild 54/54; tests 145/145; aikit 2/4 (pre-existing fails).
- No fidelity run owed (round-2 fidelity stands: KL 7.35e-4, 1013/1023 deterministic).

(End of file.)
