# IDEA-1476 STAGE-1: Cross-Layer KV Reconstruction Probe — KILLED (best affine residual 72–96%, zero viable layers)

- **Idea ID**: IDEA-1476 (dossier `arxiv-sweep-2026-11-01-03.md`: DepthGraft stride-aligned KV reconstruction — reconstruct deep-layer KV from source pairs, cut cache + prefill, no retraining; falsifier: KL over threshold, ids differ, TPOT regresses, or gains vanish on small models)
- **Cycle date**: 2026-11-06
- **Verdict**: KILLED stage-1 — per-channel affine reconstruction fails on all 27 eligible layers (best-pair mean residuals 72–83% K, 85–96% V; worst channels 100%). Zero layers meet even the 25% viability bar, let alone the 10% proceed bar. Learned/nonlinear maps would be retraining — explicitly out of scope. Spike fully reverted; only this report committed.
- **Tree state at close**: `git status` clean (no diff), spike branch created and deleted with nothing committed, stash unchanged (pre-existing peer work only), `master` ahead of `origin/master` by 34 after this report. Nothing pushed (human gate).

## 1. Mechanism (stage-1 slice)

Fit per-channel affine maps K_L[t][c] ≈ a·K_M[t][c]+b (V likewise) over real
cached tokens for every source layer M < target L on Llama-3B (K post-RoPE,
same positions both sides so RoPE angles match; V raw). Report best-source
mean/worst relative RMSE per layer. Stated falsifier: kill if no layer meets
≤25% on both K and V (nothing reconstructs without retraining); proceed if
≥3 layers meet ≤10% on both.

## 2. Predicted gain — none (diagnostic)

Stage-1 buys the go/no-go for engine integration (source-map config,
reconstructed-buffer layout, prefill early-completion) + true fidelity.
Passing would require reconstructible layers; the probe finds none.

## 3. Falsifier (stated for stage-1)

No layer with best-pair mean residual ≤25% on both K and V → kill (measured:
0/27). The 10%-proceed bar is untouched (0/27).

## 4. File anchors (current `master`, untouched except by the reverted spike)

- `src/models/qwen2.cpp` (`k_cache_/v_cache_` token-major layout read by the
  probe), `include/tge/models/qwen2.hpp` (geometry).
- Spike (reverted): `kv_stats_xlayer()` + env-gated `TGE_KV_XLAYER=1` hook
  exiting before decode; single-pass double accumulators (sx/sy/sxx/sxy/syy).

## 5. What was done (temporary spike, fully reverted)

1. `git checkout -b spike-1476-xlayer` (no commit ever made on it).
2. hpp decl + ~200-line probe + CLI hook.
3. Build clean; `tge_tests` 141/141 (read-only diagnostic).
4. `TGE_KV_XLAYER=1 generate` on Llama-3B-Q4_K_M, 2081 tokens, f32 cache —
   full 27-layer table, §6.
5. Reverted all 3 files, verified empty diff, deleted branch. Final gates on
   master: build clean, `ctest` pass, `aikit.py check` 2/4 (test + secrets
   pass; hygiene + guidelines fail pre-existing).
6. No fidelity/thinking-set owed: read-only, zero behavior change.

## 6. Measured numbers (Llama-3B, 2081 tokens; best source per layer, mean over 1024 channels)

All 27 eligible layers: K best-mean 0.72–0.83 (worst channel 1.00 throughout),
V best-mean 0.85–0.96 (worst 1.00). Best sources cluster nearby (L−1/L−2 most
common) — the probe is sensitive enough to detect locality, but locality is
far too weak: R² ≈ 0.3–0.5 at best, while reconstruction needs residuals an
order of magnitude smaller (compare: 60%-error V already prices KL 0.13,
dead). Layers with both ≤10%: 0/27. Layers with both ≤25%: 0/27.

The probe is not broken: nearby-layer preference + sub-100% means show genuine
(weak) structure, exactly what a working instrument reports on
near-orthogonal cross-layer projections. Channel permutation cannot close a
47+ point gap (channels are roughly i.i.d. across heads; the deficit is
distributional, not alignmental).

## 7. Full reimplementation recipe (kill-grade — rebuild only to re-verify)

1. hpp: `kv_stats_xlayer(int) const` decl. cpp: single-pass per-(pair,channel)
   sufficient statistics in doubles over the f32 caches; closed-form
   least-squares per pair-channel; per-layer best-source mean/worst table +
   the two verdict counters. main.cpp: env-gated call + `return 0`.
2. Gate: 2081-token f32 prefill on Llama-3B — expect 0/27 and 0/27 with the
   §6 table (nearby-layer best sources, worst channels 1.00).

## 8. Verdict and why

**KILLED stage-1, unanimously (0/27 on both bars).** Cross-layer KV are
near-orthogonal projections; per-channel affine maps explain roughly half
the variance at best and leave 72%+ relative error everywhere —
reconstruction at that error is not a smaller cache, it is a different
(random) model. Learned maps would be retraining, which the dossier
excludes; no engine work is owed. Do not revisit without a transform family
that is both retraining-free and an order of magnitude tighter — nothing on
record qualifies.

## 9. Exact commands run and their results

- `git checkout -b spike-1476-xlayer` → new branch.
- Probe (hpp + cpp + CLI); build 54/54; tests 141/141.
- `TGE_KV_XLAYER=1 generate` (Llama-3B, 2081 tok) → §6 table, 0/27 and 0/27.
- `git checkout -- <3 files>` → diff empty; `git checkout master`;
  `git branch -D spike-1476-xlayer`.
- Final: build clean, `ctest` pass, `aikit.py check` 2/4 (pre-existing fails).
- No push.
