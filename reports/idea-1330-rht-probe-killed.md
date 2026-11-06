# IDEA-1330 STAGE-1: RHT Substrate Probe for TurboQuant KV — KILLED (RHT works exactly as theory predicts, and theory is not enough)

- **Idea ID**: IDEA-1330 (dossier `scout-2026-11-tge-kernels.md`: TurboQuant KV — progressive tier, FP32 high-res window, RHT + Lloyd-Max codebook + residuals + I-frames; falsifier: KL > 1e-5, top-1 < 490/497, decode < 0.95x i8, or PPL delta > +1%)
- **Cycle date**: 2026-11-06
- **Verdict**: KILLED stage-1 — the RHT premise was tested in isolation on real post-RoPE K/V and behaves exactly as information theory predicts (Gaussianizes both to ~uniform-theory rates), but the theoretical rate itself (≈17x worse than i8 per tensor) cannot touch the dossier's bar (KL ≤ 1e-5, ~32x below shipped i8 KL). Codebook + residuals (optimistic 3–5x) still leave 20–40x. No engine integration warranted. Spike fully reverted; only this report committed.
- **Tree state at close**: `git status` clean (no diff), spike branch created and deleted with nothing committed, stash unchanged (pre-existing peer work only), `master` ahead of `origin/master` by 30 after this report. Nothing pushed (human gate).

## 1. Mechanism (stage-1 slice)

Randomized Hadamard Transform (sign flips + orthonormal FWHT) per quantization
group, then uniform symmetric 4-bit RTN; invert after dequant. Theory: RHT
spreads heavy-tailed vectors evenly, making uniform quantization near-optimal
— directly aimed at the V heavy-tail failure that killed KIVI (2-bit) and
int4 (stage-2a/1385 reports). Stage-1 asks only whether RHT recovers the
uniform-theory rate (symmetric-4bit vs symmetric-8bit RMSE = 255/15 = 17x)
on real tensors; codebook, residuals, tiers, and engine work are all gated
behind this answer.

## 2. Predicted gain — none (diagnostic)

Pass bars (stated): median RHT/i8 RMSE ratio within 2x of uniform theory
(≤34x) AND theory sufficient for the bar — i.e. PROCEED only if RHT both
works and matters. Kill if ratio ≥ 4x theory (68x, structure defeats RHT) or
RHT ≥ plain-symmetric ratio (no help).

## 3. Falsifier (stated for stage-1)

RHT works mechanistically (V: 16–19x ≈ theory 17x) but theory itself is 17x
worse than i8 per tensor while the dossier needs ~parity-or-better
(KL ≤ 1e-5 vs shipped i8 KL 3.2e-4 — the bar is 32x below the best shipped
tier). The full TurboQuant stack (codebook ~1dB, 1-bit residual ~2x,
I-frames bound drift) optimistically buys 3–5x on tensor RMSE — still
20–40x short. Killed on the arithmetic of its own bar, not on mechanism
failure.

## 4. File anchors (current `master`, untouched except by the reverted spike)

- `src/models/qwen2.cpp` — K cached post-RoPE, token-major rows (the probed
  substrate); `include/tge/models/qwen2.hpp` (`kv_dim`, head geometry).
- Spike (reverted): `kv_stats_rht()` + file-local `rht_splitmix64`/`rht_fwht`
  + env-gated `TGE_KV_RHT=1` hook exiting before decode.
- `reports/idea-1516-kivi-stage1-probe-proceed.md` (V heavy-tail evidence),
  `idea-1516-kivi-stage2a-killed.md` (KL 0.13), `idea-1385-int4kv-killed.md`
  (KL 4.6e-3) — the kills this probe arbitrates between.

## 5. What was done (temporary spike, fully reverted)

1. `git checkout -b spike-1330-rht-probe` (no commit ever made on it).
2. hpp decl + FWHT/sign helpers + 200-line probe + env hook.
3. Build clean; `tge_tests` 141/141 green (probe is read-only diagnostic).
4. `TGE_KV_RHT=1 generate` on Llama-3B-Q4_K_M, 2081 tokens, f32 cache —
   full 28-layer table, §6 (two runs; second captured all layers to file).
5. Reverted all 3 files, verified empty diff, deleted branch. Final gates on
   master: build clean, `ctest` pass, `aikit.py check` 2/4 (test + secrets
   pass; hygiene + guidelines fail pre-existing).
6. No fidelity/thinking-set owed: read-only diagnostic, zero behavior change.

## 6. Measured numbers (real K/V, Llama-3B, 2081 tokens; relRMSE per scheme)

| scheme (all layers) | K range | K/i8 ratio | V range | V/i8 ratio |
|---|---|---|---|---|
| i8 native (symmetric, per-32) | 0.0058–0.0087 | 1x | 0.0056–0.0070 | 1x |
| asymmetric 4-bit (min/max) | 0.033–0.046 | 5–6x | 0.107–0.144 | 18–20x |
| symmetric 4-bit | 0.058–0.069 | 8–9x | 0.12–0.17 | 21–25x |
| **RHT + symmetric 4-bit** | **0.092–0.096** | **11–14x** | **0.108–0.111** | **16–19x ≈ theory** |

RHT columns are nearly constant across all 28 layers (mixing erases
layer-specific structure — expected). V: RHT recovers theory (16–19x vs 17x),
modestly better than plain symmetric (21–25x). K: RHT is 2.5x WORSE than
asymmetric min/max (0.09 vs 0.04) — K was already better-than-uniform
(concentrated); spreading it toward Gaussianity destroys that advantage.
Selective RHT (V-only) was considered: 0.12→0.11 buys ~1.1x against a bar
needing ~460x. Rejected on the arithmetic, recorded here.

## 7. Full reimplementation recipe (kill-grade — rebuild only to re-verify)

1. hpp: `kv_stats_rht(int) const` decl. cpp: file-local splitmix64 + iterative
   FWHT; method looping layers × heads × channels/tokens with the four
   schemes (denominators shared); env `TGE_KV_RHT=1` hook in
   `command_generate` after the prefill print, `return 0` before decode.
2. Signs: splitmix64 reseeded identically for forward/inverse per group
   (deterministic); orthonormal scale 1/sqrt(n) applied once per direction.
3. Gate: 2081-token f32 prefill on Llama-3B — expect the §6 table (RHT K
   ~0.093–0.096, RHT V ~0.108–0.111, all layers).

## 8. Verdict and why

**KILLED stage-1, with the mechanism vindicated and the bar unreachable.**
This is the honest-negative outcome the staged design exists to produce:
RHT does exactly what theory says (both directions — helps heavy-tailed V
to theory, hurts concentrated K toward theory), and theory itself (17x worse
than i8) cannot touch KL ≤ 1e-5 when shipped i8 sits at 3.2e-4. The dossier's
falsifier trips on the arithmetic before any engine work is owed; the
codebook/residual/I-frame remainder cannot close a 20–40x gap on any
optimistic accounting. Do not revisit without a bar the shipped i8 tier can
itself pass — a bar below every shippable tier is not a gate, it is a ban
(same finding as 1581).

## 9. Exact commands run and their results

- `git checkout -b spike-1330-rht-probe` → new branch.
- Probe (hpp + cpp + CLI hook, ~250 lines); build 54/54; tests 141/141.
- `TGE_KV_RHT=1 generate` (Llama-3B, 2081 tok) ×2 → §6 table, all 28 layers.
- `git checkout -- <3 files>` → diff empty; `git checkout master`;
  `git branch -D spike-1330-rht-probe`.
- Final: build clean, `ctest` pass, `aikit.py check` 2/4 (pre-existing fails).
- No push.
