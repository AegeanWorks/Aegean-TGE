# IDEA-1516 STAGE-1: KIVI Substrate Probe — PROCEED (premise holds; V flagged for stage-2 fidelity)

- **Idea ID**: IDEA-1516 (dossier `scout-filing-2026-aegean-tge.md`: asymmetric 2-bit KV, K per-channel / V per-token, 32 fp residuals; falsifier: Llama-3B Q4_K_M at 2048 ctx, `tge fidelity` mean KL > 1e-3, top-1 < 490/497, or decode ≤ i8 → kill)
- **Cycle date**: 2026-11-06
- **Verdict**: PROCEED to stage-2 (engine integration + true fidelity under the dossier falsifier). This stage-1 probed the premise, not the product: no KV format was added, no model path touched. Spike fully reverted; only this report committed.
- **Tree state at close**: `git status` clean (no diff), spike branch created and deleted with nothing committed, stash unchanged (pre-existing peer work only), `master` ahead of `origin/master` by 19 after this report. Nothing pushed (human gate).

## 1. Mechanism (stage-1 slice)

Before building a 2-bit KV tier, check the substrate it stands on: simulate
KIVI's exact grouping — asymmetric min/max 2-bit RTN, K per-channel groups of
32 consecutive tokens, V per-token groups of head_dim — on the REAL f32 caches
from a 2081-token Llama-3.2-3B prefill (K cached post-RoPE, token-major rows),
and the same for the engine's native i8 recipe (symmetric max/127 per
32-channel block). Report per-layer relative RMSE (sqrt(err2/den2)) for both.
Kill bar (stated): any layer with 2-bit/i8 RMSE ratio above 4× the
uniform-theory 85x (= 340x) means group scales stretch pathologically and the
premise fails. Theory note: uniform data gives RMSE ratio exactly 85x
((255/3)² in MSE); real data better than 85x means concentrated (good for
2-bit), worse means outlier-stretched.

## 2. Predicted gain — none (diagnostic)

Stage-1 buys information, not tokens. Pass criterion: ratios within the bar on
all 28 layers → stage-2 (store + dequant + attention read + `--kv` tier +
fidelity + decode bench) is warranted. The fp residual window (last 32 tokens
exact) was NOT simulated — conservative toward kill, since residuals only help.

## 3. Falsifier (stated for stage-1)

KILL the KIVI premise if any layer's 2-bit/i8 RMSE ratio exceeds 340x on K or
V. Otherwise PROCEED-grade with the recipe for stage-2. (The dossier's KL/top-1
falsifier applies to stage-2, not here — tensor SNR cannot price logits.)

## 4. File anchors (current `master`, read but untouched except by the reverted spike)

- `src/models/qwen2.cpp:92-103` — i8 store recipe mirrored by the probe;
  `:1533-1557` — K cached post-RoPE, token-major memcpy into `k_cache_`.
- `include/tge/models/qwen2.hpp:384-393` — `k_cache_/v_cache_` (+half/q8 tiers).
- Spike (reverted): `Qwen2::kv_stats(int)` in qwen2.cpp + hpp decl, env-gated
  `TGE_KV_STATS=1` hook in `command_generate` after the prefill print, exit(0)
  before decode. Flag: none (env only, zero CLI surface).

## 5. What was done (temporary spike, fully reverted)

1. `git checkout -b spike-1516-kv-probe` (no commit ever made on it).
2. Added the probe (~150 lines) + decl + env hook. One self-inflicted wound:
   the first edit dropped `best = index;` from `Qwen2::argmax` (hand-retyped
   body); caught by `qwen2_argmax_rejects_null_and_empty`, restored, verified
   via `git diff` that only spike additions remained. Lesson recorded in §8.
3. Build clean; `tge_tests` 141/141 green before the probe run.
4. `TGE_KV_STATS=1 tge.exe generate --model Llama-3.2-3B-Q4_K_M --prompt
   "<2081-token paragraph>" --n 1 --threads 10 --gpu off` (f32 KV default,
   46 s prefill + ~seconds of stats): full 28-layer table, §6.
5. Reverted all 3 files (`git checkout --`), verified empty diff, deleted the
   branch. Final gates on master: build clean, `ctest` pass, `aikit.py check`
   2/4 (test + secrets pass; hygiene + guidelines fail pre-existing).
6. No fidelity/thinking-set owed: diagnostic read-only pass, zero behavior
   change (probe exits before decode; model untouched).

## 6. Measured numbers (real post-RoPE K / raw V, Llama-3B, 2081 tokens)

K = per-channel × 32-token asymmetric groups; V = per-token × 128 asymmetric
groups; i8 = engine symmetric per-32-channel blocks. relRMSE = sqrt(err2/den2).

| layer | K 2-bit | K i8 | K ratio | V 2-bit | V i8 | V ratio | K range med/worst |
|---|---|---|---|---|---|---|---|
| 0 | 2.03e-01 | 6.58e-03 | 30.9x | 7.51e-01 | 7.04e-03 | 106.6x | 2.37 / 15.06 |
| 1 | 2.01e-01 | 5.81e-03 | 34.6x | 6.31e-01 | 6.56e-03 | 96.3x | 2.90 / 18.41 |
| 2 | 2.32e-01 | 6.97e-03 | 33.3x | 6.32e-01 | 6.52e-03 | 96.9x | 3.91 / 15.93 |
| 3 | 1.93e-01 | 8.32e-03 | 23.2x | 6.45e-01 | 6.36e-03 | 101.5x | 3.29 / 17.49 |
| 4 | 1.87e-01 | 7.99e-03 | 23.4x | 5.95e-01 | 6.12e-03 | 97.3x | 2.99 / 15.41 |
| 5 | 2.05e-01 | 6.97e-03 | 29.4x | 6.40e-01 | 6.51e-03 | 98.2x | 3.36 / 18.01 |
| 6 | 1.89e-01 | 7.34e-03 | 25.8x | 5.83e-01 | 6.14e-03 | 95.0x | 3.41 / 15.45 |
| 7 | 1.85e-01 | 7.99e-03 | 23.2x | 5.79e-01 | 6.00e-03 | 96.5x | 3.45 / 17.47 |
| 8 | 1.83e-01 | 7.46e-03 | 24.6x | 5.80e-01 | 6.12e-03 | 94.8x | 3.50 / 16.35 |
| 9 | 1.81e-01 | 7.71e-03 | 23.4x | 5.88e-01 | 6.13e-03 | 96.0x | 3.41 / 19.04 |
| 10 | 1.98e-01 | 7.11e-03 | 27.9x | 6.04e-01 | 6.22e-03 | 97.2x | 3.55 / 14.06 |
| 11 | 1.77e-01 | 8.50e-03 | 20.8x | 5.91e-01 | 6.10e-03 | 96.9x | 3.32 / 16.82 |
| 12 | 1.86e-01 | 6.33e-03 | 29.4x | 5.76e-01 | 6.02e-03 | 95.7x | 3.50 / 14.12 |
| 13 | 1.86e-01 | 7.14e-03 | 26.0x | 5.83e-01 | 6.07e-03 | 96.0x | 3.50 / 15.53 |
| 14 | 1.75e-01 | 7.04e-03 | 24.9x | 5.47e-01 | 5.70e-03 | 96.0x | 3.41 / 15.89 |
| 15 | 1.82e-01 | 7.37e-03 | 24.7x | 5.37e-01 | 5.64e-03 | 95.3x | 3.30 / 13.05 |
| 16 | 1.82e-01 | 7.49e-03 | 24.3x | 6.25e-01 | 6.21e-03 | 100.7x | 3.46 / 15.32 |
| 17 | 1.75e-01 | 7.27e-03 | 24.1x | 5.56e-01 | 5.83e-03 | 95.4x | 3.16 / 15.06 |
| 18 | 1.81e-01 | 7.21e-03 | 25.2x | 5.32e-01 | 5.59e-03 | 95.2x | 3.32 / 14.35 |
| 19 | 1.84e-01 | 7.59e-03 | 24.3x | 6.01e-01 | 6.19e-03 | 97.2x | 3.51 / 15.84 |
| 20 | 1.82e-01 | 7.75e-03 | 23.4x | 5.33e-01 | 5.60e-03 | 95.2x | 3.28 / 15.14 |
| 21 | 1.73e-01 | 8.24e-03 | 20.9x | 5.47e-01 | 5.76e-03 | 95.0x | 3.06 / 15.25 |
| 22 | 1.72e-01 | 8.47e-03 | 20.3x | 5.94e-01 | 5.96e-03 | 99.6x | 2.77 / 15.67 |
| 23 | 1.78e-01 | 7.79e-03 | 22.8x | 5.42e-01 | 5.68e-03 | 95.6x | 3.34 / 14.07 |
| 24 | 1.82e-01 | 8.10e-03 | 22.5x | 5.41e-01 | 5.68e-03 | 95.2x | 3.43 / 16.02 |
| 25 | 1.74e-01 | 8.71e-03 | 20.0x | 5.93e-01 | 6.05e-03 | 98.1x | 3.03 / 20.34 |
| 26 | 1.78e-01 | 7.53e-03 | 23.6x | 5.84e-01 | 5.80e-03 | 100.6x | 3.31 / 13.17 |
| 27 | 1.67e-01 | 8.29e-03 | 20.2x | 6.15e-01 | 6.35e-03 | 96.8x | 2.96 / 16.20 |

Reading: NO layer breaches 340x (K worst 34.6x, V worst 106.6x) — the kill
bar is not tripped. K at 17–23% relRMSE beats uniform theory (33%),
so per-channel K structure holds group-32 2-bit comfortably. V at 53–75% is
worse than uniform — heavy-tailed V groups stretch per-token scales. V is the
stage-2 risk: the dossier's KL ≤ 1e-3 bar will be decided by whether attention
tolerates noisy V (sinks + residuals help) despite 60%-class tensor error.
The fp residual window (unsimulated) strictly improves on these numbers.

## 7. Full reimplementation recipe (stage-1 probe; stage-2 follows)

Stage-1 (this report): `Qwen2::kv_stats(int tokens) const` + hpp decl; loops
§5; asymmetric RTN via min/max, `llround`, 4 levels; i8 via max/127 symmetric
per-32 (engine recipe); env hook `TGE_KV_STATS` in `command_generate` after
the prefill print, `return 0` before decode; run on ≥64 cached f32 tokens.
Stage-2 (future cycle, under the dossier KL/top-1/decode falsifier): 2-bit
packed K/V tiers beside `k_cache_q8_*` (per-channel K scales over 32-token
groups + 32-slot fp residual window; per-token V scales), store path next to
`store_kv_q8`, attention read path (dequant in `q8_dot`-style kernels for both
group and head attention), `--kv` CLI value, `fidelity` pricing on Llama-3B at
2048 ctx, decode bench vs i8, ids gate under `--speculative`. Price V FIRST:
if KL fails, the V leg (§6) is the prime suspect.

## 8. Verdict and why

**PROCEED to stage-2.** The premise holds: per-channel K structure supports
group-32 2-bit at 17–23% relRMSE (better than uniform theory), and even the
heavy-tailed V stays inside the kill bar on all 28 layers. The dossier's real
falsifier (KL ≤ 1e-3) is untouched — deliberately; this cycle bought the
substrate facts it depends on. Process note: hand-retyping an existing
function body into an edit's oldString silently dropped a line (`best =
index`); the suite caught it, and the fix was verified by diff. Anchor edits
minimally; never retype what you are not changing.

## 9. Exact commands run and their results

- `git checkout -b spike-1516-kv-probe` → new branch.
- Probe + decl + env hook (3 files); one ternary typo fixed, one dropped-line
  fix (both caught by build/tests, not by review).
- `cmake --build --preset release` → 54/54; `tge_tests` → 141/141 (after the
  argmax restore; one red-green cycle in between).
- `TGE_KV_STATS=1 tge.exe generate --model Llama-3B-Q4_K_M --prompt "<2081
  tokens>" --n 1 --threads 10 --gpu off` → §6 table (46 s prefill + stats).
- `git checkout -- <3 files>` → diff empty; `git checkout master`;
  `git branch -D spike-1516-kv-probe`.
- Final: build clean, `ctest` pass, `aikit.py check` 2/4 (pre-existing fails).
- No push.
