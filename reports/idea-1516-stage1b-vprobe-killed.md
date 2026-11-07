# IDEA-1516 STAGE-1B: V per-32 Variant Probe — KILLED (0.6–0.75x, needs <0.5x; KIVI line closed barring outlier-aware coding)

- **Idea ID**: IDEA-1516 (dossier scout-filing-2026-aegean-tge.md asymmetric 2-bit KV). Stage-1 cleared the K premise and flagged V (53–75% tensor error); stage-2a killed the tier on fidelity (KL 0.13, 128x over bar). This stage-1b tests the kill report's named reopen path — "smaller groups" — at the tensor level before any engine work: V per-32-channel asymmetric groups vs the killed per-(token,head)-128 shape, same RTN, same caches.
- **Cycle date**: 2026-11-07
- **Verdict**: KILLED. Variant improves V error only 0.6–0.75x (0.40–0.46 vs 0.52–0.79 relRMSE, all 28 layers), far short of the <0.5x bar, and no layer approaches K's 17–23% league. A ~30% tensor-error cut cannot close a 128x KL gap. The KIVI line stays closed; the remaining untested door is exactly what the kill report left ajar (outlier-aware coding or higher-precision V), NOT smaller symmetric groups — this cycle exhausted that branch.
- **Tree state at close**: `git status` clean, no diff, stash untouched (pre-existing entry only). Spike branch created and deleted with nothing committed on it. `master` ahead of `origin/master` by 165 after this report. Nothing pushed (human gate).

## 1. Mechanism (as spiked, reverted)

Temp `Qwen2::kv_v_probe` + decl + `TGE_KV_VPROBE` env hook in `command_generate` (exit 0 before decode; model untouched; all reverted, branch deleted). After a 2000-token fox prefill on Llama-3.2-3B-Q4_K_M (f32 KV), per layer over all cached positions: baseline = asymmetric min/max RTN (levels lo..lo+3s) on per-(token,head) 128-value groups (the exact killed stage-2a shape); variant = same RTN on per-32-channel groups; i8 = symmetric max/127 per-32 (engine recipe). Metric: relRMSE = sqrt(err2/den2), ratios vs i8.

## 2. Predicted gain — diagnostic, none claimed

Pass bar (stated): variant < baseline/2 on EVERY layer AND variant/i8 ratio < 340x everywhere (no pathological stretch). Kill otherwise.

## 3. Falsifier outcome — kill, both halves documented

No layer breaches 340x (variant/i8 63–72x — finer groups behave sanely), so the spike is not pathological; but variant/base runs 0.58–0.75x (best layer 26: 0.396/0.556 = 0.71x), i.e. V error 0.40–0.46 absolute — still 2x worse than K's worst layer. A 30% cut on the V leg cannot plausibly move KL 0.13 toward 1e-3 (needs ~99%).

Self-correction on record: the first probe run grouped baseline per whole 1024-row (8 heads), reading 0.79–1.75 — an 8x-too-coarse error caught by comparing against the stage-1 band (53–75%). Corrected grouping reproduces stage-1 within noise on different data (0.52–0.79 here vs 0.53–0.75 there), which validates both the probe and the original measurement. Grouping granularity dominates these numbers — noted for any future tensor work.

## 4. File anchors (master = pre-spike; spike reverted, verified clean)

- `src/models/qwen2.cpp` (`Qwen2::kv_v_probe`, ~110 lines + `<cstdio>` already present; used `v_cache_` layout `(layer*context_+pos)*kv_dim`, `config_.kv_dim()`, `config_.block_count`).
- `include/tge/models/qwen2.hpp` (one decl beside `auto_kv_half`).
- `apps/cli/main.cpp` (`TGE_KV_VPROBE` hook after the prefill print, `return 0` before decode).
- Grounds: idea-1516-kivi-stage1-probe-proceed.md (§7 recipe mirrored), idea-1516-kivi-stage2a-killed.md (§8 reopen clause executed).

## 5. What was done (temporary spike, fully reverted)

1. `git checkout -b spike-1516-vprobe` (zero commits ever made on it).
2. Probe + decl + hook (~130 lines). First run exposed the grouping bug above; fixed, rebuilt, reran.
3. `cmake --build --preset release` clean (54/54); `tge_tests` 145/145 green before the probe run.
4. `TGE_KV_VPROBE=1 tge.exe generate --model Llama-3B-Q4_K_M --prompt "<fox x200>" --n 1 --threads 10 --gpu off` → §6 table (28 layers).
5. Reverted all 3 files (`git checkout --`, empty diff verified), `git checkout master`, `git branch -D spike-1516-vprobe`.
6. Final gates on master: build clean 54/54, `tge_tests` 145/145, `aikit.py check` 2/4 (test + secrets pass; hygiene + guidelines fail pre-existing).
7. No fidelity/thinking-set owed: diagnostic read-only pass, zero behavior change (probe exits before decode).

## 6. Measured numbers (Llama-3B, 2000 cached positions, all 28 layers)

Baseline (per-128) relRMSE 0.52–0.79; variant (per-32) 0.40–0.46; i8 ~0.006; variant/i8 63–72x (no breach of 340x anywhere); variant/baseline 0.58–0.75x (bar: <0.50x — missed on all 28 layers). Full table in the run log (Temp/kilo retained by session convention? No — spike output recorded here in summary; raw stderr not archived by design).

## 7. Verdict and why

**KILLED with a clean falsifier.** Smaller symmetric groups help V (~30%) but leave it at ~42% absolute error — still 2x K's worst and an order of magnitude from viability. The KIVI asymmetric-2-bit line is now closed on two independent grounds (fidelity KL + tensor ladder on two corpora). Reopen only with outlier-aware coding or higher-precision V, staged through tensor numbers first — engine work before tensor evidence is what this design prevents.

## 8. Exact commands run and their results

- `git checkout -b spike-1516-vprobe` → new branch (zero commits).
- Probe + decl + hook; build 54/54; tests 145/145.
- Probe run v1 (wrong grouping) → 0.79–1.75 baseline (caught vs stage-1 band); fix; rebuild; probe run v2 → §6.
- `git checkout -- <3 files>` → empty diff; `git checkout master`; `git branch -D`.
- Final: build clean, tests 145/145, aikit 2/4 (pre-existing fails). Report committed; no push.

(End of file.)
