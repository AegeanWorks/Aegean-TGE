# IDEA-1385 int4-KV sub-idea: 4-Bit KIVI-style KV Tier — KILLED (KL 4.6e-3 vs 1e-3, top-1 83/86)

- **Idea ID**: IDEA-1385 (dossier `scout-2026-11-03-kilo-4target.md`: project-zero ternary kernels + int4 KV + auto-calibration; falsifier: "Perplexity or VNNI t/s fails to beat Q8_0 on-target — keep calibration/int4 ideas only"). This cycle tests the explicitly keep-worthy int4-KV sub-idea. Ternary kernels (BitNet port, epic) and auto-calibration (no carrier — no calibration flags exist) are parked as out-of-cycle, noted in §8.
- **Cycle date**: 2026-11-06
- **Verdict**: KILLED by the adapted falsifier — mean KL **4.641e-3** vs the 1e-3 bar (4.6x over), top-1 **83/86** (96.5%) vs the required ≥98.6% (3 flips vs ~1 allowed). No AVX2 fast path built (unjustified after the numerics fail). Spike fully reverted; only this report committed.
- **Tree state at close**: `git status` clean (no diff), spike branch created and deleted with nothing committed, stash unchanged (pre-existing peer work only), `master` ahead of `origin/master` by 23 after this report. Nothing pushed (human gate).

## 1. Mechanism

Asymmetric 4-bit group RTN on the KV cache, KIVI-grouped: K per-channel groups
of 32 consecutive tokens (16 packed bytes + fp32 scale/min), V per-token
groups of head_dim (head_dim/2 packed bytes + fp32 scale/min), trailing-32 fp
residual window, position-monotonic + single-sequence fail-loud guards. Nibble
packing (2 values/byte, levels 0–15, step = range/15). Read path untouched:
the f32 caches double as the fidelity mirror (dequant for sealed/aged groups,
fp32 for the residual window), so the priced numerics are exactly the format's
values. Same shared-vehicle contract as the KIVI stage-2a spike.

## 2. Predicted gain — none claimed this stage

Format-pricing stage only (like KIVI stage-2a): pass bars KL ≤ 1e-3 and top-1
≥ 490/497-equivalent (86-pos: bar 85/86). Residuals included, fp32 scales
(generous vs any real format).

## 3. Falsifier (dossier-adapted for the int4 sub-idea)

KL > 1e-3 or top-1 < 490/497-equiv → kill (no AVX2 path, no decode bench).
Both arms fail with margin beyond noise (4.6x on KL; 3 flips vs ~1 allowed).

## 4. File anchors (current `master`, untouched except by the reverted spike)

- `src/models/qwen2.cpp` — store sites (prefill/verify/decode `else` chains),
  `Qwen2::load` alloc, `reset_cache`.
- `include/tge/models/qwen2.hpp` — tier members, `store_kv_q4` decl, RTN decls.
- `tests/test_dequant.cpp` — RTN unit test site (house try/catch style).
- KIVI stage-2a report (`idea-1516-kivi-stage2a-killed.md`) — the pattern this
  spike ports (2-bit KL 0.13 there vs 4-bit 0.0046 here: the expected ~30x
  step from 2 extra bits, still short of the bar).

## 5. What was done (temporary spike, fully reverted)

1. `git checkout -b spike-1385-q4kv` (no commit ever made on it).
2. hpp: tier members + `store_kv_q4` + `q4_quantize/dequantize_group` decls.
   cpp: nibble RTN pair, streaming store (V immediate, K 32-stage + seal,
   residual aging rewrite), env-gated `TGE_KV_Q4` load (eager oracle excluded,
   single-seq guard), reset, 3 store-site hooks. tests: RTN roundtrip/guard
   test.
3. Build clean; `tge_tests` **142/142** (141 + 1 new).
4. `TGE_KV_Q4=1 tge.exe fidelity` (Llama-3B, 86-pos, ctx 512) → §6 (88 s).
5. `TGE_KV_Q4=1 generate` (fox, n=16) → free-running ids diverge at token 6
   (expected for lossy KV; earlier than q2's token-9 flip — chaotic cascade
   point, not a quality ranking).
6. Reverted all 3 files, verified empty diff, deleted branch. Final gates on
   master: build clean, `ctest` pass, `aikit.py check` 2/4 (test + secrets
   pass; hygiene + guidelines fail pre-existing).

## 6. Measured numbers (Llama-3.2-3B-Q4_K_M, 86 teacher-forced positions)

| | int4 tier (spike) | bar | 2-bit precedent | i8 precedent |
|---|---|---|---|---|
| mean KL | **4.641e-3** | ≤ 1e-3 | 1.285e-1 | 3.2e-4 |
| worst KL | 3.398e-2 | — | 7.3e-1 | — |
| max \|diff\| | 1.183 | — | 7.15 | — |
| top-1 | **83/86 (96.5%)** | ≥ 98.6% | 75/86 | 31/31 |

Correctness confidence (a kill must be earned): RTN green in isolation;
position-monotonicity guard silent across 86×28 writes; failure mode is
near-correct (96.5%), not garbage — consistent with lossy-but-right
indexing; and the 2-bit→4-bit step (≈28x KL improvement for 2 extra bits)
tracks quantization theory, confirming the vehicle measures the format, not
a bug. Dossier context caveat as before: 86 teacher-forced positions at ctx
512, not 2048-ctx full runs — no plausible mechanism closes a 4.6x gap at
length (more quantized history only adds error; window fixed at 32).

## 7. Full reimplementation recipe (kill-grade — rebuild only to re-verify)

1. hpp: tier members (`k_cache_q4_` layers*groups*kv_dim*16B, `k4_scales_/mins_`,
   `v_cache_q4_` layers*ctx*heads*64B, `v4_scales_/mins_`, `k4_stage_` 32 rows,
   `k4_sealed_/_written_`); `store_kv_q4(layer,k,v,position,sequence)` decl;
   `q4_quantize/dequantize_group` decls (count%2==0, levels 0–15).
2. cpp: nibble RTN (§5 packing); streaming store per §1 (seal at (p+1)%32==0,
   age out token p−32 to dequant); env `TGE_KV_Q4` load (exclude eager,
   single-seq guard) + alloc; reset branch; 3 store-site `else if (kv_q4_)`
   hooks.
3. tests: constant-group exactness, ramp half-step bound (range/30), 128-wide
   bound, odd-count/null guards.
4. Gate: `TGE_KV_Q4=1 fidelity` on the 86-pos corpus — expect KL ≈ 4.6e-3,
   top-1 ≈ 83/86.

## 8. Verdict and why

**KILLED by measurement.** int4-KV is 28x better than 2-bit and still 4.6x
over the bar with 3 flips — the sub-8-bit KV direction is now 0-for-2 on this
workload with independent implementations agreeing. The V heavy-tail
(stage-1: 53–75% tensor error at 2-bit) survives the extra two bits enough to
break the 1e-3 budget. Parked explicitly out-of-cycle: ternary weight kernels
(BitNet port — epic, needs its own staged dossier, not a day-cycle) and
auto-calibration (no calibration flags or carrier exist in the tree — unformed
as an idea until a proponent names the mechanism). Do not revisit sub-8-bit KV
without a different grouping/precision structure staged through tensor
evidence first — engine work before tensor evidence is what the stage-1/2a
pattern exists to prevent.

## 9. Exact commands run and their results

- `git checkout -b spike-1385-q4kv` → new branch.
- Spike (hpp + cpp + tests, adapted from the stage-2a recipe).
- `cmake --build --preset release` → 54/54; `tge_tests` → 142/142.
- `TGE_KV_Q4=1 fidelity ... --tokens 200` → KL 4.641e-3, worst 3.4e-2,
  max|diff| 1.18, top-1 83/86 (88 s).
- `TGE_KV_Q4=1 generate` (fox, n=16) → ids diverge at token 6 (expected).
- `git checkout -- <3 files>` → diff empty; `git checkout master`;
  `git branch -D spike-1385-q4kv`.
- Final: build clean, `ctest` pass, `aikit.py check` 2/4 (pre-existing fails).
- No push.
