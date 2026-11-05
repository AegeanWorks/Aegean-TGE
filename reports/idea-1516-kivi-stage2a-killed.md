# IDEA-1516 STAGE-2A: KIVI 2-Bit KV Tier, Format Fidelity — KILLED (KL 0.13 vs 1e-3 bar, top-1 75/86)

- **Idea ID**: IDEA-1516 (dossier `scout-filing-2026-aegean-tge.md`: asymmetric 2-bit KV, K per-channel / V per-token, 32 fp residuals; falsifier: Llama-3B Q4_K_M at 2048 ctx, `tge fidelity` mean KL > 1e-3, top-1 < 490/497, or decode ≤ i8 → kill). Stage-1 (`idea-1516-kivi-stage1-probe-proceed.md`) cleared the substrate premise; this stage prices the format.
- **Cycle date**: 2026-11-06
- **Verdict**: KILLED by the dossier's own falsifier arm 1 — mean KL **1.285e-1** vs the 1e-3 bar (128x over), top-1 **75/86** (87.2%) vs the required ≥98.6%. No AVX2 fast path was built (unjustified after the numerics fail); no decode-speed arm reached. Spike fully reverted; only this report committed.
- **Tree state at close**: `git status` clean (no diff), spike branch created and deleted with nothing committed, stash unchanged (pre-existing peer work only), `master` ahead of `origin/master` by 20 after this report. Nothing pushed (human gate).

## 1. Mechanism (stage-2a slice)

Full KIVI-style streaming store behind an env-gated `kv_q2_` tier: K
per-channel groups of 32 consecutive tokens (8 packed bytes + fp32
scale/min), V per-token groups of head_dim (head_dim/4 packed bytes + fp32
scale/min), 32-slot fp residual window for recency, position-monotonicity and
single-sequence fail-loud guards. Read path untouched: the f32 caches double
as a fidelity mirror (dequant values for sealed/aged groups, fp32 for the
residual window), so the priced numerics are EXACTLY the format's values and
attention needs no changes for the verdict. The shared RTN pair
(`q2_quantize_group`/`q2_dequantize_group`, asymmetric min/max, 4 values/byte)
is used by store, mirror aging, and unit tests alike, so the vehicle cannot
drift from the priced values.

## 2. Predicted gain — none claimed this stage

Stage-2a buys the fidelity number the dossier demands. Pass bars: mean KL ≤
1e-3, top-1 ≥ 490/497-equivalent (86-pos corpus: bar ≈ same 98.6% rate).
Residuals (last-32 fp) included, fp32 scales (generous vs fp16).

## 3. Falsifier (from the dossier)

Mean KL > 1e-3 or top-1 < 490/497 → kill (no AVX2 path, no decode bench).
Both trip with margin: KL 128x over bar, agreement 11.4 points under.

## 4. File anchors (current `master`, untouched except by the reverted spike)

- `src/models/qwen2.cpp` — store sites (prefill/verify/decode `else` chains),
  `store_kv_q8` recipe mirrored, `Qwen2::load` alloc, `reset_cache`.
- `include/tge/models/qwen2.hpp` — cache members, `load` signature.
- `apps/cli/main.cpp` — fidelity oracle-vs-candidate harness (sequential
  positions, which the streaming staging relies on); NO CLI change was needed
  (env gate inside `load`: eager/oracle load never takes the tier).

## 5. What was done (temporary spike, fully reverted)

1. `git checkout -b spike-1516-k2-tier` (no commit ever made on it).
2. hpp: RTN decls + tier members + `store_kv_q2` decl. cpp: RTN pair, store
   (V immediate, K 32-stage + seal, residual aging rewrite), load alloc +
   env gate + single-seq guard, reset, 3 store-site hooks. tests: RTN
   roundtrip/guard test in `test_dequant.cpp`.
3. Build clean; `tge_tests` **142/142** (141 + 1 new).
4. `TGE_KV_Q2=1 tge.exe fidelity --model Llama-3B-Q4_K_M --corpus <86-pos>
   --context 512 --tokens 200 --threads 1 --gpu off` → §6 (92 s).
5. `TGE_KV_Q2=1 generate` (fox, `--n 16`) → free-running ids diverge at token
   9 (expected for lossy KV; VNNI precedent also diverges — recorded, not
   gated).
6. Reverted all 3 files, verified empty diff, deleted branch. Final gates on
   master: build clean, `ctest` pass, `aikit.py check` 2/4 (test + secrets
   pass; hygiene + guidelines fail pre-existing).

## 6. Measured numbers

`fidelity`, Llama-3.2-3B-Q4_K_M, 86 teacher-forced positions, ctx 512:

| | q2 tier (spike) | dossier bar | precedent i8 |
|---|---|---|---|
| mean KL | **1.285e-01** | ≤ 1e-3 | 3.2e-4 |
| worst KL | 7.311e-1 | — | (VNNI 2.3e-2) |
| max \|logit diff\| | 7.151 | — | (f16 5.3e-2) |
| top-1 | **75/86 (87.2%)** | ≥ 98.6% | 31/31 |

Correctness confidence (this is a kill, so it must be earned): (a) RTN unit
tests green in isolation; (b) the position-monotonicity guard never fired
across 86 sequential writes × 28 layers — the streaming path is coherent;
(c) failure mode is near-correct (87% agreement), not garbage — consistent
with lossy-but-right indexing; (d) stage-1 independently measured V at
53–75% tensor relRMSE on real K/V — KL 0.13 is exactly the expected
consequence, from a separate code path. Two independent measurements agree;
the format is correctly implemented and genuinely too lossy.

Context caveat (honest): dossier says 2048 ctx; this ran 86 teacher-forced
positions at ctx 512 (same corpus scale as the 1327/1458 runs). No plausible
mechanism closes a 128x gap at longer context — more quantized history can
only add error, and the window is fixed at 32.

## 7. Full reimplementation recipe (kill-grade — rebuild only to re-verify)

1. hpp: `q2_quantize_group`/`q2_dequantize_group` decls (count%4==0,
   4 values/byte at bits (i%4)*2, levels [lo, lo+3*scale], exact-on-constant);
   tier members (`k_cache_q2_` layers*groups*kv_dim*8B, `k2_scales_/mins_`,
   `v_cache_q2_` layers*ctx*heads*32B, `v2_scales_/mins_`,
   `k2_stage_` layers*32*kv_dim, `k2_sealed_`, `k2_written_`); `kv_q2_` flag;
   `store_kv_q2(layer, k, v, position, sequence)` decl.
2. cpp: RTN pair per §7 contract; store per §1 (V immediate, K stage+seal at
   (p+1)%32==0, mirror fp at write, residual rewrite of token p-32);
   `load`: env `TGE_KV_Q2` && !eager → flag + single-seq guard + alloc;
   `reset_cache`: q2 branch clearing all; 3 store-site `else if (kv_q2_)`
   hooks (prefill/index+sequences[token], verify/index+sequence,
   decode/index+seq 0).
3. tests: constant-group exactness, ramp half-step bound, 128-wide bound,
   count/null guards.
4. Gate: `TGE_KV_Q2=1 fidelity` on the 86-pos corpus — expect KL ≈ 1.3e-1,
   top-1 ≈ 75/86. Kill stands unless a V-grouping variant changes the
   stage-1 tensor numbers first.

## 8. Verdict and why

**KILLED by measurement, exactly as the falsifier prescribes.** Asymmetric
2-bit RTN at KIVI groupings destroys too much V information on this workload
(stage-1: 53–75% tensor error; stage-2a: KL 0.13, 11 flips in 86) for the
1e-3/98.6% bar — 128x over on KL. The K leg holds (17–23%) but KV is joint;
compounding with Q4_K_M weights (the dossier's own model choice) leaves no
headroom a residual window can recover at this density. No AVX2 fast path was
built: speed is unmeasurable for a format that fails numerics. Do not revisit
without a different V treatment (smaller groups, outlier-aware coding, or
higher-precision V) staged through stage-1 tensor numbers FIRST — engine work
before tensor evidence is what this two-stage design exists to prevent.

## 9. Exact commands run and their results

- `git checkout -b spike-1516-k2-tier` → new branch.
- Spike (hpp + cpp + tests); one interim `argmax` line-drop caught by tests
  and restored (same lesson as stage-1: never retype, always anchor).
- `cmake --build --preset release` → 54/54; `tge_tests` → 142/142.
- `TGE_KV_Q2=1 fidelity ... --tokens 200` → KL 1.285e-1, worst 7.311e-1,
  max|diff| 7.151, top-1 75/86 (92 s).
- `TGE_KV_Q2=1 generate` (fox, n=16) → ids diverge at token 9 (expected).
- `git checkout -- <3 files>` → diff empty; `git checkout master`;
  `git branch -D spike-1516-k2-tier`.
- Final: build clean, `ctest` pass, `aikit.py check` 2/4 (pre-existing fails).
- No push.
