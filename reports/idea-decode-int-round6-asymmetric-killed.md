# IDEA-DECODE-INT ROUND 6: Asymmetric per-32 Activation — KILLED (0.68x, needs <0.5x)

- **Idea**: Follow-up to rounds 2–5 (decode int8 killed 3x; per-32 validated but 4 flips short). Hypothesis: activation distributions are one-sided (post-SiLU/GELU), so symmetric int8 wastes half its codes; asymmetric min-max per-32 should roughly halve quantization error and possibly clear the 4 flips.
- **Cycle date**: 2026-11-07
- **Verdict**: KILLED. Asymmetric helps (0.60% vs 0.87% mean roundtrip error, 0.68x ratio) but far short of the <0.5x bar set for proceeding to a dot rebuild. The distributions are evidently symmetric enough that min-max buys little over amax. Reverted completely (temp probe only, no kernel built).
- **Tree state at close**: clean, no diff, stash untouched (pre-existing entry only). Spike branch created and deleted with zero commits. `master` ahead of `origin/master` by 173 after this report (prior count 172 + this one). Nothing pushed (human gate).

## 1. What was done (temporary probe, fully reverted)

Temp `TGE_ASYMPROBE` hook in `Qwen2::matvec`: per-32-group symmetric (amax/127, signed int8, matches upstream) vs asymmetric (min-max/255, unsigned 0-255) roundtrip error on live 0.5B-Q4_K_M decode rows (16-token story prompt, 360 rows). No dot changes, no numerics touched, pure measurement.

## 2. Falsifier outcome — kill, clean margin

| quantizer | mean rel | p50 | p90 |
|---|---|---|---|
| symmetric per-32 | 0.0089 | 0.0087 | 0.0102 |
| asymmetric per-32 | 0.0060 | 0.0060 | 0.0064 |

Ratio 0.68x (bar: <0.50x). The improvement is real but small — distributions are not skewed enough for min-max to matter. An asymmetric dot rebuild (new zero-point path, new fidelity gate) is not justified on a 32% quantizer win when the flips need an order-of-magnitude fix.

## 3. What this rules in/out

- OUT: asymmetric activation quantization as a decode-int fix (measured, insufficient).
- STILL OPEN: outlier-aware mixed precision (fp32 for outlier groups/channels — attacks the heavy tail directly instead of re-scaling it); per-16 uniform (unlikely to suffice alone given this result: 0.68x from full-range doubling suggests halving step size again buys <30%).
- The remaining lever with a mechanistic case is explicitly outlier-targeted, not resolution-targeted. Any future proposal must beat the round-5 per-32 baseline (p99 1.1%, 4 flips, KL 3.9e-4) on the SAME live capture harness, not on synthetic inputs.

## 4. Exact commands run and their results

- Temp probe; build 54/54; `tge_tests` 145/145 green.
- `TGE_ASYMPROBE=1 tge generate --model 0.5B-Q4_K_M --prompt "Once upon a time" --n 16` → §2 table (360 rows).
- `git checkout --` (empty diff verified); branch deleted.
- No fidelity/bench owed (quantizer-only comparison; no dot touched).

(End of file.)
