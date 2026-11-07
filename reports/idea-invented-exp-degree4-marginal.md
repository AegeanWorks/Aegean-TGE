# INVENTED: Degree-4 exp (one fewer FMA) — MARGINAL KEEP, kernel-only (3–6% SiLU; no end-to-end claim)

- **Idea**: Replace the degree-5 Taylor polynomial in `exp_avx2` with a degree-4 weighted minimax-style fit on the range-reduction interval `[-ln2/2, ln2/2]`. One FMA fewer per vector; max relative error 5.26e-6 (vs the current ~1e-6). The exp is shared by the attention softmax and SiLU.
- **Cycle date**: 2026-11-07
- **Verdict**: **MARGINAL KEEP, kernel-only.** SiLU microbenchmark best **0.262 vs 0.279 ns/element (+6.1%)**, median +3.2%; fidelity stayed in class (f32 1.261e-11, f16 2.309e-07, i8 2.370e-04, top-1 unchanged, thinking ids identical). The 512 end-to-end A/B was inconclusive (prefill 0.99–1.32x, decode 0.72–1.51x), so **no model throughput claim**. If shipped, make it an opt-in fast-exp; the degree-5 default is the bit-stability choice.
- **Tree state**: spike reverted from `include/tge/cpu/exp_avx2.hpp` and `tests/test_norm_rope.cpp`; 145/145 baseline tests rebuilt and green; no branch, no spike commit; nothing pushed.

## 1. Mechanism

`exp_avx2` range-reduces `x = n*ln2 + r`, then evaluates `exp(r)` by polynomial. The spike replaced the degree-5 Taylor coefficients with a degree-4 weighted least-squares fit on `[-ln2/2, ln2/2]` (weights `1/exp(r)` for relative error):

```text
c0 = 0.9999994399428237
c1 = 0.9999723738297991
c2 = 0.500042575219343
c3 = 0.16776648692673396
c4 = 0.04138933361355949
```

Horner: `poly = c4; poly = fma(poly, r, c3); ... c0`. One FMA fewer than the current degree-5 form; max relative error 5.26e-6 (measured on 200k points). The range reduction, clamp and `2^n` reconstruction are unchanged.

## 2. Falsifier

- **KEEP (kernel-only)** if the SiLU microbenchmark is ≥3% faster, f32 mean KL ≤1e-10, f16/i8 stay in their tiers, and thinking ids are identical.
- **KILL** if the kernel is not faster or fidelity leaves the tier.

Measured: passes the kernel and fidelity arms; end-to-end is inconclusive.

## 3. Measured numbers

### SiLU kernel microbenchmark

4,864-element buffer, 10,000 iterations, 100 warmup calls, three runs each.

| binary | run 1 | run 2 | run 3 | best | median |
|---|---:|---:|---:|---:|---:|
| degree-5 baseline | 0.285 | 0.296 | 0.279 | **0.279 ns/element** | 0.285 |
| degree-4 exp | 0.285 | 0.276 | 0.262 | **0.262 ns/element** | 0.276 |

Best-of-three is 6.1% faster; median 3.2%. The gain is one FMA per exp vector, consistent with the instruction delta, and small.

### Fidelity (spike)

| KV | mean KL | worst KL | top-1 |
|---|---:|---:|---:|
| f32 | 1.261e-11 | 4.117e-11 | 63/63 |
| f16 | 2.309e-07 | 1.342e-06 | 63/63 |
| i8 | 2.370e-04 | 9.614e-04 | 61/63 |

All in the previously recorded tiers; thinking prompt A ids are **IDENTICAL**.

### End-to-end 512 A/B (baseline vs degree-4, interleaved)

| pair | prefill base | prefill exp4 | ratio | decode base | decode exp4 | ratio |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 32.00 | 41.30 | 1.291 | 2.64 | 3.99 | 1.511 |
| 2 | 33.60 | 44.40 | 1.321 | 4.20 | 3.83 | 0.912 |
| 3 | 43.40 | 42.80 | 0.986 | 4.33 | 3.10 | 0.716 |
| **median** | — | — | **1.291** | — | — | **0.912** |

The box was heavily loaded; the ratios span both directions. This is not evidence of a model-level gain; no claim is made.

## 4. Interpretation

- The kernel is genuinely a little faster (one FMA fewer), but exp is a small share of SiLU and SiLU a small share of prefill; the model-level effect is below this box's noise.
- The fidelity cost is small enough to stay in class, but it is a real approximation: max relative error 5.3e-6 vs ~1e-6. The default f32 path is the bit-stability tier; a fast-exp should be opt-in.
- If a future cycle wants a larger exp win, reduce the range-reduction/bit-reconstruction overhead (e.g., exp2 bit trick) or the polynomial degree further with a measured KL tier, not just one FMA.

## 5. Reimplementation recipe (exact)

1. `include/tge/cpu/exp_avx2.hpp`: replace the five `c0..c5` constants and the five-FMA Horner with the four constants above and four FMAs. Keep the clamp and `2^n` reconstruction.
2. Add a temporary `silu_gated_avx2_bench` test: 4,864 elements, 10,000 iterations, 100 warmup, print ns/element (three runs each binary).
3. Gate:
   - SiLU best-of-three ≥3% faster;
   - `tge fidelity` f32 mean KL ≤1e-10, f16/i8 in class, top-1 unchanged;
   - thinking ids identical;
   - 512 A/B recorded, not used as a keep gate when the box is loaded.
4. Ship only behind an opt-in fast-exp flag; leave the degree-5 default.

## 6. Verdict and disposition

**MARGINAL KEEP, kernel-only.** The degree-4 exp removes one FMA and gives a 3–6% SiLU kernel gain with fidelity still in class, but the end-to-end A/B is inconclusive and the approximation is real. Do not claim a model throughput win; do not change the default without a bit-stability decision. The recipe is retained for an opt-in fast-exp tier.

## 7. Exact commands

```text
cmake --build --preset release                         → 54/54
.\build\release\bin\tge_tests.exe                      → 146/146 with bench; 145/145 after revert
# baseline [silu] 0.285, 0.296, 0.279 ns/element
# degree-4 [silu] 0.285, 0.276, 0.262 ns/element
tge fidelity ... --kv f32|f16|i8
tge generate ... --tokens <512 ids> --n 128 --threads 6 --gpu off
git checkout -- include/tge/cpu/exp_avx2.hpp tests/test_norm_rope.cpp
python aikit.py check aegean-tge                        → 4/4
```

No push, no merge, no force-push. Spike code is not in the tree or in any commit.
