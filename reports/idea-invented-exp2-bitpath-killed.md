# INVENTED: exp2 range-reduction path — KILLED (no consistent kernel win; fidelity fine)

- **Idea**: Replace the exp-form range reduction (`n = round(x*log2e)`, `r = x - n*ln2`, degree-5 exp polynomial) with an exp2 form: `y = x*log2e`, `n = round(y)`, `f = y - n`, and a degree-4 weighted fit of `2^f` on `[-0.5, 0.5]`. This removes the `n*ln2` multiply and one polynomial FMA.
- **Cycle date**: 2026-11-07
- **Verdict**: **KILLED on the kernel falsifier.** Alternating baseline/exp2 SiLU runs: +7.3%, −9.2%, +9.2%; median **0.293 → 0.295 ns/element** (0.7% slower), best-of-best 0.287 → 0.266 (+7.3%). No consistent win; the op-count reduction does not survive the kernel's other latency. Fidelity is fine (f32 1.233e-11, f16 2.336e-07, ids identical).
- **Tree state**: spike reverted from `include/tge/cpu/exp_avx2.hpp` and `tests/test_norm_rope.cpp`; 145/145 baseline tests rebuilt and green; no branch, no spike commit; nothing pushed.

## 1. Mechanism

```text
y = x * log2e
n = round(y)
f = y - n                       // [-0.5, 0.5]
2^f ≈ c0 + c1 f + c2 f^2 + c3 f^3 + c4 f^4
2^n via biased exponent bits
exp(x) = 2^f * 2^n
```

Degree-4 weighted fit coefficients:

```text
c0 = 0.999999439942823
c1 = 0.6931280315579599
c2 = 0.24024696235155385
c3 = 0.055870375924177
c4 = 0.009554110904974829
```

Max relative error 5.26e-6. Compared with the degree-5 exp form, this removes one multiply (`n*ln2`) and one FMA.

## 2. Falsifier

- **KEEP** if the SiLU whole-kernel benchmark is consistently ≥3% faster (median across alternating rounds) and fidelity/ids are in class.
- **KILL** if the median is flat or slower.

Measured: median 0.7% slower; kill.

## 3. Measured numbers

### SiLU kernel, alternating baseline/exp2 test binaries (3 rounds)

| round | baseline | exp2 | exp2 change |
|---:|---:|---:|---:|
| 1 | 0.287 | 0.266 | +7.3% |
| 2 | 0.293 | 0.320 | −9.2% |
| 3 | 0.325 | 0.295 | +9.2% |
| **median** | **0.293** | **0.295** | **−0.7%** |
| **best** | 0.287 | 0.266 | +7.3% |

The sign flips; the median is flat. The kernel is not throughput-limited by the one removed FMA/multiply at this point.

### Fidelity (spike)

| KV | mean KL | worst KL | top-1 |
|---|---:|---:|---:|
| f32 | 1.233e-11 | 3.757e-11 | 63/63 |
| f16 | 2.336e-07 | 1.159e-06 | 63/63 |
| i8 | 2.312e-04 | 1.080e-03 | 62/63 |

Thinking prompt A ids are **IDENTICAL**. The approximation is in class; the speed is not.

## 4. Why it lost

- The exp polynomial is not the kernel's critical path; the range reduction, bit reconstruction and SiLU division dominate or hide the FMA.
- The exp2 form changes the dependency structure: `y = x*log2e` and `f = y - n` are serial, while the exp form's `r = x - n*ln2` overlaps with the polynomial; the one-op saving does not translate.
- The cycle-17 degree-4 exp already showed a marginal 3–6% SiLU best-of-three but no end-to-end signal; exp2 does not improve on that and is less stable.

## 5. Reimplementation recipe (kill-grade, for the record)

1. `include/tge/cpu/exp_avx2.hpp`: replace the `inv_ln2`/`ln2` reduction with `y = x*log2e`, `n = round(y)`, `f = y - n`; use the degree-4 2^f coefficients above; keep the clamp and biased-exponent reconstruction.
2. Add a temporary `silu_gated_avx2_bench` (4,864 elements, 10,000 iterations, 100 warmup, print ns/element).
3. Gate: alternating baseline/spike test binaries, three rounds; require a consistent ≥3% median win. Expected: 0.293 vs 0.295 → kill.
4. Fidelity is not the blocker (f32 1.2e-11, f16 2.3e-7, ids identical); do not ship without the speed.

## 6. Verdict and disposition

**KILLED.** The exp2 path does not beat the current degree-5 exp consistently, and the degree-4 exp from cycle 17 remains the only marginal kernel option. If exp becomes a measured bottleneck, profile the range-reduction/bit path rather than removing another FMA.

## 7. Exact commands

```text
cmake --build --preset release                         → 54/54
.\build\release\bin\tge_tests.exe                      → 146/146 with bench; 145/145 after revert
# alternating rounds: 0.287/0.266, 0.293/0.320, 0.325/0.295 ns/element
tge fidelity ... --kv f32|f16|i8
git checkout -- include/tge/cpu/exp_avx2.hpp tests/test_norm_rope.cpp
python aikit.py check aegean-tge                        → 4/4
```

No push, no merge, no force-push. Spike code is not in the tree or in any commit.
