# INVENTED: SiLU reciprocal + Newton — KILLED (kernel 22% slower; numerics fine)

- **Idea**: Replace the per-element `vdivps` in `silu_gated_avx2` with `rcp_ps` plus one Newton step, `r * (2 - d*r)`, to trade a division for a reciprocal and two FMAs. Relative error ~2e-7; the scalar tail keeps exact division. The same `div` pattern is the only one left in the AVX2 kernels.
- **Cycle date**: 2026-11-07
- **Verdict**: **KILLED on the kernel falsifier.** Baseline **0.279–0.296 ns/element**; reciprocal+Newton **0.341–0.424 ns/element** (best +22% slower, worst +49%). Numerics are fine — f32 mean KL 1.098e-11, f16 2.591e-07, thinking ids identical — but a slower kernel cannot improve end-to-end. No model A/B was carried beyond the kernel.
- **Tree state**: spike reverted from `src/cpu/avx2/norm_rope_avx2.cpp` and `tests/test_norm_rope.cpp`; 145/145 baseline tests rebuilt and green; no branch, no spike commit; nothing pushed.

## 1. Mechanism

`silu_gated_avx2` computes:

```text
silu(g) = g / (1 + exp(-g))
x[i]    = silu(g[i]) * up[i]
```

The spike replaced:

```cpp
s = _mm256_div_ps(g, denom);
```

with:

```cpp
r0 = _mm256_rcp_ps(denom);
r  = r0 * (2 - denom * r0);   // one Newton step
s  = g * r;
```

`_mm256_fnmadd_ps(denom, r0, two)` provides the Newton correction. The scalar tail keeps `value / (1 + exp(-value))`.

## 2. Predicted gain / falsifier

- Prediction: a 256-bit division is expensive; replacing it with a reciprocal and two FMAs could save 1–2 cycles per 8 elements.
- Falsifier: kernel ≥5% faster; f32 mean KL ≤1e-6; thinking ids identical. If the kernel is slower, kill before any end-to-end claim.

Measured: kernel slower; kill.

## 3. Measured numbers

### Kernel microbenchmark

Temporary `silu_gated_avx2_bench` test: 4,864-element buffer, 10,000 iterations, 100 warmup calls, three runs each binary.

| binary | run 1 | run 2 | run 3 | best |
|---|---:|---:|---:|---:|
| baseline (`vdivps`) | 0.285 | 0.296 | 0.279 | **0.279 ns/element** |
| reciprocal + Newton | 0.424 | 0.369 | 0.341 | **0.341 ns/element** |

Best-of-three is 22% slower; the slowest spike run is 49% slower than the best baseline. The division is not the bottleneck on this box.

### Fidelity (spike binary)

| KV | mean KL | worst KL | top-1 |
|---|---:|---:|---:|
| f32 | 1.098e-11 | 4.774e-11 | 63/63 |
| f16 | 2.591e-07 | 1.344e-06 | 63/63 |
| i8 | 2.393e-04 | 1.041e-03 | 62/63 |

Thinking prompt A ids are **IDENTICAL** to baseline. The numerics would have been acceptable; the speed is not.

## 4. Why it lost

- **Alder Lake `vdivps` is cheaper than the instruction-count model suggests.** It is pipelined and often hidden behind the surrounding exp/multiply work.
- **The Newton step adds a dependent chain**: `rcp` latency + FMA + multiply, plus a constant register. On this kernel the divide was not the port bottleneck; the replacement increased uops and latency.
- This is the project's repeated lesson: instruction count is not port analysis. The earlier logs record five optimisations that looked cheaper on paper and measured worse; this is another.

## 5. Reimplementation recipe (kill-grade, for the record)

1. In `src/cpu/avx2/norm_rope_avx2.cpp`, replace the `s = _mm256_div_ps(g, denom)` line with:
   `r0 = _mm256_rcp_ps(denom); r = _mm256_mul_ps(r0, _mm256_fnmadd_ps(denom, r0, _mm256_set1_ps(2.0F))); s = _mm256_mul_ps(g, r);`
2. Keep the scalar tail exact.
3. Add a temporary `silu_gated_avx2_bench` test (4,864 elements, 10,000 iterations, 100 warmup, print ns/element).
4. Gate: three runs each binary; require the spike best to beat baseline best by ≥5%, then fidelity and ids. Expected: spike 0.341 vs baseline 0.279 → kill; do not ship.

## 6. Verdict and disposition

**KILLED.** The reciprocal+Newton is numerically safe but 22% slower at the kernel and would only add latency to every FFN. Leave `vdivps` in place. If SiLU becomes hot again, the lever is the exp approximation or a different kernel schedule, not the divide.

## 7. Exact commands

```text
cmake --build --preset release                         → 54/54
.\build\release\bin\tge_tests.exe                      → 146/146 with the bench test; 145/145 after revert
# baseline [silu] 0.285, 0.296, 0.279 ns/element
# spike    [silu] 0.424, 0.369, 0.341 ns/element
tge fidelity --model Llama-3B --corpus docs/architecture.md --context 512 --tokens 64 --kv f32|f16|i8
git checkout -- src/cpu/avx2/norm_rope_avx2.cpp tests/test_norm_rope.cpp
python aikit.py check aegean-tge                        → 4/4
```

No push, no merge, no force-push. Spike code is not in the tree or in any commit.
