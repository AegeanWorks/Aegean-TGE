# INVENTED: AVX2 first-index argmax shared by the sampler and speculative verifier — KEEP-GRADE (22.4x kernel; ~7% 0.5B speculative best-of; 3B neutral; ids identical)

- **Idea**: Both `sample_token` (greedy) and `Qwen2::argmax` (speculative verification, batch admission) scan 128k logits with a scalar strict-greater loop. Replace it with an AVX2 max reduction plus a first-equal vector scan. Ties keep the earliest index, so the returned token is identical to the scalar loop for finite logits.
- **Cycle date**: 2026-11-08
- **Verdict**: **KEEP-GRADE.** The kernel falls from **125.4 µs to 5.6 µs per 128k call (22.4x)** with the same result on a 2000-call A/B and a differential tie test. End-to-end speculative decode on the 0.5B improves **+6.9% on the best-of** (40.55 -> 43.35 t/s; runs are noisy) and is neutral on the 3B reference (24.31 vs 23.43 t/s, the expected ~0.3% share is inside the box drift). Ids are identical everywhere; no fidelity run is owed because the selected index and all logits are unchanged.
- **Tree state**: spike reverted from `CMakeLists.txt`, `include/tge/cpu/kernels.hpp`, `src/models/qwen2.cpp`, `src/models/sampling.cpp`, `tests/test_sampling.cpp`; `src/cpu/avx2/argmax_avx2.cpp` deleted; branch `idea/vector-argmax` deleted; `git status` clean; master rebuilt 162/162; nothing committed except this report.

## 1. Mechanism

Scalar baseline (both call sites):

```text
int best = 0; float best_value = values[0];
for (index = 1; index < count; ++index)
  if (values[index] > best_value) { best_value = values[index]; best = index; }
return best;
```

At 128,256 logits this is a serial dependency chain: measured **125 µs** per call (~1 float/cycle), although the same operations in a test translation unit auto-vectorize to 9-10 µs. The AVX2 helper:

1. Four `_mm256_max_ps` accumulators, 32 values per iteration, an 8-wide remainder, then a scalar tail; horizontal max via two shuffles.
2. A second 8-wide pass compares each block to the max with `_CMP_EQ_OQ`; a non-zero movemask returns `block + ctz(mask)` — the **first** index equal to the maximum.
3. Scalar tail from the last 8-block boundary; return 0 if the max is unreachable (cannot happen for finite values).

`Qwen2::argmax` dispatches to the helper when `features.avx2 && features.fma` (static check) and keeps the scalar loop otherwise. `sample_token`'s greedy branch does the same on `scores`. The argmax for non-greedy sampling is untouched (it does not scan for an index).

## 2. Falsifier

- **KEEP-GRADE** if the kernel is >=5x faster on 128k with an exact first-index match, ids stay identical, and the end-to-end speculative decode does not regress on the reference model.
- **KILL** if the kernel is slower, the first-index semantics differ, or any id moves.

Measured: 22.4x, exact, ids identical, 3B neutral. Keep-grade.

## 3. Measured numbers

### Kernel (same bench test, baseline vs spike binaries, 2000 calls, best of three)

| binary | total (ms) | per call | result sink |
|---|---:|---:|---:|
| baseline scalar | 250.772 | **125.4 µs** | 18060 |
| AVX2 first-index | 11.295 | **5.6 µs** | 18060 |

Speedup **22.4x**; a separate run of the spike measured 5.2 µs. The sink is identical, so the selected index matched on the timed data.

### Differential test

`argmax_avx2_matches_scalar_first_index` covers counts 1, 2, 3, 7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 65, 257 with random values, all-equal arrays (first index 0), a two-way tie at `count/2` and `count-1` (first index wins), and strictly increasing arrays (last index). Suite **164/164**.

### End-to-end speculative decode

Prompt: repeated "The quick brown fox jumps over the lazy dog." sentence; `--speculative --n 64 --threads 6 --gpu off --kv f32 --context 2048 --print-ids`; alternating arm order.

| model | baseline best | spike best | ratio (base/spike) | ids |
|---|---:|---:|---:|---|
| Qwen2.5-0.5B Q8_0 | 40.55 t/s | **43.35 t/s** | **0.935** (spike +6.9%) | identical |
| Llama-3.2-3B Q4_K_M | **24.31 t/s** | 23.43 t/s | 1.038 (base +3.8% best) | identical |

The 0.5B runs were noisy (baseline 10.6-40.6 t/s, spike 25.4-43.4 as the box warmed), so only the best-of is usable; the spike's tight 43.1/43.35/43.09 cluster against the baseline's rising 30.7/40.6 suggests a real mid-single-digit gain. The 3B prediction is ~0.76 ms of argmax per verify step against ~250 ms, i.e. ~0.3%, below this box's drift; the measured neutral result matches.

### Why no fidelity run

The helper selects the same index; it does not touch any logit, softmax, or KV value. The ids are identical on both models and the differential test pins the index semantics. Fidelity would measure noise from the generation path, not this change.

## 4. Full reimplementation recipe

1. New TU `src/cpu/avx2/argmax_avx2.cpp` (AVX2/F16C flags): `horizontal_max(__m256)` via `_mm_max_ps` on the two 128-bit halves plus `_MM_SHUFFLE(1,0,3,2)` and `_MM_SHUFFLE(2,3,0,1)`; `int argmax_avx2(const float* values, int count)` as in §1; scalar path for `count < 8`; `_mm256_cmp_ps(..., _CMP_EQ_OQ)` + `_mm256_movemask_ps` + `__builtin_ctz` for the first index; scalar tail.
2. `include/tge/cpu/kernels.hpp`: declare `int argmax_avx2(const float* values, int count);` after `float_to_half`, documenting finite values and first-index ties.
3. `CMakeLists.txt`: add the TU to `tge_core` and to `tge_set_isa_flags(AVX2 ...)`.
4. `src/models/qwen2.cpp`, `Qwen2::argmax`: keep the null/count throw; add `static const bool use_avx2 = []{ auto f = cpu::detect().features; return f.avx2 && f.fma; }();` and dispatch; keep the scalar loop as the fallback.
5. `src/models/sampling.cpp`, `sample_token` greedy branch: replace the scalar scan over `scores` with the same dispatch.
6. Tests: the differential test above; a temporary `TGE_ARGMAX_BENCH=1` test timing 2000 `Qwen2::argmax` calls over a 128,256-float row in baseline/spike binaries.
7. Run `tge_tests` (164/164 with the new case), then speculative `generate` A/B on the 0.5B and 3B with ids.

## 5. Disposition

**KEEP-GRADE, integrate as a normal feature.** It is exact, has no flag or numerical trade, replaces a 125 µs scalar reduction with 5.6 µs, and touches two token-generation call sites. The 3B reference sees no measurable end-to-end effect (the share is ~0.3%); the 0.5B speculative path sees a mid-single-digit gain. The loop keeps this report only; the change is ready for an integration task and the human push gate.

Reopen only if the argmax must handle NaN-bearing logits (the current engine guarantees finite logits; `vmaxps`/`cmpeq` would need an explicit NaN policy).
