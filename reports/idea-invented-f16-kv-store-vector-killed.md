# INVENTED: vectorize the f16 KV store with F16C — KILLED on the end-to-end bar (store 10x faster, only 0.45% of prefill)

- **Idea**: The f16 KV cache is written by a scalar `float_to_half` per value (branchy bit manipulation) in three paths: `forward`, `forward_batch` and `prefill_run`. `_mm256_cvtps_ph` converts eight floats to eight halves per instruction. The change is local to the store; no arithmetic outside the conversion moves.
- **Cycle date**: 2026-11-08
- **Verdict**: **KILLED on the stated end-to-end bar**, although the mechanism works exactly as predicted. The in-situ store drops from **612.4 ms to 61.1 ms (10.0x)** at 4081 tokens f16, but the store is only **0.50% of total prefill**, so the attributable saving is **~0.45%**. Four alternating end-to-end pairs show a median total ratio of 1.0115 raw and ~1.001 after matvec-drift normalization. The conversion is numerically safe: ids identical, fidelity f16 3.550e-07 (baseline 3.252e-07), top-1 399/399.
- **Tree state**: spike reverted from `CMakeLists.txt`, `include/tge/cpu/kernels.hpp`, `include/tge/models/qwen2.hpp`, `src/models/qwen2.cpp`, `apps/cli/main.cpp`, `tests/test_dequant.cpp`; `src/cpu/avx2/kv_store_avx2.cpp` deleted; branch `idea/f16-store-vector` deleted; `git status` clean; nothing committed except this report.

## 1. Mechanism

Three sites store K and V for one token in the f16 cache:

```text
for (int axis = 0; axis < kv_dim; ++axis) {
  k_out[axis] = cpu::float_to_half(k[axis]);
  v_out[axis] = cpu::float_to_half(v[axis]);
}
```

The scalar function handles NaN/inf, subnormal, overflow and rounding with branches. The spike adds `cpu::store_half_pair_avx2(k, v, k_out, v_out, count)` in a new AVX2 translation unit:

```text
for (index; index + 8 <= count; index += 8) {
  __m128i kh = _mm256_cvtps_ph(_mm256_loadu_ps(k + index), RNE);
  __m128i vh = _mm256_cvtps_ph(_mm256_loadu_ps(v + index), RNE);
  _mm_storeu_si128(k_out + index, kh);
  _mm_storeu_si128(v_out + index, vh);
}
tail: scalar float_to_half
```

The three model sites call it when `avx2_fma_` is set (the machine already requires F16C for `--kv f16`), with the scalar loop as the fallback. A temporary `phase_store_ns_` timer wraps the store so the component can be measured directly.

Rounding: the scalar path is round-half-up (`half + ((mantissa >> 12) & 1)`), despite its "round to nearest even" comment; F16C is ties-to-even. The only difference is exact ties with an even kept LSB — about one value in 16,384. A direct probe: `1.00048828` → scalar `0x3c01`, F16C `0x3c00`. The fidelity and id gates price this; they hold.

Predicted share: at 4081 tokens, 28 layers x 1024 kv_dim x 2 arrays x 4081 tokens = 234M conversions. The measured in-situ scalar store is 612 ms of a 122.5 s prefill (0.50%), so the ceiling is 0.5% total, not the 1.4% estimated from the isolated loop.

## 2. Falsifier

- **KEEP** if 4081-token f16 total prefill is >=1% faster in at least 4 of 6 alternating pairs with identical ids and fidelity inside the f16 tier.
- **KILL** if the attributable gain is below 1% or any id moves.

Measured: 0.45% attributable, median raw 1.0115 (normalized ~1.001), ids identical. Kill.

## 3. Measured numbers

### Store phase, direct timer (4081 tokens, f16, one run each)

| arm | prefill total (ms) | matvec (ms) | attn (ms) | norm (ms) | **store (ms)** | ids |
|---|---:|---:|---:|---:|---:|---|
| scalar | 122,482.0 | 89,306.7 | 24,143.1 | 7,641.7 | **612.4** | 578 4062 |
| F16C | 119,091.9 | 84,688.3 | 26,189.5 | 7,374.3 | **61.1** | 578 4062 |

Store speedup **10.0x**, store share **0.50% -> 0.05%**, attributable total saving **0.45%**. The rest of the total difference is matvec/window drift.

### Isolated conversion bench (4096 rows x 1024 values x 2 arrays)

| arm | time | speedup |
|---|---:|---:|
| scalar loop | 64.224 ms | — |
| `store_half_pair_avx2` | 1.996 ms | **32.17x** |

Full-entropy mantissa data: 0 mismatches over the 2048 distinct values (all rows reuse the same rows). Exact ties still differ by construction (probe above); the real KV data has ~1/16,384 tie frequency.

### End-to-end, 4096 tokens f16, four alternating pairs

| pair | total scalar (ms) | total F16C (ms) | ratio scalar/F16C | matvec ratio |
|---:|---:|---:|---:|---:|
| 1 | 179,104.6 | 162,709.5 | 1.101 | 1.064 |
| 2 | 136,369.0 | 138,111.5 | 0.987 | 0.984 |
| 3 | 132,571.7 | 128,020.5 | 1.036 | 1.037 |
| 4 | 129,873.5 | 132,326.3 | 0.981 | 0.989 |

Raw median **1.0115**; matvec-normalized **~1.001**. The matvec phase is untouched by the store, so it is the window probe; the normalized result says the measured total is not separable from noise.

### Numerics

- `tge fidelity --kv f16`, 399 positions: mean KL **3.550e-07** (baseline 3.252e-07), worst 3.335e-05 (baseline 2.709e-05), top-1 **399/399**. Still inside the f16 tier; the small increase is the tie-rounding difference.
- Ids identical on the 4081-token run and on thinking prompts A ("The quick brown fox jumps over the lazy dog. Explain why.", n=32) and B ("Write a haiku about circuits dreaming.", n=24).
- Full suite **163/163** with the temporary bench test.

## 4. Why it was killed

The change is real and safe, but the component is small: the KV store is 0.5% of f16 prefill, so a 10x store is a 0.45% engine. That is below the 1% bar and below the box's pair-to-pair noise. Keeping it would add a new ISA translation unit, a header entry, a CMake file and three conditional sites for a gain no benchmark can resolve. The recipe is ready if the share ever grows (see reopen triggers).

## 5. Full reimplementation recipe (ready to apply)

1. New TU `src/cpu/avx2/kv_store_avx2.cpp` (compiled with the existing AVX2/F16C flags):
   `#include <immintrin.h>`, `<cstdint>`, `"tge/cpu/kernels.hpp"`;
   `void tge::cpu::store_half_pair_avx2(const float* k, const float* v, std::uint16_t* k_out, std::uint16_t* v_out, int count)`:
   process eight lanes at a time with `_mm256_cvtps_ph(_mm256_loadu_ps(...), _MM_FROUND_TO_NEAREST_INT)` and `_mm_storeu_si128` for K and V; finish the tail with `float_to_half`.
2. `include/tge/cpu/kernels.hpp`: declare the function after `float_to_half`.
3. `CMakeLists.txt`: add the TU to `tge_core` sources and to the `tge_set_isa_flags(AVX2 ...)` list.
4. `src/models/qwen2.cpp`: in `forward_batch`, `forward` and `prefill_run`, replace the f16 store loop with
   `if (avx2_fma_) cpu::store_half_pair_avx2(k, v, k_out, v_out, kv_dim); else <existing loop>;`.
5. Measurement recipe: a temporary `phase_store_ns_` member (mutable, reset in `reset_phase_timers`), wrap the store in `clock_ns()`, print `store %.1f ms` in both phase lines, build a scalar-timer baseline by temporarily forcing the branch false, and compare the store phase directly. Add a `TGE_KVSTORE_BENCH=1` test with a full-entropy generator (`mantissa = (state >> 8) & 0xFFFFFF; value = mantissa / 2^24 - 0.5`) and a scalar-vs-vector mismatch count; note that 15-bit-quantized LCG values produce a misleading 6.25% mismatch because they are full of exact half-way cases.
6. Gates: `tge_tests`, `tge fidelity --kv f16`, ids at 4081 and the two thinking prompts, four alternating 4096 pairs.

## 6. Reopen triggers

- The f16 store becomes a larger share: e.g., a model with far more layers/kv_dim at the same matvec rate, a store-bound path (very short matvecs, long histories), or the scalar conversion replaced by something even slower.
- A future default changes to f16 KV for every run, making the 0.45% apply universally; even then, keep it only with a >=1% end-to-end pair result.
- Do not re-spike on this box just to re-measure the same 0.45%: the component number (612 -> 61 ms) is already on record.
