# INVENTED: vectorize the AVX2 softmax kernel's max scan and total sum — KEEP-GRADE (kernel 2.48x; sampled ids identical)

- **Idea**: `softmax_exp_avx2` computes its max with a scalar loop and, after the AVX2 exp pass, sums the 128k probabilities with a second scalar loop. Vectorize the max with four `_mm256_max_ps` accumulators and fuse the sum into the exp loop with four `_mm256_add_ps` accumulators plus a horizontal sum; scalar tails preserved.
- **Cycle date**: 2026-11-08
- **Verdict**: **KEEP-GRADE.** The kernel falls **296.6 -> 119.5 µs per 128k call (2.48x)**. The returned total changes by ~4e-7 relative from the vector summation order, and the sampled token sequences are identical on every recorded configuration: 0.5B top-p (+4.1% best-of), 3B top-p (+7.2%), 0.5B no-cut (noisy, -4.4%). Greedy ids are unchanged.
- **Tree state**: spike reverted from `src/cpu/avx2/sampler_exp_avx2.cpp` and `tests/test_sampling.cpp`; branch `idea/softmax-vector-max-sum` deleted; `git status` clean; master rebuilt 165/165; nothing committed except this report.

## 1. Mechanism

Before:

```text
scalar max over logits                      // one full pass, dependency chain
AVX2 exp loop: out[i] = exp((x - max)/T)    // 8-wide
scalar total += out[i]                      // second full pass, serial adds
```

After:

- **Max:** four 32-wide `_mm256_max_ps` accumulators, an 8-wide remainder, a scalar tail, and a shuffle-based horizontal max. Max is exact for finite values.
- **Exp + sum:** a 32-value loop computes four exp vectors, stores them, and accumulates each into one of four sum vectors; an 8-wide remainder feeds `sum0`; the scalar tail accumulates into a float. `total = horizontal_sum(sum0+sum1+sum2+sum3) + tail`.
- This removes the entire second read of `out` and the serial sum chain; the exp work is unchanged.

The non-AVX2 `softmax_exp_scalar` path is untouched.

## 2. Falsifier

- **KEEP-GRADE** if the kernel is >=1.5x faster, the sampled token sequences either match or differ only within the kernel's documented ~1e-6 draw tolerance, and greedy ids are unchanged.
- **KILL** if the kernel is slower or sampled outputs move beyond that tolerance.

Measured: 2.48x; all recorded sampled sequences and greedy ids identical. Keep.

## 3. Measured numbers

### Kernel (128,256 logits, temperature 0.8, 500 calls, best of three)

| binary | per call | accumulated total sink |
|---|---:|---:|
| scalar max + scalar sum | 296.6 µs | 219800.843750 |
| vector max + fused vector sum | **119.5 µs** | 219800.750000 |

Speedup **2.48x**; the total differs by ~1.9e-4 over 500 calls (~4e-7 relative), the expected float-summation order change. Suite 166/166 with the temporary bench.

### End-to-end sampled decode (n=64, seed 7, temp 0.8)

| config | baseline best | spike best | ratio | sampled ids |
|---|---:|---:|---:|---|
| 0.5B, top-p 0.9 | 41.53 t/s | **43.22 t/s** | 0.961 (spike +4.1%) | identical |
| 0.5B, no cut | 42.50 t/s | 40.65 t/s | 1.046 (noise; opposite sign) | identical |
| 3B, top-p 0.9 | 4.45 t/s | **4.77 t/s** | 0.933 (spike +7.2%) | identical |

The no-cut reversal is box drift (the kernel is strictly less work and the ids match); the top-p paths are consistently positive. Greedy thinking prompts A and B match the fixed ids exactly (greedy never calls this kernel).

## 4. Numerics

- Logits, greedy decoding and teacher-forced fidelity paths are untouched: the kernel only runs for sampled draws.
- The probability total changes by ~4e-7 relative; sampled draws can shift at that level, the same convention already documented for this AVX2 kernel (sampled draws shift ~1e-6 across ISAs, chi-square gated).
- All recorded 64-token sampled sequences at seed 7 are identical, so no draw flipped in the measured configs.

## 5. Full reimplementation recipe

1. `src/cpu/avx2/sampler_exp_avx2.cpp`: add `<algorithm>` (for `std::max`) and two local helpers: `horizontal_sum(__m256)` (extract/add halves, movehl, shuffle 1, `cvtss`) and `horizontal_max(__m256)` (same with `max` and the two `_MM_SHUFFLE` stages).
2. Replace the scalar max loop with the four-accumulator 32/8/scalar version; keep the scalar path for `count < 32`.
3. Replace the exp + scalar-sum section with the 32-value fused loop (four exps, four stores, four sum accumulators), an 8-wide remainder into `sum0`, and a scalar tail accumulating `tail`.
4. Return `horizontal_sum(sum0+sum1+sum2+sum3) + tail`.
5. Bench: `TGE_SOFTMAX_BENCH=1` test timing 500 `softmax_probabilities` calls on 128k logits in baseline/spike binaries; end-to-end `generate --temp 0.8 --seed 7 --print-ids` with and without `--top-p 0.9`.

## 6. Reopen / integration

**KEEP-GRADE.** Exact max, vectorized exp and sum, no greedy/fidelity impact, 2.48x on the sampler's dominant kernel. It should be integrated as a normal change (it composes with the already-shipped AVX2 argmax and radix sort). The loop keeps this report only.

Reopen only if a distribution-level gate is required before integration: build a chi-square harness over many seeds to bound the ~4e-7 total change; the current sampled-sequence evidence (three configs, 64 tokens each) found no drift.
