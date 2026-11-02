#pragma once

// Shared Cephes-style single-precision exp for AVX2 kernels.
//
// Range reduction: n = round(x * inv_ln2), r = x - n * ln2,
// exp(x) = 2^n * P(r) with a 5th-order polynomial on [-ln2/2, ln2/2].
// Relative error ~1e-6. Both the SiLU gate and the attention softmax use
// this, so they share one implementation rather than drifting apart.
#include <immintrin.h>

namespace tge::cpu {

inline __m256 exp_avx2(__m256 x) {
  const __m256 inv_ln2 = _mm256_set1_ps(1.4426950408889634F);
  const __m256 ln2 = _mm256_set1_ps(0.6931471805599453F);
  // Clamp so the 2^n reconstruction stays in normal-float range:
  // n = round(x * inv_ln2) must satisfy n >= -126 (biased exponent >= 1).
  // Softmax tails below -87 underflow to ~1e-38, negligible beside the
  // max term (1.0); clamping keeps the bit pattern a valid normal float.
  x = _mm256_max_ps(_mm256_min_ps(x, _mm256_set1_ps(88.0F)),
                    _mm256_set1_ps(-87.0F));
  const __m256 n_float = _mm256_round_ps(_mm256_mul_ps(x, inv_ln2), 0);
  const __m256 r = _mm256_sub_ps(x, _mm256_mul_ps(n_float, ln2));
  const __m256 c0 = _mm256_set1_ps(1.0F);
  const __m256 c1 = _mm256_set1_ps(1.0F);
  const __m256 c2 = _mm256_set1_ps(0.5F);
  const __m256 c3 = _mm256_set1_ps(1.0F / 6.0F);
  const __m256 c4 = _mm256_set1_ps(1.0F / 24.0F);
  const __m256 c5 = _mm256_set1_ps(1.0F / 120.0F);
  __m256 poly = _mm256_fmadd_ps(c5, r, c4);
  poly = _mm256_fmadd_ps(poly, r, c3);
  poly = _mm256_fmadd_ps(poly, r, c2);
  poly = _mm256_fmadd_ps(poly, r, c1);
  poly = _mm256_fmadd_ps(poly, r, c0);
  const __m256i n = _mm256_cvtps_epi32(n_float);
  const __m256i biased = _mm256_add_epi32(n, _mm256_set1_epi32(127));
  const __m256i bits = _mm256_slli_epi32(biased, 23);
  const __m256 scale = _mm256_castsi256_ps(bits);
  return _mm256_mul_ps(poly, scale);
}

}  // namespace tge::cpu
