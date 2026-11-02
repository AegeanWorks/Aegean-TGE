#include "tge/cpu/kernels.hpp"
#include "tge/cpu/exp_avx2.hpp"

#include <cmath>
#include <cstddef>
#include <immintrin.h>

namespace tge::cpu {

/// Interleaved (ggml NORM) rotary embedding over consecutive pairs.
/// `cos_table`/`sin_table` are the DUPLICATED tables ([c0,c0,c1,c1,...]), so
/// one contiguous load covers 4 pairs. Even lanes rotate with (x, swapped)
/// as (x*c - xs*s); odd lanes as (x*s + xs*c); the blend picks per lane.
void apply_rope_interleaved_avx2(float* values, int heads, int head_dim,
                                 int position, const float* cos_table,
                                 const float* sin_table) {
  if (values == nullptr || heads <= 0 || head_dim <= 0 ||
      cos_table == nullptr || sin_table == nullptr) {
    return;
  }
  (void)position;
  // Swap-within-pair indices: lane i reads source lane (i ^ 1).
  const __m256i swap_within_pair = _mm256_set_epi32(6, 7, 4, 5, 2, 3, 0, 1);
  const int blocked = head_dim & ~7;
  for (int head = 0; head < heads; ++head) {
    float* vector = values + static_cast<std::size_t>(head) * head_dim;
    int index = 0;
    for (; index < blocked; index += 8) {
      const __m256 cosine =
          _mm256_loadu_ps(cos_table + static_cast<std::size_t>(index));
      const __m256 sine =
          _mm256_loadu_ps(sin_table + static_cast<std::size_t>(index));
      const __m256 x = _mm256_loadu_ps(vector + index);
      const __m256 swapped =
          _mm256_permutevar8x32_ps(x, swap_within_pair);
      const __m256 even_part =
          _mm256_fmsub_ps(x, cosine, _mm256_mul_ps(swapped, sine));
      const __m256 odd_part =
          _mm256_fmadd_ps(swapped, sine, _mm256_mul_ps(x, cosine));
      _mm256_storeu_ps(vector + index,
                        _mm256_blend_ps(even_part, odd_part, 0xAA));
    }
    // Scalar tail for head_dim values that are not a multiple of 8; head
    // dimensions are even in every supported architecture, so pairs hold.
    // Scalar tail for head_dim values that are not a multiple of 8; every
    // supported architecture uses an even head dimension, so the loop
    // below always covers the tail in whole pairs.
    for (; index + 1 < head_dim; index += 2) {
      const float cosine = cos_table[index];
      const float sine = sin_table[index];
      const float even = vector[index];
      const float odd = vector[index + 1];
      vector[index] = even * cosine - odd * sine;
      vector[index + 1] = even * sine + odd * cosine;
    }
  }
}

void rms_norm_avx2(float* out, const float* in, const float* weight, int dim,
                   float eps) {
  if (out == nullptr || in == nullptr || weight == nullptr || dim <= 0) {
    return;
  }
  // Vectorised sum of squares. Order differs from the scalar loop, so the
  // result is near-identical (fidelity-priced) rather than bit-identical.
  __m256 total = _mm256_setzero_ps();
  int index = 0;
  const int blocked = dim & ~7;
  for (; index < blocked; index += 8) {
    const __m256 x = _mm256_loadu_ps(in + index);
    total = _mm256_fmadd_ps(x, x, total);
  }
  __m128 low = _mm256_castps256_ps128(total);
  __m128 high = _mm256_extractf128_ps(total, 1);
  __m128 pair = _mm_add_ps(low, high);
  pair = _mm_add_ps(pair, _mm_movehl_ps(pair, pair));
  pair = _mm_add_ss(pair, _mm_shuffle_ps(pair, pair, 1));
  float sum = _mm_cvtss_f32(pair);
  for (; index < dim; ++index) {
    sum += in[index] * in[index];
  }
  const float scale = 1.0F / std::sqrt(sum / static_cast<float>(dim) + eps);
  const __m256 factor = _mm256_set1_ps(scale);
  index = 0;
  for (; index < blocked; index += 8) {
    const __m256 x = _mm256_loadu_ps(in + index);
    const __m256 w = _mm256_loadu_ps(weight + index);
    __m256 y = _mm256_mul_ps(x, factor);
    y = _mm256_mul_ps(y, w);
    _mm256_storeu_ps(out + index, y);
  }
  for (; index < dim; ++index) {
    out[index] = in[index] * scale * weight[index];
  }
}

void apply_rope_avx2(float* values, int heads, int head_dim, int position,
                     const float* cos_table, const float* sin_table) {
  if (values == nullptr || heads <= 0 || head_dim <= 0 ||
      cos_table == nullptr || sin_table == nullptr) {
    return;
  }
  (void)position;
  const int half = head_dim / 2;
  const int blocked = half & ~7;
  for (int head = 0; head < heads; ++head) {
    float* vector = values + static_cast<std::size_t>(head) * head_dim;
    float* low = vector;
    float* high = vector + half;
    int index = 0;
    for (; index < blocked; index += 8) {
      const __m256 cosine = _mm256_loadu_ps(cos_table + index);
      const __m256 sine = _mm256_loadu_ps(sin_table + index);
      const __m256 l = _mm256_loadu_ps(low + index);
      const __m256 h = _mm256_loadu_ps(high + index);
      // out_low = l*cos - h*sin; out_high = l*sin + h*cos (FMA, one rounding).
      __m256 out_low = _mm256_fmsub_ps(l, cosine, _mm256_mul_ps(h, sine));
      __m256 out_high = _mm256_fmadd_ps(l, sine, _mm256_mul_ps(h, cosine));
      _mm256_storeu_ps(low + index, out_low);
      _mm256_storeu_ps(high + index, out_high);
    }
    for (; index < half; ++index) {
      const float cosine = cos_table[index];
      const float sine = sin_table[index];
      const float l = low[index];
      const float h = high[index];
      low[index] = l * cosine - h * sine;
      high[index] = l * sine + h * cosine;
    }
  }
}

/// Vectorised exp shared with the attention softmax; see exp_avx2.hpp.
/// The fidelity harness prices the approximation (currently ~5e-12 KL).

void silu_gated_avx2(float* gate, const float* up, int count) {
  if (gate == nullptr || up == nullptr || count <= 0) {
    return;
  }
  const __m256 one = _mm256_set1_ps(1.0F);
  int index = 0;
  const int blocked = count & ~7;
  for (; index < blocked; index += 8) {
    const __m256 g = _mm256_loadu_ps(gate + index);
    const __m256 u = _mm256_loadu_ps(up + index);
    // silu(g) = g / (1 + exp(-g)).
    const __m256 neg = _mm256_sub_ps(_mm256_setzero_ps(), g);
    const __m256 e = exp_avx2(neg);
    const __m256 denom = _mm256_add_ps(one, e);
    const __m256 s = _mm256_div_ps(g, denom);
    _mm256_storeu_ps(gate + index, _mm256_mul_ps(s, u));
  }
  for (; index < count; ++index) {
    const float value = gate[index];
    gate[index] =
        value / (1.0F + std::exp(-value)) * up[index];
  }
}

}  // namespace tge::cpu
