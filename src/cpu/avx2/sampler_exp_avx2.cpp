#include "tge/cpu/kernels.hpp"
#include "tge/cpu/exp_avx2.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <immintrin.h>

namespace tge::cpu {
namespace {

inline float horizontal_sum(__m256 values) {
  __m128 total = _mm_add_ps(_mm256_castps256_ps128(values),
                            _mm256_extractf128_ps(values, 1));
  total = _mm_add_ps(total, _mm_movehl_ps(total, total));
  total = _mm_add_ss(total, _mm_shuffle_ps(total, total, 1));
  return _mm_cvtss_f32(total);
}

inline float horizontal_max(__m256 values) {
  __m128 maximum = _mm_max_ps(_mm256_castps256_ps128(values),
                              _mm256_extractf128_ps(values, 1));
  maximum = _mm_max_ps(maximum, _mm_movehl_ps(maximum, maximum));
  maximum = _mm_max_ss(maximum, _mm_shuffle_ps(maximum, maximum, 1));
  return _mm_cvtss_f32(maximum);
}

}  // namespace

float softmax_exp_avx2(const float* logits, int count, float temperature,
                       float* out) {
  float total = 0.0F;
  if (logits == nullptr || out == nullptr || count <= 0 ||
      !(temperature > 0.0F)) {
    return total;
  }
  // Max is exact for finite values; only the scan order is vectorised.
  float maximum = logits[0];
  int index = 0;
  if (count >= 32) {
    __m256 max0 = _mm256_loadu_ps(logits);
    __m256 max1 = _mm256_loadu_ps(logits + 8);
    __m256 max2 = _mm256_loadu_ps(logits + 16);
    __m256 max3 = _mm256_loadu_ps(logits + 24);
    const int blocked = count & ~31;
    for (index = 32; index < blocked; index += 32) {
      max0 = _mm256_max_ps(max0, _mm256_loadu_ps(logits + index));
      max1 = _mm256_max_ps(max1, _mm256_loadu_ps(logits + index + 8));
      max2 = _mm256_max_ps(max2, _mm256_loadu_ps(logits + index + 16));
      max3 = _mm256_max_ps(max3, _mm256_loadu_ps(logits + index + 24));
    }
    maximum = horizontal_max(_mm256_max_ps(_mm256_max_ps(max0, max1),
                                           _mm256_max_ps(max2, max3)));
    for (; index + 8 <= count; index += 8) {
      maximum =
          std::max(maximum, horizontal_max(_mm256_loadu_ps(logits + index)));
    }
    for (; index < count; ++index) {
      maximum = std::max(maximum, logits[index]);
    }
  } else {
    for (index = 1; index < count; ++index) {
      if (logits[index] > maximum) {
        maximum = logits[index];
      }
    }
  }
  const __m256 vmax = _mm256_set1_ps(maximum);
  const __m256 vinv = _mm256_set1_ps(1.0F / temperature);
  __m256 sum0 = _mm256_setzero_ps();
  __m256 sum1 = _mm256_setzero_ps();
  __m256 sum2 = _mm256_setzero_ps();
  __m256 sum3 = _mm256_setzero_ps();
  const int blocked = count & ~31;
  for (index = 0; index < blocked; index += 32) {
    const __m256 x0 =
        _mm256_mul_ps(_mm256_sub_ps(_mm256_loadu_ps(logits + index), vmax),
                      vinv);
    const __m256 x1 =
        _mm256_mul_ps(_mm256_sub_ps(_mm256_loadu_ps(logits + index + 8), vmax),
                      vinv);
    const __m256 x2 =
        _mm256_mul_ps(_mm256_sub_ps(_mm256_loadu_ps(logits + index + 16), vmax),
                      vinv);
    const __m256 x3 =
        _mm256_mul_ps(_mm256_sub_ps(_mm256_loadu_ps(logits + index + 24), vmax),
                      vinv);
    const __m256 e0 = exp_avx2(x0);
    const __m256 e1 = exp_avx2(x1);
    const __m256 e2 = exp_avx2(x2);
    const __m256 e3 = exp_avx2(x3);
    _mm256_storeu_ps(out + index, e0);
    _mm256_storeu_ps(out + index + 8, e1);
    _mm256_storeu_ps(out + index + 16, e2);
    _mm256_storeu_ps(out + index + 24, e3);
    sum0 = _mm256_add_ps(sum0, e0);
    sum1 = _mm256_add_ps(sum1, e1);
    sum2 = _mm256_add_ps(sum2, e2);
    sum3 = _mm256_add_ps(sum3, e3);
  }
  if (index + 8 <= count) {
    const __m256 x =
        _mm256_mul_ps(_mm256_sub_ps(_mm256_loadu_ps(logits + index), vmax),
                      vinv);
    const __m256 exponent = exp_avx2(x);
    _mm256_storeu_ps(out + index, exponent);
    sum0 = _mm256_add_ps(sum0, exponent);
    index += 8;
  }
  float tail = 0.0F;
  for (; index < count; ++index) {
    out[index] = std::exp((logits[index] - maximum) / temperature);
    tail += out[index];
  }
  return horizontal_sum(_mm256_add_ps(_mm256_add_ps(sum0, sum1),
                                      _mm256_add_ps(sum2, sum3))) +
         tail;
}

}  // namespace tge::cpu
