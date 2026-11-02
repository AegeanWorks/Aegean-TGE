#include "tge/cpu/kernels.hpp"
#include "tge/cpu/exp_avx2.hpp"

#include <cmath>
#include <cstddef>
#include <immintrin.h>

namespace tge::cpu {

float softmax_exp_avx2(const float* logits, int count, float temperature,
                       float* out) {
  float total = 0.0F;
  if (logits == nullptr || out == nullptr || count <= 0 ||
      !(temperature > 0.0F)) {
    return total;
  }
  // Maximum in index order (exact like the scalar max; max is associative
  // for non-NaN, so order cannot change the value anyway).
  float maximum = logits[0];
  for (int index = 1; index < count; ++index) {
    if (logits[index] > maximum) {
      maximum = logits[index];
    }
  }
  const __m256 vmax = _mm256_set1_ps(maximum);
  const __m256 vinv = _mm256_set1_ps(1.0F / temperature);
  int index = 0;
  const int blocked = count & ~7;
  for (; index < blocked; index += 8) {
    const __m256 v = _mm256_loadu_ps(logits + index);
    const __m256 x =
        _mm256_mul_ps(_mm256_sub_ps(v, vmax), vinv);
    _mm256_storeu_ps(out + index, exp_avx2(x));
  }
  for (; index < count; ++index) {
    out[index] = std::exp((logits[index] - maximum) / temperature);
  }
  // Sum in index order, same accumulation as the scalar path.
  for (index = 0; index < count; ++index) {
    total += out[index];
  }
  return total;
}

}  // namespace tge::cpu
