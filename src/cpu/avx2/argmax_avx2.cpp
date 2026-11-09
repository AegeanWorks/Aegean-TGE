#include <immintrin.h>

#include <algorithm>
#include <cstdint>

#include "tge/cpu/kernels.hpp"

namespace tge::cpu {

namespace {

inline float horizontal_max(__m256 values) {
  __m128 lanes = _mm_max_ps(_mm256_castps256_ps128(values),
                            _mm256_extractf128_ps(values, 1));
  lanes = _mm_max_ps(lanes, _mm_shuffle_ps(lanes, lanes, _MM_SHUFFLE(1, 0, 3, 2)));
  lanes = _mm_max_ps(lanes, _mm_shuffle_ps(lanes, lanes, _MM_SHUFFLE(2, 3, 0, 1)));
  return _mm_cvtss_f32(lanes);
}

}  // namespace

int argmax_avx2(const float* values, int count) {
  if (values == nullptr || count <= 0) {
    return 0;
  }
  if (count < 8) {
    int best = 0;
    for (int index = 1; index < count; ++index) {
      if (values[index] > values[best]) {
        best = index;
      }
    }
    return best;
  }

  __m256 max0 = _mm256_loadu_ps(values);
  __m256 max1 = max0;
  __m256 max2 = max0;
  __m256 max3 = max0;
  int index = 0;
  for (; index + 32 <= count; index += 32) {
    max0 = _mm256_max_ps(max0, _mm256_loadu_ps(values + index));
    max1 = _mm256_max_ps(max1, _mm256_loadu_ps(values + index + 8));
    max2 = _mm256_max_ps(max2, _mm256_loadu_ps(values + index + 16));
    max3 = _mm256_max_ps(max3, _mm256_loadu_ps(values + index + 24));
  }
  for (; index + 8 <= count; index += 8) {
    max0 = _mm256_max_ps(max0, _mm256_loadu_ps(values + index));
  }
  const __m256 packed =
      _mm256_max_ps(_mm256_max_ps(max0, max1), _mm256_max_ps(max2, max3));
  float maximum = horizontal_max(packed);
  for (; index < count; ++index) {
    maximum = std::max(maximum, values[index]);
  }

  // First index equal to the maximum: the scalar strict-greater loop keeps the
  // earliest occurrence on ties.
  const __m256 target = _mm256_set1_ps(maximum);
  int scan = 0;
  for (; scan + 8 <= count; scan += 8) {
    const __m256 equal = _mm256_cmp_ps(_mm256_loadu_ps(values + scan), target,
                                       _CMP_EQ_OQ);
    const int mask = _mm256_movemask_ps(equal);
    if (mask != 0) {
      return scan + __builtin_ctz(static_cast<unsigned int>(mask));
    }
  }
  for (; scan < count; ++scan) {
    if (values[scan] == maximum) {
      return scan;
    }
  }
  return 0;
}

}  // namespace tge::cpu
