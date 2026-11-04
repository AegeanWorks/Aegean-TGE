#include "tge/cpu/gemm_epilogue.hpp"

#include <cstddef>

#include <immintrin.h>

namespace tge::cpu {

void clamp_avx2(float* data, std::size_t count, float vmin, float vmax) {
  if (data == nullptr || count == 0) return;
  const __m256 lo = _mm256_set1_ps(vmin);
  const __m256 hi = _mm256_set1_ps(vmax);
  std::size_t i = 0;
  const std::size_t stepped = count & ~static_cast<std::size_t>(7);
  for (; i < stepped; i += 8) {
    __m256 v = _mm256_loadu_ps(data + i);
    v = _mm256_max_ps(v, lo);
    v = _mm256_min_ps(v, hi);
    _mm256_storeu_ps(data + i, v);
  }
  for (; i < count; ++i) {
    data[i] = clamp_scalar_one(data[i], vmin, vmax);
  }
}

}  // namespace tge::cpu
