#include "tge/cpu/kernels.hpp"

#include <cstddef>

#include <immintrin.h>

namespace tge::cpu {

void copy_stream_avx2(float* dst, const float* src, std::size_t count) {
  std::size_t index = 0;
  for (; index + 8 <= count; index += 8) {
    _mm256_storeu_ps(dst + index, _mm256_loadu_ps(src + index));
  }
  for (; index < count; ++index) {
    dst[index] = src[index];
  }
}

float sum_stream_avx2(const float* src, std::size_t count) {
  __m256 acc_low = _mm256_setzero_ps();
  __m256 acc_high = _mm256_setzero_ps();

  std::size_t index = 0;
  for (; index + 16 <= count; index += 16) {
    acc_low = _mm256_add_ps(acc_low, _mm256_loadu_ps(src + index));
    acc_high = _mm256_add_ps(acc_high, _mm256_loadu_ps(src + index + 8));
  }

  const __m256 combined = _mm256_add_ps(acc_low, acc_high);
  alignas(32) float lanes[8];
  _mm256_store_ps(lanes, combined);

  float total = 0.0F;
  for (int lane = 0; lane < 8; ++lane) {
    total += lanes[lane];
  }
  for (; index < count; ++index) {
    total += src[index];
  }
  return total;
}

}  // namespace tge::cpu
