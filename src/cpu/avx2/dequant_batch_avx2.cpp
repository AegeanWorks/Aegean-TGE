#include "tge/cpu/kernels.hpp"

#include <algorithm>
#include <cstddef>
#include <vector>

#include <immintrin.h>

#include "tge/core/thread_pool.hpp"

namespace tge::cpu {
namespace {

inline float dot_f32(const float* left, const float* right, int count) {
  __m256 acc0 = _mm256_setzero_ps();
  __m256 acc1 = _mm256_setzero_ps();

  int index = 0;
  for (; index + 16 <= count; index += 16) {
    acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(left + index),
                           _mm256_loadu_ps(right + index), acc0);
    acc1 = _mm256_fmadd_ps(_mm256_loadu_ps(left + index + 8),
                           _mm256_loadu_ps(right + index + 8), acc1);
  }
  for (; index + 8 <= count; index += 8) {
    acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(left + index),
                           _mm256_loadu_ps(right + index), acc0);
  }

  const __m256 sum = _mm256_add_ps(acc0, acc1);
  __m128 total = _mm_add_ps(_mm256_castps256_ps128(sum),
                            _mm256_extractf128_ps(sum, 1));
  total = _mm_hadd_ps(total, total);
  total = _mm_hadd_ps(total, total);
  float result = _mm_cvtss_f32(total);
  for (; index < count; ++index) {
    result += left[index] * right[index];
  }
  return result;
}

}  // namespace

void gemv_dequant_batch_avx2(const DequantBatchArgs& args,
                             const GemvConfig& config) {
  if (args.rows <= 0 || args.cols <= 0 || args.tokens <= 0) {
    return;
  }

  const std::size_t row_bytes = quant_row_bytes(args.kind, args.cols);
  const int rows = args.rows;
  const int cols = args.cols;
  const int tokens = args.tokens;

  const int threads = resolve_thread_count(config.threads);
  core::ThreadPool& pool =
      config.pool != nullptr ? *config.pool : core::acquire_pool(threads);

  int band_rows = rows / (threads * 4);
  if (band_rows < 1) {
    band_rows = 1;
  }
  const int bands = (rows + band_rows - 1) / band_rows;

  pool.parallel_for(bands, 1, [&](int band) {
    // One scratch row per worker: the dequantized weights are reused by every
    // token in the group, which is exactly why this beats decoding per token.
    std::vector<float> weights(static_cast<std::size_t>(cols));
    const int begin = band * band_rows;
    const int end = std::min(rows, begin + band_rows);
    const std::uint8_t* row =
        args.data + static_cast<std::size_t>(begin) * row_bytes;

    for (int index = begin; index < end; ++index) {
      dequantize_row_avx2(args.kind, row, weights.data(), cols);
      for (int token = 0; token < tokens; ++token) {
        const float* input = args.input + static_cast<std::size_t>(token) * cols;
        args.out[static_cast<std::size_t>(token) * rows + index] =
            dot_f32(weights.data(), input, cols);
      }
      row += row_bytes;
    }
  });
}

}  // namespace tge::cpu
