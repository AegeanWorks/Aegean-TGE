#include "tge/cpu/kernels.hpp"

#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <vector>

#include <immintrin.h>

#include "tge/core/thread_pool.hpp"

namespace tge::cpu {
namespace {

inline float dot_f32(const float* left, const float* right, int count) {
  // Four accumulators hide the ~4-5 cycle FMA latency; two starve it.
  // Tail reduction uses shuffles (1 uop each) instead of hadd (3 uops).
  __m256 acc0 = _mm256_setzero_ps();
  __m256 acc1 = _mm256_setzero_ps();
  __m256 acc2 = _mm256_setzero_ps();
  __m256 acc3 = _mm256_setzero_ps();

  int index = 0;
  for (; index + 32 <= count; index += 32) {
    acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(left + index),
                           _mm256_loadu_ps(right + index), acc0);
    acc1 = _mm256_fmadd_ps(_mm256_loadu_ps(left + index + 8),
                           _mm256_loadu_ps(right + index + 8), acc1);
    acc2 = _mm256_fmadd_ps(_mm256_loadu_ps(left + index + 16),
                           _mm256_loadu_ps(right + index + 16), acc2);
    acc3 = _mm256_fmadd_ps(_mm256_loadu_ps(left + index + 24),
                           _mm256_loadu_ps(right + index + 24), acc3);
  }
  for (; index + 8 <= count; index += 8) {
    acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(left + index),
                           _mm256_loadu_ps(right + index), acc0);
  }

  const __m256 sum01 = _mm256_add_ps(acc0, acc1);
  const __m256 sum23 = _mm256_add_ps(acc2, acc3);
  const __m256 sum = _mm256_add_ps(sum01, sum23);
  __m128 total = _mm_add_ps(_mm256_castps256_ps128(sum),
                            _mm256_extractf128_ps(sum, 1));
  total = _mm_add_ps(total, _mm_movehl_ps(total, total));
  total = _mm_add_ss(total, _mm_shuffle_ps(total, total, 1));
  float result = _mm_cvtss_f32(total);
  for (; index < count; ++index) {
    result += left[index] * right[index];
  }
  return result;
}

inline void dot_pair_f32(const float* left0, const float* left1,
                         const float* right, int count, float& result0,
                         float& result1) {
  __m256 a0 = _mm256_setzero_ps();
  __m256 a1 = _mm256_setzero_ps();
  __m256 a2 = _mm256_setzero_ps();
  __m256 a3 = _mm256_setzero_ps();
  __m256 b0 = _mm256_setzero_ps();
  __m256 b1 = _mm256_setzero_ps();
  __m256 b2 = _mm256_setzero_ps();
  __m256 b3 = _mm256_setzero_ps();
  int index = 0;
  for (; index + 32 <= count; index += 32) {
    const __m256 x0 = _mm256_loadu_ps(right + index);
    const __m256 x1 = _mm256_loadu_ps(right + index + 8);
    const __m256 x2 = _mm256_loadu_ps(right + index + 16);
    const __m256 x3 = _mm256_loadu_ps(right + index + 24);
    a0 = _mm256_fmadd_ps(_mm256_loadu_ps(left0 + index), x0, a0);
    a1 = _mm256_fmadd_ps(_mm256_loadu_ps(left0 + index + 8), x1, a1);
    a2 = _mm256_fmadd_ps(_mm256_loadu_ps(left0 + index + 16), x2, a2);
    a3 = _mm256_fmadd_ps(_mm256_loadu_ps(left0 + index + 24), x3, a3);
    b0 = _mm256_fmadd_ps(_mm256_loadu_ps(left1 + index), x0, b0);
    b1 = _mm256_fmadd_ps(_mm256_loadu_ps(left1 + index + 8), x1, b1);
    b2 = _mm256_fmadd_ps(_mm256_loadu_ps(left1 + index + 16), x2, b2);
    b3 = _mm256_fmadd_ps(_mm256_loadu_ps(left1 + index + 24), x3, b3);
  }
  for (; index + 8 <= count; index += 8) {
    const __m256 x = _mm256_loadu_ps(right + index);
    a0 = _mm256_fmadd_ps(_mm256_loadu_ps(left0 + index), x, a0);
    b0 = _mm256_fmadd_ps(_mm256_loadu_ps(left1 + index), x, b0);
  }
  const __m256 sum_a = _mm256_add_ps(_mm256_add_ps(a0, a1),
                                     _mm256_add_ps(a2, a3));
  const __m256 sum_b = _mm256_add_ps(_mm256_add_ps(b0, b1),
                                     _mm256_add_ps(b2, b3));
  __m128 total_a = _mm_add_ps(_mm256_castps256_ps128(sum_a),
                              _mm256_extractf128_ps(sum_a, 1));
  __m128 total_b = _mm_add_ps(_mm256_castps256_ps128(sum_b),
                              _mm256_extractf128_ps(sum_b, 1));
  total_a = _mm_add_ps(total_a, _mm_movehl_ps(total_a, total_a));
  total_b = _mm_add_ps(total_b, _mm_movehl_ps(total_b, total_b));
  total_a = _mm_add_ss(total_a, _mm_shuffle_ps(total_a, total_a, 1));
  total_b = _mm_add_ss(total_b, _mm_shuffle_ps(total_b, total_b, 1));
  result0 = _mm_cvtss_f32(total_a);
  result1 = _mm_cvtss_f32(total_b);
  for (; index < count; ++index) {
    result0 += left0[index] * right[index];
    result1 += left1[index] * right[index];
  }
}

}  // namespace

void gemv_q4k_fused_avx2(const DequantBatchArgs& args,
                         const GemvConfig& config) {
  if (args.kind != QuantKind::Q4_K || args.tokens != 1) {
    gemv_dequant_batch_avx2(args, config);
    return;
  }
  if (args.rows <= 0 || args.cols <= 0 || args.tokens <= 0) {
    return;
  }
  if (args.data == nullptr || args.input == nullptr || args.out == nullptr) {
    throw std::invalid_argument("gemv_q4k_fused: null data/input/out");
  }

  const std::size_t row_bytes = quant_row_bytes(args.kind, args.cols);
  const int rows = args.rows;
  const int cols = args.cols;

  const int threads = resolve_thread_count(config.threads);
  core::ThreadPool& pool =
      config.pool != nullptr ? *config.pool : core::acquire_pool(threads);

  int band_rows = rows / (threads * 4);
  if (band_rows < 1) {
    band_rows = 1;
  }
  const int bands = (rows + band_rows - 1) / band_rows;

  pool.parallel_for(bands, 1, [&](int band) {
    const int begin = band * band_rows;
    const int end = std::min(rows, begin + band_rows);
    const std::uint8_t* row =
        args.data + static_cast<std::size_t>(begin) * row_bytes;
    for (int index = begin; index < end; ++index) {
      args.out[index] = dot_q4k_row_avx2(row, args.input, cols);
      row += row_bytes;
    }
  });
}

void gemv_q6k_fused_avx2(const DequantBatchArgs& args,
                         const GemvConfig& config) {
  if (args.kind != QuantKind::Q6_K || args.tokens != 1) {
    gemv_dequant_batch_avx2(args, config);
    return;
  }
  if (args.rows <= 0 || args.cols <= 0 || args.tokens <= 0) {
    return;
  }
  if (args.data == nullptr || args.input == nullptr || args.out == nullptr) {
    throw std::invalid_argument("gemv_q6k_fused: null data/input/out");
  }

  const std::size_t row_bytes = quant_row_bytes(args.kind, args.cols);
  const int rows = args.rows;
  const int cols = args.cols;

  const int threads = resolve_thread_count(config.threads);
  core::ThreadPool& pool =
      config.pool != nullptr ? *config.pool : core::acquire_pool(threads);

  int band_rows = rows / (threads * 4);
  if (band_rows < 1) {
    band_rows = 1;
  }
  const int bands = (rows + band_rows - 1) / band_rows;

  pool.parallel_for(bands, 1, [&](int band) {
    const int begin = band * band_rows;
    const int end = std::min(rows, begin + band_rows);
    const std::uint8_t* row =
        args.data + static_cast<std::size_t>(begin) * row_bytes;
    for (int index = begin; index < end; ++index) {
      args.out[index] = dot_q6k_row_avx2(row, args.input, cols);
      row += row_bytes;
    }
  });
}

void gemv_dequant_batch_avx2(const DequantBatchArgs& args,
                             const GemvConfig& config) {
  if (args.rows <= 0 || args.cols <= 0 || args.tokens <= 0) {
    return;
  }
  if (args.data == nullptr || args.input == nullptr || args.out == nullptr) {
    throw std::invalid_argument("gemv_dequant_batch: null data/input/out");
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
    const int begin = band * band_rows;
    const int end = std::min(rows, begin + band_rows);
    const std::uint8_t* row =
        args.data + static_cast<std::size_t>(begin) * row_bytes;

    // One scratch row per worker thread, sized once and reused across calls:
    // the dequantized weights are reused by every token in the group. A fresh
    // std::vector per band would reallocate and zero-fill cols floats on every
    // call (~bands x 169 calls per decode token); the fill is pure waste
    // since dequantize_row_avx2 overwrites every element.
    thread_local std::vector<float> scratch;
    if (scratch.size() < static_cast<std::size_t>(cols)) {
      scratch.resize(static_cast<std::size_t>(cols));
    }
    float* weights = scratch.data();

    for (int index = begin; index < end; ++index) {
      dequantize_row_avx2(args.kind, row, weights, cols);
      for (int token = 0; token < tokens; ++token) {
        const float* input = args.input + static_cast<std::size_t>(token) * cols;
        args.out[static_cast<std::size_t>(token) * rows + index] =
            dot_f32(weights, input, cols);
      }
      row += row_bytes;
    }
  });
}

void gemv_dequant_batch_tiled_avx2(const DequantBatchArgs& args,
                                   const GemvConfig& config) {
  if (args.rows <= 0 || args.cols <= 0 || args.tokens <= 0) {
    return;
  }
  if (args.data == nullptr || args.input == nullptr || args.out == nullptr) {
    throw std::invalid_argument("gemv_dequant_batch_tiled: null data/input/out");
  }
  if (args.rows < 2 || args.rows % 2 != 0) {
    gemv_dequant_batch_avx2(args, config);
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
  if ((band_rows & 1) != 0) {
    ++band_rows;
  }
  const int bands = (rows + band_rows - 1) / band_rows;
  pool.parallel_for(bands, 1, [&](int band) {
    thread_local std::vector<float> scratch0;
    thread_local std::vector<float> scratch1;
    if (scratch0.size() < static_cast<std::size_t>(cols)) {
      scratch0.resize(static_cast<std::size_t>(cols));
      scratch1.resize(static_cast<std::size_t>(cols));
    }
    const int begin = band * band_rows;
    const int end = std::min(rows, begin + band_rows);
    const std::uint8_t* row0 =
        args.data + static_cast<std::size_t>(begin) * row_bytes;
    for (int row = begin; row < end; row += 2) {
      const std::uint8_t* row1 = row0 + row_bytes;
      dequantize_row_avx2(args.kind, row0, scratch0.data(), cols);
      dequantize_row_avx2(args.kind, row1, scratch1.data(), cols);
      for (int token = 0; token < tokens; ++token) {
        const float* input =
            args.input + static_cast<std::size_t>(token) * cols;
        float value0 = 0.0F;
        float value1 = 0.0F;
        dot_pair_f32(scratch0.data(), scratch1.data(), input, cols, value0,
                     value1);
        args.out[static_cast<std::size_t>(token) * rows + row] = value0;
        args.out[static_cast<std::size_t>(token) * rows + row + 1] = value1;
      }
      row0 = row1 + row_bytes;
    }
  });
}

}  // namespace tge::cpu
