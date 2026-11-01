#include "tge/cpu/kernels.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <vector>

#include <immintrin.h>

#include "tge/core/thread_pool.hpp"

namespace tge::cpu {
namespace {

inline float horizontal_sum(__m256 values) {
  __m128 total = _mm_add_ps(_mm256_castps256_ps128(values),
                            _mm256_extractf128_ps(values, 1));
  total = _mm_hadd_ps(total, total);
  total = _mm_hadd_ps(total, total);
  return _mm_cvtss_f32(total);
}

inline float quantize_block(const float* source, std::int8_t* destination) {
  float maximum = 0.0F;
  for (int index = 0; index < kQ8BlockElements; ++index) {
    maximum = std::max(maximum, std::fabs(source[index]));
  }
  const float scale = maximum / 127.0F;
  const float inverse = scale != 0.0F ? 1.0F / scale : 0.0F;
  for (int index = 0; index < kQ8BlockElements; ++index) {
    const float scaled = source[index] * inverse;
    const float rounded = scaled >= 0.0F ? scaled + 0.5F : scaled - 0.5F;
    const int value = static_cast<int>(rounded);
    destination[index] = static_cast<std::int8_t>(
        value > 127 ? 127 : (value < -127 ? -127 : value));
  }
  return scale;
}

/// Sum of each group of four signed bytes, as eight int32 lanes.
inline __m256i lane_sums(__m256i bytes, __m256i ones8, __m256i ones16) {
  const __m256i pairs = _mm256_maddubs_epi16(ones8, bytes);
  return _mm256_madd_epi16(pairs, ones16);
}

constexpr int kGroup = 6;

}  // namespace

void gemv_q8_batch_vnni(const Q8BatchArgs& args, const GemvConfig& config) {
  if (args.rows <= 0 || args.cols <= 0 || args.tokens <= 0) {
    return;
  }

  const int blocks = args.cols / kQ8BlockElements;
  const std::size_t row_bytes =
      static_cast<std::size_t>(blocks) * kQ8BlockBytes;
  const int rows = args.rows;
  const int cols = args.cols;
  const int tokens = args.tokens;

  // Reused across calls: allocating per matvec showed up as a real cost when
  // the model calls this ~150 times per chunk.
  thread_local std::vector<std::int8_t> quantized;
  thread_local std::vector<float> scales;
  const std::size_t value_count =
      static_cast<std::size_t>(tokens) * static_cast<std::size_t>(cols);
  const std::size_t scale_count =
      static_cast<std::size_t>(tokens) * static_cast<std::size_t>(blocks);
  if (quantized.size() < value_count) {
    quantized.resize(value_count);
  }
  if (scales.size() < scale_count) {
    scales.resize(scale_count);
  }

  // The workers must read the calling thread's buffers, so take plain pointers:
  // naming the thread_local vectors inside the lambda would give each worker its
  // own empty instance.
  std::int8_t* quantized_data = quantized.data();
  float* scale_data = scales.data();

  for (int token = 0; token < tokens; ++token) {
    const float* source = args.input + static_cast<std::size_t>(token) * cols;
    std::int8_t* target =
        quantized_data + static_cast<std::size_t>(token) * cols;
    for (int block = 0; block < blocks; ++block) {
      scale_data[static_cast<std::size_t>(token) * blocks + block] = quantize_block(
          source + block * kQ8BlockElements,
          target + block * kQ8BlockElements);
    }
  }

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
    const std::uint8_t* weight_row =
        args.data + static_cast<std::size_t>(begin) * row_bytes;
    const __m256i bias = _mm256_set1_epi8(static_cast<char>(0x80));
    const __m256i ones8 = _mm256_set1_epi8(1);
    const __m256i ones16 = _mm256_set1_epi16(1);

    for (int row = begin; row < end; ++row) {
      int token = 0;

      for (; token + kGroup <= tokens; token += kGroup) {
        __m256 accumulators[kGroup];
        for (int index = 0; index < kGroup; ++index) {
          accumulators[index] = _mm256_setzero_ps();
        }

        for (int block = 0; block < blocks; ++block) {
          const std::uint8_t* entry = weight_row + block * kQ8BlockBytes;
          std::uint16_t scale_bits = 0;
          std::memcpy(&scale_bits, entry, sizeof(scale_bits));
          const float weight_scale = half_to_float(scale_bits);

          const __m256i weights = _mm256_loadu_si256(
              reinterpret_cast<const __m256i*>(entry + 2));
          // The activation is the biased operand, so the correction depends
          // only on the weights and is computed once per block.
          const __m256i correction =
              _mm256_slli_epi32(lane_sums(weights, ones8, ones16), 7);

          for (int index = 0; index < kGroup; ++index) {
            const __m256i values = _mm256_loadu_si256(
                reinterpret_cast<const __m256i*>(quantized_data +
                                                 static_cast<std::size_t>(
                                                     token + index) *
                                                     cols +
                                                 block * kQ8BlockElements));
            const __m256i biased = _mm256_xor_si256(values, bias);
            __m256i sums =
                _mm256_dpbusd_epi32(_mm256_setzero_si256(), biased, weights);
            sums = _mm256_sub_epi32(sums, correction);

            const float scale =
                weight_scale *
                scale_data[static_cast<std::size_t>(token + index) * blocks + block];
            accumulators[index] = _mm256_fmadd_ps(
                _mm256_cvtepi32_ps(sums), _mm256_set1_ps(scale),
                accumulators[index]);
          }
        }

        for (int index = 0; index < kGroup; ++index) {
          args.out[static_cast<std::size_t>(token + index) * rows + row] =
              horizontal_sum(accumulators[index]);
        }
      }

      for (; token < tokens; ++token) {
        __m256 total = _mm256_setzero_ps();
        for (int block = 0; block < blocks; ++block) {
          const std::uint8_t* entry = weight_row + block * kQ8BlockBytes;
          std::uint16_t scale_bits = 0;
          std::memcpy(&scale_bits, entry, sizeof(scale_bits));
          const float weight_scale = half_to_float(scale_bits);

          const __m256i weights = _mm256_loadu_si256(
              reinterpret_cast<const __m256i*>(entry + 2));
          const __m256i correction =
              _mm256_slli_epi32(lane_sums(weights, ones8, ones16), 7);
          const __m256i values = _mm256_loadu_si256(
              reinterpret_cast<const __m256i*>(
                  quantized_data + static_cast<std::size_t>(token) * cols +
                  block * kQ8BlockElements));
          const __m256i biased = _mm256_xor_si256(values, bias);

          __m256i sums =
              _mm256_dpbusd_epi32(_mm256_setzero_si256(), biased, weights);
          sums = _mm256_sub_epi32(sums, correction);

          const float scale =
              weight_scale *
              scale_data[static_cast<std::size_t>(token) * blocks + block];
          total = _mm256_fmadd_ps(_mm256_cvtepi32_ps(sums),
                                  _mm256_set1_ps(scale), total);
        }
        args.out[static_cast<std::size_t>(token) * rows + row] =
            horizontal_sum(total);
      }

      weight_row += row_bytes;
    }
  });
}

}  // namespace tge::cpu
