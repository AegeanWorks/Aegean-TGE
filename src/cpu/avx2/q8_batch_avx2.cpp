#include "tge/cpu/kernels.hpp"

#include <algorithm>
#include <cstddef>
#include <cstring>

#include <immintrin.h>

#include "tge/core/thread_pool.hpp"

namespace tge::cpu {
namespace {

/// Eight signed bytes widened to eight floats.
inline __m256 widen(const std::int8_t* values) {
  const __m128i packed =
      _mm_loadl_epi64(reinterpret_cast<const __m128i*>(values));
  return _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(packed));
}

inline float horizontal_sum(__m256 values) {
  __m128 total = _mm_add_ps(_mm256_castps256_ps128(values),
                            _mm256_extractf128_ps(values, 1));
  total = _mm_hadd_ps(total, total);
  total = _mm_hadd_ps(total, total);
  return _mm_cvtss_f32(total);
}

/// One output row against `kGroup` tokens at once.
///
/// The weight block is widened once and then applied to every token in the
/// group: six folds of the same 32 weights. That amortization is the whole
/// point of batching, so the token loop is inside the block loop on purpose.
/// Six is a register-budget choice: six accumulators plus four weight vectors
/// plus the per-token temporaries fit the sixteen YMM registers without
/// spilling, and eight does not.
constexpr int kGroup = 6;

}  // namespace

void gemv_q8_batch_avx2(const Q8BatchArgs& args, const GemvConfig& config) {
  if (args.rows <= 0 || args.cols <= 0 || args.tokens <= 0) {
    return;
  }

  const int blocks = args.cols / kQ8BlockElements;
  const std::size_t row_bytes =
      static_cast<std::size_t>(blocks) * kQ8BlockBytes;
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
    const std::uint8_t* weight_row =
        args.data + static_cast<std::size_t>(begin) * row_bytes;

    for (int row = begin; row < end; ++row) {
      int token = 0;

      for (; token + kGroup <= tokens; token += kGroup) {
        __m256 accumulators[kGroup];
        const float* inputs[kGroup];
        for (int index = 0; index < kGroup; ++index) {
          accumulators[index] = _mm256_setzero_ps();
          inputs[index] = args.input +
                          static_cast<std::size_t>(token + index) * cols;
        }

        for (int block = 0; block < blocks; ++block) {
          const std::uint8_t* entry = weight_row + block * kQ8BlockBytes;
          std::uint16_t scale_bits = 0;
          std::memcpy(&scale_bits, entry, sizeof(scale_bits));
          const __m256 scale = _mm256_set1_ps(half_to_float(scale_bits));

          const auto* values = reinterpret_cast<const std::int8_t*>(entry + 2);
          const __m256 w0 = widen(values + 0);
          const __m256 w1 = widen(values + 8);
          const __m256 w2 = widen(values + 16);
          const __m256 w3 = widen(values + 24);

          const int offset = block * kQ8BlockElements;
          for (int index = 0; index < kGroup; ++index) {
            const float* x = inputs[index] + offset;
            __m256 sum = _mm256_mul_ps(w0, _mm256_loadu_ps(x));
            sum = _mm256_fmadd_ps(w1, _mm256_loadu_ps(x + 8), sum);
            sum = _mm256_fmadd_ps(w2, _mm256_loadu_ps(x + 16), sum);
            sum = _mm256_fmadd_ps(w3, _mm256_loadu_ps(x + 24), sum);
            accumulators[index] =
                _mm256_fmadd_ps(scale, sum, accumulators[index]);
          }
        }

        for (int index = 0; index < kGroup; ++index) {
          args.out[static_cast<std::size_t>(token + index) * rows + row] =
              horizontal_sum(accumulators[index]);
        }
      }

      // Tail tokens that do not fill a group.
      for (; token < tokens; ++token) {
        const float* input = args.input + static_cast<std::size_t>(token) * cols;
        __m256 total = _mm256_setzero_ps();
        for (int block = 0; block < blocks; ++block) {
          const std::uint8_t* entry = weight_row + block * kQ8BlockBytes;
          std::uint16_t scale_bits = 0;
          std::memcpy(&scale_bits, entry, sizeof(scale_bits));
          const __m256 scale = _mm256_set1_ps(half_to_float(scale_bits));

          const auto* values = reinterpret_cast<const std::int8_t*>(entry + 2);
          const float* x = input + block * kQ8BlockElements;
          __m256 sum = _mm256_mul_ps(widen(values + 0), _mm256_loadu_ps(x));
          sum = _mm256_fmadd_ps(widen(values + 8), _mm256_loadu_ps(x + 8), sum);
          sum = _mm256_fmadd_ps(widen(values + 16), _mm256_loadu_ps(x + 16), sum);
          sum = _mm256_fmadd_ps(widen(values + 24), _mm256_loadu_ps(x + 24), sum);
          total = _mm256_fmadd_ps(scale, sum, total);
        }
        args.out[static_cast<std::size_t>(token) * rows + row] =
            horizontal_sum(total);
      }

      weight_row += row_bytes;
    }
  });
}

}  // namespace tge::cpu
