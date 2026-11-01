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
/// group: four folds of the same 32 weights. That amortization is the whole
/// point of batching, so the token loop is inside the block loop on purpose.
constexpr int kGroup = 4;

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
  core::ThreadPool& pool = core::acquire_pool(threads);

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
        __m256 acc0 = _mm256_setzero_ps();
        __m256 acc1 = _mm256_setzero_ps();
        __m256 acc2 = _mm256_setzero_ps();
        __m256 acc3 = _mm256_setzero_ps();

        const float* in0 = args.input + static_cast<std::size_t>(token) * cols;
        const float* in1 = in0 + cols;
        const float* in2 = in1 + cols;
        const float* in3 = in2 + cols;

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
          const float* x0 = in0 + offset;
          const float* x1 = in1 + offset;
          const float* x2 = in2 + offset;
          const float* x3 = in3 + offset;

          __m256 s0 = _mm256_mul_ps(w0, _mm256_loadu_ps(x0));
          s0 = _mm256_fmadd_ps(w1, _mm256_loadu_ps(x0 + 8), s0);
          s0 = _mm256_fmadd_ps(w2, _mm256_loadu_ps(x0 + 16), s0);
          s0 = _mm256_fmadd_ps(w3, _mm256_loadu_ps(x0 + 24), s0);
          acc0 = _mm256_fmadd_ps(scale, s0, acc0);

          __m256 s1 = _mm256_mul_ps(w0, _mm256_loadu_ps(x1));
          s1 = _mm256_fmadd_ps(w1, _mm256_loadu_ps(x1 + 8), s1);
          s1 = _mm256_fmadd_ps(w2, _mm256_loadu_ps(x1 + 16), s1);
          s1 = _mm256_fmadd_ps(w3, _mm256_loadu_ps(x1 + 24), s1);
          acc1 = _mm256_fmadd_ps(scale, s1, acc1);

          __m256 s2 = _mm256_mul_ps(w0, _mm256_loadu_ps(x2));
          s2 = _mm256_fmadd_ps(w1, _mm256_loadu_ps(x2 + 8), s2);
          s2 = _mm256_fmadd_ps(w2, _mm256_loadu_ps(x2 + 16), s2);
          s2 = _mm256_fmadd_ps(w3, _mm256_loadu_ps(x2 + 24), s2);
          acc2 = _mm256_fmadd_ps(scale, s2, acc2);

          __m256 s3 = _mm256_mul_ps(w0, _mm256_loadu_ps(x3));
          s3 = _mm256_fmadd_ps(w1, _mm256_loadu_ps(x3 + 8), s3);
          s3 = _mm256_fmadd_ps(w2, _mm256_loadu_ps(x3 + 16), s3);
          s3 = _mm256_fmadd_ps(w3, _mm256_loadu_ps(x3 + 24), s3);
          acc3 = _mm256_fmadd_ps(scale, s3, acc3);
        }

        args.out[static_cast<std::size_t>(token + 0) * rows + row] =
            horizontal_sum(acc0);
        args.out[static_cast<std::size_t>(token + 1) * rows + row] =
            horizontal_sum(acc1);
        args.out[static_cast<std::size_t>(token + 2) * rows + row] =
            horizontal_sum(acc2);
        args.out[static_cast<std::size_t>(token + 3) * rows + row] =
            horizontal_sum(acc3);
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
