#include "tge/cpu/kernels.hpp"

#include <algorithm>
#include <cstddef>
#include <cstring>
#include <stdexcept>

#include <immintrin.h>

#include "tge/core/thread_pool.hpp"

namespace tge::cpu {
namespace {

/// One output row of a Q8_0 matrix against a float activation vector.
///
/// Per 34-byte block: a half-precision scale, then 32 signed bytes. The int8
/// payload is widened to int32 and then to float so the multiply stays in the
/// vector unit; the scale is applied once per block rather than per element.
///
/// An integer-domain variant (quantize the activation too, then `maddubs` with
/// no widening at all) was implemented and measured: 31.0 tokens/s median
/// against 30.3-34.0 for this version, which is inside the noise band on this
/// machine. It was reverted because it costs exact token parity with llama.cpp
/// and buys nothing measurable. Decode here is limited by memory and per-call
/// synchronisation, not by the widening chain.
inline float dot_row_q8(const std::uint8_t* row, const float* vector,
                        int blocks) {
  __m256 total = _mm256_setzero_ps();

  for (int block = 0; block < blocks; ++block) {
    const std::uint8_t* entry = row + block * kQ8BlockBytes;
    std::uint16_t scale_bits = 0;
    std::memcpy(&scale_bits, entry, sizeof(scale_bits));
    const __m256 scale = _mm256_set1_ps(half_to_float(scale_bits));

    const auto* values = reinterpret_cast<const std::int8_t*>(entry + 2);
    const float* vector_block = vector + block * kQ8BlockElements;

    __m256 accumulator = _mm256_setzero_ps();
    for (int index = 0; index < kQ8BlockElements; index += 8) {
      const __m128i packed =
          _mm_loadl_epi64(reinterpret_cast<const __m128i*>(values + index));
      const __m256i widened = _mm256_cvtepi8_epi32(packed);
      const __m256 as_float = _mm256_cvtepi32_ps(widened);
      accumulator = _mm256_fmadd_ps(as_float,
                                    _mm256_loadu_ps(vector_block + index),
                                    accumulator);
    }
    total = _mm256_fmadd_ps(scale, accumulator, total);
  }

  const __m128 low = _mm256_castps256_ps128(total);
  const __m128 high = _mm256_extractf128_ps(total, 1);
  __m128 sum = _mm_add_ps(low, high);
  sum = _mm_add_ps(sum, _mm_movehl_ps(sum, sum));
  sum = _mm_add_ss(sum, _mm_shuffle_ps(sum, sum, 1));
  return _mm_cvtss_f32(sum);
}

}  // namespace

void gemv_q8_0_avx2(const Q8GemvArgs& args, const GemvConfig& config) {
  if (args.rows <= 0 || args.cols <= 0) {
    return;
  }
  if (args.cols % kQ8BlockElements != 0) {
    throw std::invalid_argument("gemv_q8_0: cols not a whole block");
  }

  const int threads = resolve_thread_count(config.threads);
  core::ThreadPool& pool =
      config.pool != nullptr ? *config.pool : core::acquire_pool(threads);

  const int blocks = args.cols / kQ8BlockElements;
  const std::size_t row_bytes =
      static_cast<std::size_t>(blocks) * kQ8BlockBytes;

  int band_rows = args.rows / (threads * 4);
  if (band_rows < 1) {
    band_rows = 1;
  }
  const int bands = (args.rows + band_rows - 1) / band_rows;

  pool.parallel_for(bands, 1, [&](int band) {
    const int begin = band * band_rows;
    const int end = std::min(args.rows, begin + band_rows);
    const std::uint8_t* row =
        args.data + static_cast<std::size_t>(begin) * row_bytes;
    for (int index = begin; index < end; ++index) {
      args.out[index] = dot_row_q8(row, args.vector, blocks);
      row += row_bytes;
    }
  });
}

}  // namespace tge::cpu
