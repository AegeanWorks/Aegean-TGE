#include "tge/cpu/kernels.hpp"

#include <algorithm>
#include <cstddef>
#include <cstring>
#include <stdexcept>
#include <vector>

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
  total = _mm_add_ps(total, _mm_movehl_ps(total, total));
  total = _mm_add_ss(total, _mm_shuffle_ps(total, total, 1));
  return _mm_cvtss_f32(total);
}

/// One output row against `kGroup` tokens at once.
///
/// The weight block is widened once and then applied to every token in the
/// group: eight folds of the same 32 weights. That amortization is the whole
/// point of batching, so the token loop is inside the block loop on purpose.
/// Eight was measured against six on this compiler/CPU (+6-7% prefill tok/s,
/// 46/48 pairs); the old claim that eight spills the YMM registers did not
/// hold. See the 2026-11-21 measurement-log entry. The full-group case uses a
/// compile-time-unrolled fold so the eight accumulators stay in registers
/// (another +2% prefill, 40/48 pairs) rather than a runtime-indexed stack array.
constexpr int kGroup = 8;

}  // namespace

void gemv_q8_batch_avx2(const Q8BatchArgs& args, const GemvConfig& config) {
  if (args.rows <= 0 || args.cols <= 0 || args.tokens <= 0) {
    return;
  }
  if (args.cols % kQ8BlockElements != 0) {
    throw std::invalid_argument("gemv_q8_batch: cols not a whole block");
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

  int band_rows = rows / (threads * 8);
  if (band_rows < 1) {
    band_rows = 1;
  }
  const int bands = (rows + band_rows - 1) / band_rows;

  pool.parallel_for(bands, 1, [&](int band) {
    // One scale row per worker, filled once per weight row and reused by every
    // token group. The same branchy half_to_float was previously recomputed for
    // each of the (tokens/kGroup) groups walking the identical row.
    thread_local std::vector<float> scale_row;
    if (static_cast<int>(scale_row.size()) < blocks) {
      scale_row.resize(static_cast<std::size_t>(blocks));
    }
    const int begin = band * band_rows;
    const int end = std::min(rows, begin + band_rows);
    const std::uint8_t* weight_row =
        args.data + static_cast<std::size_t>(begin) * row_bytes;

    for (int row = begin; row < end; ++row) {
      int token = 0;
      // Convert this row's scales once; every token group reuses them.
      for (int b = 0; b < blocks; ++b) {
        std::uint16_t sb = 0;
        std::memcpy(&sb, weight_row + b * kQ8BlockBytes, sizeof(sb));
        scale_row[static_cast<std::size_t>(b)] = half_to_float(sb);
      }

      // Full and partial groups share one widened weight block across every
      // token in the group. The old tail re-read the same block once per
      // token, so 5 tokens cost more than 6 (5 weight reads vs 1). A 64-token
      // prefill chunk is 10 full groups plus a 4-token tail, so every chunk
      // paid it; the speculative verifier's 6-token pass never did.
      while (token < tokens) {
        const int group = std::min(kGroup, tokens - token);

        if (group == kGroup) {
          // Fully-unrolled full group: named accumulators stay in YMM registers
          // (the generic path's runtime-indexed array spills them to the stack
          // every block). Per-token FMA order is identical => byte-identical.
          __m256 a0 = _mm256_setzero_ps();
          __m256 a1 = _mm256_setzero_ps();
          __m256 a2 = _mm256_setzero_ps();
          __m256 a3 = _mm256_setzero_ps();
          __m256 a4 = _mm256_setzero_ps();
          __m256 a5 = _mm256_setzero_ps();
          __m256 a6 = _mm256_setzero_ps();
          __m256 a7 = _mm256_setzero_ps();
          const float* x0 = args.input + static_cast<std::size_t>(token + 0) * cols;
          const float* x1 = args.input + static_cast<std::size_t>(token + 1) * cols;
          const float* x2 = args.input + static_cast<std::size_t>(token + 2) * cols;
          const float* x3 = args.input + static_cast<std::size_t>(token + 3) * cols;
          const float* x4 = args.input + static_cast<std::size_t>(token + 4) * cols;
          const float* x5 = args.input + static_cast<std::size_t>(token + 5) * cols;
          const float* x6 = args.input + static_cast<std::size_t>(token + 6) * cols;
          const float* x7 = args.input + static_cast<std::size_t>(token + 7) * cols;
          for (int block = 0; block < blocks; ++block) {
            const std::uint8_t* entry = weight_row + block * kQ8BlockBytes;
            const __m256 scale =
                _mm256_set1_ps(scale_row[static_cast<std::size_t>(block)]);
            const auto* values = reinterpret_cast<const std::int8_t*>(entry + 2);
            const __m256 w0 = widen(values + 0);
            const __m256 w1 = widen(values + 8);
            const __m256 w2 = widen(values + 16);
            const __m256 w3 = widen(values + 24);
            const int offset = block * kQ8BlockElements;
#define TGE_FOLD(a, x)                                                        \
  {                                                                           \
    __m256 sum = _mm256_mul_ps(w0, _mm256_loadu_ps(x + offset));              \
    sum = _mm256_fmadd_ps(w1, _mm256_loadu_ps(x + offset + 8), sum);           \
    sum = _mm256_fmadd_ps(w2, _mm256_loadu_ps(x + offset + 16), sum);          \
    sum = _mm256_fmadd_ps(w3, _mm256_loadu_ps(x + offset + 24), sum);          \
    (a) = _mm256_fmadd_ps(scale, sum, (a));                                   \
  }
            TGE_FOLD(a0, x0);
            TGE_FOLD(a1, x1);
            TGE_FOLD(a2, x2);
            TGE_FOLD(a3, x3);
            TGE_FOLD(a4, x4);
            TGE_FOLD(a5, x5);
            TGE_FOLD(a6, x6);
            TGE_FOLD(a7, x7);
#undef TGE_FOLD
          }
          args.out[static_cast<std::size_t>(token + 0) * rows + row] = horizontal_sum(a0);
          args.out[static_cast<std::size_t>(token + 1) * rows + row] = horizontal_sum(a1);
          args.out[static_cast<std::size_t>(token + 2) * rows + row] = horizontal_sum(a2);
          args.out[static_cast<std::size_t>(token + 3) * rows + row] = horizontal_sum(a3);
          args.out[static_cast<std::size_t>(token + 4) * rows + row] = horizontal_sum(a4);
          args.out[static_cast<std::size_t>(token + 5) * rows + row] = horizontal_sum(a5);
          args.out[static_cast<std::size_t>(token + 6) * rows + row] = horizontal_sum(a6);
          args.out[static_cast<std::size_t>(token + 7) * rows + row] = horizontal_sum(a7);
          token += kGroup;
          continue;
        }

        __m256 accumulators[kGroup];
        const float* inputs[kGroup];
        for (int index = 0; index < group; ++index) {
          accumulators[index] = _mm256_setzero_ps();
          inputs[index] = args.input +
                          static_cast<std::size_t>(token + index) * cols;
        }

        for (int block = 0; block < blocks; ++block) {
          const std::uint8_t* entry = weight_row + block * kQ8BlockBytes;
          const __m256 scale =
              _mm256_set1_ps(scale_row[static_cast<std::size_t>(block)]);

          const auto* values = reinterpret_cast<const std::int8_t*>(entry + 2);
          const __m256 w0 = widen(values + 0);
          const __m256 w1 = widen(values + 8);
          const __m256 w2 = widen(values + 16);
          const __m256 w3 = widen(values + 24);

          const int offset = block * kQ8BlockElements;
          for (int index = 0; index < group; ++index) {
            const float* x = inputs[index] + offset;
            __m256 sum = _mm256_mul_ps(w0, _mm256_loadu_ps(x));
            sum = _mm256_fmadd_ps(w1, _mm256_loadu_ps(x + 8), sum);
            sum = _mm256_fmadd_ps(w2, _mm256_loadu_ps(x + 16), sum);
            sum = _mm256_fmadd_ps(w3, _mm256_loadu_ps(x + 24), sum);
            accumulators[index] =
                _mm256_fmadd_ps(scale, sum, accumulators[index]);
          }
        }

        for (int index = 0; index < group; ++index) {
          args.out[static_cast<std::size_t>(token + index) * rows + row] =
              horizontal_sum(accumulators[index]);
        }
        token += group;
      }

      weight_row += row_bytes;
    }
  });
}

void gemv_q8_batch_tiled_avx2(const Q8BatchArgs& args,
                              const GemvConfig& config) {
  if (args.rows <= 0 || args.cols <= 0 || args.tokens <= 0) {
    return;
  }
  if (args.cols % kQ8BlockElements != 0) {
    throw std::invalid_argument("gemv_q8_batch_tiled: cols not a whole block");
  }
  if (args.rows < 2 || args.rows % 2 != 0 || args.tokens % 8 != 0) {
    gemv_q8_batch_avx2(args, config);
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

  int band_rows = rows / (threads * 8);
  if (band_rows < 1) {
    band_rows = 1;
  }
  if ((band_rows & 1) != 0) {
    ++band_rows;
  }
  const int bands = (rows + band_rows - 1) / band_rows;

  pool.parallel_for(bands, 1, [&](int band) {
    thread_local std::vector<float> scale0;
    thread_local std::vector<float> scale1;
    if (static_cast<int>(scale0.size()) < blocks) {
      scale0.resize(static_cast<std::size_t>(blocks));
      scale1.resize(static_cast<std::size_t>(blocks));
    }
    const int begin = band * band_rows;
    const int end = std::min(rows, begin + band_rows);
    const std::uint8_t* weight0 =
        args.data + static_cast<std::size_t>(begin) * row_bytes;
    for (int row = begin; row < end; row += 2) {
      const std::uint8_t* weight1 = weight0 + row_bytes;
      for (int block = 0; block < blocks; ++block) {
        std::uint16_t bits0 = 0;
        std::uint16_t bits1 = 0;
        std::memcpy(&bits0, weight0 + block * kQ8BlockBytes, 2);
        std::memcpy(&bits1, weight1 + block * kQ8BlockBytes, 2);
        scale0[static_cast<std::size_t>(block)] = half_to_float(bits0);
        scale1[static_cast<std::size_t>(block)] = half_to_float(bits1);
      }
      for (int token = 0; token < tokens; token += 8) {
        __m256 r0a0 = _mm256_setzero_ps();
        __m256 r0a1 = _mm256_setzero_ps();
        __m256 r0a2 = _mm256_setzero_ps();
        __m256 r0a3 = _mm256_setzero_ps();
        __m256 r0a4 = _mm256_setzero_ps();
        __m256 r0a5 = _mm256_setzero_ps();
        __m256 r0a6 = _mm256_setzero_ps();
        __m256 r0a7 = _mm256_setzero_ps();
        alignas(32) __m256 r1[8];
        for (__m256& value : r1) {
          value = _mm256_setzero_ps();
        }
        const float* inputs[8];
        for (int index = 0; index < 8; ++index) {
          inputs[index] = args.input +
                          static_cast<std::size_t>(token + index) * cols;
        }
        for (int block = 0; block < blocks; ++block) {
          const std::uint8_t* entry = weight0 + block * kQ8BlockBytes;
          const std::uint8_t* entry1 = weight1 + block * kQ8BlockBytes;
          const auto* values = reinterpret_cast<const std::int8_t*>(entry + 2);
          const auto* values1 = reinterpret_cast<const std::int8_t*>(entry1 + 2);
          const __m256 w0 = widen(values + 0);
          const __m256 w1 = widen(values + 8);
          const __m256 w2 = widen(values + 16);
          const __m256 w3 = widen(values + 24);
          const __m256 v0 = widen(values1 + 0);
          const __m256 v1 = widen(values1 + 8);
          const __m256 v2 = widen(values1 + 16);
          const __m256 v3 = widen(values1 + 24);
          const __m256 s0 = _mm256_set1_ps(scale0[static_cast<std::size_t>(block)]);
          const __m256 s1 = _mm256_set1_ps(scale1[static_cast<std::size_t>(block)]);
          const int offset = block * kQ8BlockElements;
#define TGE_FOLD8(a, x, p0, p1, p2, p3, scale)                                  \
  {                                                                             \
    __m256 sum = _mm256_mul_ps(p0, _mm256_loadu_ps(x + offset));                \
    sum = _mm256_fmadd_ps(p1, _mm256_loadu_ps(x + offset + 8), sum);             \
    sum = _mm256_fmadd_ps(p2, _mm256_loadu_ps(x + offset + 16), sum);            \
    sum = _mm256_fmadd_ps(p3, _mm256_loadu_ps(x + offset + 24), sum);            \
    (a) = _mm256_fmadd_ps(scale, sum, (a));                                      \
  }
          TGE_FOLD8(r0a0, inputs[0], w0, w1, w2, w3, s0);
          TGE_FOLD8(r0a1, inputs[1], w0, w1, w2, w3, s0);
          TGE_FOLD8(r0a2, inputs[2], w0, w1, w2, w3, s0);
          TGE_FOLD8(r0a3, inputs[3], w0, w1, w2, w3, s0);
          TGE_FOLD8(r0a4, inputs[4], w0, w1, w2, w3, s0);
          TGE_FOLD8(r0a5, inputs[5], w0, w1, w2, w3, s0);
          TGE_FOLD8(r0a6, inputs[6], w0, w1, w2, w3, s0);
          TGE_FOLD8(r0a7, inputs[7], w0, w1, w2, w3, s0);
          for (int index = 0; index < 8; ++index) {
            __m256 sum = _mm256_mul_ps(v0, _mm256_loadu_ps(inputs[index] + offset));
            sum = _mm256_fmadd_ps(v1, _mm256_loadu_ps(inputs[index] + offset + 8), sum);
            sum = _mm256_fmadd_ps(v2, _mm256_loadu_ps(inputs[index] + offset + 16), sum);
            sum = _mm256_fmadd_ps(v3, _mm256_loadu_ps(inputs[index] + offset + 24), sum);
            r1[static_cast<std::size_t>(index)] =
                _mm256_fmadd_ps(s1, sum, r1[static_cast<std::size_t>(index)]);
          }
#undef TGE_FOLD8
        }
        args.out[static_cast<std::size_t>(token + 0) * rows + row + 0] = horizontal_sum(r0a0);
        args.out[static_cast<std::size_t>(token + 1) * rows + row + 0] = horizontal_sum(r0a1);
        args.out[static_cast<std::size_t>(token + 2) * rows + row + 0] = horizontal_sum(r0a2);
        args.out[static_cast<std::size_t>(token + 3) * rows + row + 0] = horizontal_sum(r0a3);
        args.out[static_cast<std::size_t>(token + 4) * rows + row + 0] = horizontal_sum(r0a4);
        args.out[static_cast<std::size_t>(token + 5) * rows + row + 0] = horizontal_sum(r0a5);
        args.out[static_cast<std::size_t>(token + 6) * rows + row + 0] = horizontal_sum(r0a6);
        args.out[static_cast<std::size_t>(token + 7) * rows + row + 0] = horizontal_sum(r0a7);
        for (int index = 0; index < 8; ++index) {
          args.out[static_cast<std::size_t>(token + index) * rows + row + 1] =
              horizontal_sum(r1[static_cast<std::size_t>(index)]);
        }
      }
      weight0 = weight1 + row_bytes;
    }
  });
}

}  // namespace tge::cpu
