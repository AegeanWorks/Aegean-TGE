#include "tge/cpu/kernels.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

#include <immintrin.h>

#include "tge/core/thread_pool.hpp"

namespace tge::cpu {
namespace {

// Transient Q8_K activation block. Same field order as ggml's block_q8_K
// (llama.cpp ggml/src/ggml-common.h at cd8427d); wdata only, never
// serialized, so no format-compat burden beyond this file.
struct Q8KBlock {
  float d = 0.0F;
  std::int8_t qs[256] = {};
  std::int16_t bsums[16] = {};
};
static_assert(sizeof(Q8KBlock) ==
              sizeof(float) + 256 + 16 * sizeof(std::int16_t));

// Q4_K super-block field order (d, dmin, 12 scale bytes, 128 nibble bytes).
struct Q4KBlock {
  std::uint16_t d_bits = 0;
  std::uint16_t dmin_bits = 0;
  std::uint8_t scales[12] = {};
  std::uint8_t qs[128] = {};
};
static_assert(sizeof(Q4KBlock) == 144);

// Q6_K super-block field order (128 low bytes, 64 high bytes, 16 int8
// scales, fp16 d).
struct Q6KBlock {
  std::uint8_t ql[128] = {};
  std::uint8_t qh[64] = {};
  std::int8_t scales[16] = {};
  std::uint16_t d_bits = 0;
};
static_assert(sizeof(Q6KBlock) == 210);

// Round-half-away-from-zero via the 2^23+2^22 mantissa trick (ggml
// nearest_int at cd8427d; valid for |fval| <= 2^22, far above any
// activation/scale ratio here).
inline int nearest_int(float fval) {
  float shifted = fval + 12582912.0F;
  int bits = 0;
  std::memcpy(&bits, &shifted, sizeof(bits));
  return (bits & 0x007FFFFF) - 0x00400000;
}

// Scalar per-row Q8_K activation quantizer. Semantics match llama.cpp
// quantize_row_q8_K_ref at cd8427d exactly (negated int8 quants via a
// negative iscale, bsums per 16, d = 1/iscale), including the zero-block
// path — except the whole block is cleared first so bsums are
// deterministic even when d == 0 kills their contribution anyway.
void quantize_row_q8k(const float* x, Q8KBlock* y, int count) {
  const int blocks = count / 256;
  for (int i = 0; i < blocks; ++i) {
    std::memset(&y[i], 0, sizeof(y[i]));
    float max = 0.0F;
    float amax = 0.0F;
    for (int j = 0; j < 256; ++j) {
      const float ax = std::fabs(x[j]);
      if (ax > amax) {
        amax = ax;
        max = x[j];
      }
    }
    if (amax == 0.0F) {
      x += 256;
      continue;
    }
    const float iscale = -127.0F / max;
    for (int j = 0; j < 256; ++j) {
      const int v = nearest_int(iscale * x[j]);
      y[i].qs[j] = static_cast<std::int8_t>(v < 127 ? v : 127);
    }
    for (int j = 0; j < 16; ++j) {
      int sum = 0;
      for (int k = 0; k < 16; ++k) {
        sum += y[i].qs[j * 16 + k];
      }
      y[i].bsums[j] = static_cast<std::int16_t>(sum);
    }
    y[i].d = 1.0F / iscale;
    x += 256;
  }
}

inline __m256i mm256_set_m128i(__m128i lo, __m128i hi) {
  return _mm256_inserti128_si256(_mm256_castsi128_si256(lo), hi, 1);
}

inline float hsum_float_8(__m256 value) {
  __m128 sum =
      _mm_add_ps(_mm256_castps256_ps128(value), _mm256_extractf128_ps(value, 1));
  sum = _mm_add_ps(sum, _mm_movehl_ps(sum, sum));
  sum = _mm_add_ss(sum, _mm_movehdup_ps(sum));
  return _mm_cvtss_f32(sum);
}

// Shuffle tables copied from ggml's get_scale_shuffle_k4 / get_scale_shuffle
// (arch/x86/quants.c at cd8427d): broadcast patterns for the K-quant scale
// bytes into 16-bit lanes.
inline __m256i k4_shuffle_mask(int i) {
  // Row i of ggml's get_scale_shuffle_k4 table (arch/x86/quants.c at
  // cd8427d) is 16x the byte pair (2i, 2i+1). This builds the mask
  // directly — it is the shuffle CONTROL, applied to the scales vector
  // below, never a scale value itself.
  const int lo = 2 * i;
  const int hi = 2 * i + 1;
  return _mm256_set1_epi16(static_cast<short>((hi << 8) | lo));
}

inline __m128i k_scale_shuffle(int i) {
  static const std::uint8_t table[128] = {
      0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1,
      2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3,
      4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5,
      6, 6, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7, 7,
      8, 8, 8, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 9, 9,
      10, 10, 10, 10, 10, 10, 10, 10, 11, 11, 11, 11, 11, 11, 11, 11,
      12, 12, 12, 12, 12, 12, 12, 12, 13, 13, 13, 13, 13, 13, 13, 13,
      14, 14, 14, 14, 14, 14, 14, 14, 15, 15, 15, 15, 15, 15, 15, 15};
  return _mm_loadu_si128(reinterpret_cast<const __m128i*>(table) + i);
}

// Integer Q4_K x Q8_K dot, AVX2 path transcribed from
// ggml_vec_dot_q4_K_q8_K (arch/x86/quants.c at cd8427d): nibbles stay
// unsigned, two maddubs+madd per 64 weights, one float scale per
// 256-superblock, ±bias via precomputed bsums. The single hadd sits on
// the cold bias path; the hot loop is hadd-free.
float dot_q4k_q8k(const Q4KBlock* x, const Q8KBlock* y, int blocks) {
  static const std::uint32_t kmask1 = 0x3F3F3F3F;
  static const std::uint32_t kmask2 = 0x0F0F0F0F;
  static const std::uint32_t kmask3 = 0x03030303;

  const __m256i m4 = _mm256_set1_epi8(0x0F);
  __m256 acc = _mm256_setzero_ps();
  __m128 acc_m = _mm_setzero_ps();
  std::uint32_t utmp[4] = {};

  for (int i = 0; i < blocks; ++i) {
    const float d = y[i].d * half_to_float(x[i].d_bits);
    const float dmin = -y[i].d * half_to_float(x[i].dmin_bits);

    std::memcpy(utmp, x[i].scales, 12);
    utmp[3] = ((utmp[2] >> 4) & kmask2) | (((utmp[1] >> 6) & kmask3) << 4);
    const std::uint32_t uaux = utmp[1] & kmask1;
    utmp[1] = (utmp[2] & kmask2) | (((utmp[0] >> 6) & kmask3) << 4);
    utmp[2] = uaux;
    utmp[0] &= kmask1;

    const std::uint8_t* q4 = x[i].qs;
    const std::int8_t* q8 = y[i].qs;

    const __m256i mins_and_scales = _mm256_cvtepu8_epi16(
        _mm_set_epi32(static_cast<int>(utmp[3]), static_cast<int>(utmp[2]),
                      static_cast<int>(utmp[1]), static_cast<int>(utmp[0])));

    const __m256i q8sums =
        _mm256_loadu_si256(reinterpret_cast<const __m256i*>(y[i].bsums));
    const __m128i q8s = _mm_hadd_epi16(_mm256_extracti128_si256(q8sums, 0),
                                      _mm256_extracti128_si256(q8sums, 1));
    const __m128i prod = _mm_madd_epi16(
        _mm256_extracti128_si256(mins_and_scales, 1), q8s);
    acc_m = _mm_fmadd_ps(_mm_set1_ps(dmin), _mm_cvtepi32_ps(prod), acc_m);

    const __m128i sc128 = _mm256_extracti128_si256(mins_and_scales, 0);
    const __m256i scales = mm256_set_m128i(sc128, sc128);

    __m256i sumi = _mm256_setzero_si256();
    for (int j = 0; j < 4; ++j) {
      const __m256i scale_l = _mm256_shuffle_epi8(scales, k4_shuffle_mask(2 * j));
      const __m256i scale_h =
          _mm256_shuffle_epi8(scales, k4_shuffle_mask(2 * j + 1));

      const __m256i q4bits =
          _mm256_loadu_si256(reinterpret_cast<const __m256i*>(q4));
      q4 += 32;
      const __m256i q4l = _mm256_and_si256(q4bits, m4);
      const __m256i q4h =
          _mm256_and_si256(_mm256_srli_epi16(q4bits, 4), m4);

      const __m256i q8l =
          _mm256_loadu_si256(reinterpret_cast<const __m256i*>(q8));
      q8 += 32;
      __m256i p16l = _mm256_maddubs_epi16(q4l, q8l);
      p16l = _mm256_madd_epi16(scale_l, p16l);

      const __m256i q8h =
          _mm256_loadu_si256(reinterpret_cast<const __m256i*>(q8));
      q8 += 32;
      __m256i p16h = _mm256_maddubs_epi16(q4h, q8h);
      p16h = _mm256_madd_epi16(scale_h, p16h);

      sumi = _mm256_add_epi32(sumi, _mm256_add_epi32(p16l, p16h));
    }

    acc = _mm256_fmadd_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(sumi), acc);
  }

  acc_m = _mm_add_ps(acc_m, _mm_movehl_ps(acc_m, acc_m));
  acc_m = _mm_add_ss(acc_m, _mm_movehdup_ps(acc_m));
  return hsum_float_8(acc) + _mm_cvtss_f32(acc_m);
}

// Integer Q6_K x Q8_K dot, AVX2 path transcribed from
// ggml_vec_dot_q6_K_q8_K (arch/x86/quants.c at cd8427d): the -32 offset
// is handled once per superblock via precomputed bsums
// (q8sclsub << 5); no horizontal reduction in the hot loop.
float dot_q6k_q8k(const Q6KBlock* x, const Q8KBlock* y, int blocks) {
  const __m256i m3 = _mm256_set1_epi8(3);
  const __m256i m15 = _mm256_set1_epi8(15);

  __m256 acc = _mm256_setzero_ps();

  for (int i = 0; i < blocks; ++i) {
    const float d = y[i].d * half_to_float(x[i].d_bits);

    const std::uint8_t* q4 = x[i].ql;
    const std::uint8_t* qh = x[i].qh;
    const std::int8_t* q8 = y[i].qs;

    const __m256i q8sums =
        _mm256_loadu_si256(reinterpret_cast<const __m256i*>(y[i].bsums));
    const __m128i scales =
        _mm_loadu_si128(reinterpret_cast<const __m128i*>(x[i].scales));
    const __m256i scales_16 = _mm256_cvtepi8_epi16(scales);
    const __m256i q8sclsub =
        _mm256_slli_epi32(_mm256_madd_epi16(q8sums, scales_16), 5);

    __m256i sumi = _mm256_setzero_si256();
    int is = 0;
    for (int j = 0; j < 2; ++j) {
      const __m256i q4bits1 =
          _mm256_loadu_si256(reinterpret_cast<const __m256i*>(q4));
      q4 += 32;
      const __m256i q4bits2 =
          _mm256_loadu_si256(reinterpret_cast<const __m256i*>(q4));
      q4 += 32;
      const __m256i q4bitsH =
          _mm256_loadu_si256(reinterpret_cast<const __m256i*>(qh));
      qh += 32;

      const __m256i q4h_0 =
          _mm256_slli_epi16(_mm256_and_si256(q4bitsH, m3), 4);
      const __m256i q4h_1 = _mm256_slli_epi16(
          _mm256_and_si256(q4bitsH, _mm256_set1_epi8(12)), 2);
      const __m256i q4h_2 =
          _mm256_and_si256(q4bitsH, _mm256_set1_epi8(48));
      const __m256i q4h_3 = _mm256_srli_epi16(
          _mm256_and_si256(q4bitsH, _mm256_set1_epi8(-64)), 2);

      const __m256i q4_0 =
          _mm256_or_si256(_mm256_and_si256(q4bits1, m15), q4h_0);
      const __m256i q4_1 =
          _mm256_or_si256(_mm256_and_si256(q4bits2, m15), q4h_1);
      const __m256i q4_2 = _mm256_or_si256(
          _mm256_and_si256(_mm256_srli_epi16(q4bits1, 4), m15), q4h_2);
      const __m256i q4_3 = _mm256_or_si256(
          _mm256_and_si256(_mm256_srli_epi16(q4bits2, 4), m15), q4h_3);

      const __m256i q8_0 =
          _mm256_loadu_si256(reinterpret_cast<const __m256i*>(q8));
      q8 += 32;
      const __m256i q8_1 =
          _mm256_loadu_si256(reinterpret_cast<const __m256i*>(q8));
      q8 += 32;
      const __m256i q8_2 =
          _mm256_loadu_si256(reinterpret_cast<const __m256i*>(q8));
      q8 += 32;
      const __m256i q8_3 =
          _mm256_loadu_si256(reinterpret_cast<const __m256i*>(q8));
      q8 += 32;

      __m256i p16_0 = _mm256_maddubs_epi16(q4_0, q8_0);
      __m256i p16_1 = _mm256_maddubs_epi16(q4_1, q8_1);
      __m256i p16_2 = _mm256_maddubs_epi16(q4_2, q8_2);
      __m256i p16_3 = _mm256_maddubs_epi16(q4_3, q8_3);

      const __m128i scale_0 = _mm_shuffle_epi8(scales, k_scale_shuffle(is));
      const __m128i scale_1 = _mm_shuffle_epi8(scales, k_scale_shuffle(is + 1));
      const __m128i scale_2 = _mm_shuffle_epi8(scales, k_scale_shuffle(is + 2));
      const __m128i scale_3 = _mm_shuffle_epi8(scales, k_scale_shuffle(is + 3));
      is += 4;

      p16_0 = _mm256_madd_epi16(_mm256_cvtepi8_epi16(scale_0), p16_0);
      p16_1 = _mm256_madd_epi16(_mm256_cvtepi8_epi16(scale_1), p16_1);
      p16_2 = _mm256_madd_epi16(_mm256_cvtepi8_epi16(scale_2), p16_2);
      p16_3 = _mm256_madd_epi16(_mm256_cvtepi8_epi16(scale_3), p16_3);

      sumi = _mm256_add_epi32(sumi, _mm256_add_epi32(p16_0, p16_1));
      sumi = _mm256_add_epi32(sumi, _mm256_add_epi32(p16_2, p16_3));
    }

    sumi = _mm256_sub_epi32(sumi, q8sclsub);
    acc = _mm256_fmadd_ps(_mm256_broadcast_ss(&d), _mm256_cvtepi32_ps(sumi),
                          acc);
  }

  return hsum_float_8(acc);
}

}  // namespace

// Integer Q4_K/Q6_K x Q8_K batch prototype behind
// `bench batch --kernel kq8` (see measurement-log §57). Same banding and
// worker discipline as gemv_dequant_batch_avx2, but each token row is
// quantized to Q8_K once per call (scalar, matching upstream cost) and
// every (row, token) dot runs the integer maddubs path above. tokens == 1
// takes the integer path too — unlike the float batch kernel — so the x1
// guard compares paths, not identical loops.
void gemv_dequant_batch_kq8_avx2(const DequantBatchArgs& args,
                                 const GemvConfig& config) {
  if (args.rows <= 0 || args.cols <= 0 || args.tokens <= 0) {
    return;
  }
  if ((args.kind != QuantKind::Q4_K && args.kind != QuantKind::Q6_K) ||
      args.cols % 256 != 0) {
    gemv_dequant_batch_avx2(args, config);
    return;
  }

  const std::size_t row_bytes = quant_row_bytes(args.kind, args.cols);
  const int rows = args.rows;
  const int cols = args.cols;
  const int tokens = args.tokens;
  const int blocks = cols / 256;

  // Activation quant is per token row and independent of the weight rows,
  // so it runs once per call, shared read-only across bands. At 64x8192
  // this is ~600 KB: L3-resident, amortized over every weight row.
  std::vector<Q8KBlock> quantized(
      static_cast<std::size_t>(tokens) * static_cast<std::size_t>(blocks));
  for (int token = 0; token < tokens; ++token) {
    quantize_row_q8k(
        args.input + static_cast<std::size_t>(token) * cols,
        quantized.data() + static_cast<std::size_t>(token) * blocks, cols);
  }

  const int threads = resolve_thread_count(config.threads);
  core::ThreadPool& pool =
      config.pool != nullptr ? *config.pool : core::acquire_pool(threads);

  int band_rows = rows / (threads * 4);
  if (band_rows < 1) {
    band_rows = 1;
  }
  const int bands = (rows + band_rows - 1) / band_rows;
  const bool is_q4k = args.kind == QuantKind::Q4_K;

  pool.parallel_for(bands, 1, [&](int band) {
    const int begin = band * band_rows;
    const int end = std::min(rows, begin + band_rows);
    const std::uint8_t* row =
        args.data + static_cast<std::size_t>(begin) * row_bytes;

    for (int index = begin; index < end; ++index) {
      for (int token = 0; token < tokens; ++token) {
        const Q8KBlock* q8 =
            quantized.data() + static_cast<std::size_t>(token) * blocks;
        const float dot =
            is_q4k
                ? dot_q4k_q8k(reinterpret_cast<const Q4KBlock*>(row), q8,
                              blocks)
                : dot_q6k_q8k(reinterpret_cast<const Q6KBlock*>(row), q8,
                              blocks);
        args.out[static_cast<std::size_t>(token) * rows + index] = dot;
      }
      row += row_bytes;
    }
  });
}

}  // namespace tge::cpu
