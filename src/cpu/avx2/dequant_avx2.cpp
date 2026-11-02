#include "tge/cpu/kernels.hpp"

#include <cstddef>
#include <cstring>

#include <immintrin.h>

namespace tge::cpu {
namespace {

constexpr int kKBlockElements = 256;
constexpr std::size_t kQ4_KBlockBytes = 144;
constexpr std::size_t kQ5_KBlockBytes = 176;
constexpr std::size_t kQ6_KBlockBytes = 210;
constexpr std::size_t kQ5_0BlockBytes = 22;

/// Unpacks one 6-bit scale and its 6-bit minimum from the 12-byte scale table.
/// Mirrors the scalar path exactly; see `dequantize_q4_k` for the derivation.
inline void unpack_scale_min(int index, const std::uint8_t* scales,
                             std::uint8_t& scale, std::uint8_t& minimum) {
  if (index < 4) {
    scale = scales[index] & 63u;
    minimum = scales[index + 4] & 63u;
  } else {
    scale = static_cast<std::uint8_t>((scales[index + 4] & 0x0Fu) |
                                      ((scales[index - 4] >> 6) << 4));
    minimum = static_cast<std::uint8_t>((scales[index + 4] >> 4) |
                                        ((scales[index] >> 6) << 4));
  }
}

/// Writes 32 floats from 32 packed bytes, one nibble each.
///
/// Multiply then subtract, never fused, so the result matches the scalar path
/// bit for bit and token parity survives.
inline void write32(const std::uint8_t* quants, bool high_nibble, float scale,
                    float offset, float* out) {
  const __m256i bytes =
      _mm256_loadu_si256(reinterpret_cast<const __m256i*>(quants));
  const __m256i nibbles =
      high_nibble
          ? _mm256_and_si256(_mm256_srli_epi16(bytes, 4), _mm256_set1_epi8(0x0F))
          : _mm256_and_si256(bytes, _mm256_set1_epi8(0x0F));

  const __m256 factor = _mm256_set1_ps(scale);
  const __m256 base = _mm256_set1_ps(offset);

  const __m128i first = _mm256_castsi256_si128(nibbles);
  const __m128i second = _mm256_extracti128_si256(nibbles, 1);

  const __m256 lanes[4] = {
      _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(first)),
      _mm256_cvtepi32_ps(
          _mm256_cvtepi8_epi32(_mm_srli_si128(first, 8))),
      _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(second)),
      _mm256_cvtepi32_ps(
          _mm256_cvtepi8_epi32(_mm_srli_si128(second, 8))),
  };
  for (int lane = 0; lane < 4; ++lane) {
    _mm256_storeu_ps(out + lane * 8,
                     _mm256_sub_ps(_mm256_mul_ps(lanes[lane], factor), base));
  }
}

void dequantize_q4_k_avx2(const std::uint8_t* source, float* destination,
                          int cols) {
  const int blocks = cols / kKBlockElements;
  for (int block = 0; block < blocks; ++block) {
    const std::uint8_t* entry = source + block * kQ4_KBlockBytes;
    std::uint16_t scale_bits = 0;
    std::uint16_t minimum_bits = 0;
    std::memcpy(&scale_bits, entry, sizeof(scale_bits));
    std::memcpy(&minimum_bits, entry + 2, sizeof(minimum_bits));
    const float d = half_to_float(scale_bits);
    const float dmin = half_to_float(minimum_bits);

    const std::uint8_t* scales = entry + 4;
    const std::uint8_t* quants = entry + 16;
    float* out = destination + block * kKBlockElements;

    int scale_index = 0;
    for (int chunk = 0; chunk < kKBlockElements; chunk += 64) {
      std::uint8_t sc = 0;
      std::uint8_t m = 0;
      unpack_scale_min(scale_index + 0, scales, sc, m);
      const float d1 = d * sc;
      const float m1 = dmin * m;
      unpack_scale_min(scale_index + 1, scales, sc, m);
      const float d2 = d * sc;
      const float m2 = dmin * m;

      // All 32 bytes hold the low nibbles of the first 32 values, then the same
      // 32 bytes hold the high nibbles of the next 32. Splitting the byte range
      // between the two halves instead produces plausible but wrong values.
      write32(quants, /*high_nibble=*/false, d1, m1, out);
      write32(quants, /*high_nibble=*/true, d2, m2, out + 32);
      out += 64;
      quants += 32;
      scale_index += 2;
    }
  }
}

}  // namespace

/// Q6_K super-block: 128 low bytes + 64 high bytes + 16 int8 scales + fp16 d.
/// Each 6-bit weight is 4 bits from `low` plus 2 bits from `high`, minus 32,
/// times d * the int8 scale. All arithmetic is multiply-only in scalar order
/// (d*sc)*q with no FMA, so the result is bit-identical to the scalar oracle.
/// Loads are explicitly unaligned: model buffers are only 16-byte aligned
/// (see §28i), so no vector pointer is ever formed over them. The 16-bit
/// shifts are exact per byte: srli pairs bytes (2i, 2i+1) and the 0x0F/0x03
/// masks keep only the wanted bits; the and-before-shift order keeps the
/// slli spill-free (values < 4 never carry into the next byte).
inline void dequantize_group_q6_k(const std::uint8_t* low,
                                  const std::uint8_t* high, int low_shift,
                                  int high_shift, float factor, float* out) {
  const __m128i low8 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(low));
  const __m128i high8 =
      _mm_loadl_epi64(reinterpret_cast<const __m128i*>(high));
  const __m128i lo_nibbles = _mm_and_si128(
      _mm_srli_epi16(low8, low_shift), _mm_set1_epi8(0x0F));
  const __m128i hi_bits = _mm_slli_epi16(
      _mm_and_si128(_mm_srli_epi16(high8, high_shift),
                    _mm_set1_epi8(0x03)),
      4);
  const __m128i combined =
      _mm_sub_epi8(_mm_or_si128(lo_nibbles, hi_bits), _mm_set1_epi8(32));
  const __m256 values =
      _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(combined));
  _mm256_storeu_ps(out, _mm256_mul_ps(_mm256_set1_ps(factor), values));
}

void dequantize_q6_k_avx2(const std::uint8_t* source, float* destination,
                          int cols) {
  const int blocks = cols / kKBlockElements;
  for (int block = 0; block < blocks; ++block) {
    const std::uint8_t* entry = source + block * kQ6_KBlockBytes;
    const std::uint8_t* low = entry;
    const std::uint8_t* high = entry + 128;
    const auto* scales = reinterpret_cast<const std::int8_t*>(entry + 192);
    std::uint16_t scale_bits = 0;
    std::memcpy(&scale_bits, entry + 208, sizeof(scale_bits));
    const float d = half_to_float(scale_bits);

    float* out = destination + block * kKBlockElements;
    for (int half = 0; half < kKBlockElements; half += 128) {
      for (int index = 0; index < 32; index += 8) {
        const int within = index / 16;
        // The scalar path writes out[index], [+32], [+64], [+96]: low nibbles
        // of low[index] and low[index+32], then the high nibbles of the same
        // bytes, with high bits shifting 0, 2, 4, 6.
        const float f1 = d * static_cast<float>(scales[within + 0]);
        const float f2 = d * static_cast<float>(scales[within + 2]);
        const float f3 = d * static_cast<float>(scales[within + 4]);
        const float f4 = d * static_cast<float>(scales[within + 6]);
        dequantize_group_q6_k(low + index, high + index, 0, 0, f1,
                              out + index);
        dequantize_group_q6_k(low + index + 32, high + index, 0, 2, f2,
                              out + index + 32);
        dequantize_group_q6_k(low + index, high + index, 4, 4, f3,
                              out + index + 64);
        dequantize_group_q6_k(low + index + 32, high + index, 4, 6, f4,
                              out + index + 96);
      }
      out += 128;
      low += 64;
      high += 32;
      scales += 8;
    }
  }
}

/// Q5_0 legacy block: fp16 d + 32 high bits + 16 quant bytes. Each weight is
/// a nibble plus bit ((hb >> i) & 1) << 4, minus 16, times d. The bit spread
/// comes from a 256-entry LUT (byte k of entry b is ((b >> k) & 1) << 4), so
/// no variable per-element shift exists. Multiply-only in scalar order d*q,
/// no FMA: bit-identical to the scalar oracle. All loads explicitly
/// unaligned (§28i: model buffers are 16-byte aligned, never form a vector
/// pointer over them).
inline const std::uint64_t* q5_0_bit_spread_lut() {
  static const std::uint64_t* table = [] {
    static std::uint64_t entries[256];
    for (int byte = 0; byte < 256; ++byte) {
      std::uint64_t packed = 0;
      for (int bit = 0; bit < 8; ++bit) {
        if ((byte >> bit) & 1) {
          packed |= static_cast<std::uint64_t>(0x10) << (bit * 8);
        }
      }
      entries[byte] = packed;
    }
    return entries;
  }();
  return table;
}

inline void dequantize_group_q5_0(const std::uint8_t* quants,
                                  const std::uint64_t* lut, std::uint32_t bits_lo,
                                  std::uint32_t bits_hi, float factor,
                                  float* out_lo, float* out_hi) {
  const __m128i bytes =
      _mm_loadl_epi64(reinterpret_cast<const __m128i*>(quants));
  const __m128i nibble_mask = _mm_set1_epi8(0x0F);
  const __m128i bias = _mm_set1_epi8(16);
  const __m256 scale = _mm256_set1_ps(factor);
  std::uint64_t spread_lo = 0;
  std::uint64_t spread_hi = 0;
  std::memcpy(&spread_lo, lut + (bits_lo & 0xFFu), sizeof(spread_lo));
  std::memcpy(&spread_hi, lut + (bits_hi & 0xFFu), sizeof(spread_hi));
  const __m128i lo = _mm_sub_epi8(
      _mm_or_si128(_mm_and_si128(bytes, nibble_mask),
                   _mm_set_epi64x(0, static_cast<long long>(spread_lo))),
      bias);
  const __m128i hi = _mm_sub_epi8(
      _mm_or_si128(
          _mm_and_si128(_mm_srli_epi16(bytes, 4), nibble_mask),
          _mm_set_epi64x(0, static_cast<long long>(spread_hi))),
      bias);
  const __m256 lo_values =
      _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(lo));
  const __m256 hi_values =
      _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(hi));
  _mm256_storeu_ps(out_lo, _mm256_mul_ps(scale, lo_values));
  _mm256_storeu_ps(out_hi, _mm256_mul_ps(scale, hi_values));
}

void dequantize_q5_0_avx2(const std::uint8_t* source, float* destination,
                          int cols) {
  constexpr int kLegacyElements = 32;
  const int blocks = cols / kLegacyElements;
  const std::uint64_t* lut = q5_0_bit_spread_lut();
  for (int block = 0; block < blocks; ++block) {
    const std::uint8_t* entry = source + block * kQ5_0BlockBytes;
    std::uint16_t scale_bits = 0;
    std::memcpy(&scale_bits, entry, sizeof(scale_bits));
    const float d = half_to_float(scale_bits);
    std::uint32_t high_bits = 0;
    std::memcpy(&high_bits, entry + 2, sizeof(high_bits));
    const std::uint8_t* quants = entry + 6;
    float* out = destination + block * kLegacyElements;
    // The scalar path pairs low bit (hb >> i) with high bit (hb >> (i + 16)):
    // bit_high = (hb >> (i + 12)) & 0x10 selects bit 4 of (hb >> (i + 12)),
    // i.e. source bit i + 16. The LUT index is the 8-bit window directly.
    dequantize_group_q5_0(quants, lut, high_bits >> 0, high_bits >> 16, d,
                          out, out + 16);
  dequantize_group_q5_0(quants + 8, lut, high_bits >> 8, high_bits >> 24, d,
                        out + 8, out + 24);
  }
}

/// Q5_1 legacy block: fp16 d + fp16 m + 32 high bits + 16 quant bytes.
/// Same nibble/bit extraction as Q5_0 (shared LUT), but the combination is
/// d * q + m with NO bias subtraction: mul-then-add in scalar order, no
/// FMA, so the result is bit-identical to the scalar oracle.
inline void dequantize_group_q5_1(const std::uint8_t* quants,
                                  const std::uint64_t* lut, std::uint32_t bits_lo,
                                  std::uint32_t bits_hi, float d, float m,
                                  float* out_lo, float* out_hi) {
  const __m128i bytes =
      _mm_loadl_epi64(reinterpret_cast<const __m128i*>(quants));
  const __m128i nibble_mask = _mm_set1_epi8(0x0F);
  const __m256 scale = _mm256_set1_ps(d);
  const __m256 minimum = _mm256_set1_ps(m);
  std::uint64_t spread_lo = 0;
  std::uint64_t spread_hi = 0;
  std::memcpy(&spread_lo, lut + (bits_lo & 0xFFu), sizeof(spread_lo));
  std::memcpy(&spread_hi, lut + (bits_hi & 0xFFu), sizeof(spread_hi));
  const __m128i lo = _mm_or_si128(
      _mm_and_si128(bytes, nibble_mask),
      _mm_set_epi64x(0, static_cast<long long>(spread_lo)));
  const __m128i hi = _mm_or_si128(
      _mm_and_si128(_mm_srli_epi16(bytes, 4), nibble_mask),
      _mm_set_epi64x(0, static_cast<long long>(spread_hi)));
  const __m256 lo_values =
      _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(lo));
  const __m256 hi_values =
      _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(hi));
  // (d * q) + m, in this order, no FMA: matches the scalar exactly.
  _mm256_storeu_ps(out_lo,
                   _mm256_add_ps(_mm256_mul_ps(scale, lo_values), minimum));
  _mm256_storeu_ps(out_hi,
                   _mm256_add_ps(_mm256_mul_ps(scale, hi_values), minimum));
}

void dequantize_q5_1_avx2(const std::uint8_t* source, float* destination,
                          int cols) {
  constexpr int kLegacyElements = 32;
  constexpr std::size_t kQ5_1BlockBytes = 24;
  const int blocks = cols / kLegacyElements;
  const std::uint64_t* lut = q5_0_bit_spread_lut();
  for (int block = 0; block < blocks; ++block) {
    const std::uint8_t* entry = source + block * kQ5_1BlockBytes;
    std::uint16_t scale_bits = 0;
    std::uint16_t minimum_bits = 0;
    std::memcpy(&scale_bits, entry, sizeof(scale_bits));
    std::memcpy(&minimum_bits, entry + 2, sizeof(minimum_bits));
    const float d = half_to_float(scale_bits);
    const float m = half_to_float(minimum_bits);
    std::uint32_t high_bits = 0;
    std::memcpy(&high_bits, entry + 4, sizeof(high_bits));
    const std::uint8_t* quants = entry + 8;
    float* out = destination + block * kLegacyElements;
    // Same bit pairing as Q5_0 (low bit hb >> i, high bit hb >> (i + 16)).
  dequantize_group_q5_1(quants, lut, high_bits >> 0, high_bits >> 16, d, m,
                        out, out + 16);
  dequantize_group_q5_1(quants + 8, lut, high_bits >> 8, high_bits >> 24, d,
                        m, out + 8, out + 24);
  }
}

/// Q5_K super-block: fp16 d + fp16 dmin + 12 scale bytes + 64 high bytes +
/// 128 low bytes. Four 64-weight chunks; each chunk reads one 32-byte low
/// row twice (low nibbles, then high nibbles of the same bytes) plus one
/// high bit per weight from a shared 32-byte row: bit (2c) for the low
/// half, bit (2c + 1) for the high half, chunk c. Multiply then subtract,
/// never fused, so the result matches the scalar path bit for bit.
inline void dequantize_group_q5_k(const std::uint8_t* quants,
                                  const std::uint8_t* high, int nibble_shift,
                                  int bit, float scale, float minimum,
                                  float* out) {
  const __m256i bytes =
      _mm256_loadu_si256(reinterpret_cast<const __m256i*>(quants));
  const __m256i nibbles =
      _mm256_and_si256(_mm256_srli_epi16(bytes, nibble_shift),
                       _mm256_set1_epi8(0x0F));

  // One high bit per weight from the shared row: (high[i] >> bit) & 1,
  // shifted into the 0x10 position. cmpeq yields 0xFF/0x00 bytes, masked
  // down and ORed with the nibbles before widening. All loads unaligned
  // (§28i: model buffers are 16-byte aligned, never form a vector pointer).
  const __m256i high_bytes =
      _mm256_loadu_si256(reinterpret_cast<const __m256i*>(high));
  const __m256i bit_mask = _mm256_set1_epi8(static_cast<char>(1 << bit));
  const __m256i present =
      _mm256_cmpeq_epi8(_mm256_and_si256(high_bytes, bit_mask), bit_mask);
  const __m256i combined = _mm256_or_si256(
      nibbles, _mm256_and_si256(present, _mm256_set1_epi8(0x10)));

  const __m256 factor = _mm256_set1_ps(scale);
  const __m256 base = _mm256_set1_ps(minimum);
  const __m128i first = _mm256_castsi256_si128(combined);
  const __m128i second = _mm256_extracti128_si256(combined, 1);
  const __m256 lanes[4] = {
      _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(first)),
      _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(first, 8))),
      _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(second)),
      _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(second, 8))),
  };
  for (int lane = 0; lane < 4; ++lane) {
    _mm256_storeu_ps(out + lane * 8,
                     _mm256_sub_ps(_mm256_mul_ps(lanes[lane], factor), base));
  }
}

void dequantize_q5_k_avx2(const std::uint8_t* source, float* destination,
                          int cols) {
  const int blocks = cols / kKBlockElements;
  for (int block = 0; block < blocks; ++block) {
    const std::uint8_t* entry = source + block * kQ5_KBlockBytes;
    std::uint16_t scale_bits = 0;
    std::uint16_t minimum_bits = 0;
    std::memcpy(&scale_bits, entry, sizeof(scale_bits));
    std::memcpy(&minimum_bits, entry + 2, sizeof(minimum_bits));
    const float d = half_to_float(scale_bits);
    const float dmin = half_to_float(minimum_bits);

    const std::uint8_t* scales = entry + 4;
    const std::uint8_t* high = entry + 16;
    const std::uint8_t* low = entry + 48;
    float* out = destination + block * kKBlockElements;

    int scale_index = 0;
    int bit = 0;
    for (int chunk = 0; chunk < kKBlockElements; chunk += 64) {
      std::uint8_t sc = 0;
      std::uint8_t m = 0;
      unpack_scale_min(scale_index + 0, scales, sc, m);
      const float d1 = d * sc;
      const float m1 = dmin * m;
      unpack_scale_min(scale_index + 1, scales, sc, m);
      const float d2 = d * sc;
      const float m2 = dmin * m;

      // Same 32 low bytes feed both halves (low nibbles, then high nibbles);
      // the high-bit row is shared across all four chunks, one bit pair per
      // chunk. Matches dequantize_q5_k's lo += 32 with a stationary high row.
      dequantize_group_q5_k(low, high, 0, bit, d1, m1, out);
      dequantize_group_q5_k(low, high, 4, bit + 1, d2, m2, out + 32);
      out += 64;
      low += 32;
      scale_index += 2;
      bit += 2;
    }
  }
}

void dequantize_row_avx2(QuantKind kind, const std::uint8_t* source,
                         float* destination, int cols) {
  if (kind == QuantKind::Q4_K) {
    dequantize_q4_k_avx2(source, destination, cols);
    return;
  }
  if (kind == QuantKind::Q6_K) {
    dequantize_q6_k_avx2(source, destination, cols);
    return;
  }
  if (kind == QuantKind::Q5_0) {
    dequantize_q5_0_avx2(source, destination, cols);
    return;
  }
  if (kind == QuantKind::Q5_1) {
    dequantize_q5_1_avx2(source, destination, cols);
    return;
  }
  if (kind == QuantKind::Q5_K) {
    dequantize_q5_k_avx2(source, destination, cols);
    return;
  }
  // The other formats are not hot yet; the scalar path stays authoritative.
  dequantize_row(kind, source, destination, cols);
}

}  // namespace tge::cpu
