#include "tge/cpu/kernels.hpp"

#include <cstddef>
#include <cstring>

#include <immintrin.h>

namespace tge::cpu {
namespace {

constexpr int kKBlockElements = 256;
constexpr std::size_t kQ4_KBlockBytes = 144;

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

void dequantize_row_avx2(QuantKind kind, const std::uint8_t* source,
                         float* destination, int cols) {
  if (kind == QuantKind::Q4_K) {
    dequantize_q4_k_avx2(source, destination, cols);
    return;
  }
  // The other formats are not hot yet; the scalar path stays authoritative.
  dequantize_row(kind, source, destination, cols);
}

}  // namespace tge::cpu
