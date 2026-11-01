#include "tge/cpu/kernels.hpp"

#include <cstddef>
#include <cstring>

namespace tge::cpu {
namespace {

// Super-block sizes from the GGML format. A K-quant block always covers 256
// weights, so a row that is not a multiple of 256 cannot be decoded.
constexpr int kKBlockElements = 256;
constexpr std::size_t kQ4_KBlockBytes = 144;
constexpr std::size_t kQ5_KBlockBytes = 176;
constexpr std::size_t kQ6_KBlockBytes = 210;

// Legacy 32-element blocks. "Q4_K_M" is a mixed recipe: it keeps the small
// tensors, including token embeddings, in these formats.
constexpr int kLegacyBlockElements = 32;
constexpr std::size_t kQ4_0BlockBytes = 18;
constexpr std::size_t kQ5_0BlockBytes = 22;

void dequantize_q4_0(const std::uint8_t* source, float* destination, int cols) {
  const int blocks = cols / kLegacyBlockElements;
  for (int block = 0; block < blocks; ++block) {
    const std::uint8_t* entry = source + block * kQ4_0BlockBytes;
    std::uint16_t scale_bits = 0;
    std::memcpy(&scale_bits, entry, sizeof(scale_bits));
    const float d = half_to_float(scale_bits);
    const std::uint8_t* quants = entry + 2;
    float* out = destination + block * kLegacyBlockElements;
    for (int index = 0; index < 16; ++index) {
      out[index] = d * (static_cast<float>(quants[index] & 0x0F) - 8.0F);
      out[index + 16] = d * (static_cast<float>(quants[index] >> 4) - 8.0F);
    }
  }
}

void dequantize_q5_0(const std::uint8_t* source, float* destination, int cols) {
  const int blocks = cols / kLegacyBlockElements;
  for (int block = 0; block < blocks; ++block) {
    const std::uint8_t* entry = source + block * kQ5_0BlockBytes;
    std::uint16_t scale_bits = 0;
    std::memcpy(&scale_bits, entry, sizeof(scale_bits));
    const float d = half_to_float(scale_bits);

    std::uint32_t high_bits = 0;
    std::memcpy(&high_bits, entry + 2, sizeof(high_bits));
    const std::uint8_t* quants = entry + 6;
    float* out = destination + block * kLegacyBlockElements;

    for (int index = 0; index < 16; ++index) {
      const std::uint32_t bit_low = ((high_bits >> index) << 4) & 0x10u;
      const std::uint32_t bit_high = (high_bits >> (index + 12)) & 0x10u;
      const int low = static_cast<int>((quants[index] & 0x0F) | bit_low) - 16;
      const int high = static_cast<int>((quants[index] >> 4) | bit_high) - 16;
      out[index] = d * static_cast<float>(low);
      out[index + 16] = d * static_cast<float>(high);
    }
  }
}

/// Unpacks one 6-bit scale and its 6-bit minimum from the 12-byte scale table.
/// The packing changes shape halfway through, which is the part that is easy to
/// get wrong.
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

void dequantize_q4_k(const std::uint8_t* source, float* destination, int cols) {
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

      for (int index = 0; index < 32; ++index) {
        out[index] = d1 * static_cast<float>(quants[index] & 0x0F) - m1;
      }
      for (int index = 0; index < 32; ++index) {
        out[32 + index] = d2 * static_cast<float>(quants[index] >> 4) - m2;
      }
      out += 64;
      quants += 32;
      scale_index += 2;
    }
  }
}

void dequantize_q5_k(const std::uint8_t* source, float* destination, int cols) {
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
    std::uint8_t bit1 = 1;
    std::uint8_t bit2 = 2;
    for (int chunk = 0; chunk < kKBlockElements; chunk += 64) {
      std::uint8_t sc = 0;
      std::uint8_t m = 0;
      unpack_scale_min(scale_index + 0, scales, sc, m);
      const float d1 = d * sc;
      const float m1 = dmin * m;
      unpack_scale_min(scale_index + 1, scales, sc, m);
      const float d2 = d * sc;
      const float m2 = dmin * m;

      for (int index = 0; index < 32; ++index) {
        const int value = static_cast<int>(low[index] & 0x0F) +
                          ((high[index] & bit1) != 0 ? 16 : 0);
        out[index] = d1 * static_cast<float>(value) - m1;
      }
      for (int index = 0; index < 32; ++index) {
        const int value = static_cast<int>(low[index] >> 4) +
                          ((high[index] & bit2) != 0 ? 16 : 0);
        out[32 + index] = d2 * static_cast<float>(value) - m2;
      }
      out += 64;
      low += 32;
      scale_index += 2;
      bit1 = static_cast<std::uint8_t>(bit1 << 2);
      bit2 = static_cast<std::uint8_t>(bit2 << 2);
    }
  }
}

void dequantize_q6_k(const std::uint8_t* source, float* destination, int cols) {
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
      for (int index = 0; index < 32; ++index) {
        const int within = index / 16;
        const auto q1 = static_cast<std::int8_t>(
            ((low[index] & 0x0F) | (((high[index] >> 0) & 3) << 4))) - 32;
        const auto q2 = static_cast<std::int8_t>(
            ((low[index + 32] & 0x0F) | (((high[index] >> 2) & 3) << 4))) - 32;
        const auto q3 = static_cast<std::int8_t>(
            ((low[index] >> 4) | (((high[index] >> 4) & 3) << 4))) - 32;
        const auto q4 = static_cast<std::int8_t>(
            ((low[index + 32] >> 4) | (((high[index] >> 6) & 3) << 4))) - 32;

        out[index] = d * scales[within + 0] * q1;
        out[index + 32] = d * scales[within + 2] * q2;
        out[index + 64] = d * scales[within + 4] * q3;
        out[index + 96] = d * scales[within + 6] * q4;
      }
      out += 128;
      low += 64;
      high += 32;
      scales += 8;
    }
  }
}

}  // namespace

std::size_t quant_row_bytes(QuantKind kind, int cols) {
  switch (kind) {
    case QuantKind::Q8_0:
      return static_cast<std::size_t>(cols / kQ8BlockElements) * kQ8BlockBytes;
    case QuantKind::Q4_K:
      return static_cast<std::size_t>(cols / kKBlockElements) * kQ4_KBlockBytes;
    case QuantKind::Q5_K:
      return static_cast<std::size_t>(cols / kKBlockElements) * kQ5_KBlockBytes;
    case QuantKind::Q6_K:
      return static_cast<std::size_t>(cols / kKBlockElements) * kQ6_KBlockBytes;
    case QuantKind::Q4_0:
      return static_cast<std::size_t>(cols / kLegacyBlockElements) *
             kQ4_0BlockBytes;
    case QuantKind::Q5_0:
      return static_cast<std::size_t>(cols / kLegacyBlockElements) *
             kQ5_0BlockBytes;
  }
  return 0;
}

void dequantize_row(QuantKind kind, const std::uint8_t* source,
                    float* destination, int cols) {
  switch (kind) {
    case QuantKind::Q4_K:
      dequantize_q4_k(source, destination, cols);
      return;
    case QuantKind::Q5_K:
      dequantize_q5_k(source, destination, cols);
      return;
    case QuantKind::Q6_K:
      dequantize_q6_k(source, destination, cols);
      return;
    case QuantKind::Q4_0:
      dequantize_q4_0(source, destination, cols);
      return;
    case QuantKind::Q5_0:
      dequantize_q5_0(source, destination, cols);
      return;
    case QuantKind::Q8_0:
      break;
  }
  // Q8_0 has a fused kernel; this path exists only so the switch is total.
  const int blocks = cols / kQ8BlockElements;
  for (int block = 0; block < blocks; ++block) {
    const std::uint8_t* entry = source + block * kQ8BlockBytes;
    std::uint16_t scale_bits = 0;
    std::memcpy(&scale_bits, entry, sizeof(scale_bits));
    const float scale = half_to_float(scale_bits);
    const auto* values = reinterpret_cast<const std::int8_t*>(entry + 2);
    float* out = destination + block * kQ8BlockElements;
    for (int index = 0; index < kQ8BlockElements; ++index) {
      out[index] = static_cast<float>(values[index]) * scale;
    }
  }
}

}  // namespace tge::cpu
