#include <cstdint>
#include <cstring>
#include <vector>

#include "harness.hpp"
#include "tge/cpu/kernels.hpp"

namespace {

constexpr int kBlockElements = 256;
constexpr std::size_t kQ4_KBlockBytes = 144;

/// Builds Q4_K rows with a valid fp16 super-block scale and minimum, and random
/// payload bytes. The scales are fixed rather than random so the comparison
/// cannot be confounded by infinities or NaNs in the fp16 exponent.
std::vector<std::uint8_t> make_q4_k_matrix(int blocks, std::uint32_t seed) {
  std::vector<std::uint8_t> data(
      static_cast<std::size_t>(blocks) * kQ4_KBlockBytes);

  std::uint32_t state = seed | 1u;
  const auto next = [&state]() {
    state = state * 1664525u + 1013904223u;
    return state;
  };

  for (int block = 0; block < blocks; ++block) {
    std::uint8_t* entry = data.data() + static_cast<std::size_t>(block) * kQ4_KBlockBytes;
    // 0x2C00 is 0.0625 in fp16: a small positive scale for both d and dmin.
    entry[0] = 0x00;
    entry[1] = 0x2C;
    entry[2] = 0x00;
    entry[3] = 0x2C;
    for (std::size_t index = 4; index < kQ4_KBlockBytes; ++index) {
      entry[index] = static_cast<std::uint8_t>((next() >> 24) & 0xFFu);
    }
  }
  return data;
}

}  // namespace

TGE_TEST(q4k_avx2_matches_scalar_exactly) {
  const std::vector<std::uint8_t> data = make_q4_k_matrix(1, 0x51ED270Bu);

  std::vector<float> scalar(kBlockElements, 0.0F);
  std::vector<float> vectorised(kBlockElements, 0.0F);

  tge::cpu::dequantize_row(tge::cpu::QuantKind::Q4_K, data.data(), scalar.data(),
                           kBlockElements);
  tge::cpu::dequantize_row_avx2(tge::cpu::QuantKind::Q4_K, data.data(),
                                vectorised.data(), kBlockElements);

  // Bit-identical, not merely close: the kernel is a speed change, not a
  // precision trade, and anything else would move the logits.
  TGE_CHECK(std::memcmp(scalar.data(), vectorised.data(),
                        scalar.size() * sizeof(float)) == 0);
}

TGE_TEST(q4k_avx2_matches_scalar_across_blocks) {
  const int blocks = 3;
  const int cols = blocks * kBlockElements;
  const std::vector<std::uint8_t> data = make_q4_k_matrix(blocks, 0x1234ABCDu);

  std::vector<float> scalar(static_cast<std::size_t>(cols), 0.0F);
  std::vector<float> vectorised(static_cast<std::size_t>(cols), 0.0F);

  tge::cpu::dequantize_row(tge::cpu::QuantKind::Q4_K, data.data(), scalar.data(),
                           cols);
  tge::cpu::dequantize_row_avx2(tge::cpu::QuantKind::Q4_K, data.data(),
                                vectorised.data(), cols);

  TGE_CHECK(std::memcmp(scalar.data(), vectorised.data(),
                        scalar.size() * sizeof(float)) == 0);
}

TGE_TEST(q4k_avx2_nibble_halves_are_distinct) {
  // A row where every low nibble is 0 and every high nibble is 15. If the two
  // halves were taken from the wrong byte range, or swapped, the block would
  // come out uniform and this would not hold.
  std::vector<std::uint8_t> data(kQ4_KBlockBytes, 0xF0);
  data[0] = 0x00;
  data[1] = 0x2C;
  data[2] = 0x00;
  data[3] = 0x2C;

  std::vector<float> values(kBlockElements, 0.0F);
  tge::cpu::dequantize_row_avx2(tge::cpu::QuantKind::Q4_K, data.data(),
                                values.data(), kBlockElements);

  // The low nibble is zero, so the first half evaluates to -m1 and the second
  // to 15*d2 - m2. They differ unless the two halves were read from the wrong
  // byte range or swapped.
  TGE_CHECK(values[0] != values[32]);
  TGE_CHECK(values[32] > values[0]);
}
