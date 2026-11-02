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

namespace {

constexpr std::size_t kQ6_KBlockBytes = 210;

/// Q6_K rows with valid fp16 d and random payload/scales. Scales are plain
/// int8 (no inf/nan domain exists), so only d is fixed to a small value.
std::vector<std::uint8_t> make_q6_k_matrix(int blocks, std::uint32_t seed) {
  std::vector<std::uint8_t> data(
      static_cast<std::size_t>(blocks) * kQ6_KBlockBytes);

  std::uint32_t state = seed | 1u;
  const auto next = [&state]() {
    state = state * 1664525u + 1013904223u;
    return state;
  };

  for (int block = 0; block < blocks; ++block) {
    std::uint8_t* entry = data.data() + static_cast<std::size_t>(block) * kQ6_KBlockBytes;
    entry[208] = 0x00;
    entry[209] = 0x2C;  // 0.0625 in fp16.
    for (std::size_t index = 0; index < kQ6_KBlockBytes; ++index) {
      if (index == 208 || index == 209) {
        continue;
      }
      entry[index] = static_cast<std::uint8_t>((next() >> 24) & 0xFFu);
    }
  }
  return data;
}

}  // namespace

TGE_TEST(q6k_avx2_matches_scalar_exactly) {
  for (const int blocks : {1, 3}) {
    const int cols = blocks * kBlockElements;
    const std::vector<std::uint8_t> data = make_q6_k_matrix(blocks, 0x6B6B6B6Bu);

    std::vector<float> scalar(static_cast<std::size_t>(cols), 0.0F);
    std::vector<float> vectorised(static_cast<std::size_t>(cols), 0.0F);

    tge::cpu::dequantize_row(tge::cpu::QuantKind::Q6_K, data.data(),
                             scalar.data(), cols);
    tge::cpu::dequantize_row_avx2(tge::cpu::QuantKind::Q6_K, data.data(),
                                  vectorised.data(), cols);

  // Bit-identical: multiply-only in scalar order, no FMA anywhere.
  TGE_CHECK(std::memcmp(scalar.data(), vectorised.data(),
                        scalar.size() * sizeof(float)) == 0);
  }
}

TGE_TEST(q51_scalar_matches_hand_computation) {
  // One block, d = 1.0, m = 2.0, hand-placed nibbles and high bits.
  // out[i] = q*d + m, so every value is directly checkable.
  std::vector<std::uint8_t> data(22 + 2, 0);  // +2 headroom, unused
  data.resize(24);
  data[0] = 0x00;
  data[1] = 0x3C;  // 1.0 in fp16.
  data[2] = 0x00;
  data[3] = 0x40;  // 2.0 in fp16.
  // qh: bit 0 and bit 17 set -> lo[0] gains 0x10, hi[1] gains 0x10.
  // (bit i feeds weight i; bit i + 16 feeds weight i + 16.)
  data[4] = 0x01;
  data[5] = 0x00;
  data[6] = 0x02;
  data[7] = 0x00;
  // quants: qs[0] = 0x21 (lo 1, hi 2), qs[1] = 0x43 (lo 3, hi 4).
  data[8] = 0x21;
  data[9] = 0x43;
  std::vector<float> out(32, -999.0F);
  tge::cpu::dequantize_row(tge::cpu::QuantKind::Q5_1, data.data(), out.data(),
                           32);
  TGE_CHECK_NEAR(out[0], 1.0F * (1 + 16) + 2.0F, 0.0F);
  TGE_CHECK_NEAR(out[1], 1.0F * 3 + 2.0F, 0.0F);
  TGE_CHECK_NEAR(out[16], 1.0F * 2 + 2.0F, 0.0F);
  TGE_CHECK_NEAR(out[17], 1.0F * (4 + 16) + 2.0F, 0.0F);
  TGE_CHECK_NEAR(out[2], 1.0F * 0 + 2.0F, 0.0F);
}

// Cross-check against ggml's own dequantizer: the two blocks and outputs
// below were produced by ggml's quantize_row_q5_1_ref + dequantize_row_q5_1
// (llama.cpp @ cd8427d, same toolchain) and embedded here, so no file or
// second binary is needed to re-verify. Bit-identical: this is the oracle
// the AVX2 path must match.
TGE_TEST(q5_1_matches_ggml_reference) {
  const char* hex =
      "a52ff3be14f07f203a88c2115091942d7fd9e488e3b7550fae2f9fbffcf557fb0"
      "5070f71c200938cb342bbdc96975748";
  const float expected[64] = {
      -0.5428466796875F, -0.78173828125F, 0.4127197265625F, -1.61785888671875F,
      0.173828125F, -1.61785888671875F, -1.259521484375F, -0.18450927734375F,
      0.05438232421875F, -0.66229248046875F, -1.259521484375F, -0.78173828125F,
      0.53216552734375F, 1.00994873046875F, 0.77105712890625F, 1.96551513671875F,
      0.53216552734375F, 1.12939453125F, 1.607177734375F, 0.29327392578125F,
      0.77105712890625F, 1.24884033203125F, 1.24884033203125F, -1.4984130859375F,
      -0.90118408203125F, -0.18450927734375F, -0.0650634765625F, -0.78173828125F,
      -0.0650634765625F, 1.48773193359375F, -1.14007568359375F, -1.7373046875F,
      -1.3052978515625F, -1.0653076171875F, 1.8145751953125F, 0.1346435546875F,
      0.254638671875F, 0.0146484375F, 0.3746337890625F, 1.45458984375F,
      0.3746337890625F, -1.665283203125F, 1.3345947265625F, -0.46533203125F,
      0.734619140625F, 0.8546142578125F, 0.8546142578125F, 0.974609375F,
      0.0146484375F, 0.0146484375F, 0.0146484375F, -1.0653076171875F,
      1.45458984375F, -1.9052734375F, 1.0946044921875F, -0.9453125F,
      1.3345947265625F, 0.49462890625F, -0.5853271484375F, 1.5745849609375F,
      1.0946044921875F, 1.0946044921875F, 0.6146240234375F, 0.49462890625F};
  std::uint8_t data[48];
  for (int index = 0; index < 48; ++index) {
    const char digits[3] = {hex[2 * index], hex[2 * index + 1], 0};
    auto nibble = [](char c) {
      return c >= '0' && c <= '9' ? c - '0' : c - 'a' + 10;
    };
    data[index] =
        static_cast<std::uint8_t>((nibble(digits[0]) << 4) | nibble(digits[1]));
  }
  float actual[64];
  tge::cpu::dequantize_row(tge::cpu::QuantKind::Q5_1, data, actual, 64);
  TGE_CHECK(std::memcmp(actual, expected, sizeof(expected)) == 0);
}

namespace {

constexpr std::size_t kQ5_1BlockBytes = 24;

/// Q5_1 rows with valid fp16 d/m and random high bits / quant bytes.
std::vector<std::uint8_t> make_q5_1_matrix(int blocks, std::uint32_t seed) {
  std::vector<std::uint8_t> data(
      static_cast<std::size_t>(blocks) * kQ5_1BlockBytes);

  std::uint32_t state = seed | 1u;
  const auto next = [&state]() {
    state = state * 1664525u + 1013904223u;
    return state;
  };

  for (int block = 0; block < blocks; ++block) {
    std::uint8_t* entry = data.data() + static_cast<std::size_t>(block) * kQ5_1BlockBytes;
    entry[0] = 0x00;
    entry[1] = 0x2C;  // d = 0.0625 in fp16.
    entry[2] = 0x00;
    entry[3] = 0x3C;  // m = 1.0 in fp16.
    for (std::size_t index = 4; index < kQ5_1BlockBytes; ++index) {
      entry[index] = static_cast<std::uint8_t>((next() >> 24) & 0xFFu);
    }
  }
  return data;
}

}  // namespace

TGE_TEST(q51_avx2_matches_scalar_exactly) {
  for (const int blocks : {1, 3}) {
    const int cols = blocks * 32;
    const std::vector<std::uint8_t> data = make_q5_1_matrix(blocks, 0x51515151u);

    std::vector<float> scalar(static_cast<std::size_t>(cols), 0.0F);
    std::vector<float> vectorised(static_cast<std::size_t>(cols), 0.0F);

    tge::cpu::dequantize_row(tge::cpu::QuantKind::Q5_1, data.data(),
                             scalar.data(), cols);
    tge::cpu::dequantize_row_avx2(tge::cpu::QuantKind::Q5_1, data.data(),
                                  vectorised.data(), cols);

  // Bit-identical: mul-then-add in scalar order, no FMA anywhere.
  TGE_CHECK(std::memcmp(scalar.data(), vectorised.data(),
                        scalar.size() * sizeof(float)) == 0);
  }
}

namespace {

constexpr std::size_t kQ5_KBlockBytes = 176;
constexpr int kQ5_KBlockElements = 256;

/// Q5_K rows with valid fp16 d/dmin and random scales / high / low bytes.
std::vector<std::uint8_t> make_q5_k_matrix(int blocks, std::uint32_t seed) {
  std::vector<std::uint8_t> data(
      static_cast<std::size_t>(blocks) * kQ5_KBlockBytes);

  std::uint32_t state = seed | 1u;
  const auto next = [&state]() {
    state = state * 1664525u + 1013904223u;
    return state;
  };

  for (int block = 0; block < blocks; ++block) {
    std::uint8_t* entry = data.data() + static_cast<std::size_t>(block) * kQ5_KBlockBytes;
    entry[0] = 0x00;
    entry[1] = 0x2C;  // d = 0.0625 in fp16.
    entry[2] = 0x00;
    entry[3] = 0x3C;  // dmin = 1.0 in fp16.
    for (std::size_t index = 4; index < kQ5_KBlockBytes; ++index) {
      entry[index] = static_cast<std::uint8_t>((next() >> 24) & 0xFFu);
    }
  }
  return data;
}

}  // namespace

TGE_TEST(q5k_avx2_matches_scalar_exactly) {
  for (const int blocks : {1, 3}) {
    const int cols = blocks * kQ5_KBlockElements;
    const std::vector<std::uint8_t> data = make_q5_k_matrix(blocks, 0x5B5B5B5Du);

    std::vector<float> scalar(static_cast<std::size_t>(cols), 0.0F);
    std::vector<float> vectorised(static_cast<std::size_t>(cols), 0.0F);

    tge::cpu::dequantize_row(tge::cpu::QuantKind::Q5_K, data.data(),
                             scalar.data(), cols);
    tge::cpu::dequantize_row_avx2(tge::cpu::QuantKind::Q5_K, data.data(),
                                  vectorised.data(), cols);

    // Bit-identical: mul-then-sub in scalar order, no FMA anywhere.
    TGE_CHECK(std::memcmp(scalar.data(), vectorised.data(),
                          scalar.size() * sizeof(float)) == 0);
  }
}

namespace {

constexpr std::size_t kQ5_0BlockBytes = 22;
constexpr int kLegacyElements = 32;

/// Q5_0 rows with valid fp16 d and random high bits / quant bytes.
std::vector<std::uint8_t> make_q5_0_matrix(int blocks, std::uint32_t seed) {
  std::vector<std::uint8_t> data(
      static_cast<std::size_t>(blocks) * kQ5_0BlockBytes);

  std::uint32_t state = seed | 1u;
  const auto next = [&state]() {
    state = state * 1664525u + 1013904223u;
    return state;
  };

  for (int block = 0; block < blocks; ++block) {
    std::uint8_t* entry = data.data() + static_cast<std::size_t>(block) * kQ5_0BlockBytes;
    entry[0] = 0x00;
    entry[1] = 0x2C;  // 0.0625 in fp16.
    for (std::size_t index = 2; index < kQ5_0BlockBytes; ++index) {
      entry[index] = static_cast<std::uint8_t>((next() >> 24) & 0xFFu);
    }
  }
  return data;
}

}  // namespace

TGE_TEST(q50_avx2_matches_scalar_exactly) {
  for (const int blocks : {1, 3}) {
    const int cols = blocks * kLegacyElements;
    const std::vector<std::uint8_t> data = make_q5_0_matrix(blocks, 0x50505050u);

    std::vector<float> scalar(static_cast<std::size_t>(cols), 0.0F);
    std::vector<float> vectorised(static_cast<std::size_t>(cols), 0.0F);

    tge::cpu::dequantize_row(tge::cpu::QuantKind::Q5_0, data.data(),
                             scalar.data(), cols);
    tge::cpu::dequantize_row_avx2(tge::cpu::QuantKind::Q5_0, data.data(),
                                  vectorised.data(), cols);

    // Bit-identical: multiply-only in scalar order, no FMA anywhere.
    TGE_CHECK(std::memcmp(scalar.data(), vectorised.data(),
                          scalar.size() * sizeof(float)) == 0);
  }
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
