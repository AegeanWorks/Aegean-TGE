#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

#include "harness.hpp"
#include "tge/core/dispatch.hpp"
#include "tge/cpu/kernels.hpp"
#include "tge/tensor/ops.hpp"

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

void expect_dequant_batch_matches(tge::cpu::QuantKind kind,
                                    const std::uint8_t* matrix, int rows,
                                    int cols, int tokens, std::uint32_t seed) {
  std::vector<float> input(static_cast<std::size_t>(tokens) * cols);
  std::vector<float> reference(static_cast<std::size_t>(tokens) * rows, 0.0F);
  std::vector<float> optimized(static_cast<std::size_t>(tokens) * rows, 0.0F);

  std::uint32_t state = seed | 1u;
  for (float& value : input) {
    state = state * 1664525u + 1013904223u;
    value = (static_cast<float>((state >> 8) & 0xFFFFu) / 32768.0F) - 1.0F;
  }

  tge::cpu::DequantBatchArgs args;
  args.data = matrix;
  args.kind = kind;
  args.rows = rows;
  args.cols = cols;
  args.input = input.data();
  args.tokens = tokens;

  args.out = reference.data();
  tge::cpu::gemv_dequant_batch_scalar(args, tge::cpu::GemvConfig{1});

  args.out = optimized.data();
  tge::core::ThreadPool pool(tge::cpu::resolve_thread_count(0));
  tge::kernels().gemv_dequant_batch(
      args, tge::cpu::GemvConfig{tge::cpu::resolve_thread_count(0), &pool});

  // Panel tiling reorders the reduction, so this is near rather than exact:
  // same tolerance discipline as the q8 batch differential.
  const float worst = tge::ops::max_abs_diff(
      reference.data(), optimized.data(),
      static_cast<std::int64_t>(reference.size()));
  const float tolerance = 1e-4F * static_cast<float>(cols) + 1e-3F;
  if (!(worst <= tolerance)) {
    std::fprintf(stderr,
                 "    dequant batch kind=%d rows=%d cols=%d tokens=%d worst=%g "
                 "tolerance=%g\n",
                 static_cast<int>(kind), rows, cols, tokens,
                 static_cast<double>(worst), static_cast<double>(tolerance));
    tgetest::fail("dequant batch matches scalar", __FILE__, __LINE__);
  }
}

TGE_TEST(dequant_batch_matches_scalar_reference) {
  // Column counts cross the 1024 panel grid (exact, tail, wide) at prefill
  // width and at decode width, for both hot K-quant formats on this file.
  // Sized for three full rows: the batch kernel strides rows by
  // quant_row_bytes, so a single-row matrix would read out of bounds.
  constexpr int kRows = 3;
  for (const int cols : {1024, 3072, 3328, 8192}) {
    for (const int tokens : {1, 7, 64}) {
      const std::vector<std::uint8_t> q4k =
          make_q4_k_matrix(kRows * cols / kBlockElements, 0xB4710201u);
      expect_dequant_batch_matches(tge::cpu::QuantKind::Q4_K, q4k.data(),
                                   kRows, cols, tokens, 0x11111111u);
      const std::vector<std::uint8_t> q6k =
          make_q6_k_matrix(kRows * cols / kBlockElements, 0xB4710601u);
      expect_dequant_batch_matches(tge::cpu::QuantKind::Q6_K, q6k.data(),
                                   kRows, cols, tokens, 0x66666666u);
    }
  }
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
  // to 15*d2 - m2. They differ unless the two halves were taken from the wrong
  // byte range, or swapped.
  TGE_CHECK(values[0] != values[32]);
  TGE_CHECK(values[32] > values[0]);
}

void expect_kq8_matches(tge::cpu::QuantKind kind,
                          const std::uint8_t* matrix, int rows, int cols,
                          int tokens, std::uint32_t seed, int input_mode) {
  // input_mode: 0 = zeros (exact-zero tripwire), 1 = uniform random
  // (quantization price), 2 = constant 0.5 (transcription proof: the Q8
  // quantizer is near-exact on constants, so any residual above 1e-3 is a
  // porting bug, not activation-quantization price).
  std::vector<float> input(static_cast<std::size_t>(tokens) * cols);
  std::vector<float> reference(static_cast<std::size_t>(tokens) * rows, 0.0F);
  std::vector<float> integer(static_cast<std::size_t>(tokens) * rows, 0.0F);

  if (input_mode == 1) {
    std::uint32_t state = seed | 1u;
    for (float& value : input) {
      state = state * 1664525u + 1013904223u;
      value = (static_cast<float>((state >> 8) & 0xFFFFu) / 32768.0F) - 1.0F;
    }
  } else if (input_mode == 2) {
    std::fill(input.begin(), input.end(), 0.5F);
  }

  tge::cpu::DequantBatchArgs args;
  args.data = matrix;
  args.kind = kind;
  args.rows = rows;
  args.cols = cols;
  args.input = input.data();
  args.tokens = tokens;

  // Oracle: the shipped float batch path (TGE-verified bit-exact vs its own
  // AVX2 twin). The integer path must track it up to the Q8-activation
  // quantization price — any transcription bug lands on top of the price.
  args.out = reference.data();
  tge::cpu::gemv_dequant_batch_scalar(args, tge::cpu::GemvConfig{1});

  args.out = integer.data();
  tge::core::ThreadPool pool(tge::cpu::resolve_thread_count(0));
  tge::cpu::gemv_dequant_batch_kq8_avx2(
      args, tge::cpu::GemvConfig{tge::cpu::resolve_thread_count(0), &pool});

  // Three-way triangulation lives in its own test below
  // (dequant_batch_kq8_avx2_matches_generic), defined after the kq8ref
  // namespace.

  if (input_mode == 0) {
    // Zero activations quantize to d == 0, so every dot is exactly 0
    // regardless of the weight bytes: a hard tripwire for pointer and
    // layout transcription errors.
    for (const float value : integer) {
      if (value != 0.0F) {
        std::fprintf(stderr,
                     "    kq8 zero-input kind=%d rows=%d cols=%d tokens=%d "
                     "value=%g\n",
                     static_cast<int>(kind), rows, cols, tokens,
                     static_cast<double>(value));
        tgetest::fail("kq8 zero input exactly zero", __FILE__, __LINE__);
      }
    }
    return;
  }

  const float worst = tge::ops::max_abs_diff(
      reference.data(), integer.data(),
      static_cast<std::int64_t>(reference.size()));
  // The test matrices are random bytes, so reference magnitudes are
  // arbitrary (scales/d are unconstrained); only a relative price is
  // meaningful. Measured activation-quantization price on these adversarial
  // inputs is <= ~1% (generic scalar vs float oracle); the 5% gate locks
  // that price with margin while failing gross transcription slips (the
  // k4-shuffle-table corruption measured 20-80x; Q4K AVX2 vs independent
  // generic scalar is gated separately at 1e-3).
  float reference_peak = 0.0F;
  for (const float value : reference) {
    const float magnitude = value < 0.0F ? -value : value;
    if (magnitude > reference_peak) {
      reference_peak = magnitude;
    }
  }
  const float relative =
      reference_peak > 0.0F ? worst / reference_peak : 0.0F;
  // Constant inputs quantize near-exactly, so the gate is tight: any
  // residual above it is transcription, not price. Random inputs carry the
  // activation-quantization price (large on adversarial random weights —
  // see §57), gated loosely to lock it while catching gross slips.
  const float relative_tolerance = input_mode == 2 ? 1e-3F : 0.05F;
  if (!(relative <= relative_tolerance)) {
    std::fprintf(stderr,
                 "    kq8 batch kind=%d rows=%d cols=%d tokens=%d "
                 "worst=%g refpeak=%g relative=%g tolerance=%g\n",
                 static_cast<int>(kind), rows, cols, tokens,
                  static_cast<double>(worst), static_cast<double>(reference_peak),
                  static_cast<double>(relative),
                  static_cast<double>(relative_tolerance));
    tgetest::fail("kq8 batch matches float oracle", __FILE__, __LINE__);
  }
}

namespace kq8ref {
// Test-local Q4_K/Q8_K block layouts (field order identical to ggml at
// cd8427d and to the prototype TU; transient only).
struct Q4KBlock {
  std::uint16_t d_bits = 0;
  std::uint16_t dmin_bits = 0;
  std::uint8_t scales[12] = {};
  std::uint8_t qs[128] = {};
};
struct Q8KBlockR {
  float d = 0.0F;
  std::int8_t qs[256] = {};
  std::int16_t bsums[16] = {};
};
static_assert(sizeof(Q4KBlock) == 144);
static_assert(sizeof(Q8KBlockR) == 292);

// Scalar port of ggml_vec_dot_q4_K_q8_K_generic
// (ggml/src/ggml-cpu/quants.c at cd8427d). Independent oracle for the AVX2
// transcription: same math, no shuffles, no FMA (plain order left to the
// compiler is fine here — the comparison tolerance absorbs ulps).
float dot_q4k_q8k_generic(const Q4KBlock* x, const Q8KBlockR* y, int blocks) {
  static const std::uint32_t kmask1 = 0x3F3F3F3F;
  static const std::uint32_t kmask2 = 0x0F0F0F0F;
  static const std::uint32_t kmask3 = 0x03030303;

  std::uint32_t utmp[4] = {};
  const std::uint8_t* scales = reinterpret_cast<const std::uint8_t*>(&utmp[0]);
  const std::uint8_t* mins = reinterpret_cast<const std::uint8_t*>(&utmp[2]);

  std::int8_t aux8[256] = {};
  std::int16_t aux16[8] = {};
  float sums[8] = {};
  std::int32_t aux32[8] = {};

  float sumf = 0.0F;
  for (int i = 0; i < blocks; ++i) {
    const std::uint8_t* q4 = x[i].qs;
    const std::int8_t* q8 = y[i].qs;
    std::memset(aux32, 0, sizeof(aux32));
    std::int8_t* a = aux8;
    for (int j = 0; j < 4; ++j) {
      for (int l = 0; l < 32; ++l) {
        a[l] = static_cast<std::int8_t>(q4[l] & 0x0F);
      }
      a += 32;
      for (int l = 0; l < 32; ++l) {
        a[l] = static_cast<std::int8_t>(q4[l] >> 4);
      }
      a += 32;
      q4 += 32;
    }
    std::memcpy(utmp, x[i].scales, 12);
    utmp[3] = ((utmp[2] >> 4) & kmask2) | (((utmp[1] >> 6) & kmask3) << 4);
    const std::uint32_t uaux = utmp[1] & kmask1;
    utmp[1] = (utmp[2] & kmask2) | (((utmp[0] >> 6) & kmask3) << 4);
    utmp[2] = uaux;
    utmp[0] &= kmask1;

    int sumi = 0;
    for (int j = 0; j < 16; ++j) {
      sumi += y[i].bsums[j] * mins[j / 2];
    }
    a = aux8;
    int is = 0;
    for (int j = 0; j < 8; ++j) {
      const std::int32_t scale = scales[is++];
      for (int rep = 0; rep < 4; ++rep) {
        for (int l = 0; l < 8; ++l) {
          aux16[l] = static_cast<std::int16_t>(q8[l] * a[l]);
        }
        for (int l = 0; l < 8; ++l) {
          aux32[l] += scale * aux16[l];
        }
        q8 += 8;
        a += 8;
      }
    }
    const float d =
        tge::cpu::half_to_float(x[i].d_bits) * y[i].d;
    for (int l = 0; l < 8; ++l) {
      sums[l] += d * static_cast<float>(aux32[l]);
    }
    const float dmin =
        tge::cpu::half_to_float(x[i].dmin_bits) * y[i].d;
    sumf -= dmin * static_cast<float>(sumi);
  }
  for (int l = 0; l < 8; ++l) {
    sumf += sums[l];
  }
  return sumf;
}

// Test-local copy of the prototype's scalar activation quantizer (same
// semantics; determinism across the two copies is itself a check).
inline int kq8ref_nearest_int(float fval) {
  float shifted = fval + 12582912.0F;
  int bits = 0;
  std::memcpy(&bits, &shifted, sizeof(bits));
  return (bits & 0x007FFFFF) - 0x00400000;
}

inline void quantize(const float* x, Q8KBlockR* y, int count) {
  const int blocks = count / 256;
  for (int i = 0; i < blocks; ++i) {
    std::memset(&y[i], 0, sizeof(y[i]));
    float max = 0.0F;
    float amax = 0.0F;
    for (int j = 0; j < 256; ++j) {
      const float ax = x[j] < 0.0F ? -x[j] : x[j];
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
      const int v = kq8ref_nearest_int(iscale * x[j]);
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
}  // namespace kq8ref

// Three-way triangulation for the Q4_K integer path: the generic scalar
// port of ggml's own dot shares the quantizer input with the AVX2 path but
// not its intrinsics. generic==AVX2 plus both-off-float means a
// formula/premise gap; generic-off-AVX2 means a transcription bug.
TGE_TEST(dequant_batch_kq8_avx2_matches_generic) {
  constexpr int kRows = 3;
  for (const int cols : {1024, 8192}) {
    const int blocks = cols / 256;
    const std::vector<std::uint8_t> q4k =
        make_q4_k_matrix(kRows * cols / kBlockElements, 0xB4710201u);
    for (const int tokens : {1, 7}) {
      // Random inputs (uniform) and constant inputs (0.5).
      for (int mode = 1; mode <= 2; ++mode) {
        std::vector<float> input(static_cast<std::size_t>(tokens) * cols);
        if (mode == 1) {
          std::uint32_t state = 0x11111111u;
          for (float& value : input) {
            state = state * 1664525u + 1013904223u;
            value =
                (static_cast<float>((state >> 8) & 0xFFFFu) / 32768.0F) - 1.0F;
          }
        } else {
          std::fill(input.begin(), input.end(), 0.5F);
        }

        tge::cpu::DequantBatchArgs args;
        args.data = q4k.data();
        args.kind = tge::cpu::QuantKind::Q4_K;
        args.rows = kRows;
        args.cols = cols;
        args.input = input.data();
        args.tokens = tokens;
        std::vector<float> reference(
            static_cast<std::size_t>(tokens) * kRows, 0.0F);
        std::vector<float> integer(
            static_cast<std::size_t>(tokens) * kRows, 0.0F);
        args.out = reference.data();
        tge::cpu::gemv_dequant_batch_scalar(args, tge::cpu::GemvConfig{1});
        args.out = integer.data();
        tge::core::ThreadPool pool(tge::cpu::resolve_thread_count(0));
        tge::cpu::gemv_dequant_batch_kq8_avx2(
            args, tge::cpu::GemvConfig{tge::cpu::resolve_thread_count(0),
                                       &pool});

        std::vector<kq8ref::Q8KBlockR> wdata(
            static_cast<std::size_t>(tokens) *
            static_cast<std::size_t>(blocks));
        for (int token = 0; token < tokens; ++token) {
          kq8ref::quantize(
              input.data() + static_cast<std::size_t>(token) * cols,
              wdata.data() + static_cast<std::size_t>(token) * blocks, cols);
        }
        float worst_avx = 0.0F;
        float worst_price = 0.0F;
        float peak = 0.0F;
        for (int row = 0; row < kRows; ++row) {
          const auto* weights = reinterpret_cast<const kq8ref::Q4KBlock*>(
              q4k.data() + static_cast<std::size_t>(row) *
                               static_cast<std::size_t>(cols) / 256 * 144);
          for (int token = 0; token < tokens; ++token) {
            const float generic = kq8ref::dot_q4k_q8k_generic(
                weights,
                wdata.data() + static_cast<std::size_t>(token) * blocks,
                blocks);
            const float avx =
                integer[static_cast<std::size_t>(token) * kRows + row];
            const float ref =
                reference[static_cast<std::size_t>(token) * kRows + row];
            const float da =
                generic - avx < 0.0F ? avx - generic : generic - avx;
            const float dp =
                generic - ref < 0.0F ? ref - generic : generic - ref;
            if (da > worst_avx) {
              worst_avx = da;
            }
            if (dp > worst_price) {
              worst_price = dp;
            }
            const float m = ref < 0.0F ? -ref : ref;
            if (m > peak) {
              peak = m;
            }
          }
        }
        const float rel_avx = peak > 0.0F ? worst_avx / peak : 0.0F;
        const float rel_price = peak > 0.0F ? worst_price / peak : 0.0F;
        if (!(rel_avx <= 1e-3F)) {
          std::fprintf(stderr,
                       "    kq8 tri cols=%d tokens=%d mode=%d peak=%g "
                       "avx-vs-generic=%g generic-vs-float=%g\n",
                       cols, tokens, mode, static_cast<double>(peak),
                       static_cast<double>(rel_avx),
                       static_cast<double>(rel_price));
          tgetest::fail("kq8 avx2 matches generic scalar", __FILE__,
                        __LINE__);
        }
      }
    }
  }
}

TGE_TEST(dequant_batch_kq8_matches_float_oracle) {
  // Same matrix as the shipped batch differential. bsums tails at cols 3328
  // and the tokens == 1 integer path are both covered.
  constexpr int kRows = 3;
  for (const int cols : {1024, 3072, 3328, 8192}) {
    for (const int tokens : {1, 7, 64}) {
      const std::vector<std::uint8_t> q4k =
          make_q4_k_matrix(kRows * cols / kBlockElements, 0xB4710201u);
      expect_kq8_matches(tge::cpu::QuantKind::Q4_K, q4k.data(), kRows, cols,
                         tokens, 0x11111111u, 1);
      expect_kq8_matches(tge::cpu::QuantKind::Q4_K, q4k.data(), kRows, cols,
                         tokens, 0u, 0);
      expect_kq8_matches(tge::cpu::QuantKind::Q4_K, q4k.data(), kRows, cols,
                         tokens, 0u, 2);
      const std::vector<std::uint8_t> q6k =
          make_q6_k_matrix(kRows * cols / kBlockElements, 0xB4710601u);
      expect_kq8_matches(tge::cpu::QuantKind::Q6_K, q6k.data(), kRows, cols,
                         tokens, 0x66666666u, 1);
      expect_kq8_matches(tge::cpu::QuantKind::Q6_K, q6k.data(), kRows, cols,
                         tokens, 0u, 0);
      expect_kq8_matches(tge::cpu::QuantKind::Q6_K, q6k.data(), kRows, cols,
                         tokens, 0u, 2);
    }
  }
}
