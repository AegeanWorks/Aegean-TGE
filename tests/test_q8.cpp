#include <cstdint>
#include <cstddef>
#include <vector>

#include "harness.hpp"
#include "tge/core/dispatch.hpp"
#include "tge/cpu/kernels.hpp"
#include "tge/tensor/ops.hpp"

namespace {

/// Builds a Q8_0 matrix in the on-disk layout: 34-byte blocks of one fp16 scale
/// followed by 32 signed bytes.
std::vector<std::uint8_t> make_q8_matrix(int rows, int cols, std::uint32_t seed) {
  const int blocks_per_row = cols / tge::cpu::kQ8BlockElements;
  std::vector<std::uint8_t> data(
      static_cast<std::size_t>(rows) * blocks_per_row * tge::cpu::kQ8BlockBytes);

  std::uint32_t state = seed | 1u;
  const auto next = [&state]() {
    state = state * 1664525u + 1013904223u;
    return state;
  };

  for (int row = 0; row < rows; ++row) {
    for (int block = 0; block < blocks_per_row; ++block) {
      std::uint8_t* entry =
          data.data() +
          (static_cast<std::size_t>(row) * blocks_per_row + block) *
              tge::cpu::kQ8BlockBytes;
      // A small positive fp16 scale: exponent 10, mantissa 0 is 0.0625.
      const std::uint16_t scale_bits = static_cast<std::uint16_t>(0x2C00u);
      entry[0] = static_cast<std::uint8_t>(scale_bits & 0xFFu);
      entry[1] = static_cast<std::uint8_t>(scale_bits >> 8);
      for (int index = 0; index < tge::cpu::kQ8BlockElements; ++index) {
        entry[2 + index] = static_cast<std::uint8_t>((next() >> 24) & 0xFFu);
      }
    }
  }
  return data;
}

void expect_q8_gemv_matches(int rows, int cols, std::uint32_t seed) {
  const std::vector<std::uint8_t> matrix = make_q8_matrix(rows, cols, seed);
  std::vector<float> vector(static_cast<std::size_t>(cols));
  std::vector<float> reference(static_cast<std::size_t>(rows), 0.0F);
  std::vector<float> optimized(static_cast<std::size_t>(rows), 0.0F);

  std::uint32_t state = 0xBEEF1234u;
  for (float& value : vector) {
    state = state * 1664525u + 1013904223u;
    value = (static_cast<float>((state >> 8) & 0xFFFFu) / 32768.0F) - 1.0F;
  }

  tge::cpu::Q8GemvArgs args;
  args.data = matrix.data();
  args.rows = rows;
  args.cols = cols;
  args.vector = vector.data();

  args.out = reference.data();
  tge::cpu::gemv_q8_0_scalar(args, tge::cpu::GemvConfig{1});

  args.out = optimized.data();
  tge::kernels().gemv_q8(args, tge::cpu::GemvConfig{tge::cpu::resolve_decode_threads()});

  const float worst = tge::ops::max_abs_diff(
      reference.data(), optimized.data(), static_cast<std::int64_t>(rows));
  const float tolerance = 1e-4F * static_cast<float>(cols) + 1e-3F;
  if (!(worst <= tolerance)) {
    std::fprintf(stderr, "    q8 gemv rows=%d cols=%d worst=%g tolerance=%g\n",
                 rows, cols, static_cast<double>(worst),
                 static_cast<double>(tolerance));
    tgetest::fail("q8 gemv matches scalar", __FILE__, __LINE__);
  }
}

}  // namespace

TGE_TEST(q8_gemv_matches_scalar_reference) {
  expect_q8_gemv_matches(1, 32, 1u);
  expect_q8_gemv_matches(7, 64, 2u);
  expect_q8_gemv_matches(64, 896, 3u);
  expect_q8_gemv_matches(129, 256, 4u);
}

TGE_TEST(half_conversion_matches_known_values) {
  TGE_CHECK_NEAR(tge::cpu::half_to_float(0x0000), 0.0F, 1e-9);
  TGE_CHECK_NEAR(tge::cpu::half_to_float(0x3C00), 1.0F, 1e-6);
  TGE_CHECK_NEAR(tge::cpu::half_to_float(0x4000), 2.0F, 1e-6);
  TGE_CHECK_NEAR(tge::cpu::half_to_float(0xBC00), -1.0F, 1e-6);
  TGE_CHECK_NEAR(tge::cpu::half_to_float(0x3555), 0.33325195F, 1e-6);
}

TGE_TEST(decode_thread_count_is_bounded) {
  const int decode_threads = tge::cpu::resolve_decode_threads();
  const int logical = tge::cpu::resolve_thread_count(0);
  TGE_CHECK(decode_threads >= 1);
  TGE_CHECK(decode_threads <= logical);
}
