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
  tge::core::ThreadPool pool(tge::cpu::resolve_decode_threads());
  tge::kernels().gemv_q8(
      args, tge::cpu::GemvConfig{tge::cpu::resolve_decode_threads(), &pool});

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

void expect_q8_batch_matches(int rows, int cols, int tokens,
                             std::uint32_t seed) {
  const std::vector<std::uint8_t> matrix = make_q8_matrix(rows, cols, seed);
  std::vector<float> input(static_cast<std::size_t>(tokens) * cols);
  std::vector<float> reference(static_cast<std::size_t>(tokens) * rows, 0.0F);
  std::vector<float> optimized(static_cast<std::size_t>(tokens) * rows, 0.0F);

  std::uint32_t state = 0x1234ABCDu;
  for (float& value : input) {
    state = state * 1664525u + 1013904223u;
    value = (static_cast<float>((state >> 8) & 0xFFFFu) / 32768.0F) - 1.0F;
  }

  tge::cpu::Q8BatchArgs args;
  args.data = matrix.data();
  args.rows = rows;
  args.cols = cols;
  args.input = input.data();
  args.tokens = tokens;

  args.out = reference.data();
  tge::cpu::gemv_q8_batch_scalar(args, tge::cpu::GemvConfig{1});

  args.out = optimized.data();
  tge::core::ThreadPool pool(tge::cpu::resolve_thread_count(0));
  tge::kernels().gemv_q8_batch(
      args,
      tge::cpu::GemvConfig{tge::cpu::resolve_thread_count(0), &pool});

  const float worst = tge::ops::max_abs_diff(
      reference.data(), optimized.data(),
      static_cast<std::int64_t>(reference.size()));
  const float tolerance = 1e-4F * static_cast<float>(cols) + 1e-3F;
  if (!(worst <= tolerance)) {
    std::fprintf(stderr,
                 "    q8 batch rows=%d cols=%d tokens=%d worst=%g tolerance=%g\n",
                 rows, cols, tokens, static_cast<double>(worst),
                 static_cast<double>(tolerance));
    tgetest::fail("q8 batch matches scalar", __FILE__, __LINE__);
  }
}

}  // namespace

TGE_TEST(q8_gemv_matches_scalar_reference) {
  expect_q8_gemv_matches(1, 32, 1u);
  expect_q8_gemv_matches(7, 64, 2u);
  expect_q8_gemv_matches(64, 896, 3u);
  expect_q8_gemv_matches(129, 256, 4u);
}

TGE_TEST(q8_batch_matches_scalar_reference) {
  // Token counts around the four-wide group and the full-group case, so the
  // tail path is exercised rather than assumed.
  expect_q8_batch_matches(1, 32, 1, 11u);
  expect_q8_batch_matches(7, 64, 3, 12u);
  expect_q8_batch_matches(32, 128, 4, 13u);
  expect_q8_batch_matches(33, 256, 7, 14u);
  expect_q8_batch_matches(64, 896, 16, 15u);
}

TGE_TEST(q8_rejects_partial_block) {
  // Regression: cols%32 tails were silently dropped while the prefix
  // benchmarked as exact. Both ISAs must fail loudly instead.
  const std::vector<std::uint8_t> matrix = make_q8_matrix(2, 32, 1u);
  std::vector<float> vector(48, 0.5F);
  std::vector<float> out(2, 0.0F);
  tge::cpu::Q8GemvArgs args;
  args.data = matrix.data();
  args.rows = 2;
  args.cols = 48;
  args.vector = vector.data();
  args.out = out.data();
  bool threw_scalar = false;
  try {
    tge::cpu::gemv_q8_0_scalar(args, tge::cpu::GemvConfig{1});
  } catch (...) {
    threw_scalar = true;
  }
  TGE_CHECK(threw_scalar);
  bool threw_dispatched = false;
  try {
    tge::kernels().gemv_q8(args, tge::cpu::GemvConfig{1});
  } catch (...) {
    threw_dispatched = true;
  }
  TGE_CHECK(threw_dispatched);

  tge::cpu::Q8BatchArgs bargs;
  std::vector<float> input(48, 0.5F);
  std::vector<float> bout(2, 0.0F);
  bargs.data = matrix.data();
  bargs.rows = 2;
  bargs.cols = 48;
  bargs.input = input.data();
  bargs.tokens = 1;
  bargs.out = bout.data();
  bool threw_bscalar = false;
  try {
    tge::cpu::gemv_q8_batch_scalar(bargs, tge::cpu::GemvConfig{1});
  } catch (...) {
    threw_bscalar = true;
  }
  TGE_CHECK(threw_bscalar);
  bool threw_bdispatched = false;
  try {
    tge::kernels().gemv_q8_batch(bargs, tge::cpu::GemvConfig{1});
  } catch (...) {
    threw_bdispatched = true;
  }
  TGE_CHECK(threw_bdispatched);
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
