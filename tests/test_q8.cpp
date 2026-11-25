#include <cstdint>
#include <cstddef>
#include <cstring>
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

TGE_TEST(q8_batch_tiled_matches_shipped_bitwise) {
  const int rows[] = {8, 33, 64};
  const int cols[] = {32, 64, 896};
  const int tokens[] = {8, 8, 64};
  for (int shape = 0; shape < 3; ++shape) {
    const std::vector<std::uint8_t> matrix =
        make_q8_matrix(rows[shape], cols[shape], 201u + shape);
    std::vector<float> input(static_cast<std::size_t>(tokens[shape]) * cols[shape]);
    for (std::size_t index = 0; index < input.size(); ++index) {
      input[index] = static_cast<float>(index % 29 - 14) / 14.0F;
    }
    std::vector<float> reference(input.size() * rows[shape] / cols[shape]);
    std::vector<float> tiled(reference.size());
    tge::cpu::Q8BatchArgs args;
    args.data = matrix.data();
    args.rows = rows[shape];
    args.cols = cols[shape];
    args.input = input.data();
    args.tokens = tokens[shape];
    args.out = reference.data();
    tge::core::ThreadPool pool(4, false);
    const tge::cpu::GemvConfig config{4, &pool};
    tge::cpu::gemv_q8_batch_avx2(args, config);
    args.out = tiled.data();
    tge::cpu::gemv_q8_batch_tiled_avx2(args, config);
    TGE_CHECK(std::memcmp(reference.data(), tiled.data(),
                          reference.size() * sizeof(float)) == 0);
  }
}

TGE_TEST(q8_group_matches_individual_bitwise) {
  constexpr int kCols = 896;
  std::vector<float> input(static_cast<std::size_t>(kCols));
  for (int index = 0; index < kCols; ++index) {
    input[static_cast<std::size_t>(index)] =
        static_cast<float>(index % 31 - 15) / 15.0F;
  }
  const auto check = [&](const int* rows, int threads) {
    std::vector<std::uint8_t> matrices[3];
    tge::cpu::Q8GemvArgs args[3];
    std::vector<float> reference[3];
    std::vector<float> grouped[3];
    for (int matrix = 0; matrix < 3; ++matrix) {
      matrices[matrix] = make_q8_matrix(rows[matrix], kCols, 71u + matrix);
      reference[matrix].resize(static_cast<std::size_t>(rows[matrix]));
      grouped[matrix].resize(static_cast<std::size_t>(rows[matrix]));
      args[matrix].data = matrices[matrix].data();
      args[matrix].rows = rows[matrix];
      args[matrix].cols = kCols;
      args[matrix].vector = input.data();
      args[matrix].out = reference[matrix].data();
    }
    tge::core::ThreadPool pool(threads, false);
    const tge::cpu::GemvConfig config{threads, &pool};
    for (int matrix = 0; matrix < 3; ++matrix) {
      tge::cpu::gemv_q8_0_avx2(args[matrix], config);
      args[matrix].out = grouped[matrix].data();
    }
    tge::cpu::gemv_q8_0_group_avx2(args, 3, config);
    for (int matrix = 0; matrix < 3; ++matrix) {
      TGE_CHECK(std::memcmp(reference[matrix].data(), grouped[matrix].data(),
                            reference[matrix].size() * sizeof(float)) == 0);
    }
  };
  const int normal[3] = {896, 128, 896};
  check(normal, 4);
  // 95 rows at 12 threads produce 95 one-row bands per matrix: 285 tasks,
  // forcing the checked stack-to-heap migration rather than overflowing.
  const int overflow[3] = {95, 95, 95};
  check(overflow, 12);
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
  bool threw_grouped = false;
  try {
    tge::cpu::gemv_q8_0_group_avx2(&args, 1, tge::cpu::GemvConfig{1});
  } catch (...) {
    threw_grouped = true;
  }
  TGE_CHECK(threw_grouped);

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

TGE_TEST(half_conversion_rounds_exact_ties_to_even) {
  // Regression: float_to_half is documented "round to nearest even" but added
  // the round bit unconditionally, ignoring the kept LSB and sticky bits. At an
  // exact half-way value whose lower half has an even mantissa LSB, it rounded
  // up to odd. Build inputs from exact bit patterns so the tie is precise.
  const auto from_bits = [](std::uint32_t bits) {
    float value = 0.0F;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
  };
  // 1.0 = 0x3F800000 -> 0x3C00. mantissa 0x1000 is exactly half-way between
  // 0x3C00 (even LSB) and 0x3C01; nearest-even keeps 0x3C00.
  TGE_CHECK(tge::cpu::float_to_half(from_bits(0x3F801000u)) == 0x3C00u);
  // mantissa 0x3000 is the tie between 0x3C01 (odd) and 0x3C02 (even);
  // nearest-even rounds up to 0x3C02.
  TGE_CHECK(tge::cpu::float_to_half(from_bits(0x3F803000u)) == 0x3C02u);
  // Round bit set plus any sticky bit is strictly above the tie -> round up.
  TGE_CHECK(tge::cpu::float_to_half(from_bits(0x3F801001u)) == 0x3C01u);
  // Round bit clear is strictly below the tie -> keep the lower half.
  TGE_CHECK(tge::cpu::float_to_half(from_bits(0x3F800FFFu)) == 0x3C00u);
  // Subnormal branch (unbiased -15): guard bit set, sticky clear, even kept
  // LSB is the tie between half 0x0200 and 0x0201 -> nearest-even keeps 0x0200.
  TGE_CHECK(tge::cpu::float_to_half(from_bits(0x38002000u)) == 0x0200u);
  TGE_CHECK(tge::cpu::float_to_half(from_bits(0x38006000u)) == 0x0202u);
}

TGE_TEST(decode_thread_count_is_bounded) {
  const int decode_threads = tge::cpu::resolve_decode_threads();
  const int logical = tge::cpu::resolve_thread_count(0);
  TGE_CHECK(decode_threads >= 1);
  TGE_CHECK(decode_threads <= logical);
}
