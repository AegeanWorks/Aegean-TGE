#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <vector>

#include "harness.hpp"
#include "tge/core/dispatch.hpp"
#include "tge/cpu/kernels.hpp"
#include "tge/tensor/ops.hpp"

namespace {

void fill(float* data, std::size_t count, std::uint32_t seed) {
  std::uint32_t state = seed | 1u;
  for (std::size_t index = 0; index < count; ++index) {
    state = state * 1664525u + 1013904223u;
    data[index] =
        (static_cast<float>((state >> 9) & 0x7FFFu) / 16384.0F) - 1.0F;
  }
}

void expect_gemm_matches_reference(int m, int n, int k, int threads) {
  std::vector<float> a(static_cast<std::size_t>(m) * k);
  std::vector<float> b(static_cast<std::size_t>(k) * n);
  std::vector<float> reference(static_cast<std::size_t>(m) * n);
  std::vector<float> optimized(static_cast<std::size_t>(m) * n);

  fill(a.data(), a.size(), 0x51ED270Bu);
  fill(b.data(), b.size(), 0x9E3779B9u);

  tge::cpu::GemmArgs args;
  args.m = m;
  args.n = n;
  args.k = k;
  args.a = a.data();
  args.lda = k;
  args.b = b.data();
  args.ldb = n;
  args.c = reference.data();
  args.ldc = n;

  tge::ops::gemm_ref(args, tge::cpu::GemmConfig{1});

  args.c = optimized.data();
  tge::kernels().gemm_f32(args, tge::cpu::GemmConfig{threads});

  const float worst = tge::ops::max_abs_diff(
      reference.data(), optimized.data(),
      static_cast<std::int64_t>(reference.size()));
  const float tolerance = 1e-5F * static_cast<float>(k) + 1e-4F;

  if (!(worst <= tolerance)) {
    std::fprintf(stderr,
                 "    m=%d n=%d k=%d threads=%d worst=%g tolerance=%g kernel=%s\n",
                 m, n, k, threads, static_cast<double>(worst),
                 static_cast<double>(tolerance),
                 tge::kernels().gemm_name.c_str());
    tgetest::fail("gemm matches reference", __FILE__, __LINE__);
  }
}

void expect_gemv_matches_reference(int rows, int cols, int threads) {
  std::vector<float> matrix(static_cast<std::size_t>(rows) * cols);
  std::vector<float> vector(static_cast<std::size_t>(cols));
  std::vector<float> reference(static_cast<std::size_t>(rows));
  std::vector<float> optimized(static_cast<std::size_t>(rows));

  fill(matrix.data(), matrix.size(), 0xA5A5A5u);
  fill(vector.data(), vector.size(), 0x5A5A5Au);

  tge::cpu::GemvArgs args;
  args.rows = rows;
  args.cols = cols;
  args.matrix = matrix.data();
  args.ldm = cols;
  args.vector = vector.data();
  args.out = reference.data();

  tge::ops::gemv_ref(args, tge::cpu::GemvConfig{1});

  args.out = optimized.data();
  tge::kernels().gemv_f32(args, tge::cpu::GemvConfig{threads});

  const float worst = tge::ops::max_abs_diff(
      reference.data(), optimized.data(), static_cast<std::int64_t>(rows));
  const float tolerance = 1e-5F * static_cast<float>(cols) + 1e-4F;

  if (!(worst <= tolerance)) {
    std::fprintf(stderr,
                 "    gemv rows=%d cols=%d threads=%d worst=%g tolerance=%g\n",
                 rows, cols, threads, static_cast<double>(worst),
                 static_cast<double>(tolerance));
    tgetest::fail("gemv matches reference", __FILE__, __LINE__);
  }
}

}  // namespace

TGE_TEST(gemm_handles_degenerate_shapes) {
  expect_gemm_matches_reference(1, 1, 1, 1);
  expect_gemm_matches_reference(1, 16, 1, 1);
  expect_gemm_matches_reference(6, 16, 1, 1);
  expect_gemm_matches_reference(5, 15, 1, 1);
}

TGE_TEST(gemm_matches_reference_off_tile_edges) {
  expect_gemm_matches_reference(7, 17, 3, 1);
  expect_gemm_matches_reference(13, 29, 31, 1);
  expect_gemm_matches_reference(5, 33, 65, 1);
}

TGE_TEST(gemm_matches_reference_on_tiles) {
  expect_gemm_matches_reference(6, 16, 16, 1);
  expect_gemm_matches_reference(12, 32, 64, 1);
  expect_gemm_matches_reference(96, 128, 128, 1);
}

TGE_TEST(gemm_matches_reference_across_k_blocks) {
  // K larger than the 256-deep packing block exercises accumulation across
  // depth blocks, which is where a store instead of an add would show up.
  expect_gemm_matches_reference(96, 96, 1024, 1);
  expect_gemm_matches_reference(17, 33, 601, 1);
}

TGE_TEST(gemm_matches_reference_with_threads) {
  // 1000 rows over 144-row blocks leaves an uneven final block, so the
  // threaded path has to handle a short panel.
  expect_gemm_matches_reference(1000, 300, 700, 4);
  expect_gemm_matches_reference(1000, 300, 700, tge::cpu::resolve_thread_count(0));
}

TGE_TEST(gemv_matches_reference) {
  expect_gemv_matches_reference(1, 1, 1);
  expect_gemv_matches_reference(3, 7, 1);
  expect_gemv_matches_reference(16, 16, 1);
  expect_gemv_matches_reference(129, 257, 1);
  expect_gemv_matches_reference(37, 1023, 1);
}

TGE_TEST(gemv_matches_reference_with_threads) {
  const int workers = tge::cpu::resolve_thread_count(0);
  expect_gemv_matches_reference(1000, 700, workers);
  expect_gemv_matches_reference(999, 511, 3);
}

TGE_TEST(streaming_dispatch_never_reaches_past_its_features) {
  constexpr int count = 4099;
  std::vector<float> source(count);
  std::vector<float> dispatched(count);
  std::vector<float> scalar(count);
  fill(source.data(), source.size(), 0x1BADB002u);

  // The dispatch path must always be callable: it selects a portable kernel
  // when AVX2 is absent instead of executing an unsupported instruction.
  tge::kernels().copy_stream(dispatched.data(), source.data(), count);
  tge::cpu::copy_stream_scalar(scalar.data(), source.data(), count);
  TGE_CHECK_NEAR(
      tge::ops::max_abs_diff(dispatched.data(), scalar.data(), count), 0.0F,
      1e-9);

  const float via_dispatch = tge::kernels().sum_stream(source.data(), count);
  const float via_scalar = tge::cpu::sum_stream_scalar(source.data(), count);
  TGE_CHECK_NEAR(via_dispatch, via_scalar, 2.0F);
}

TGE_TEST(vector_operations_are_correct) {
  float a[4] = {1.0F, 2.0F, 3.0F, 4.0F};
  const float b[4] = {4.0F, 3.0F, 2.0F, 1.0F};

  TGE_CHECK_NEAR(tge::ops::dot(a, b, 4), 20.0F, 1e-6);

  tge::ops::add_inplace(a, b, 4);
  TGE_CHECK_NEAR(a[0], 5.0F, 1e-6);
  TGE_CHECK_NEAR(a[3], 5.0F, 1e-6);

  tge::ops::scale_inplace(a, 0.5F, 4);
  TGE_CHECK_NEAR(a[0], 2.5F, 1e-6);
}

TGE_TEST(streaming_kernels_agree_with_scalar) {
  constexpr int count = 4099;
  std::vector<float> source(count);
  std::vector<float> target(count, 0.0F);
  fill(source.data(), source.size(), 0xABCDEFu);

  tge::cpu::copy_stream_avx2(target.data(), source.data(), count);
  TGE_CHECK_NEAR(tge::ops::max_abs_diff(source.data(), target.data(), count),
                 0.0F, 1e-9);

  float expected = 0.0F;
  for (int index = 0; index < count; ++index) {
    expected += source[static_cast<std::size_t>(index)];
  }
  TGE_CHECK_NEAR(
      tge::cpu::sum_stream_avx2(source.data(), count), expected, 1e-1);
}
