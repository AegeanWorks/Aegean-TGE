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
