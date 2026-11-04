#include <vector>

#include "harness.hpp"
#include "tge/cpu/gemm_epilogue.hpp"
#include "tge/cpu/kernels.hpp"

TGE_TEST(gemm_args_clamp_defaults_off) {
  tge::cpu::GemmArgs args;
  TGE_CHECK(!args.do_clamp);
}

TGE_TEST(gemm_clamp_scalar_bounds) {
  std::vector<float> data = {-3.0F, -0.5F, 0.0F, 0.7F, 2.0F};
  tge::cpu::clamp_scalar(data.data(), data.size(), -1.0F, 1.0F);
  TGE_CHECK(data[0] == -1.0F);
  TGE_CHECK(data[1] == -0.5F);
  TGE_CHECK(data[2] == 0.0F);
  TGE_CHECK(data[3] == 0.7F);
  TGE_CHECK(data[4] == 1.0F);
}

TGE_TEST(gemm_clamp_avx2_matches_scalar) {
  std::vector<float> base(67);
  for (std::size_t i = 0; i < base.size(); ++i) {
    base[i] = static_cast<float>(i) * 0.13F - 4.0F;
  }
  std::vector<float> a = base, b = base;
  tge::cpu::clamp_scalar(a.data(), a.size(), -2.0F, 3.0F);
  tge::cpu::clamp_avx2(b.data(), b.size(), -2.0F, 3.0F);
  TGE_REQUIRE(a.size() == b.size());
  for (std::size_t i = 0; i < a.size(); ++i) TGE_CHECK(a[i] == b[i]);
}

TGE_TEST(gemm_clamp_wide_range_identity) {
  std::vector<float> data = {1.0F, -1.0F, 0.25F};
  std::vector<float> copy = data;
  tge::cpu::clamp_avx2(data.data(), data.size(), -1e30F, 1e30F);
  for (std::size_t i = 0; i < data.size(); ++i) TGE_CHECK(data[i] == copy[i]);
}
