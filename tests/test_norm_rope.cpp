#include <cmath>
#include <cstdint>
#include <vector>

#include "harness.hpp"
#include "tge/cpu/kernels.hpp"

namespace {

std::vector<float> make_values(int count, std::uint32_t seed) {
  std::vector<float> values(static_cast<std::size_t>(count));
  std::uint32_t state = seed | 1u;
  for (int index = 0; index < count; ++index) {
    state = state * 1664525u + 1013904223u;
    values[static_cast<std::size_t>(index)] =
        (static_cast<float>((state >> 8) & 0xFFFFu) / 32768.0F - 1.0F) * 0.5F;
  }
  return values;
}

}  // namespace

TGE_TEST(norm_rope_avx2_matches_scalar) {
  for (const int dim : {32, 64, 896}) {
    const std::vector<float> input = make_values(dim, 11u);
    const std::vector<float> weight = make_values(dim, 12u);
    std::vector<float> scalar_out(static_cast<std::size_t>(dim));
    std::vector<float> vector_out(static_cast<std::size_t>(dim));
    tge::cpu::rms_norm_scalar(scalar_out.data(), input.data(), weight.data(),
                              dim, 1e-6F);
    tge::cpu::rms_norm_avx2(vector_out.data(), input.data(), weight.data(),
                            dim, 1e-6F);
    for (int index = 0; index < dim; ++index) {
      TGE_CHECK_NEAR(vector_out[index], scalar_out[index], 1e-5);
    }
  }

  for (const int heads : {2, 14}) {
    for (const int head_dim : {64, 128}) {
      const int half = head_dim / 2;
      const std::vector<float> cosines = make_values(half, 21u);
      const std::vector<float> sines = make_values(half, 22u);
      std::vector<float> scalar_values =
          make_values(heads * head_dim, 23u);
      std::vector<float> vector_values = scalar_values;
      tge::cpu::apply_rope_scalar(scalar_values.data(), heads, head_dim, 7,
                                  cosines.data(), sines.data());
      tge::cpu::apply_rope_avx2(vector_values.data(), heads, head_dim, 7,
                                cosines.data(), sines.data());
      for (std::size_t index = 0;
           index < static_cast<std::size_t>(heads * head_dim); ++index) {
        TGE_CHECK_NEAR(vector_values[index], scalar_values[index], 1e-5);
      }
    }
  }
}

TGE_TEST(rope_interleaved_avx2_matches_scalar) {
  for (const int heads : {2, 8}) {
    for (const int head_dim : {64, 128}) {
      const int half = head_dim / 2;
      const std::vector<float> cosines = make_values(half, 41u);
      const std::vector<float> sines = make_values(half, 42u);
      // Duplicated tables as apply_rope_interleaved_* expects them.
      std::vector<float> cos_dup(static_cast<std::size_t>(half) * 2);
      std::vector<float> sin_dup(static_cast<std::size_t>(half) * 2);
      for (int index = 0; index < half; ++index) {
        cos_dup[static_cast<std::size_t>(index) * 2] = cosines[index];
        cos_dup[static_cast<std::size_t>(index) * 2 + 1] = cosines[index];
        sin_dup[static_cast<std::size_t>(index) * 2] = sines[index];
        sin_dup[static_cast<std::size_t>(index) * 2 + 1] = sines[index];
      }
      std::vector<float> scalar_values = make_values(heads * head_dim, 43u);
      std::vector<float> vector_values = scalar_values;
      tge::cpu::apply_rope_interleaved_scalar(
          scalar_values.data(), heads, head_dim, 7, cos_dup.data(),
          sin_dup.data());
      tge::cpu::apply_rope_interleaved_avx2(
          vector_values.data(), heads, head_dim, 7, cos_dup.data(),
          sin_dup.data());
      for (std::size_t index = 0;
           index < static_cast<std::size_t>(heads * head_dim); ++index) {
        TGE_CHECK_NEAR(vector_values[index], scalar_values[index], 1e-5);
      }
    }
  }

  // Hand-computed: pair (2, 3) with cos = 0, sin = 1 is a +90-degree turn:
  // (0, 1) -> (-1, 0), while pair (0, 1) is the identity.
  {
    float values[4] = {1.0F, 0.0F, 0.0F, 1.0F};
    const float cos_dup[4] = {1.0F, 1.0F, 0.0F, 0.0F};
    const float sin_dup[4] = {0.0F, 0.0F, 1.0F, 1.0F};
    tge::cpu::apply_rope_interleaved_scalar(values, 1, 4, 0, cos_dup,
                                            sin_dup);
    TGE_CHECK_NEAR(values[0], 1.0F, 1e-6);
    TGE_CHECK_NEAR(values[1], 0.0F, 1e-6);
    TGE_CHECK_NEAR(values[2], -1.0F, 1e-6);
    TGE_CHECK_NEAR(values[3], 0.0F, 1e-6);
  }
}

TGE_TEST(silu_gated_avx2_matches_scalar) {
  for (const int count : {1, 7, 32, 4864}) {
    std::vector<float> gate = make_values(count, 31u);
    const std::vector<float> up = make_values(count, 32u);
    std::vector<float> scalar_gate = gate;
    std::vector<float> vector_gate = gate;
    tge::cpu::silu_gated_scalar(scalar_gate.data(), up.data(), count);
    tge::cpu::silu_gated_avx2(vector_gate.data(), up.data(), count);
    for (int index = 0; index < count; ++index) {
      TGE_CHECK_NEAR(vector_gate[index], scalar_gate[index], 1e-4);
    }
  }
}
