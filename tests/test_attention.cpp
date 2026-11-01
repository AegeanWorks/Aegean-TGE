#include <algorithm>
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
        (static_cast<float>((state >> 8) & 0xFFFFu) / 32768.0F - 1.0F) * 0.25F;
  }
  return values;
}

}  // namespace

TGE_TEST(attention_avx2_matches_scalar) {
  // The vectorised path changes summation order and fuses multiply-add, so the
  // test is a differential against the scalar oracle, not an equality check.
  for (const int span : {1, 7, 64, 257}) {
    for (const int head_dim : {64, 128}) {
      const int stride = head_dim;
      const std::vector<float> query = make_values(head_dim, 1u);
      const std::vector<float> keys = make_values(span * stride, 2u);
      const std::vector<float> values = make_values(span * stride, 3u);

      std::vector<float> scalar_scores(static_cast<std::size_t>(span));
      std::vector<float> vector_scores(static_cast<std::size_t>(span));
      std::vector<float> scalar_out(static_cast<std::size_t>(head_dim));
      std::vector<float> vector_out(static_cast<std::size_t>(head_dim));

      tge::cpu::AttentionArgs scalar_args;
      scalar_args.query = query.data();
      scalar_args.keys = keys.data();
      scalar_args.values = values.data();
      scalar_args.kv_stride = stride;
      scalar_args.span = span;
      scalar_args.head_dim = head_dim;
      scalar_args.scale = 1.0F / std::sqrt(static_cast<float>(head_dim));
      scalar_args.scores = scalar_scores.data();
      scalar_args.out = scalar_out.data();

      tge::cpu::AttentionArgs vector_args = scalar_args;
      vector_args.scores = vector_scores.data();
      vector_args.out = vector_out.data();

      tge::cpu::attention_head_scalar(scalar_args);
      tge::cpu::attention_head_avx2(vector_args);

      for (int index = 0; index < span; ++index) {
        TGE_CHECK_NEAR(scalar_scores[index], vector_scores[index], 1e-4);
      }
      for (int index = 0; index < head_dim; ++index) {
        TGE_CHECK_NEAR(scalar_out[index], vector_out[index], 1e-4);
      }
    }
  }
}

TGE_TEST(half_round_trip) {
  const float values[] = {0.0F, 1.0F, -1.0F, 0.5F, 2.0F, 1024.0F, 65504.0F};
  for (const float value : values) {
    const std::uint16_t half = tge::cpu::float_to_half(value);
    const float back = tge::cpu::half_to_float(half);
    TGE_CHECK_NEAR(value, back, std::max(std::fabs(value) * 0.001F, 1e-6F));
  }
  TGE_CHECK(tge::cpu::float_to_half(INFINITY) == 0x7C00u);
}

TGE_TEST(attention_half_matches_float) {
  for (const int span : {1, 7, 64, 257}) {
    for (const int head_dim : {64, 128}) {
      const int stride = head_dim;
      const std::vector<float> query = make_values(head_dim, 7u);
      const std::vector<float> keys = make_values(span * stride, 8u);
      const std::vector<float> values = make_values(span * stride, 9u);
      std::vector<std::uint16_t> half_keys(keys.size());
      std::vector<std::uint16_t> half_values(values.size());
      for (std::size_t index = 0; index < keys.size(); ++index) {
        half_keys[index] = tge::cpu::float_to_half(keys[index]);
        half_values[index] = tge::cpu::float_to_half(values[index]);
      }

      std::vector<float> float_scores(static_cast<std::size_t>(span));
      std::vector<float> half_scores(static_cast<std::size_t>(span));
      std::vector<float> avx_scores(static_cast<std::size_t>(span));
      std::vector<float> float_out(static_cast<std::size_t>(head_dim));
      std::vector<float> half_out(static_cast<std::size_t>(head_dim));
      std::vector<float> avx_out(static_cast<std::size_t>(head_dim));

      tge::cpu::AttentionArgs base;
      base.query = query.data();
      base.kv_stride = stride;
      base.span = span;
      base.head_dim = head_dim;
      base.scale = 1.0F / std::sqrt(static_cast<float>(head_dim));

      tge::cpu::AttentionArgs f32 = base;
      f32.keys = keys.data();
      f32.values = values.data();
      f32.scores = float_scores.data();
      f32.out = float_out.data();

      tge::cpu::AttentionArgs f16 = base;
      f16.half = true;
      f16.keys_half = half_keys.data();
      f16.values_half = half_values.data();
      f16.scores = half_scores.data();
      f16.out = half_out.data();

      tge::cpu::AttentionArgs avx = f16;
      avx.scores = avx_scores.data();
      avx.out = avx_out.data();

      tge::cpu::attention_head_scalar(f32);
      tge::cpu::attention_head_scalar(f16);
      tge::cpu::attention_head_avx2(avx);

      for (int index = 0; index < span; ++index) {
        TGE_CHECK_NEAR(half_scores[index], avx_scores[index], 1e-4);
        TGE_CHECK_NEAR(float_scores[index], half_scores[index], 5e-2);
      }
      for (int index = 0; index < head_dim; ++index) {
        TGE_CHECK_NEAR(half_out[index], avx_out[index], 1e-4);
        TGE_CHECK_NEAR(float_out[index], half_out[index], 5e-2);
      }
    }
  }
}
TGE_TEST(attention_group_avx2_matches_scalar) {
  for (const int count : {1, 7, 64, 257}) {
    for (const int group_heads : {1, 7}) {
      for (const int head_dim : {64, 128}) {
        const int stride = head_dim;
        const std::vector<float> queries =
            make_values(group_heads * head_dim, 4u);
        const std::vector<float> keys = make_values(count * stride, 5u);
        const std::vector<float> values = make_values(count * stride, 6u);

        std::vector<float> scalar_maxima(static_cast<std::size_t>(group_heads));
        std::vector<float> vector_maxima(static_cast<std::size_t>(group_heads));
        std::vector<float> scalar_sums(static_cast<std::size_t>(group_heads));
        std::vector<float> vector_sums(static_cast<std::size_t>(group_heads));
        const std::size_t accumulators =
            static_cast<std::size_t>(group_heads) * head_dim;
        std::vector<float> scalar_acc(accumulators);
        std::vector<float> vector_acc(accumulators);

        tge::cpu::AttentionGroupArgs scalar_args;
        scalar_args.queries = queries.data();
        scalar_args.keys = keys.data();
        scalar_args.values = values.data();
        scalar_args.kv_stride = stride;
        scalar_args.count = count;
        scalar_args.head_dim = head_dim;
        scalar_args.group_heads = group_heads;
        scalar_args.scale = 1.0F / std::sqrt(static_cast<float>(head_dim));
        scalar_args.maxima = scalar_maxima.data();
        scalar_args.sums = scalar_sums.data();
        scalar_args.accumulators = scalar_acc.data();

        tge::cpu::AttentionGroupArgs vector_args = scalar_args;
        vector_args.maxima = vector_maxima.data();
        vector_args.sums = vector_sums.data();
        vector_args.accumulators = vector_acc.data();

        tge::cpu::attention_group_scalar(scalar_args);
        tge::cpu::attention_group_avx2(vector_args);

        for (int head = 0; head < group_heads; ++head) {
          TGE_CHECK_NEAR(scalar_maxima[head], vector_maxima[head], 1e-4);
          TGE_CHECK_NEAR(scalar_sums[head], vector_sums[head], 1e-4);
        }
        for (std::size_t index = 0; index < accumulators; ++index) {
          TGE_CHECK_NEAR(scalar_acc[index], vector_acc[index], 1e-4);
        }
      }
    }
  }
}
