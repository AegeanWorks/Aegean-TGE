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
