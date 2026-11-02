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

/// Quantizes one 32-value row slice to int8, mirroring the model store path
/// (max / 127, clamped to [-127, 127]).
float quantize_q8_values(const float* source, std::int8_t* destination) {
  float maximum = 0.0F;
  for (int index = 0; index < 32; ++index) {
    const float value = source[index] >= 0.0F ? source[index] : -source[index];
    maximum = maximum > value ? maximum : value;
  }
  const float scale = maximum / 127.0F;
  const float inverse = scale != 0.0F ? 1.0F / scale : 0.0F;
  for (int index = 0; index < 32; ++index) {
    const float scaled = source[index] * inverse;
    const float rounded = scaled >= 0.0F ? scaled + 0.5F : scaled - 0.5F;
    const int value = static_cast<int>(rounded);
    destination[index] = static_cast<std::int8_t>(
        value > 127 ? 127 : (value < -127 ? -127 : value));
  }
  return scale;
}

void quantize_q8_rows(const std::vector<float>& source, int stride,
                      std::vector<std::int8_t>& values,
                      std::vector<float>& scales) {
  const int rows = static_cast<int>(source.size() / stride);
  const int blocks = stride / 32;
  values.resize(source.size());
  scales.resize(static_cast<std::size_t>(rows) * blocks);
  for (int row = 0; row < rows; ++row) {
    for (int block = 0; block < blocks; ++block) {
      scales[static_cast<std::size_t>(row) * blocks + block] =
          quantize_q8_values(
              source.data() + static_cast<std::size_t>(row) * stride +
                  block * 32,
              values.data() + static_cast<std::size_t>(row) * stride +
                  block * 32);
    }
  }
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

TGE_TEST(attention_i8_matches_float) {
  // Int8 KV prices a quantization (like --kv f16), so scalar-int8 agrees
  // with scalar-float loosely while the AVX2 int8 path must track its own
  // scalar oracle tightly.
  for (const int span : {1, 7, 64}) {
    for (const int head_dim : {64, 128}) {
      const int stride = head_dim;
      const std::vector<float> query = make_values(head_dim, 10u);
      const std::vector<float> keys = make_values(span * stride, 11u);
      const std::vector<float> values = make_values(span * stride, 12u);
      std::vector<std::int8_t> q8_keys;
      std::vector<std::int8_t> q8_values;
      std::vector<float> key_scales;
      std::vector<float> value_scales;
      quantize_q8_rows(keys, stride, q8_keys, key_scales);
      quantize_q8_rows(values, stride, q8_values, value_scales);

      std::vector<float> float_scores(static_cast<std::size_t>(span));
      std::vector<float> int_scores(static_cast<std::size_t>(span));
      std::vector<float> avx_scores(static_cast<std::size_t>(span));
      std::vector<float> float_out(static_cast<std::size_t>(head_dim));
      std::vector<float> int_out(static_cast<std::size_t>(head_dim));
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

      tge::cpu::AttentionArgs i8 = base;
      i8.q8 = true;
      i8.keys_q8 = q8_keys.data();
      i8.values_q8 = q8_values.data();
      i8.keys_q8_scales = key_scales.data();
      i8.values_q8_scales = value_scales.data();
      i8.scores = int_scores.data();
      i8.out = int_out.data();

      tge::cpu::AttentionArgs avx = i8;
      avx.scores = avx_scores.data();
      avx.out = avx_out.data();

      tge::cpu::attention_head_scalar(f32);
      tge::cpu::attention_head_scalar(i8);
      tge::cpu::attention_head_avx2(avx);

      for (int index = 0; index < span; ++index) {
        TGE_CHECK_NEAR(int_scores[index], avx_scores[index], 1e-4);
        TGE_CHECK_NEAR(float_scores[index], int_scores[index], 5e-2);
      }
      for (int index = 0; index < head_dim; ++index) {
        TGE_CHECK_NEAR(int_out[index], avx_out[index], 1e-4);
        TGE_CHECK_NEAR(float_out[index], int_out[index], 5e-2);
      }
    }
  }
}

TGE_TEST(attention_group_i8_matches_float) {
  for (const int count : {1, 7, 64}) {
    for (const int group_heads : {1, 3}) {
      for (const int head_dim : {64, 128}) {
        const int stride = head_dim;
        const std::vector<float> queries =
            make_values(group_heads * head_dim, 13u);
        const std::vector<float> keys = make_values(count * stride, 14u);
        const std::vector<float> values = make_values(count * stride, 15u);
        std::vector<std::int8_t> q8_keys;
        std::vector<std::int8_t> q8_values;
        std::vector<float> key_scales;
        std::vector<float> value_scales;
        quantize_q8_rows(keys, stride, q8_keys, key_scales);
        quantize_q8_rows(values, stride, q8_values, value_scales);

        std::vector<float> scalar_maxima(static_cast<std::size_t>(group_heads));
        std::vector<float> vector_maxima(static_cast<std::size_t>(group_heads));
        std::vector<float> scalar_sums(static_cast<std::size_t>(group_heads));
        std::vector<float> vector_sums(static_cast<std::size_t>(group_heads));
        std::vector<float> ref_sums(static_cast<std::size_t>(group_heads));
        const std::size_t accumulators =
            static_cast<std::size_t>(group_heads) * head_dim;
        std::vector<float> scalar_acc(accumulators);
        std::vector<float> vector_acc(accumulators);
        std::vector<float> ref_acc(accumulators);

        tge::cpu::AttentionGroupArgs base;
        base.queries = queries.data();
        base.kv_stride = stride;
        base.count = count;
        base.head_dim = head_dim;
        base.group_heads = group_heads;
        base.scale = 1.0F / std::sqrt(static_cast<float>(head_dim));

        tge::cpu::AttentionGroupArgs f32 = base;
        f32.keys = keys.data();
        f32.values = values.data();
        f32.maxima = scalar_maxima.data();
        f32.sums = ref_sums.data();
        f32.accumulators = ref_acc.data();

        tge::cpu::AttentionGroupArgs i8 = base;
        i8.q8 = true;
        i8.keys_q8 = q8_keys.data();
        i8.values_q8 = q8_values.data();
        i8.keys_q8_scales = key_scales.data();
        i8.values_q8_scales = value_scales.data();
        i8.maxima = vector_maxima.data();
        i8.sums = vector_sums.data();
        i8.accumulators = vector_acc.data();

        tge::cpu::AttentionGroupArgs avx = i8;
        avx.maxima = scalar_maxima.data();
        avx.sums = scalar_sums.data();
        avx.accumulators = scalar_acc.data();

        tge::cpu::attention_group_scalar(f32);
        tge::cpu::attention_group_scalar(i8);
        tge::cpu::attention_group_avx2(avx);

        for (int head = 0; head < group_heads; ++head) {
          TGE_CHECK_NEAR(scalar_maxima[head], vector_maxima[head], 1e-4);
          TGE_CHECK_NEAR(ref_sums[head], vector_sums[head], 5e-2);
        }
        for (std::size_t index = 0; index < accumulators; ++index) {
          TGE_CHECK_NEAR(scalar_acc[index], vector_acc[index], 1e-4);
          TGE_CHECK_NEAR(ref_acc[index], vector_acc[index], 5e-2);
        }
      }
    }
  }
}
