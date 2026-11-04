#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

#include "harness.hpp"
#include "tge/cpu/kernels.hpp"
#include "tge/models/qwen2.hpp"

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

TGE_TEST(sparse_index_dense_when_short) {
  // At or below sink + window the builder returns the full range, so the
  // model keeps the exact dense path instead of a gathered copy.
  const tge::models::SparseConfig config{true, 4, 512, 4};
  for (const int span : {1, 7, 64, 516}) {
    const std::vector<int> kept = tge::models::build_sparse_index(span, config);
    TGE_CHECK(static_cast<int>(kept.size()) == span);
    for (int position = 0; position < span; ++position) {
      TGE_CHECK(kept[static_cast<std::size_t>(position)] == position);
    }
  }
  // Disabled config is always dense.
  const tge::models::SparseConfig off{false, 4, 512, 4};
  const std::vector<int> kept = tge::models::build_sparse_index(2048, off);
  TGE_CHECK(static_cast<int>(kept.size()) == 2048);
}

TGE_TEST(sparse_index_selects_sink_stride_window) {
  const tge::models::SparseConfig config{true, 4, 512, 4};
  for (const int span : {1024, 2048, 8192}) {
    const std::vector<int> kept = tge::models::build_sparse_index(span, config);
    TGE_CHECK(!kept.empty());
    // Sorted and strictly increasing (unique).
    for (std::size_t index = 1; index < kept.size(); ++index) {
      TGE_CHECK(kept[index] > kept[index - 1]);
    }
    TGE_CHECK(kept.front() == 0);
    TGE_CHECK(kept.back() == span - 1);
    // Sink head is identity.
    for (int position = 0; position < 4; ++position) {
      TGE_CHECK(kept[static_cast<std::size_t>(position)] == position);
    }
    // Window tail is identity.
    for (int position = 0; position < 512; ++position) {
      TGE_CHECK(kept[kept.size() - 512 + position] == span - 512 + position);
    }
    // Middle is every 4th position starting at the sink end.
    const int middle = span - 4 - 512;
    const int middle_kept = (middle + 4 - 1) / 4;
    TGE_CHECK(static_cast<int>(kept.size()) == 4 + middle_kept + 512);
    for (int index = 0; index < middle_kept; ++index) {
      TGE_CHECK(kept[static_cast<std::size_t>(4 + index)] == 4 + index * 4);
    }
  }
}

TGE_TEST(sparse_gather_matches_subset_reference) {
  // The model's sparse path gathers kept rows into compact buffers and runs
  // the unchanged scalar kernel over them. The reference below computes the
  // same subset attention directly, so the test catches gather offset bugs.
  const tge::models::SparseConfig config{true, 4, 64, 4};
  const int span = 600;
  const int head_dim = 64;
  const std::vector<float> query = make_values(head_dim, 21u);
  const std::vector<float> keys = make_values(span * head_dim, 22u);
  const std::vector<float> values = make_values(span * head_dim, 23u);
  const std::vector<int> kept = tge::models::build_sparse_index(span, config);
  TGE_CHECK(static_cast<int>(kept.size()) < span);

  std::vector<float> compact_keys(kept.size() * head_dim);
  std::vector<float> compact_values(kept.size() * head_dim);
  for (std::size_t index = 0; index < kept.size(); ++index) {
    const std::size_t row = static_cast<std::size_t>(kept[index]) * head_dim;
    std::copy(keys.begin() + row, keys.begin() + row + head_dim,
              compact_keys.begin() + index * head_dim);
    std::copy(values.begin() + row, values.begin() + row + head_dim,
              compact_values.begin() + index * head_dim);
  }

  std::vector<float> scores(kept.size());
  std::vector<float> out(head_dim);
  tge::cpu::AttentionArgs args;
  args.query = query.data();
  args.keys = compact_keys.data();
  args.values = compact_values.data();
  args.kv_stride = head_dim;
  args.span = static_cast<int>(kept.size());
  args.head_dim = head_dim;
  args.scale = 1.0F / std::sqrt(static_cast<float>(head_dim));
  args.scores = scores.data();
  args.out = out.data();
  tge::cpu::attention_head_scalar(args);

  // Direct reference over the same kept positions.
  float maximum = -INFINITY;
  std::vector<float> raw(kept.size());
  for (std::size_t index = 0; index < kept.size(); ++index) {
    const std::size_t row = static_cast<std::size_t>(kept[index]) * head_dim;
    float product = 0.0F;
    for (int axis = 0; axis < head_dim; ++axis) {
      product += query[axis] * keys[row + axis];
    }
    raw[index] = product * args.scale;
    maximum = std::max(maximum, raw[index]);
  }
  float total = 0.0F;
  for (float value : raw) {
    total += std::exp(value - maximum);
  }
  for (int axis = 0; axis < head_dim; ++axis) {
    float expected = 0.0F;
    for (std::size_t index = 0; index < kept.size(); ++index) {
      const std::size_t row = static_cast<std::size_t>(kept[index]) * head_dim;
      expected += std::exp(raw[index] - maximum) / total *
                  values[row + axis];
    }
    TGE_CHECK_NEAR(out[axis], expected, 1e-4);
  }
}

TGE_TEST(attention_rejects_bad_shapes) {
  // Regression: stride<head_dim read overlapping rows, half+q8 was
  // ambiguous, and q8 with a ragged width read the wrong scale row —
  // all silently. Both ISAs must fail loudly instead.
  const int head_dim = 64;
  const int stride = 64;
  const int span = 4;
  const std::vector<float> query = make_values(head_dim, 1u);
  const std::vector<float> keys = make_values(span * stride, 2u);
  const std::vector<float> values = make_values(span * stride, 3u);
  std::vector<float> scores(static_cast<std::size_t>(span));
  std::vector<float> out(static_cast<std::size_t>(head_dim));
  auto valid = [&]() {
    tge::cpu::AttentionArgs args;
    args.query = query.data();
    args.keys = keys.data();
    args.values = values.data();
    args.kv_stride = stride;
    args.span = span;
    args.head_dim = head_dim;
    args.scores = scores.data();
    args.out = out.data();
    return args;
  };
  auto throws_both = [&](tge::cpu::AttentionArgs args) {
    bool threw_scalar = false;
    try {
      tge::cpu::attention_head_scalar(args);
    } catch (...) {
      threw_scalar = true;
    }
    bool threw_avx2 = false;
    try {
      tge::cpu::attention_head_avx2(args);
    } catch (...) {
      threw_avx2 = true;
    }
    return threw_scalar && threw_avx2;
  };
  {  // span 0: softmax over nothing (1/0) is not a defined result.
    auto args = valid();
    args.span = 0;
    TGE_CHECK(throws_both(args));
  }
  {  // stride below head_dim: rows overlap.
    auto args = valid();
    args.kv_stride = head_dim - 1;
    TGE_CHECK(throws_both(args));
  }
  {  // null query.
    auto args = valid();
    args.query = nullptr;
    TGE_CHECK(throws_both(args));
  }
  {  // half and q8 together: which buffer wins is undefined.
    auto args = valid();
    args.half = true;
    args.q8 = true;
    TGE_CHECK(throws_both(args));
  }
  {  // q8 with a ragged width: scale blocks misalign.
    auto args = valid();
    args.q8 = true;
    args.head_dim = 48;
    TGE_CHECK(throws_both(args));
  }
}
