#include "tge/cpu/kernels.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>

namespace tge::cpu {
namespace {

void validate_attention_args(const AttentionArgs& args) {
  if (args.head_dim <= 0 || args.span <= 0) {
    throw std::invalid_argument("attention: non-positive head_dim/span");
  }
  if (args.kv_stride < args.head_dim) {
    // Rows would overlap: token t's row is read from [t*stride, t*stride + D).
    throw std::invalid_argument("attention: kv_stride < head_dim");
  }
  if (args.query == nullptr || args.scores == nullptr || args.out == nullptr) {
    throw std::invalid_argument("attention: null query/scores/out");
  }
  if (args.half && args.q8) {
    throw std::invalid_argument("attention: half and q8 both set");
  }
  if (args.q8) {
    // Scale blocks parallel the values one per 32; a ragged width silently
    // reads the wrong scale row.
    if (args.head_dim % 32 != 0 || args.kv_stride % 32 != 0) {
      throw std::invalid_argument("attention: q8 needs 32-element blocks");
    }
    if (args.keys_q8 == nullptr || args.values_q8 == nullptr ||
        args.keys_q8_scales == nullptr || args.values_q8_scales == nullptr) {
      throw std::invalid_argument("attention: null q8 buffers");
    }
  } else if (args.half) {
    if (args.keys_half == nullptr || args.values_half == nullptr) {
      throw std::invalid_argument("attention: null half buffers");
    }
  } else {
    if (args.keys == nullptr || args.values == nullptr) {
      throw std::invalid_argument("attention: null float buffers");
    }
  }
}

}  // namespace

void attention_head_scalar(const AttentionArgs& args) {
  validate_attention_args(args);
  const int head_dim = args.head_dim;
  const int stride = args.kv_stride;
  const int span = args.span;
  const bool half = args.half;
  const std::uint16_t* half_keys = args.keys_half;
  const std::uint16_t* half_values = args.values_half;
  // Int8 KV form: one float scale per 32 values, strided with the rows.
  const bool q8 = args.q8;
  const std::int8_t* q8_keys = args.keys_q8;
  const std::int8_t* q8_values = args.values_q8;
  const int scale_stride = stride / 32;

  float maximum = -INFINITY;
  for (int token = 0; token < span; ++token) {
    const std::size_t row = static_cast<std::size_t>(token) * stride;
    float product = 0.0F;
    for (int index = 0; index < head_dim; ++index) {
      float key;
      if (q8) {
        const std::size_t block =
            static_cast<std::size_t>(token) * scale_stride + index / 32;
        key = static_cast<float>(q8_keys[row + index]) *
              args.keys_q8_scales[block];
      } else {
        key =
            half ? half_to_float(half_keys[row + index]) : args.keys[row + index];
      }
      product += args.query[index] * key;
    }
    const float scaled = product * args.scale;
    args.scores[static_cast<std::size_t>(token)] = scaled;
    maximum = std::max(maximum, scaled);
  }

  float total = 0.0F;
  for (int token = 0; token < span; ++token) {
    const float value =
        std::exp(args.scores[static_cast<std::size_t>(token)] - maximum);
    args.scores[static_cast<std::size_t>(token)] = value;
    total += value;
  }
  const float inverse = 1.0F / total;

  for (int index = 0; index < head_dim; ++index) {
    args.out[index] = 0.0F;
  }
  for (int token = 0; token < span; ++token) {
    const float weight =
        args.scores[static_cast<std::size_t>(token)] * inverse;
    const std::size_t row = static_cast<std::size_t>(token) * stride;
    const float* value = (half || q8) ? nullptr : args.values + row;
    for (int index = 0; index < head_dim; ++index) {
      float element;
      if (q8) {
        const std::size_t block =
            static_cast<std::size_t>(token) * scale_stride + index / 32;
        element = static_cast<float>(q8_values[row + index]) *
                  args.values_q8_scales[block];
      } else {
        element =
            half ? half_to_float(half_values[row + index]) : value[index];
      }
      args.out[index] += weight * element;
    }
  }
}

void attention_group_scalar(const AttentionGroupArgs& args) {
  const int head_dim = args.head_dim;
  const int stride = args.kv_stride;
  const int count = args.count;
  const bool half = args.half;
  const std::uint16_t* half_keys = args.keys_half;
  const std::uint16_t* half_values = args.values_half;
  const bool q8 = args.q8;
  const std::int8_t* q8_keys = args.keys_q8;
  const std::int8_t* q8_values = args.values_q8;
  const int scale_stride = stride / 32;

  for (int head = 0; head < args.group_heads; ++head) {
    const float* query =
        args.queries + static_cast<std::size_t>(head) * head_dim;
    float* accumulators =
        args.accumulators + static_cast<std::size_t>(head) * head_dim;

    float maximum = -INFINITY;
    float sum = 0.0F;
    for (int index = 0; index < head_dim; ++index) {
      accumulators[index] = 0.0F;
    }

    for (int token = 0; token < count; ++token) {
      const std::size_t row = static_cast<std::size_t>(token) * stride;
      float product = 0.0F;
      for (int index = 0; index < head_dim; ++index) {
        float key;
        if (q8) {
          const std::size_t block =
              static_cast<std::size_t>(token) * scale_stride + index / 32;
          key = static_cast<float>(q8_keys[row + index]) *
                args.keys_q8_scales[block];
        } else {
          key = half ? half_to_float(half_keys[row + index])
                     : args.keys[row + index];
        }
        product += query[index] * key;
      }
      const float score = product * args.scale;
      const float* value = (half || q8) ? nullptr : args.values + row;

      if (score > maximum) {
        const float alpha = std::exp(maximum - score);
        sum = sum * alpha + 1.0F;
        for (int index = 0; index < head_dim; ++index) {
          float element;
          if (q8) {
            const std::size_t block =
                static_cast<std::size_t>(token) * scale_stride + index / 32;
            element = static_cast<float>(q8_values[row + index]) *
                      args.values_q8_scales[block];
          } else {
            element = half ? half_to_float(half_values[row + index])
                           : value[index];
          }
          accumulators[index] = accumulators[index] * alpha + element;
        }
        maximum = score;
      } else {
        const float beta = std::exp(score - maximum);
        sum += beta;
        for (int index = 0; index < head_dim; ++index) {
          float element;
          if (q8) {
            const std::size_t block =
                static_cast<std::size_t>(token) * scale_stride + index / 32;
            element = static_cast<float>(q8_values[row + index]) *
                      args.values_q8_scales[block];
          } else {
            element = half ? half_to_float(half_values[row + index])
                           : value[index];
          }
          accumulators[index] += beta * element;
        }
      }
    }

    args.maxima[head] = maximum;
    args.sums[head] = sum;
  }
}

}  // namespace tge::cpu
