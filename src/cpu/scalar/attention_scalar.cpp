#include "tge/cpu/kernels.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>

namespace tge::cpu {

void attention_head_scalar(const AttentionArgs& args) {
  const int head_dim = args.head_dim;
  const int stride = args.kv_stride;
  const int span = args.span;

  float maximum = -INFINITY;
  for (int token = 0; token < span; ++token) {
    const float* key = args.keys + static_cast<std::size_t>(token) * stride;
    float product = 0.0F;
    for (int index = 0; index < head_dim; ++index) {
      product += args.query[index] * key[index];
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
    const float* value = args.values + static_cast<std::size_t>(token) * stride;
    for (int index = 0; index < head_dim; ++index) {
      args.out[index] += weight * value[index];
    }
  }
}

void attention_group_scalar(const AttentionGroupArgs& args) {
  const int head_dim = args.head_dim;
  const int stride = args.kv_stride;
  const int count = args.count;

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
      const float* key = args.keys + static_cast<std::size_t>(token) * stride;
      float product = 0.0F;
      for (int index = 0; index < head_dim; ++index) {
        product += query[index] * key[index];
      }
      const float score = product * args.scale;
      const float* value =
          args.values + static_cast<std::size_t>(token) * stride;

      if (score > maximum) {
        const float alpha = std::exp(maximum - score);
        sum = sum * alpha + 1.0F;
        for (int index = 0; index < head_dim; ++index) {
          accumulators[index] = accumulators[index] * alpha + value[index];
        }
        maximum = score;
      } else {
        const float beta = std::exp(score - maximum);
        sum += beta;
        for (int index = 0; index < head_dim; ++index) {
          accumulators[index] += beta * value[index];
        }
      }
    }

    args.maxima[head] = maximum;
    args.sums[head] = sum;
  }
}

}  // namespace tge::cpu
