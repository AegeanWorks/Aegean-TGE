#include "tge/cpu/kernels.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>

#include <immintrin.h>

namespace tge::cpu {
namespace {

inline float horizontal_sum(__m256 values) {
  __m128 total = _mm_add_ps(_mm256_castps256_ps128(values),
                            _mm256_extractf128_ps(values, 1));
  total = _mm_hadd_ps(total, total);
  total = _mm_hadd_ps(total, total);
  return _mm_cvtss_f32(total);
}

inline float dot_avx2(const float* left, const float* right, int count) {
  __m256 acc0 = _mm256_setzero_ps();
  __m256 acc1 = _mm256_setzero_ps();
  int index = 0;
  for (; index + 16 <= count; index += 16) {
    acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(left + index),
                           _mm256_loadu_ps(right + index), acc0);
    acc1 = _mm256_fmadd_ps(_mm256_loadu_ps(left + index + 8),
                           _mm256_loadu_ps(right + index + 8), acc1);
  }
  for (; index + 8 <= count; index += 8) {
    acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(left + index),
                           _mm256_loadu_ps(right + index), acc0);
  }
  float result = horizontal_sum(_mm256_add_ps(acc0, acc1));
  for (; index < count; ++index) {
    result += left[index] * right[index];
  }
  return result;
}

}  // namespace

void attention_head_avx2(const AttentionArgs& args) {
  const int head_dim = args.head_dim;
  const int stride = args.kv_stride;
  const int span = args.span;
  const bool half = args.half;
  const std::uint16_t* half_keys = args.keys_half;
  const std::uint16_t* half_values = args.values_half;

  float maximum = -INFINITY;
  for (int token = 0; token < span; ++token) {
    const std::size_t row = static_cast<std::size_t>(token) * stride;
    float dot = 0.0F;
    if (half) {
      const std::uint16_t* key = half_keys + row;
      __m256 acc0 = _mm256_setzero_ps();
      __m256 acc1 = _mm256_setzero_ps();
      int index = 0;
      for (; index + 16 <= head_dim; index += 16) {
        const __m256 k0 = _mm256_cvtph_ps(
            _mm_loadu_si128(reinterpret_cast<const __m128i*>(key + index)));
        const __m256 k1 = _mm256_cvtph_ps(
            _mm_loadu_si128(reinterpret_cast<const __m128i*>(key + index + 8)));
        acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(args.query + index), k0, acc0);
        acc1 =
            _mm256_fmadd_ps(_mm256_loadu_ps(args.query + index + 8), k1, acc1);
      }
      for (; index + 8 <= head_dim; index += 8) {
        const __m256 k = _mm256_cvtph_ps(
            _mm_loadu_si128(reinterpret_cast<const __m128i*>(key + index)));
        acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(args.query + index), k, acc0);
      }
      dot = horizontal_sum(_mm256_add_ps(acc0, acc1));
      for (; index < head_dim; ++index) {
        dot += args.query[index] * half_to_float(key[index]);
      }
    } else {
      dot = dot_avx2(args.query, args.keys + row, head_dim);
    }
    const float scaled = dot * args.scale;
    args.scores[static_cast<std::size_t>(token)] = scaled;
    maximum = std::max(maximum, scaled);
  }

  // The exponential stays scalar: a vectorised approximation would change the
  // distribution, so it needs its own KL price first.
  float total = 0.0F;
  for (int token = 0; token < span; ++token) {
    const float value =
        std::exp(args.scores[static_cast<std::size_t>(token)] - maximum);
    args.scores[static_cast<std::size_t>(token)] = value;
    total += value;
  }
  const float inverse = 1.0F / total;

  int index = 0;
  for (; index + 8 <= head_dim; index += 8) {
    _mm256_storeu_ps(args.out + index, _mm256_setzero_ps());
  }
  for (; index < head_dim; ++index) {
    args.out[index] = 0.0F;
  }

  for (int token = 0; token < span; ++token) {
    const float scalar_weight =
        args.scores[static_cast<std::size_t>(token)] * inverse;
    const std::size_t row = static_cast<std::size_t>(token) * stride;
    const __m256 weight = _mm256_set1_ps(scalar_weight);
    if (half) {
      const std::uint16_t* value = half_values + row;
      int lane = 0;
      for (; lane + 8 <= head_dim; lane += 8) {
        const __m256 v = _mm256_cvtph_ps(
            _mm_loadu_si128(reinterpret_cast<const __m128i*>(value + lane)));
        _mm256_storeu_ps(
            args.out + lane,
            _mm256_fmadd_ps(weight, v, _mm256_loadu_ps(args.out + lane)));
      }
      for (; lane < head_dim; ++lane) {
        args.out[lane] += scalar_weight * half_to_float(value[lane]);
      }
    } else {
      const float* value = args.values + row;
      int lane = 0;
      for (; lane + 8 <= head_dim; lane += 8) {
        _mm256_storeu_ps(
            args.out + lane,
            _mm256_fmadd_ps(weight, _mm256_loadu_ps(value + lane),
                            _mm256_loadu_ps(args.out + lane)));
      }
      for (; lane < head_dim; ++lane) {
        args.out[lane] += scalar_weight * value[lane];
      }
    }
  }
}

void attention_group_avx2(const AttentionGroupArgs& args) {
  const int head_dim = args.head_dim;
  const int stride = args.kv_stride;
  const int count = args.count;
  const bool half = args.half;
  const std::uint16_t* half_keys = args.keys_half;
  const std::uint16_t* half_values = args.values_half;

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
      float score;
      if (half) {
        const std::uint16_t* key = half_keys + row;
        __m256 acc0 = _mm256_setzero_ps();
        __m256 acc1 = _mm256_setzero_ps();
        int index = 0;
        for (; index + 16 <= head_dim; index += 16) {
          const __m256 k0 = _mm256_cvtph_ps(
              _mm_loadu_si128(reinterpret_cast<const __m128i*>(key + index)));
          const __m256 k1 = _mm256_cvtph_ps(
              _mm_loadu_si128(reinterpret_cast<const __m128i*>(key + index + 8)));
          acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(query + index), k0, acc0);
          acc1 = _mm256_fmadd_ps(_mm256_loadu_ps(query + index + 8), k1, acc1);
        }
        for (; index + 8 <= head_dim; index += 8) {
          const __m256 k = _mm256_cvtph_ps(
              _mm_loadu_si128(reinterpret_cast<const __m128i*>(key + index)));
          acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(query + index), k, acc0);
        }
        float dot = horizontal_sum(_mm256_add_ps(acc0, acc1));
        for (; index < head_dim; ++index) {
          dot += query[index] * half_to_float(key[index]);
        }
        score = dot * args.scale;
      } else {
        score = dot_avx2(query, args.keys + row, head_dim) * args.scale;
      }

      if (score > maximum) {
        const float alpha = std::exp(maximum - score);
        const __m256 scale = _mm256_set1_ps(alpha);
        sum = sum * alpha + 1.0F;
        if (half) {
          const std::uint16_t* value = half_values + row;
          int index = 0;
          for (; index + 8 <= head_dim; index += 8) {
            const __m256 v = _mm256_cvtph_ps(
                _mm_loadu_si128(reinterpret_cast<const __m128i*>(value + index)));
            _mm256_storeu_ps(
                accumulators + index,
                _mm256_fmadd_ps(scale, _mm256_loadu_ps(accumulators + index), v));
          }
          for (; index < head_dim; ++index) {
            accumulators[index] =
                accumulators[index] * alpha + half_to_float(value[index]);
          }
        } else {
          const float* value = args.values + row;
          int index = 0;
          for (; index + 8 <= head_dim; index += 8) {
            _mm256_storeu_ps(
                accumulators + index,
                _mm256_fmadd_ps(scale, _mm256_loadu_ps(accumulators + index),
                                _mm256_loadu_ps(value + index)));
          }
          for (; index < head_dim; ++index) {
            accumulators[index] = accumulators[index] * alpha + value[index];
          }
        }
        maximum = score;
      } else {
        const float beta = std::exp(score - maximum);
        const __m256 scale = _mm256_set1_ps(beta);
        sum += beta;
        if (half) {
          const std::uint16_t* value = half_values + row;
          int index = 0;
          for (; index + 8 <= head_dim; index += 8) {
            const __m256 v = _mm256_cvtph_ps(
                _mm_loadu_si128(reinterpret_cast<const __m128i*>(value + index)));
            _mm256_storeu_ps(
                accumulators + index,
                _mm256_fmadd_ps(scale, v, _mm256_loadu_ps(accumulators + index)));
          }
          for (; index < head_dim; ++index) {
            accumulators[index] += beta * half_to_float(value[index]);
          }
        } else {
          const float* value = args.values + row;
          int index = 0;
          for (; index + 8 <= head_dim; index += 8) {
            _mm256_storeu_ps(
                accumulators + index,
                _mm256_fmadd_ps(scale, _mm256_loadu_ps(value + index),
                                _mm256_loadu_ps(accumulators + index)));
          }
          for (; index < head_dim; ++index) {
            accumulators[index] += beta * value[index];
          }
        }
      }
    }

    args.maxima[head] = maximum;
    args.sums[head] = sum;
  }
}

}  // namespace tge::cpu
