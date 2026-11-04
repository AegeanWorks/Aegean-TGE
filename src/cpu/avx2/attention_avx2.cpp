#include "tge/cpu/kernels.hpp"
#include "tge/cpu/exp_avx2.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>

#include <immintrin.h>

namespace tge::cpu {
namespace {

inline float horizontal_sum(__m256 values) {
  __m128 total = _mm_add_ps(_mm256_castps256_ps128(values),
                            _mm256_extractf128_ps(values, 1));
  total = _mm_add_ps(total, _mm_movehl_ps(total, total));
  total = _mm_add_ss(total, _mm_shuffle_ps(total, total, 1));
  return _mm_cvtss_f32(total);
}

inline float dot_avx2(const float* left, const float* right, int count) {
  // Four accumulators hide FMA latency even for head_dim 128 (4x32 = 4 iters).
  __m256 acc0 = _mm256_setzero_ps();
  __m256 acc1 = _mm256_setzero_ps();
  __m256 acc2 = _mm256_setzero_ps();
  __m256 acc3 = _mm256_setzero_ps();
  int index = 0;
  for (; index + 32 <= count; index += 32) {
    acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(left + index),
                           _mm256_loadu_ps(right + index), acc0);
    acc1 = _mm256_fmadd_ps(_mm256_loadu_ps(left + index + 8),
                           _mm256_loadu_ps(right + index + 8), acc1);
    acc2 = _mm256_fmadd_ps(_mm256_loadu_ps(left + index + 16),
                           _mm256_loadu_ps(right + index + 16), acc2);
    acc3 = _mm256_fmadd_ps(_mm256_loadu_ps(left + index + 24),
                           _mm256_loadu_ps(right + index + 24), acc3);
  }
  for (; index + 8 <= count; index += 8) {
    acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(left + index),
                           _mm256_loadu_ps(right + index), acc0);
  }
  const __m256 sum =
      _mm256_add_ps(_mm256_add_ps(acc0, acc1), _mm256_add_ps(acc2, acc3));
  float result = horizontal_sum(sum);
  for (; index < count; ++index) {
    result += left[index] * right[index];
  }
  return result;
}

/// Widens eight signed bytes to eight floats.
inline __m256 widen_i8(const std::int8_t* values) {
  const __m128i packed =
      _mm_loadl_epi64(reinterpret_cast<const __m128i*>(values));
  return _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(packed));
}

/// Dot product of a float query against one int8 KV row with per-32 scales.
/// Same widen idiom as the Q8 batch kernels; the tail handles head dims that
/// are not a multiple of 32 (production head_dim 128 always divides evenly).
inline float q8_dot_avx2(const float* query, const std::int8_t* keys,
                         const float* scales, int count) {
  __m256 acc0 = _mm256_setzero_ps();
  __m256 acc1 = _mm256_setzero_ps();
  int index = 0;
  for (; index + 32 <= count; index += 32) {
    const __m256 s = _mm256_set1_ps(scales[index / 32]);
    for (int lane = 0; lane < 32; lane += 8) {
      const __m256 w =
          _mm256_mul_ps(widen_i8(keys + index + lane), s);
      if ((lane & 16) == 0) {
        acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(query + index + lane), w, acc0);
      } else {
        acc1 = _mm256_fmadd_ps(_mm256_loadu_ps(query + index + lane), w, acc1);
      }
    }
  }
  float dot = horizontal_sum(_mm256_add_ps(acc0, acc1));
  for (; index < count; ++index) {
    dot += query[index] * static_cast<float>(keys[index]) *
           scales[index / 32];
  }
  return dot;
}

void validate_attention_args(const AttentionArgs& args) {
  if (args.head_dim <= 0 || args.span <= 0) {
    throw std::invalid_argument("attention: non-positive head_dim/span");
  }
  if (args.kv_stride < args.head_dim) {
    throw std::invalid_argument("attention: kv_stride < head_dim");
  }
  if (args.query == nullptr || args.scores == nullptr || args.out == nullptr) {
    throw std::invalid_argument("attention: null query/scores/out");
  }
  if (args.half && args.q8) {
    throw std::invalid_argument("attention: half and q8 both set");
  }
  if (args.q8) {
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

void attention_head_avx2(const AttentionArgs& args) {
  validate_attention_args(args);
  const int head_dim = args.head_dim;
  const int stride = args.kv_stride;
  const int span = args.span;
  const bool half = args.half;
  const std::uint16_t* half_keys = args.keys_half;
  const std::uint16_t* half_values = args.values_half;
  const bool q8 = args.q8;
  const int scale_stride = stride / 32;

  float maximum = -INFINITY;
  for (int token = 0; token < span; ++token) {
    const std::size_t row = static_cast<std::size_t>(token) * stride;
    float dot = 0.0F;
    if (q8) {
      dot = q8_dot_avx2(args.query, args.keys_q8 + row,
                        args.keys_q8_scales +
                            static_cast<std::size_t>(token) * scale_stride,
                        head_dim);
    } else if (half) {
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

  // Softmax via the shared vectorised exp (priced by the fidelity harness,
  // differential-tested against the scalar oracle at 1e-4).
  float total = 0.0F;
  {
    const __m256 vmax = _mm256_set1_ps(maximum);
    __m256 vsum = _mm256_setzero_ps();
    int token = 0;
    const int blocked = span & ~7;
    for (; token < blocked; token += 8) {
      const __m256 s =
          _mm256_loadu_ps(args.scores + static_cast<std::size_t>(token));
      const __m256 shifted = _mm256_sub_ps(s, vmax);
      const __m256 e = exp_avx2(shifted);
      _mm256_storeu_ps(args.scores + static_cast<std::size_t>(token), e);
      vsum = _mm256_add_ps(vsum, e);
    }
    total = horizontal_sum(vsum);
    for (; token < span; ++token) {
      const float value =
          std::exp(args.scores[static_cast<std::size_t>(token)] - maximum);
      args.scores[static_cast<std::size_t>(token)] = value;
      total += value;
    }
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
    if (q8) {
      const std::int8_t* value = args.values_q8 + row;
      const float* vscales = args.values_q8_scales +
                             static_cast<std::size_t>(token) * scale_stride;
      int lane = 0;
      for (; lane + 32 <= head_dim; lane += 32) {
        const __m256 s = _mm256_set1_ps(vscales[lane / 32]);
        for (int k = 0; k < 32; k += 8) {
          const __m256 v =
              _mm256_mul_ps(widen_i8(value + lane + k), s);
          _mm256_storeu_ps(
              args.out + lane + k,
              _mm256_fmadd_ps(weight, v,
                              _mm256_loadu_ps(args.out + lane + k)));
        }
      }
      for (; lane < head_dim; ++lane) {
        args.out[lane] += scalar_weight *
                          static_cast<float>(value[lane]) *
                          vscales[lane / 32];
      }
    } else if (half) {
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
  const bool q8 = args.q8;
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
      float score;
      if (q8) {
        score = q8_dot_avx2(query, args.keys_q8 + row,
                            args.keys_q8_scales +
                                static_cast<std::size_t>(token) * scale_stride,
                            head_dim) *
                args.scale;
      } else if (half) {
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
        if (q8) {
          const std::int8_t* value = args.values_q8 + row;
          const float* vscales = args.values_q8_scales +
                                 static_cast<std::size_t>(token) * scale_stride;
          int index = 0;
          for (; index + 32 <= head_dim; index += 32) {
            const __m256 s = _mm256_set1_ps(vscales[index / 32]);
            for (int k = 0; k < 32; k += 8) {
              const __m256 v =
                  _mm256_mul_ps(widen_i8(value + index + k), s);
              _mm256_storeu_ps(
                  accumulators + index + k,
                  _mm256_fmadd_ps(scale,
                                  _mm256_loadu_ps(accumulators + index + k),
                                  v));
            }
          }
          for (; index < head_dim; ++index) {
            accumulators[index] =
                accumulators[index] * alpha +
                static_cast<float>(value[index]) * vscales[index / 32];
          }
        } else if (half) {
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
        if (q8) {
          const std::int8_t* value = args.values_q8 + row;
          const float* vscales = args.values_q8_scales +
                                 static_cast<std::size_t>(token) * scale_stride;
          int index = 0;
          for (; index + 32 <= head_dim; index += 8) {
            const __m256 v = _mm256_mul_ps(
                widen_i8(value + index),
                _mm256_set1_ps(vscales[index / 32]));
            _mm256_storeu_ps(
                accumulators + index,
                _mm256_fmadd_ps(scale, v,
                                _mm256_loadu_ps(accumulators + index)));
          }
          for (; index < head_dim; ++index) {
            accumulators[index] += beta * static_cast<float>(value[index]) *
                                   vscales[index / 32];
          }
        } else if (half) {
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
