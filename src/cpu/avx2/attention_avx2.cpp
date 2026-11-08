#include "tge/cpu/kernels.hpp"
#include "tge/cpu/exp_avx2.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <vector>

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

void attention_head_static_sparse_avx2(const AttentionArgs& args, int sink,
                                       int window, int stride) {
  validate_attention_args(args);
  if (args.half || args.q8) {
    throw std::invalid_argument("static sparse: half/q8 not supported");
  }
  if (sink < 0 || window < 1 || stride < 2) {
    throw std::invalid_argument("static sparse: bad sink/window/stride");
  }
  const int head_dim = args.head_dim;
  const int span = args.span;
  const int kv_stride = args.kv_stride;

  // Kept-set geometry identical to build_sparse_index (ascending): sink
  // verbatim [0,sink_end), strided middle [sink_end,window_start), window
  // tail [max(window_start,sink_end),span). Ordinal i maps to a resident
  // position arithmetically — no index build, no compact buffers.
  const int sink_end = sink < span ? sink : span;
  const int window_start = span - window;
  const int tail_start = window_start > sink_end ? window_start : sink_end;
  const int middle_count =
      window_start > sink_end
          ? (window_start - sink_end + stride - 1) / stride
          : 0;
  const int tail_count = span > tail_start ? span - tail_start : 0;
  const int kept_count = sink_end + middle_count + tail_count;
  if (kept_count <= 0) {
    throw std::invalid_argument("static sparse: empty kept set");
  }
  auto kept_position = [=](int ordinal) {
    if (ordinal < sink_end) {
      return ordinal;
    }
    const int middle = ordinal - sink_end;
    if (middle < middle_count) {
      return sink_end + middle * stride;
    }
    return tail_start + (middle - middle_count);
  };

  float maximum = -INFINITY;
  for (int kept = 0; kept < kept_count; ++kept) {
    const std::size_t row =
        static_cast<std::size_t>(kept_position(kept)) * kv_stride;
    const float dot = dot_avx2(args.query, args.keys + row, head_dim);
    const float scaled = dot * args.scale;
    args.scores[static_cast<std::size_t>(kept)] = scaled;
    maximum = std::max(maximum, scaled);
  }

  // Softmax via the shared vectorised exp (same block as the dense kernel,
  // over the compact kept count).
  float total = 0.0F;
  {
    const __m256 vmax = _mm256_set1_ps(maximum);
    __m256 vsum = _mm256_setzero_ps();
    int kept = 0;
    const int blocked = kept_count & ~7;
    for (; kept < blocked; kept += 8) {
      const __m256 s =
          _mm256_loadu_ps(args.scores + static_cast<std::size_t>(kept));
      const __m256 shifted = _mm256_sub_ps(s, vmax);
      const __m256 e = exp_avx2(shifted);
      _mm256_storeu_ps(args.scores + static_cast<std::size_t>(kept), e);
      vsum = _mm256_add_ps(vsum, e);
    }
    total = horizontal_sum(vsum);
    for (; kept < kept_count; ++kept) {
      const float value =
          std::exp(args.scores[static_cast<std::size_t>(kept)] - maximum);
      args.scores[static_cast<std::size_t>(kept)] = value;
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

  for (int kept = 0; kept < kept_count; ++kept) {
    const float scalar_weight =
        args.scores[static_cast<std::size_t>(kept)] * inverse;
    const std::size_t row =
        static_cast<std::size_t>(kept_position(kept)) * kv_stride;
    const float* value = args.values + row;
    const __m256 weight = _mm256_set1_ps(scalar_weight);
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

void attention_head_static_sparse_fp16_avx2(const AttentionArgs& args,
                                              int sink, int window,
                                              int stride) {
  validate_attention_args(args);
  if (args.q8) {
    throw std::invalid_argument("static sparse: q8 not supported");
  }
  if (!args.half) {
    throw std::invalid_argument("static sparse fp16: half flag required");
  }
  if (sink < 0 || window < 1 || stride < 2) {
    throw std::invalid_argument("static sparse: bad sink/window/stride");
  }
  const int head_dim = args.head_dim;
  const int span = args.span;
  const int kv_stride = args.kv_stride;
  const std::uint16_t* half_keys = args.keys_half;
  const std::uint16_t* half_values = args.values_half;

  const int sink_end = sink < span ? sink : span;
  const int window_start = span - window;
  const int tail_start = window_start > sink_end ? window_start : sink_end;
  const int middle_count =
      window_start > sink_end ? (window_start - sink_end + stride - 1) / stride
                              : 0;
  const int tail_count = span > tail_start ? span - tail_start : 0;
  const int kept_count = sink_end + middle_count + tail_count;
  if (kept_count <= 0) {
    throw std::invalid_argument("static sparse: empty kept set");
  }
  auto kept_position = [=](int ordinal) {
    if (ordinal < sink_end) {
      return ordinal;
    }
    const int middle = ordinal - sink_end;
    if (middle < middle_count) {
      return sink_end + middle * stride;
    }
    return tail_start + (middle - middle_count);
  };

  float maximum = -INFINITY;
  for (int kept = 0; kept < kept_count; ++kept) {
    const std::size_t row =
        static_cast<std::size_t>(kept_position(kept)) * kv_stride;
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
    float dot = horizontal_sum(_mm256_add_ps(acc0, acc1));
    for (; index < head_dim; ++index) {
      dot += args.query[index] * half_to_float(key[index]);
    }
    const float scaled = dot * args.scale;
    args.scores[static_cast<std::size_t>(kept)] = scaled;
    maximum = std::max(maximum, scaled);
  }

  float total = 0.0F;
  {
    const __m256 vmax = _mm256_set1_ps(maximum);
    __m256 vsum = _mm256_setzero_ps();
    int kept = 0;
    const int blocked = kept_count & ~7;
    for (; kept < blocked; kept += 8) {
      const __m256 s =
          _mm256_loadu_ps(args.scores + static_cast<std::size_t>(kept));
      const __m256 shifted = _mm256_sub_ps(s, vmax);
      const __m256 e = exp_avx2(shifted);
      _mm256_storeu_ps(args.scores + static_cast<std::size_t>(kept), e);
      vsum = _mm256_add_ps(vsum, e);
    }
    total = horizontal_sum(vsum);
    for (; kept < kept_count; ++kept) {
      const float value =
          std::exp(args.scores[static_cast<std::size_t>(kept)] - maximum);
      args.scores[static_cast<std::size_t>(kept)] = value;
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

  for (int kept = 0; kept < kept_count; ++kept) {
    const float scalar_weight =
        args.scores[static_cast<std::size_t>(kept)] * inverse;
    const std::size_t row =
        static_cast<std::size_t>(kept_position(kept)) * kv_stride;
    const std::uint16_t* value = half_values + row;
    const __m256 weight = _mm256_set1_ps(scalar_weight);
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
          for (; index + 8 <= head_dim; index += 8) {
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

void attention_block_head_avx2(const AttentionBlockArgs& args, int head) {
  if (args.queries == nullptr || args.scores == nullptr || args.out == nullptr ||
      args.tokens <= 0 || args.head_dim <= 0 || args.head_dim % 8 != 0 ||
      args.context <= 0 || head < 0) {
    return;
  }
  if (args.half) {
    if (args.keys_half == nullptr || args.values_half == nullptr) {
      return;
    }
  } else if (args.keys == nullptr || args.values == nullptr) {
    return;
  }

  const int head_dim = args.head_dim;
  const int tokens = args.tokens;
  const int start = args.start_position;
  const int last_key = start + tokens - 1;
  const std::size_t head_offset =
      static_cast<std::size_t>(head) * static_cast<std::size_t>(head_dim);

  // f16 K/V rows are converted once per key row, then reused for every query
  // in the block. The old loop converted the same row inside each query.
  constexpr int kStackHeadDim = 256;
  alignas(32) float stack_row[kStackHeadDim];
  std::vector<float> heap_row;
  float* row_f32 = stack_row;
  if (head_dim > kStackHeadDim) {
    heap_row.resize(static_cast<std::size_t>(head_dim));
    row_f32 = heap_row.data();
  }
  const auto widen_row = [&](const std::uint16_t* source) {
    int index = 0;
    for (; index + 8 <= head_dim; index += 8) {
      _mm256_storeu_ps(
          row_f32 + index,
          _mm256_cvtph_ps(_mm_loadu_si128(
              reinterpret_cast<const __m128i*>(source + index))));
    }
    for (; index < head_dim; ++index) {
      row_f32[index] = half_to_float(source[index]);
    }
  };

  // Pass 1: QK scores. One key row feeds every query in the block that can
  // attend to it, so the row is loaded once instead of once per query.
  for (int key_index = 0; key_index <= last_key; ++key_index) {
    const std::size_t row =
        static_cast<std::size_t>(key_index) * args.kv_stride;
    const float* key_f32 = nullptr;
    const std::uint16_t* key_half = nullptr;
    if (args.half) {
      key_half = args.keys_half + row + args.kv_head_offset;
      widen_row(key_half);
    } else {
      key_f32 = args.keys + row + args.kv_head_offset;
    }
    const int first_query = std::max(0, key_index - start);
    for (int token = first_query; token < tokens; ++token) {
      const float* query =
          args.queries +
          static_cast<std::size_t>(token) * args.query_stride + head_offset;
      float score = 0.0F;
      if (args.half) {
        __m256 acc0 = _mm256_setzero_ps();
        __m256 acc1 = _mm256_setzero_ps();
        int index = 0;
        for (; index + 16 <= head_dim; index += 16) {
          const __m256 k0 = _mm256_loadu_ps(row_f32 + index);
          const __m256 k1 = _mm256_loadu_ps(row_f32 + index + 8);
          acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(query + index), k0, acc0);
          acc1 =
              _mm256_fmadd_ps(_mm256_loadu_ps(query + index + 8), k1, acc1);
        }
        for (; index + 8 <= head_dim; index += 8) {
          const __m256 k = _mm256_loadu_ps(row_f32 + index);
          acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(query + index), k, acc0);
        }
        score = horizontal_sum(_mm256_add_ps(acc0, acc1));
        for (; index < head_dim; ++index) {
          score += query[index] * row_f32[index];
        }
        score *= args.scale;
      } else {
        score = dot_avx2(query, key_f32, head_dim) * args.scale;
      }
      args.scores[static_cast<std::size_t>(token) * args.score_stride +
                  key_index] = score;
    }
  }

  // Softmax per query over its causal prefix; the max/exp form mirrors
  // attention_head so the two paths agree to summation-order noise.
  for (int token = 0; token < tokens; ++token) {
    float* scores = args.scores +
                    static_cast<std::size_t>(token) * args.score_stride;
    const int span = start + token + 1;
    float maximum = -INFINITY;
    for (int index = 0; index < span; ++index) {
      maximum = std::max(maximum, scores[index]);
    }
    const __m256 vmax = _mm256_set1_ps(maximum);
    __m256 vsum = _mm256_setzero_ps();
    int index = 0;
    const int blocked = span & ~7;
    for (; index < blocked; index += 8) {
      const __m256 shifted = _mm256_sub_ps(_mm256_loadu_ps(scores + index), vmax);
      const __m256 e = exp_avx2(shifted);
      _mm256_storeu_ps(scores + index, e);
      vsum = _mm256_add_ps(vsum, e);
    }
    float total = horizontal_sum(vsum);
    for (; index < span; ++index) {
      const float e = std::exp(scores[index] - maximum);
      scores[index] = e;
      total += e;
    }
    const float inverse = 1.0F / total;
    for (index = 0; index < span; ++index) {
      scores[index] *= inverse;
    }
  }

  // Pass 2: value accumulation. One value row feeds every query in the block.
  for (int token = 0; token < tokens; ++token) {
    float* accumulator = args.out +
                         static_cast<std::size_t>(token) * args.query_stride +
                         head_offset;
    int index = 0;
    for (; index + 8 <= head_dim; index += 8) {
      _mm256_storeu_ps(accumulator + index, _mm256_setzero_ps());
    }
    for (; index < head_dim; ++index) {
      accumulator[index] = 0.0F;
    }
  }
  for (int key_index = 0; key_index <= last_key; ++key_index) {
    const std::size_t row =
        static_cast<std::size_t>(key_index) * args.kv_stride;
    const float* value_f32 = nullptr;
    const std::uint16_t* value_half = nullptr;
    if (args.half) {
      value_half = args.values_half + row + args.kv_head_offset;
      widen_row(value_half);
    } else {
      value_f32 = args.values + row + args.kv_head_offset;
    }
    const int first_query = std::max(0, key_index - start);
    for (int token = first_query; token < tokens; ++token) {
      const float weight =
          args.scores[static_cast<std::size_t>(token) * args.score_stride +
                      key_index];
      if (weight == 0.0F) {
        continue;
      }
      float* accumulator = args.out +
                           static_cast<std::size_t>(token) * args.query_stride +
                           head_offset;
      const __m256 w = _mm256_set1_ps(weight);
      int index = 0;
      if (args.half) {
        for (; index + 8 <= head_dim; index += 8) {
          const __m256 value = _mm256_loadu_ps(row_f32 + index);
          _mm256_storeu_ps(accumulator + index,
                           _mm256_fmadd_ps(w, value,
                                           _mm256_loadu_ps(accumulator + index)));
        }
        for (; index < head_dim; ++index) {
          accumulator[index] += weight * row_f32[index];
        }
      } else {
        for (; index + 8 <= head_dim; index += 8) {
          _mm256_storeu_ps(
              accumulator + index,
              _mm256_fmadd_ps(w, _mm256_loadu_ps(value_f32 + index),
                              _mm256_loadu_ps(accumulator + index)));
        }
        for (; index < head_dim; ++index) {
          accumulator[index] += weight * value_f32[index];
        }
      }
    }
  }
}

}  // namespace tge::cpu
