#include "tge/cpu/kernels.hpp"

#include <algorithm>
#include <cmath>

namespace tge::cpu {

void rms_norm_scalar(float* out, const float* in, const float* weight, int dim,
                     float eps) {
  if (out == nullptr || in == nullptr || weight == nullptr || dim <= 0) {
    return;
  }
  float sum = 0.0F;
  for (int index = 0; index < dim; ++index) {
    sum += in[index] * in[index];
  }
  const float scale = 1.0F / std::sqrt(sum / static_cast<float>(dim) + eps);
  for (int index = 0; index < dim; ++index) {
    out[index] = in[index] * scale * weight[index];
  }
}

void apply_rope_scalar(float* values, int heads, int head_dim, int position,
                       const float* cos_table, const float* sin_table) {  if (values == nullptr || heads <= 0 || head_dim <= 0 ||
      cos_table == nullptr || sin_table == nullptr) {
    return;
  }
  const int half = head_dim / 2;
  for (int head = 0; head < heads; ++head) {
    float* vector = values + static_cast<std::size_t>(head) * head_dim;
    for (int index = 0; index < half; ++index) {
      const float cosine = cos_table[index];
      const float sine = sin_table[index];
      const float low = vector[index];
      const float high = vector[index + half];
      vector[index] = low * cosine - high * sine;
      vector[index + half] = low * sine + high * cosine;
    }
  }
  (void)position;
}

void silu_gated_scalar(float* gate, const float* up, int count) {
  if (gate == nullptr || up == nullptr || count <= 0) {
    return;
  }
  for (int index = 0; index < count; ++index) {
    const float value = gate[index];
    gate[index] = value / (1.0F + std::exp(-value)) * up[index];
  }
}

float softmax_exp_scalar(const float* logits, int count, float temperature,
                         float* out) {
  float total = 0.0F;
  if (logits == nullptr || out == nullptr || count <= 0 ||
      !(temperature > 0.0F)) {
    return total;
  }
  float maximum = logits[0];
  for (int index = 1; index < count; ++index) {
    maximum = std::max(maximum, logits[index]);
  }
  const float inverse_temperature = 1.0F / temperature;
  for (int index = 0; index < count; ++index) {
    const float probability =
        std::exp((logits[index] - maximum) * inverse_temperature);
    out[index] = probability;
    total += probability;
  }
  return total;
}

void apply_rope_interleaved_scalar(float* values, int heads, int head_dim,
                                   int position, const float* cos_table,
                                   const float* sin_table) {
  if (values == nullptr || heads <= 0 || head_dim <= 0 ||
      cos_table == nullptr || sin_table == nullptr) {
    return;
  }
  // ggml NORM: consecutive pairs (x[2k], x[2k+1]) rotate by table entry k.
  // The duplicated tables hold each coefficient twice, so loading either
  // slot of the pair reads the same value.
  for (int head = 0; head < heads; ++head) {
    float* vector = values + static_cast<std::size_t>(head) * head_dim;
    for (int index = 0; index < head_dim; index += 2) {
      const float cosine = cos_table[index];
      const float sine = sin_table[index];
      const float even = vector[index];
      const float odd = vector[index + 1];
      vector[index] = even * cosine - odd * sine;
      vector[index + 1] = even * sine + odd * cosine;
    }
  }
  (void)position;
}

}  // namespace tge::cpu
