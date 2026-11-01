#include "tge/models/qwen2.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <stdexcept>

#include "tge/core/dispatch.hpp"

namespace tge::models {
namespace {

// Context is allocated up front for both caches. 2048 tokens keeps 24 layers of
// KV under 60 MiB total, which matters on a machine with 2 GB free; the model
// metadata advertises far more than this decoder needs to demonstrate decode.
constexpr int kMaxContext = 2048;

float silu(float value) { return value / (1.0F + std::exp(-value)); }

}  // namespace

std::unique_ptr<Qwen2> Qwen2::load(const std::string& path, std::string& error) {
  std::unique_ptr<Qwen2> model(new Qwen2());
  try {
    model->file_ = gguf::File::open(path);

    const std::string architecture =
        model->file_.get_string("general.architecture", "");
    if (architecture.empty()) {
      throw std::runtime_error("model has no general.architecture");
    }
    const std::string prefix = architecture + ".";

    Qwen2Config& config = model->config_;
    config.block_count =
        static_cast<int>(model->file_.get_u64(prefix + "block_count", 0));
    config.embedding_length =
        static_cast<int>(model->file_.get_u64(prefix + "embedding_length", 0));
    config.feed_forward_length =
        static_cast<int>(model->file_.get_u64(prefix + "feed_forward_length", 0));
    config.head_count =
        static_cast<int>(model->file_.get_u64(prefix + "attention.head_count", 0));
    config.head_count_kv = static_cast<int>(
        model->file_.get_u64(prefix + "attention.head_count_kv", config.head_count));
    config.context_length =
        static_cast<int>(model->file_.get_u64(prefix + "context_length", 0));
    config.rms_epsilon = static_cast<float>(
        model->file_.get_f64(prefix + "layer_norm_rms_epsilon", 1e-6));
    config.rope_theta = static_cast<float>(
        model->file_.get_f64(prefix + "rope.freq_base", 10000.0));

    if (config.block_count <= 0 || config.embedding_length <= 0 ||
        config.head_count <= 0 || config.head_count_kv <= 0) {
      throw std::runtime_error("model configuration is incomplete");
    }
    config.head_dim = static_cast<int>(model->file_.get_u64(
        prefix + "attention.key_length",
        config.embedding_length / config.head_count));

    model->token_embd_ = model->matrix("token_embd.weight");
    model->output_ = model->matrix("output.weight");
    model->output_norm_ = model->vector("output_norm.weight",
                                        config.embedding_length);
    config.vocab_size = model->output_.rows;

    model->context_ = std::min(config.context_length > 0 ? config.context_length
                                                         : kMaxContext,
                               kMaxContext);

    model->layers_.resize(static_cast<std::size_t>(config.block_count));
    char name[96];
    for (int index = 0; index < config.block_count; ++index) {
      Layer& layer = model->layers_[static_cast<std::size_t>(index)];
      const auto make = [index, &name](const char* suffix) {
        std::snprintf(name, sizeof(name), "blk.%d.%s", index, suffix);
        return std::string(name);
      };
      layer.attn_norm = model->vector(make("attn_norm.weight").c_str(),
                                      config.embedding_length);
      layer.ffn_norm = model->vector(make("ffn_norm.weight").c_str(),
                                     config.embedding_length);
      layer.wq = model->matrix(make("attn_q.weight").c_str());
      layer.wk = model->matrix(make("attn_k.weight").c_str());
      layer.wv = model->matrix(make("attn_v.weight").c_str());
      layer.wo = model->matrix(make("attn_output.weight").c_str());
      layer.w_gate = model->matrix(make("ffn_gate.weight").c_str());
      layer.w_up = model->matrix(make("ffn_up.weight").c_str());
      layer.w_down = model->matrix(make("ffn_down.weight").c_str());

      layer.q_bias = model->vector(make("attn_q.bias").c_str(),
                                   config.embedding_length);
      layer.k_bias = model->vector(make("attn_k.bias").c_str(),
                                   config.kv_dim());
      layer.v_bias = model->vector(make("attn_v.bias").c_str(),
                                   config.kv_dim());
    }

    const int dim = config.embedding_length;
    const int kv_dim = config.kv_dim();
    const std::size_t cache_elements =
        static_cast<std::size_t>(config.block_count) * model->context_ * kv_dim;
    model->k_cache_.assign(cache_elements, 0.0F);
    model->v_cache_.assign(cache_elements, 0.0F);

    model->x_.assign(static_cast<std::size_t>(dim), 0.0F);
    model->h_.assign(static_cast<std::size_t>(dim), 0.0F);
    model->q_.assign(static_cast<std::size_t>(dim), 0.0F);
    model->k_.assign(static_cast<std::size_t>(kv_dim), 0.0F);
    model->v_.assign(static_cast<std::size_t>(kv_dim), 0.0F);
    model->attn_.assign(static_cast<std::size_t>(dim), 0.0F);
    model->gate_.assign(static_cast<std::size_t>(config.feed_forward_length),
                        0.0F);
    model->up_.assign(static_cast<std::size_t>(config.feed_forward_length),
                      0.0F);
    model->logits_.assign(static_cast<std::size_t>(config.vocab_size), 0.0F);
    model->scores_.assign(static_cast<std::size_t>(model->context_), 0.0F);

    model->rope_freq_.resize(static_cast<std::size_t>(config.head_dim / 2));
    for (int index = 0; index < config.head_dim / 2; ++index) {
      model->rope_freq_[static_cast<std::size_t>(index)] =
          std::pow(config.rope_theta,
                   -2.0F * static_cast<float>(index) /
                       static_cast<float>(config.head_dim));
    }

    return model;
  } catch (const std::exception& exception) {
    error = exception.what();
    return nullptr;
  }
}

Qwen2::Matrix Qwen2::matrix(const char* name) const {
  const gguf::TensorInfo* info = file_.tensor(name);
  if (info == nullptr) {
    throw std::runtime_error(std::string("missing tensor ") + name);
  }
  if (!gguf::type_supported(info->type)) {
    throw std::runtime_error(std::string("unsupported type for ") + name + ": " +
                             gguf::type_name(info->type));
  }
  if (info->shape.size() != 2) {
    throw std::runtime_error(std::string("tensor is not a matrix: ") + name);
  }

  Matrix result;
  result.data = file_.tensor_data(*info);
  result.cols = static_cast<int>(info->shape[0]);
  result.rows = static_cast<int>(info->shape[1]);
  result.type = info->type;
  return result;
}

const float* Qwen2::vector(const char* name, int expected) const {
  const gguf::TensorInfo* info = file_.tensor(name);
  if (info == nullptr) {
    throw std::runtime_error(std::string("missing tensor ") + name);
  }
  if (info->type != gguf::TensorType::F32) {
    throw std::runtime_error(std::string("expected F32 for ") + name);
  }
  if (static_cast<int>(info->elements) != expected) {
    throw std::runtime_error(std::string("wrong length for ") + name);
  }
  return reinterpret_cast<const float*>(file_.tensor_data(*info));
}

void Qwen2::matvec(const Matrix& weight, const float* input,
                   float* output) const {
  // Decode is memory-bound and synchronisation-sensitive, so it uses the
  // decode pool rather than one worker per logical processor. Sizing each call
  // to the matrix was measured and gave no gain over a flat count, so the
  // simpler rule stands.
  const int workers =
      threads_ > 0 ? threads_ : cpu::resolve_decode_threads();
  const cpu::GemvConfig config{workers};
  if (weight.type == gguf::TensorType::Q8_0) {
    cpu::Q8GemvArgs args;
    args.data = weight.data;
    args.rows = weight.rows;
    args.cols = weight.cols;
    args.vector = input;
    args.out = output;
    kernels().gemv_q8(args, config);
    return;
  }
  if (weight.type == gguf::TensorType::F32) {
    cpu::GemvArgs args;
    args.rows = weight.rows;
    args.cols = weight.cols;
    args.matrix = reinterpret_cast<const float*>(weight.data);
    args.ldm = weight.cols;
    args.vector = input;
    args.out = output;
    kernels().gemv_f32(args, config);
    return;
  }
  throw std::runtime_error("matvec: unsupported weight type");
}

void Qwen2::dequant_row(const Matrix& weight, int row, float* output) const {
  if (row < 0 || row >= weight.rows) {
    throw std::runtime_error("embedding row out of range");
  }
  if (weight.type == gguf::TensorType::F32) {
    std::memcpy(output, reinterpret_cast<const float*>(weight.data) +
                           static_cast<std::size_t>(row) * weight.cols,
                sizeof(float) * static_cast<std::size_t>(weight.cols));
    return;
  }

  const int blocks = weight.cols / cpu::kQ8BlockElements;
  const std::size_t row_bytes = static_cast<std::size_t>(blocks) * cpu::kQ8BlockBytes;
  const std::uint8_t* row_data = weight.data + static_cast<std::size_t>(row) * row_bytes;

  for (int block = 0; block < blocks; ++block) {
    const std::uint8_t* entry = row_data + block * cpu::kQ8BlockBytes;
    std::uint16_t scale_bits = 0;
    std::memcpy(&scale_bits, entry, sizeof(scale_bits));
    const float scale = cpu::half_to_float(scale_bits);
    const auto* values = reinterpret_cast<const std::int8_t*>(entry + 2);
    float* destination = output + block * cpu::kQ8BlockElements;
    for (int index = 0; index < cpu::kQ8BlockElements; ++index) {
      destination[index] = static_cast<float>(values[index]) * scale;
    }
  }
}

void Qwen2::rms_norm(float* out, const float* in, const float* weight) const {
  const int dim = config_.embedding_length;
  float sum = 0.0F;
  for (int index = 0; index < dim; ++index) {
    sum += in[index] * in[index];
  }
  const float scale = 1.0F / std::sqrt(sum / static_cast<float>(dim) +
                                       config_.rms_epsilon);
  for (int index = 0; index < dim; ++index) {
    out[index] = in[index] * scale * weight[index];
  }
}

void Qwen2::apply_rope(float* values, int heads, int position) const {
  const int half = config_.head_dim / 2;
  for (int head = 0; head < heads; ++head) {
    float* vector = values + static_cast<std::size_t>(head) * config_.head_dim;
    for (int index = 0; index < half; ++index) {
      const float angle =
          static_cast<float>(position) * rope_freq_[static_cast<std::size_t>(index)];
      const float cosine = std::cos(angle);
      const float sine = std::sin(angle);
      // Qwen2 rotates the first half of the head against the second half
      // (NeoX layout), not adjacent pairs.
      const float low = vector[index];
      const float high = vector[index + half];
      vector[index] = low * cosine - high * sine;
      vector[index + half] = low * sine + high * cosine;
    }
  }
}

void Qwen2::attention(int layer, int position) {
  const int heads = config_.head_count;
  const int kv_heads = config_.head_count_kv;
  const int head_dim = config_.head_dim;
  const int kv_dim = config_.kv_dim();
  const int group = heads / kv_heads;
  const float scale = 1.0F / std::sqrt(static_cast<float>(head_dim));
  const int span = position + 1;

  const std::size_t layer_base =
      static_cast<std::size_t>(layer) * context_ * kv_dim;
  const float* k_base = k_cache_.data() + layer_base;
  const float* v_base = v_cache_.data() + layer_base;

  for (int head = 0; head < heads; ++head) {
    const float* query = q_.data() + static_cast<std::size_t>(head) * head_dim;
    const int kv_head = head / group;
    const std::size_t kv_head_offset = static_cast<std::size_t>(kv_head) * head_dim;

    float maximum = -INFINITY;
    for (int token = 0; token < span; ++token) {
      const float* key =
          k_base + static_cast<std::size_t>(token) * kv_dim + kv_head_offset;
      float product = 0.0F;
      for (int index = 0; index < head_dim; ++index) {
        product += query[index] * key[index];
      }
      scores_[static_cast<std::size_t>(token)] = product * scale;
      maximum = std::max(maximum, product * scale);
    }

    float total = 0.0F;
    for (int token = 0; token < span; ++token) {
      const float value =
          std::exp(scores_[static_cast<std::size_t>(token)] - maximum);
      scores_[static_cast<std::size_t>(token)] = value;
      total += value;
    }
    const float inverse = 1.0F / total;

    float* out = attn_.data() + static_cast<std::size_t>(head) * head_dim;
    for (int index = 0; index < head_dim; ++index) {
      out[index] = 0.0F;
    }
    for (int token = 0; token < span; ++token) {
      const float weight = scores_[static_cast<std::size_t>(token)] * inverse;
      const float* value =
          v_base + static_cast<std::size_t>(token) * kv_dim + kv_head_offset;
      for (int index = 0; index < head_dim; ++index) {
        out[index] += weight * value[index];
      }
    }
  }
}

const float* Qwen2::forward(int token, int position) {
  const int dim = config_.embedding_length;
  const int kv_dim = config_.kv_dim();

  dequant_row(token_embd_, token, x_.data());

  for (int index = 0; index < config_.block_count; ++index) {
    const Layer& layer = layers_[static_cast<std::size_t>(index)];

    rms_norm(h_.data(), x_.data(), layer.attn_norm);
    matvec(layer.wq, h_.data(), q_.data());
    matvec(layer.wk, h_.data(), k_.data());
    matvec(layer.wv, h_.data(), v_.data());
    for (int axis = 0; axis < dim; ++axis) {
      q_[static_cast<std::size_t>(axis)] += layer.q_bias[axis];
    }
    for (int axis = 0; axis < kv_dim; ++axis) {
      k_[static_cast<std::size_t>(axis)] += layer.k_bias[axis];
      v_[static_cast<std::size_t>(axis)] += layer.v_bias[axis];
    }

    apply_rope(q_.data(), config_.head_count, position);
    apply_rope(k_.data(), config_.head_count_kv, position);

    const std::size_t layer_base =
        static_cast<std::size_t>(index) * context_ * kv_dim;
    const std::size_t slot =
        layer_base + static_cast<std::size_t>(position) * kv_dim;
    std::memcpy(k_cache_.data() + slot, k_.data(),
                sizeof(float) * static_cast<std::size_t>(kv_dim));
    std::memcpy(v_cache_.data() + slot, v_.data(),
                sizeof(float) * static_cast<std::size_t>(kv_dim));

    attention(index, position);

    matvec(layer.wo, attn_.data(), h_.data());
    for (int axis = 0; axis < dim; ++axis) {
      x_[static_cast<std::size_t>(axis)] += h_[axis];
    }

    rms_norm(h_.data(), x_.data(), layer.ffn_norm);
    matvec(layer.w_gate, h_.data(), gate_.data());
    matvec(layer.w_up, h_.data(), up_.data());
    for (int axis = 0; axis < config_.feed_forward_length; ++axis) {
      gate_[static_cast<std::size_t>(axis)] =
          silu(gate_[static_cast<std::size_t>(axis)]) * up_[axis];
    }
    matvec(layer.w_down, gate_.data(), h_.data());
    for (int axis = 0; axis < dim; ++axis) {
      x_[static_cast<std::size_t>(axis)] += h_[axis];
    }
  }

  rms_norm(h_.data(), x_.data(), output_norm_);
  matvec(output_, h_.data(), logits_.data());
  return logits_.data();
}

void Qwen2::reset_cache() {
  std::fill(k_cache_.begin(), k_cache_.end(), 0.0F);
  std::fill(v_cache_.begin(), v_cache_.end(), 0.0F);
}

int Qwen2::argmax(const float* values, int count) {
  int best = 0;
  float best_value = values[0];
  for (int index = 1; index < count; ++index) {
    if (values[index] > best_value) {
      best_value = values[index];
      best = index;
    }
  }
  return best;
}

}  // namespace tge::models
