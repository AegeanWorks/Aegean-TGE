#include "tge/models/qwen2.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>

#include "tge/core/dispatch.hpp"
#include "tge/core/thread_pool.hpp"
#include "tge/memory/budget.hpp"
#include "tge/models/plan.hpp"

namespace tge::models {
namespace {

// Context is allocated up front for both caches. The default keeps the KV cache
// small enough to coexist with other applications on a 16 GiB machine; callers
// that want more ask for it, and the request is capped by the model itself.
constexpr int kDefaultContext = 4096;
constexpr int kHardMaxContext = 32768;

// Prompt chunk. The batch kernel re-reads the activation rows once per output
// row, so the chunk has to stay small enough that they live in cache. 64 rows
// of the widest activation (4864 floats) is about 1.2 MiB per worker. Aligning
// it to the kernel's six-token group was measured and was slightly worse, so
// the tail is left to the single-token path.
constexpr int kPrefillChunk = 64;

// Grouped-query attention splits decode positions into at most this many
// chunks; the scratch is sized for the maximum.
constexpr int kMaxGqaChunks = 16;

float silu(float value) { return value / (1.0F + std::exp(-value)); }

/// Maps a supported weight type onto the kernel's quant kind. Q8_0 has fused
/// kernels of its own, so it is not routed through this path.
bool to_quant_kind(gguf::TensorType type, cpu::QuantKind& kind) {
  switch (type) {
    case gguf::TensorType::Q4_K:
      kind = cpu::QuantKind::Q4_K;
      return true;
    case gguf::TensorType::Q5_K:
      kind = cpu::QuantKind::Q5_K;
      return true;
    case gguf::TensorType::Q6_K:
      kind = cpu::QuantKind::Q6_K;
      return true;
    case gguf::TensorType::Q4_0:
      kind = cpu::QuantKind::Q4_0;
      return true;
    case gguf::TensorType::Q5_0:
      kind = cpu::QuantKind::Q5_0;
      return true;
    default:
      return false;
  }
}

}  // namespace

Qwen2Config read_config(const gguf::File& file) {
  const std::string architecture = file.get_string("general.architecture", "");
  if (architecture.empty()) {
    throw std::runtime_error("model has no general.architecture");
  }
  const std::string prefix = architecture + ".";

  Qwen2Config config;
  config.block_count = static_cast<int>(file.get_u64(prefix + "block_count", 0));
  config.embedding_length =
      static_cast<int>(file.get_u64(prefix + "embedding_length", 0));
  config.feed_forward_length =
      static_cast<int>(file.get_u64(prefix + "feed_forward_length", 0));
  config.head_count =
      static_cast<int>(file.get_u64(prefix + "attention.head_count", 0));
  config.head_count_kv = static_cast<int>(
      file.get_u64(prefix + "attention.head_count_kv", config.head_count));
  config.context_length =
      static_cast<int>(file.get_u64(prefix + "context_length", 0));
  config.rms_epsilon = static_cast<float>(
      file.get_f64(prefix + "layer_norm_rms_epsilon", 1e-6));
  config.rope_theta =
      static_cast<float>(file.get_f64(prefix + "rope.freq_base", 10000.0));

  // The tokenizer metadata names the ids that end a sequence. A file without
  // them keeps -1, which never matches a real token.
  const auto read_token_id = [&file](const char* key) {
    const gguf::Value* value = file.find(key);
    return value == nullptr ? -1 : static_cast<int>(value->as_u64());
  };
  config.eos_token_id = read_token_id("tokenizer.ggml.eos_token_id");
  config.eot_token_id = read_token_id("tokenizer.ggml.eot_token_id");

  const auto add_end = [&config](int id) {
    if (id < 0) {
      return;
    }
    for (const int existing : config.end_token_ids) {
      if (existing == id) {
        return;
      }
    }
    config.end_token_ids.push_back(id);
  };
  add_end(config.eos_token_id);
  add_end(config.eot_token_id);
  if (const gguf::Value* ids = file.find("tokenizer.ggml.eos_token_ids");
      ids != nullptr && ids->type == gguf::ValueType::Array) {
    for (const gguf::Value& value : ids->elements) {
      add_end(static_cast<int>(value.as_u64()));
    }
  }
  // Qwen uses <|endoftext|> as BOS, padding and a valid end-of-text token, so
  // a generation that emits it has ended even when it is not the eos id.
  if (architecture.find("qwen") != std::string::npos) {
    if (const gguf::Value* bos = file.find("tokenizer.ggml.bos_token_id");
        bos != nullptr) {
      add_end(static_cast<int>(bos->as_u64()));
    }
  }

  if (config.block_count <= 0 || config.embedding_length <= 0 ||
      config.head_count <= 0 || config.head_count_kv <= 0) {
    throw std::runtime_error("model configuration is incomplete");
  }
  config.head_dim = static_cast<int>(file.get_u64(
      prefix + "attention.key_length",
      config.embedding_length / config.head_count));

  // Grouped-query attention divides the query heads into whole KV groups, and
  // every kernel needs a usable head width. A malformed file is rejected here
  // instead of dividing by zero or silently mis-grouping later.
  if (config.head_dim <= 0) {
    throw std::runtime_error("model reports no usable head dimension");
  }
  if (config.head_count % config.head_count_kv != 0) {
    throw std::runtime_error(
        "attention head count is not a multiple of the KV head count");
  }

  // The vocabulary size is the output projection's row count.
  if (const gguf::TensorInfo* output = file.tensor("output.weight");
      output != nullptr && output->shape.size() == 2) {
    config.vocab_size = static_cast<int>(output->shape[1]);
  }
  return config;
}

std::unique_ptr<Qwen2> Qwen2::load(const std::string& path, std::string& error,
                                   int context_limit, bool eager_vocabulary) {
  std::unique_ptr<Qwen2> model(new Qwen2());
  try {
    // The body needs no metadata arrays; building the vocabulary here would cost
    // more than mapping the weights. Unless the caller says it will want them.
    model->file_ = gguf::File::open(path, eager_vocabulary);
    model->config_ = read_config(model->file_);
    Qwen2Config& config = model->config_;

    model->token_embd_ = model->matrix("token_embd.weight");
    model->output_ = model->matrix("output.weight");
    model->output_norm_ = model->vector("output_norm.weight",
                                        config.embedding_length);

    const int model_context =
        config.context_length > 0 ? config.context_length : kDefaultContext;
    const int requested = context_limit > 0 ? context_limit : kDefaultContext;
    model->context_ = std::min(model_context, std::min(requested, kHardMaxContext));
    if (model->context_ < 1) {
      throw std::runtime_error("model reports no usable context length");
    }

    // The same rule `tge plan` reports, applied rather than printed: an
    // over-large context shrinks to what the budget supports instead of failing
    // on allocation after the work has started.
    const ModelPlan plan = analyse(model->file_, model->context_);
    const PlacementDecision decision =
        place(plan, static_cast<std::int64_t>(memory::MemoryBudget{}.snapshot().usable),
              model->context_, 1);
    if (decision.max_context > 0 && decision.max_context < model->context_) {
      model->context_ = decision.max_context;
    }

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
    const std::size_t prefill_rows = static_cast<std::size_t>(
        std::min(kPrefillChunk, model->context_));
    // Attention writes one score slice per parallel task: one per query head
    // for decode, one per chunk token for prefill.
    const std::size_t score_rows =
        std::max(static_cast<std::size_t>(config.head_count), prefill_rows);
    model->scores_.assign(score_rows * static_cast<std::size_t>(model->context_),
                          0.0F);
    const int kv_heads = config.head_count_kv;
    const int group = config.head_count / kv_heads;
    const std::size_t gqa_states = static_cast<std::size_t>(kv_heads) *
                                   kMaxGqaChunks *
                                   static_cast<std::size_t>(group);
    model->group_maxima_.assign(gqa_states, 0.0F);
    model->group_sums_.assign(gqa_states, 0.0F);
    model->group_acc_.assign(
        gqa_states * static_cast<std::size_t>(config.head_dim), 0.0F);
    model->reduction_acc_.assign(
        static_cast<std::size_t>(config.head_dim), 0.0F);
    model->batch_x_.assign(prefill_rows * static_cast<std::size_t>(dim), 0.0F);
    model->batch_h_.assign(prefill_rows * static_cast<std::size_t>(dim), 0.0F);
    model->batch_q_.assign(prefill_rows * static_cast<std::size_t>(dim), 0.0F);
    model->batch_attn_.assign(prefill_rows * static_cast<std::size_t>(dim), 0.0F);
    model->batch_k_.assign(prefill_rows * static_cast<std::size_t>(kv_dim), 0.0F);
    model->batch_v_.assign(prefill_rows * static_cast<std::size_t>(kv_dim), 0.0F);
    model->batch_gate_.assign(
        prefill_rows * static_cast<std::size_t>(config.feed_forward_length), 0.0F);
    model->batch_up_.assign(
        prefill_rows * static_cast<std::size_t>(config.feed_forward_length), 0.0F);

    // Every layer and head rotates by the same angle at a given position, so
    // the trigonometry is done once here instead of per head per layer.
    const int half = config.head_dim / 2;
    std::vector<float> frequencies(static_cast<std::size_t>(half));
    for (int index = 0; index < half; ++index) {
      frequencies[static_cast<std::size_t>(index)] =
          std::pow(config.rope_theta,
                   -2.0F * static_cast<float>(index) /
                       static_cast<float>(config.head_dim));
    }
    model->rope_cos_.resize(static_cast<std::size_t>(model->context_) *
                            static_cast<std::size_t>(half));
    model->rope_sin_.resize(model->rope_cos_.size());
    for (int position = 0; position < model->context_; ++position) {
      const std::size_t base =
          static_cast<std::size_t>(position) * static_cast<std::size_t>(half);
      for (int index = 0; index < half; ++index) {
        const float angle = static_cast<float>(position) *
                            frequencies[static_cast<std::size_t>(index)];
        model->rope_cos_[base + static_cast<std::size_t>(index)] =
            std::cos(angle);
        model->rope_sin_[base + static_cast<std::size_t>(index)] =
            std::sin(angle);
      }
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

  // The Q8 kernel works in whole 32-element blocks and the K-quants in whole
  // 256-element super-blocks; a row that is not a multiple would silently lose
  // its tail.
  const int block = gguf::type_block_size(result.type);
  if (block == 0 || result.cols % block != 0) {
    throw std::runtime_error(std::string("row width is not a whole ") +
                             gguf::type_name(result.type) + " block: " + name);
  }
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
  cpu::QuantKind kind;
  if (to_quant_kind(weight.type, kind)) {
    cpu::DequantBatchArgs args;
    args.data = weight.data;
    args.kind = kind;
    args.rows = weight.rows;
    args.cols = weight.cols;
    args.input = input;
    args.tokens = 1;
    args.out = output;
    kernels().gemv_dequant_batch(args, config);
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

void Qwen2::batch_matvec(const Matrix& weight, const float* input,
                         float* output, int tokens) const {
  if (tokens <= 0) {
    return;
  }
  // Prompt processing is compute-bound, so it takes every logical processor
  // rather than the smaller decode pool.
  const int workers = threads_ > 0 ? threads_ : cpu::resolve_thread_count(0);
  const cpu::GemvConfig config{workers};

  if (weight.type == gguf::TensorType::Q8_0) {
    cpu::Q8BatchArgs args;
    args.data = weight.data;
    args.rows = weight.rows;
    args.cols = weight.cols;
    args.input = input;
    args.tokens = tokens;
    args.out = output;
    kernels().gemv_q8_batch(args, config);
    return;
  }
  cpu::QuantKind kind;
  if (to_quant_kind(weight.type, kind)) {
    cpu::DequantBatchArgs args;
    args.data = weight.data;
    args.kind = kind;
    args.rows = weight.rows;
    args.cols = weight.cols;
    args.input = input;
    args.tokens = tokens;
    args.out = output;
    kernels().gemv_dequant_batch(args, config);
    return;
  }
  if (weight.type == gguf::TensorType::F32) {
    cpu::GemvArgs args;
    args.rows = weight.rows;
    args.cols = weight.cols;
    args.matrix = reinterpret_cast<const float*>(weight.data);
    args.ldm = weight.cols;
    for (int token = 0; token < tokens; ++token) {
      args.vector = input + static_cast<std::size_t>(token) * weight.cols;
      args.out = output + static_cast<std::size_t>(token) * weight.rows;
      kernels().gemv_f32(args, config);
    }
    return;
  }
  throw std::runtime_error("batch_matvec: unsupported weight type");
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

  // The embedding table can be any supported format, and "Q4_K_M" deliberately
  // mixes them, so the row stride has to come from the type rather than from an
  // assumed Q8_0 block.
  cpu::QuantKind kind = cpu::QuantKind::Q8_0;
  switch (weight.type) {
    case gguf::TensorType::Q8_0:
      kind = cpu::QuantKind::Q8_0;
      break;
    default:
      if (!to_quant_kind(weight.type, kind)) {
        throw std::runtime_error("embedding has an unsupported type");
      }
      break;
  }

  const std::size_t row_bytes = cpu::quant_row_bytes(kind, weight.cols);
  cpu::dequantize_row(kind,
                      weight.data + static_cast<std::size_t>(row) * row_bytes,
                      output, weight.cols);
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
  const std::size_t base =
      static_cast<std::size_t>(position) * static_cast<std::size_t>(half);
  const float* cosines = rope_cos_.data() + base;
  const float* sines = rope_sin_.data() + base;
  for (int head = 0; head < heads; ++head) {
    float* vector = values + static_cast<std::size_t>(head) * config_.head_dim;
    for (int index = 0; index < half; ++index) {
      const float cosine = cosines[index];
      const float sine = sines[index];
      // Qwen2 rotates the first half of the head against the second half
      // (NeoX layout), not adjacent pairs.
      const float low = vector[index];
      const float high = vector[index + half];
      vector[index] = low * cosine - high * sine;
      vector[index + half] = low * sine + high * cosine;
    }
  }
}

void Qwen2::attention_head(int layer, int position, int head,
                           const float* query, float* out,
                           float* scores) const {
  const int head_dim = config_.head_dim;
  const int kv_dim = config_.kv_dim();
  const int group = config_.head_count / config_.head_count_kv;
  const int kv_head = head / group;
  const std::size_t kv_head_offset =
      static_cast<std::size_t>(kv_head) * head_dim;
  const std::size_t layer_base =
      static_cast<std::size_t>(layer) * context_ * kv_dim;

  cpu::AttentionArgs args;
  args.query = query;
  args.keys = k_cache_.data() + layer_base + kv_head_offset;
  args.values = v_cache_.data() + layer_base + kv_head_offset;
  args.kv_stride = kv_dim;
  args.span = position + 1;
  args.head_dim = head_dim;
  args.scale = 1.0F / std::sqrt(static_cast<float>(head_dim));
  args.scores = scores;
  args.out = out;

  if (attention_mode_ == AttentionMode::Scalar) {
    cpu::attention_head_scalar(args);
  } else {
    kernels().attention_head(args);
  }
}

void Qwen2::attention(int layer, int position, const float* queries,
                      float* out_buffer) {
  const int heads = config_.head_count;
  const int head_dim = config_.head_dim;
  const int threads = threads_ > 0 ? threads_ : cpu::resolve_decode_threads();

  const bool use_gqa =
      attention_mode_ == AttentionMode::Gqa ||
      (attention_mode_ == AttentionMode::Auto && position >= 256);
  if (use_gqa && threads > 1) {
    attention_gqa(layer, position, queries, out_buffer);
    return;
  }

  // At short context the pool round trip costs more than the work, so keep the
  // heads inline until there is enough history to pay for it.
  if (threads <= 1 || position < 64) {
    for (int head = 0; head < heads; ++head) {
      attention_head(layer, position, head,
                     queries + static_cast<std::size_t>(head) * head_dim,
                     out_buffer + static_cast<std::size_t>(head) * head_dim,
                     scores_.data());
    }
    return;
  }

  core::ThreadPool& pool = core::acquire_pool(threads);
  pool.parallel_for(heads, 1, [&](int head) {
    attention_head(layer, position, head,
                   queries + static_cast<std::size_t>(head) * head_dim,
                   out_buffer + static_cast<std::size_t>(head) * head_dim,
                   scores_.data() +
                       static_cast<std::size_t>(head) * context_);
  });
}

void Qwen2::attention_gqa(int layer, int position, const float* queries,
                          float* out) {
  const int heads = config_.head_count;
  const int kv_heads = config_.head_count_kv;
  const int head_dim = config_.head_dim;
  const int kv_dim = config_.kv_dim();
  const int group = heads / kv_heads;
  const int span = position + 1;
  const int threads = threads_ > 0 ? threads_ : cpu::resolve_decode_threads();
  const int chunks =
      std::min(kMaxGqaChunks,
               std::max(1, (threads * 2 + kv_heads - 1) / kv_heads));
  const int chunk_size = (span + chunks - 1) / chunks;
  const int actual_chunks = (span + chunk_size - 1) / chunk_size;
  const std::size_t layer_base =
      static_cast<std::size_t>(layer) * context_ * kv_dim;

  core::ThreadPool& pool = core::acquire_pool(threads);
  pool.parallel_for(kv_heads * actual_chunks, 1, [&](int task) {
    const int kv_head = task / actual_chunks;
    const int chunk = task % actual_chunks;
    const int start = chunk * chunk_size;
    const int count = std::min(chunk_size, span - start);
    const std::size_t state =
        (static_cast<std::size_t>(kv_head) * actual_chunks + chunk) * group;
    const std::size_t head_offset =
        static_cast<std::size_t>(kv_head) * head_dim;
    const std::size_t row_offset = static_cast<std::size_t>(start) * kv_dim;

    cpu::AttentionGroupArgs args;
    args.queries =
        queries + static_cast<std::size_t>(kv_head) * group * head_dim;
    args.keys = k_cache_.data() + layer_base + head_offset + row_offset;
    args.values = v_cache_.data() + layer_base + head_offset + row_offset;
    args.kv_stride = kv_dim;
    args.count = count;
    args.head_dim = head_dim;
    args.group_heads = group;
    args.scale = 1.0F / std::sqrt(static_cast<float>(head_dim));
    args.maxima = group_maxima_.data() + state;
    args.sums = group_sums_.data() + state;
    args.accumulators = group_acc_.data() + state * head_dim;
    kernels().attention_group(args);
  });

  // Merge the per-chunk online-softmax states for each query head.
  for (int head = 0; head < heads; ++head) {
    const int kv_head = head / group;
    const int group_head = head % group;
    bool initialized = false;
    float maximum = -INFINITY;
    float sum = 0.0F;
    for (int index = 0; index < head_dim; ++index) {
      reduction_acc_[static_cast<std::size_t>(index)] = 0.0F;
    }

    for (int chunk = 0; chunk < actual_chunks; ++chunk) {
      const std::size_t state =
          (static_cast<std::size_t>(kv_head) * actual_chunks + chunk) * group +
          static_cast<std::size_t>(group_head);
      const float chunk_max = group_maxima_[state];
      const float chunk_sum = group_sums_[state];
      if (!(chunk_sum > 0.0F)) {
        continue;
      }
      const float* chunk_acc = group_acc_.data() + state * head_dim;
      if (!initialized) {
        maximum = chunk_max;
        sum = chunk_sum;
        for (int index = 0; index < head_dim; ++index) {
          reduction_acc_[static_cast<std::size_t>(index)] = chunk_acc[index];
        }
        initialized = true;
        continue;
      }

      const float merged_max = std::max(maximum, chunk_max);
      const float alpha = std::exp(maximum - merged_max);
      const float beta = std::exp(chunk_max - merged_max);
      sum = sum * alpha + chunk_sum * beta;
      for (int index = 0; index < head_dim; ++index) {
        reduction_acc_[static_cast<std::size_t>(index)] =
            reduction_acc_[static_cast<std::size_t>(index)] * alpha +
            chunk_acc[index] * beta;
      }
      maximum = merged_max;
    }

    float* head_out = out + static_cast<std::size_t>(head) * head_dim;
    if (initialized && sum > 0.0F) {
      const float inverse = 1.0F / sum;
      for (int index = 0; index < head_dim; ++index) {
        head_out[index] =
            reduction_acc_[static_cast<std::size_t>(index)] * inverse;
      }
    } else {
      for (int index = 0; index < head_dim; ++index) {
        head_out[index] = 0.0F;
      }
    }
  }
}

void Qwen2::attention_batch(int layer, int start_position, int tokens) {
  const int heads = config_.head_count;
  const int head_dim = config_.head_dim;
  const int dim = config_.embedding_length;
  const int threads = threads_ > 0 ? threads_ : cpu::resolve_thread_count(0);

  core::ThreadPool& pool = core::acquire_pool(threads);
  pool.parallel_for(tokens, 1, [&](int token) {
    const int position = start_position + token;
    const float* queries =
        batch_q_.data() + static_cast<std::size_t>(token) * dim;
    float* out = batch_attn_.data() + static_cast<std::size_t>(token) * dim;
    float* scores =
        scores_.data() + static_cast<std::size_t>(token) * context_;
    for (int head = 0; head < heads; ++head) {
      attention_head(layer, position, head,
                     queries + static_cast<std::size_t>(head) * head_dim,
                     out + static_cast<std::size_t>(head) * head_dim, scores);
    }
  });
}

const float* Qwen2::forward(int token, int position) {
  // Guard the public entry point: an out-of-range position would write past the
  // KV cache, and an out-of-range token would read past the embedding table.
  if (token < 0 || token >= config_.vocab_size) {
    throw std::runtime_error("forward: token outside the vocabulary");
  }
  if (position < 0 || position >= context_) {
    throw std::runtime_error("forward: position outside the allocated context");
  }

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

    attention(index, position, q_.data(), attn_.data());

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

const float* Qwen2::prefill(const int* tokens, int count, int start_position) {
  if (tokens == nullptr || count <= 0) {
    throw std::runtime_error("prefill: empty prompt");
  }
  if (start_position < 0 || start_position + count > context_) {
    throw std::runtime_error("prefill: prompt does not fit in the context");
  }
  for (int index = 0; index < count; ++index) {
    if (tokens[index] < 0 || tokens[index] >= config_.vocab_size) {
      throw std::runtime_error("prefill: token outside the vocabulary");
    }
  }

  const int dim = config_.embedding_length;
  const int kv_dim = config_.kv_dim();
  const int ffn = config_.feed_forward_length;
  const int capacity = static_cast<int>(batch_x_.size() / static_cast<std::size_t>(dim));

  int consumed = 0;
  int last_rows = 0;
  while (consumed < count) {
    const int chunk = std::min(capacity, count - consumed);
    const int* chunk_tokens = tokens + consumed;
    const int chunk_start = start_position + consumed;
    last_rows = chunk;

    for (int token = 0; token < chunk; ++token) {
      dequant_row(token_embd_, chunk_tokens[token],
                  batch_x_.data() + static_cast<std::size_t>(token) * dim);
    }

    for (int index = 0; index < config_.block_count; ++index) {
      const Layer& layer = layers_[static_cast<std::size_t>(index)];

      for (int token = 0; token < chunk; ++token) {
        rms_norm(batch_h_.data() + static_cast<std::size_t>(token) * dim,
                 batch_x_.data() + static_cast<std::size_t>(token) * dim,
                 layer.attn_norm);
      }

      batch_matvec(layer.wq, batch_h_.data(), batch_q_.data(), chunk);
      batch_matvec(layer.wk, batch_h_.data(), batch_k_.data(), chunk);
      batch_matvec(layer.wv, batch_h_.data(), batch_v_.data(), chunk);

      for (int token = 0; token < chunk; ++token) {
        float* q = batch_q_.data() + static_cast<std::size_t>(token) * dim;
        float* k = batch_k_.data() + static_cast<std::size_t>(token) * kv_dim;
        float* v = batch_v_.data() + static_cast<std::size_t>(token) * kv_dim;
        for (int axis = 0; axis < dim; ++axis) {
          q[axis] += layer.q_bias[axis];
        }
        for (int axis = 0; axis < kv_dim; ++axis) {
          k[axis] += layer.k_bias[axis];
          v[axis] += layer.v_bias[axis];
        }

        const int position = chunk_start + token;
        apply_rope(q, config_.head_count, position);
        apply_rope(k, config_.head_count_kv, position);

        const std::size_t slot =
            static_cast<std::size_t>(index) * context_ * kv_dim +
            static_cast<std::size_t>(position) * kv_dim;
        std::memcpy(k_cache_.data() + slot, k,
                    sizeof(float) * static_cast<std::size_t>(kv_dim));
        std::memcpy(v_cache_.data() + slot, v,
                    sizeof(float) * static_cast<std::size_t>(kv_dim));
      }

      // Every key and value in this chunk is in the cache before attention
      // runs, so each token sees exactly the causal prefix it should, and the
      // tokens can be attended in parallel.
      attention_batch(index, chunk_start, chunk);

      batch_matvec(layer.wo, batch_attn_.data(), batch_h_.data(), chunk);
      for (int token = 0; token < chunk; ++token) {
        float* x = batch_x_.data() + static_cast<std::size_t>(token) * dim;
        const float* residual =
            batch_h_.data() + static_cast<std::size_t>(token) * dim;
        for (int axis = 0; axis < dim; ++axis) {
          x[axis] += residual[axis];
        }
      }

      for (int token = 0; token < chunk; ++token) {
        rms_norm(batch_h_.data() + static_cast<std::size_t>(token) * dim,
                 batch_x_.data() + static_cast<std::size_t>(token) * dim,
                 layer.ffn_norm);
      }

      batch_matvec(layer.w_gate, batch_h_.data(), batch_gate_.data(), chunk);
      batch_matvec(layer.w_up, batch_h_.data(), batch_up_.data(), chunk);
      const std::size_t activation_elements =
          static_cast<std::size_t>(chunk) * static_cast<std::size_t>(ffn);
      for (std::size_t position = 0; position < activation_elements; ++position) {
        batch_gate_[position] = silu(batch_gate_[position]) * batch_up_[position];
      }

      batch_matvec(layer.w_down, batch_gate_.data(), batch_h_.data(), chunk);
      for (int token = 0; token < chunk; ++token) {
        float* x = batch_x_.data() + static_cast<std::size_t>(token) * dim;
        const float* residual =
            batch_h_.data() + static_cast<std::size_t>(token) * dim;
        for (int axis = 0; axis < dim; ++axis) {
          x[axis] += residual[axis];
        }
      }
    }

    consumed += chunk;
  }

  // Only the final prompt token needs logits to seed decoding.
  const std::size_t last = static_cast<std::size_t>(last_rows - 1) * dim;
  std::memcpy(x_.data(), batch_x_.data() + last,
              sizeof(float) * static_cast<std::size_t>(dim));
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
