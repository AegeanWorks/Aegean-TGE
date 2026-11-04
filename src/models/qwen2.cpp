#include "tge/models/qwen2.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cctype>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>

#include "tge/core/dispatch.hpp"
#include "tge/core/thread_pool.hpp"
#include "tge/gpu/backend.hpp"
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

inline std::uint64_t clock_ns() {
  return static_cast<std::uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(
          std::chrono::steady_clock::now().time_since_epoch())
          .count());
}

/// Quantizes one 32-value KV row slice to int8 with a single float scale.
/// Same recipe as the Q8 batch path (max / 127, clamped to [-127, 127]) so
/// the attention kernels share one dequantization idiom.
inline float quantize_kv_block(const float* source, std::int8_t* destination) {
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

/// Writes one position's K and V vectors to the int8 cache with per-32
/// scales. `slot` is the element offset of the position; the scale slot is
/// the same offset divided by 32 (kv_dim is always a multiple of 32).
inline void store_kv_q8(const float* k, const float* v, int kv_dim,
                         std::size_t slot, std::int8_t* k_out,
                         std::int8_t* v_out, float* k_scales, float* v_scales) {
  const int blocks = kv_dim / 32;
  const std::size_t scale_slot = slot / 32;
  for (int block = 0; block < blocks; ++block) {
    k_scales[scale_slot + block] = quantize_kv_block(
        k + block * 32, k_out + slot + block * 32);
    v_scales[scale_slot + block] = quantize_kv_block(
        v + block * 32, v_out + slot + block * 32);
  }
}

/// Maps a supported weight type onto the kernel's quant kind. Q8_0 has fused
/// kernels of its own, so it is not routed through this path.
bool to_quant_kind(gguf::TensorType type, cpu::QuantKind& kind) {  switch (type) {
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
    case gguf::TensorType::Q5_1:
      kind = cpu::QuantKind::Q5_1;
      return true;
    default:
      return false;
  }
}

}  // namespace

void Qwen2::set_sparse_config(const SparseConfig& config) {
  if (config.enabled && config.stride < 2) {
    throw std::runtime_error("sparse: stride must be >= 2");
  }
  if (config.enabled && (config.sink < 0 || config.window < 1)) {
    throw std::runtime_error("sparse: sink must be >= 0 and window >= 1");
  }
  sparse_ = config;
}

std::vector<int> build_sparse_index(int span, const SparseConfig& config) {
  std::vector<int> kept;
  if (span <= 0) {
    return kept;
  }
  if (!config.enabled || span <= config.sink + config.window) {
    kept.resize(static_cast<std::size_t>(span));
    for (int position = 0; position < span; ++position) {
      kept[static_cast<std::size_t>(position)] = position;
    }
    return kept;
  }
  const int stride = config.stride < 2 ? 2 : config.stride;
  kept.reserve(static_cast<std::size_t>(config.sink + config.window +
                                        span / stride + 1));
  const int sink_end = std::min(config.sink, span);
  for (int position = 0; position < sink_end; ++position) {
    kept.push_back(position);
  }
  const int window_start = span - config.window;
  for (int position = sink_end; position < window_start; position += stride) {
    kept.push_back(position);
  }
  for (int position = std::max(window_start, sink_end); position < span;
       ++position) {
    kept.push_back(position);
  }
  return kept;
}

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
      file.get_f64(prefix + "attention.layer_norm_rms_epsilon", -1.0));
  if (config.rms_epsilon < 0.0F) {
    // Older files (and the first Qwen2 exports) name the key without the
    // `attention.` infix; the value is the same contract either way.
    config.rms_epsilon = static_cast<float>(
        file.get_f64(prefix + "layer_norm_rms_epsilon", 1e-6));
  }
  config.rope_theta =
      static_cast<float>(file.get_f64(prefix + "rope.freq_base", 10000.0));
  // Interleaved RoPE pairs consecutive elements (ggml NORM); NeoX pairs the
  // halves. Only families verified against the reference enable it here.
  std::string arch_lower = architecture;
  for (char& letter : arch_lower) {
    letter =
        static_cast<char>(std::tolower(static_cast<unsigned char>(letter)));
  }
  config.rope_interleaved = arch_lower.find("llama") != std::string::npos;

  // YaRN magnitude scale (`rope.scaling.attn_factor`, Phi-3's 1.19024):
  // the reference multiplies every cos/sin entry by it. Absent key means
  // 1.0, which leaves every existing model bit-identical.
  config.rope_attn_factor = static_cast<float>(
      file.get_f64(prefix + "rope.scaling.attn_factor", 1.0));
  // Long/short factor selection pivots on the original context length;
  // 0 falls back to the file context at load, mirroring n_ctx_train.
  config.rope_orig_context = static_cast<int>(
      file.get_u64(prefix + "rope.scaling.original_context_length", 0));

  // Only the no-blend scalings run here: linear, su/longrope (explicit
  // per-dim factors, possibly absent), or no key at all. Classic YaRN
  // interpolation blending (ext_factor) has no path yet and stops loudly
  // instead of running unblended.
  const std::string scaling_type =
      file.get_string(prefix + "rope.scaling.type", "");
  if (!scaling_type.empty() && scaling_type != "linear" &&
      scaling_type != "su" && scaling_type != "longrope") {
    throw std::runtime_error("unsupported rope scaling type: " + scaling_type);
  }

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
  if (config.embedding_length % config.head_count != 0) {
    throw std::runtime_error(
        "embedding length is not a multiple of the head count");
  }

  // A file that rotates only part of each head (a `rope.dimension_count`
  // below the head width) would silently mis-rotate under the full-head
  // tables built at load. Every file seen so far rotates the whole head;
  // anything else stops here until a partial-RoPE path exists.
  const int rope_dim = static_cast<int>(
      file.get_u64(prefix + "rope.dimension_count", 0));
  if (rope_dim > 0 && rope_dim != config.head_dim) {
    throw std::runtime_error("partial RoPE is not supported");
  }

  // YaRN factor tables ride along in phi3 files. The tables are recorded so
  // a later long-context path can price them; short runs ignore them the
  // same way the harness prices every other unapplied trade.
  config.has_yarn_factors =
      file.tensor("rope_factors_short.weight") != nullptr ||
      file.tensor("rope_factors_long.weight") != nullptr;

  // The vocabulary size is the output projection's row count. With tied
  // word embeddings there is no output tensor, so fall back to the
  // embedding table's row count (same layout, verified row-per-token).
  if (const gguf::TensorInfo* output = file.tensor("output.weight");
      output != nullptr && output->shape.size() == 2) {
    config.vocab_size = static_cast<int>(output->shape[1]);
  } else if (const gguf::TensorInfo* embd = file.tensor("token_embd.weight");
             embd != nullptr && embd->shape.size() == 2) {
    config.vocab_size = static_cast<int>(embd->shape[1]);
  }
  return config;
}

std::unique_ptr<Qwen2> Qwen2::load(const std::string& path, std::string& error,
                                   int context_limit, bool eager_vocabulary,
                                   bool kv_half, bool kv_q8, int sequences) {
  std::unique_ptr<Qwen2> model(new Qwen2());
  try {
    // The body needs no metadata arrays; building the vocabulary here would cost
    // more than mapping the weights. Unless the caller says it will want them.
    model->file_ = gguf::File::open(path, eager_vocabulary);
    model->config_ = read_config(model->file_);
    Qwen2Config& config = model->config_;

    model->token_embd_ = model->matrix("token_embd.weight");
    if (model->file_.tensor("output.weight") != nullptr) {
      model->output_ = model->matrix("output.weight");
    } else {
      // Tied word embeddings (e.g. Llama-3.2): the file ships no separate
      // output projection and the token embedding rows double as output
      // rows. GGUF stores embeddings row-per-token, exactly the layout the
      // output matvec expects, so the matrix aliases the same bytes.
      model->output_ = model->token_embd_;
    }
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
    if (!decision.weights_fit) {
      throw std::runtime_error("plan: " + decision.reason);
    }
    if (decision.max_context == 0) {
      throw std::runtime_error("plan: " + decision.reason);
    }
    if (decision.max_context < model->context_) {
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
      if (model->file_.tensor(make("attn_q.weight").c_str()) != nullptr) {
        layer.wq = model->matrix(make("attn_q.weight").c_str());
        layer.wk = model->matrix(make("attn_k.weight").c_str());
        layer.wv = model->matrix(make("attn_v.weight").c_str());
      } else {
        // Phi-3 fuses the triple: one `attn_qkv` tensor holds the query rows,
        // then the key rows, then the value rows. The query block is a full
        // embedding wide; each key/value block is one KV group wide. The
        // split is by construction, so a fused tensor of any other geometry
        // stops the load instead of running mis-sliced.
        const std::string fused = make("attn_qkv.weight");
        const gguf::TensorInfo* fused_info = model->file_.tensor(fused);
        if (fused_info == nullptr || fused_info->shape.size() != 2 ||
            static_cast<int>(fused_info->shape[0]) != config.embedding_length ||
            static_cast<int>(fused_info->shape[1]) !=
                config.embedding_length + 2 * config.kv_dim()) {
          throw std::runtime_error(
              "block " + std::to_string(index) +
              " has neither attn_q.weight nor a fused attn_qkv.weight");
        }
        layer.wq =
            model->matrix_slice(fused.c_str(), 0, config.embedding_length);
        layer.wk = model->matrix_slice(fused.c_str(), config.embedding_length,
                                       config.kv_dim());
        layer.wv = model->matrix_slice(
            fused.c_str(), config.embedding_length + config.kv_dim(),
            config.kv_dim());
      }
      layer.wo = model->matrix(make("attn_output.weight").c_str());
      if (model->file_.tensor(make("ffn_gate.weight").c_str()) != nullptr) {
        layer.w_gate = model->matrix(make("ffn_gate.weight").c_str());
        layer.w_up = model->matrix(make("ffn_up.weight").c_str());
      } else {
        // Phi-3 fuses the pair: one `ffn_up` tensor holds the gate rows
        // followed by the up rows (the HuggingFace `gate_up_proj` order the
        // GGUF converter preserves). Anything that is not exactly two
        // feed-forward blocks stops here instead of guessing.
        const std::string fused = make("ffn_up.weight");
        const gguf::TensorInfo* fused_info = model->file_.tensor(fused);
        if (fused_info == nullptr || fused_info->shape.size() != 2 ||
            static_cast<int>(fused_info->shape[1]) !=
                2 * config.feed_forward_length) {
          throw std::runtime_error(
              "block " + std::to_string(index) +
              " has neither ffn_gate.weight nor a fused ffn_up.weight");
        }
        layer.w_gate = model->matrix_slice(fused.c_str(), 0,
                                          config.feed_forward_length);
        layer.w_up = model->matrix_slice(fused.c_str(),
                                        config.feed_forward_length,
                                        config.feed_forward_length);
      }
      layer.w_down = model->matrix(make("ffn_down.weight").c_str());

      // QKV bias is optional: older files (e.g. Llama-3.2) omit it, in which
      // case every layer shares one zero vector instead of faulting.
      // Reading through the same pointers keeps the hot loops branch-free.
      if (model->file_.tensor(make("attn_q.bias").c_str()) != nullptr) {
        layer.q_bias = model->vector(make("attn_q.bias").c_str(),
                                     config.embedding_length);
        layer.k_bias = model->vector(make("attn_k.bias").c_str(),
                                     config.kv_dim());
        layer.v_bias = model->vector(make("attn_v.bias").c_str(),
                                     config.kv_dim());
      } else {
        if (model->zero_bias_.size() !=
            static_cast<std::size_t>(config.embedding_length)) {
          model->zero_bias_.assign(
              static_cast<std::size_t>(config.embedding_length), 0.0F);
        }
        layer.q_bias = model->zero_bias_.data();
        layer.k_bias = model->zero_bias_.data();
        layer.v_bias = model->zero_bias_.data();
      }
    }

    const int dim = config.embedding_length;
    const int kv_dim = config.kv_dim();
    model->sequence_count_ = std::max(1, sequences);
    const std::size_t cache_elements =
        static_cast<std::size_t>(model->sequence_count_) *
        static_cast<std::size_t>(config.block_count) * model->context_ * kv_dim;
    model->kv_half_ = kv_half;
    model->kv_q8_ = kv_q8 && !kv_half;
    if (kv_half) {
      model->k_cache_half_.assign(cache_elements, 0);
      model->v_cache_half_.assign(cache_elements, 0);
    } else if (model->kv_q8_) {
      model->k_cache_q8_.assign(cache_elements, 0);
      model->v_cache_q8_.assign(cache_elements, 0);
      model->k_cache_q8_scales_.assign(cache_elements / 32, 0.0F);
      model->v_cache_q8_scales_.assign(cache_elements / 32, 0.0F);
    } else {
      model->k_cache_.assign(cache_elements, 0.0F);
      model->v_cache_.assign(cache_elements, 0.0F);
    }

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
    model->logits_batch_.assign(
        static_cast<std::size_t>(std::min(kPrefillChunk, model->context_)) *
            static_cast<std::size_t>(config.vocab_size),
        0.0F);
    model->verify_logits_.assign(
        static_cast<std::size_t>(kSpeculativeBatch) *
            static_cast<std::size_t>(config.vocab_size),
        0.0F);
    const std::size_t prefill_rows = static_cast<std::size_t>(
        std::min(kPrefillChunk, model->context_));
    // Attention writes one score slice per parallel task. Decode uses one
    // per query head; prefill uses one per (chunk token, head) so short
    // prompts still fill the pool (4 tokens x 14 heads = 56 tasks).
    const std::size_t score_rows =
        prefill_rows * static_cast<std::size_t>(config.head_count);
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

    // YaRN per-dim factors, mirroring get_rope_factors: the long set serves
    // runs past the original context length, the short set everything below
    // it — and a missing selected set means all-ones, even when the other
    // exists. The Phi export carries neither per-block tensor, so both
    // paths are standard angles; only the magnitude scale below applies.
    {
      const int orig_ctx = config.rope_orig_context > 0
                               ? config.rope_orig_context
                               : config.context_length;
      char probe[96];
      std::snprintf(probe, sizeof(probe), "blk.0.rope_factors_%s.weight",
                    model->context_ > orig_ctx ? "long" : "short");
      const gguf::TensorInfo* factors = model->file_.tensor(probe);
      if (factors != nullptr) {
        if (factors->type != gguf::TensorType::F32 ||
            static_cast<int>(factors->elements) != half) {
          throw std::runtime_error(
              std::string("wrong shape for YaRN factors: ") + probe);
        }
        const float* values = reinterpret_cast<const float*>(
            model->file_.tensor_data(*factors));
        for (int index = 0; index < half; ++index) {
          if (values[index] <= 0.0F) {
            throw std::runtime_error(
                std::string("non-positive YaRN factor: ") + probe);
          }
          frequencies[static_cast<std::size_t>(index)] /= values[index];
        }
      }
    }

    model->rope_cos_.resize(static_cast<std::size_t>(model->context_) *
                            static_cast<std::size_t>(half));
    model->rope_sin_.resize(model->rope_cos_.size());
    // Duplicated copies for the interleaved AVX2 kernel: each coefficient
    // twice in a row so one vector load covers 4 consecutive pairs.
    model->rope_cos_dup_.resize(model->rope_cos_.size() * 2);
    model->rope_sin_dup_.resize(model->rope_cos_.size() * 2);
    for (int position = 0; position < model->context_; ++position) {
      const std::size_t base =
          static_cast<std::size_t>(position) * static_cast<std::size_t>(half);
      for (int index = 0; index < half; ++index) {
        // YaRN magnitude scale: the reference multiplies every table entry
        // by attn_factor (Phi-3's 1.19024), which sharpens attention by its
        // square. 1.0 on every other file, so the multiply is free there.
        const float angle = static_cast<float>(position) *
                            frequencies[static_cast<std::size_t>(index)];
        const float cosine =
            std::cos(angle) * config.rope_attn_factor;
        const float sine =
            std::sin(angle) * config.rope_attn_factor;
        model->rope_cos_[base + static_cast<std::size_t>(index)] = cosine;
        model->rope_sin_[base + static_cast<std::size_t>(index)] = sine;
        model->rope_cos_dup_[(base + static_cast<std::size_t>(index)) * 2] =
            cosine;
        model->rope_cos_dup_[(base + static_cast<std::size_t>(index)) * 2 + 1] =
            cosine;
        model->rope_sin_dup_[(base + static_cast<std::size_t>(index)) * 2] =
            sine;
        model->rope_sin_dup_[(base + static_cast<std::size_t>(index)) * 2 + 1] =
            sine;
      }
    }

    return model;
  } catch (const std::exception& exception) {
    error = exception.what();
    return nullptr;
  }
}

void Qwen2::set_threads(int threads) {
  if (threads == threads_ && decode_pool_ && prefill_pool_) {
    return;
  }
  threads_ = threads;
  decode_pool_.reset();
  prefill_pool_.reset();
}

core::ThreadPool& Qwen2::decode_pool() const {
  if (!decode_pool_) {
    const int size = threads_ > 0 ? threads_ : cpu::resolve_decode_threads();
    decode_pool_ = std::make_unique<core::ThreadPool>(size, pin_threads_);
  }
  return *decode_pool_;
}

core::ThreadPool& Qwen2::prefill_pool() const {
  if (!prefill_pool_) {
    prefill_pool_ =
        std::make_unique<core::ThreadPool>(prefill_threads(), false);
  }
  return *prefill_pool_;
}

int Qwen2::prefill_threads() const {
  if (prefill_threads_override_ > 0) {
    return prefill_threads_override_;
  }
  const int hw = cpu::resolve_thread_count(0);
  // Phase-specific plan: prefill is compute-bound, so it takes every logical
  // core even when the user asked decode down to a few threads for the
  // bandwidth wall. Unpinned: E-cores contribute throughput here and the
  // dynamic work-stealing absorbs their slower pace.
  if (threads_ > 0) {
    return std::max(threads_, hw);
  }
  return hw;
}

void Qwen2::set_prefill_threads(int threads) {
  if (threads == prefill_threads_override_ && prefill_pool_) {
    return;
  }
  prefill_threads_override_ = threads;
  prefill_pool_.reset();
}

void Qwen2::set_pin_threads(bool pin) {
  if (pin == pin_threads_ && decode_pool_) {
    return;
  }
  pin_threads_ = pin;
  decode_pool_.reset();
}

bool Qwen2::wants_gpu_batch(int tokens, gguf::TensorType type) const {
  const bool supported = type == gguf::TensorType::Q8_0 ||
                         type == gguf::TensorType::F32;
  const bool clean = !vnni_ && !scalar_kernels_ &&
                     attention_mode_ != AttentionMode::Scalar;
  return gpu::eligible_batch(use_gpu_requested_, tokens, supported, clean);
}

bool Qwen2::wants_gpu_gqa(int span) const {
  const int threads = threads_ > 0 ? threads_ : cpu::resolve_decode_threads();
  // uses_gqa() already encodes the grouped schedule floor (span >= 256 with
  // threads > 1); the sparse subset keeps its exact gather path and the
  // scalar arm keeps the oracle, so neither ever offloads.
  const bool follows = uses_gqa(span - 1);
  const bool dense = !uses_sparse(span - 1) && !scalar_kernels_ &&
                     attention_mode_ != AttentionMode::Scalar;
  return gpu::eligible_gqa(use_gpu_requested_, span, threads, follows, dense);
}

bool Qwen2::wants_gpu_output(gguf::TensorType type) const {
  const bool supported = type == gguf::TensorType::Q8_0 ||
                         type == gguf::TensorType::F32;
  const bool clean = !vnni_ && !scalar_kernels_ &&
                     attention_mode_ != AttentionMode::Scalar;
  return gpu::eligible_decode_output(use_gpu_requested_, supported, clean);
}

const float* Qwen2::output_logits(float* hidden) {
  const int dim = config_.embedding_length;
  // Q8_0 rides the device today; the F32 twin is unwritten, so F32 weights
  // keep the CPU path even when the wish holds.
  if (output_.type == gguf::TensorType::Q8_0 &&
      wants_gpu_output(output_.type) && gpu::use_gpu()) {
    const std::size_t row_bytes =
        static_cast<std::size_t>(dim / cpu::kQ8BlockElements) *
        cpu::kQ8BlockBytes;
    gpu::GemvQ8Result dispatched = gpu::gemv_q8(
        output_.data, static_cast<std::size_t>(output_.rows) * row_bytes,
        output_.rows, dim, row_bytes, hidden, logits_.data());
    if (dispatched.ok) {
      phase_gpu_ns_ +=
          dispatched.alloc_ns + dispatched.h2d_ns + dispatched.submit_ns +
          dispatched.d2h_ns;
      phase_gpu_bytes_ += gpu::hidden_bytes(dim) +
                          gpu::logits_bytes(config_.vocab_size);
      return logits_.data();
    }
  }
  const cpu::Q8Quantized q_out =
      vnni_ ? quantize_activations(hidden, 1, dim) : cpu::Q8Quantized{};
  matvec(output_, hidden, logits_.data(), vnni_ ? &q_out : nullptr);
  return logits_.data();
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
  if (info->shape[0] == 0 || info->shape[0] > 2147483647ULL ||
      info->shape[1] == 0 || info->shape[1] > 2147483647ULL) {
    throw std::runtime_error(std::string("tensor shape out of range: ") + name);
  }
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

Qwen2::Matrix Qwen2::matrix_slice(const char* name, int row_begin,
                                  int rows) const {
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
  if (info->shape[0] == 0 || info->shape[0] > 2147483647ULL ||
      info->shape[1] == 0 || info->shape[1] > 2147483647ULL) {
    throw std::runtime_error(std::string("tensor shape out of range: ") + name);
  }
  const int total_rows = static_cast<int>(info->shape[1]);
  if (row_begin < 0 || rows <= 0 || row_begin + rows > total_rows) {
    throw std::runtime_error(std::string("row slice is out of range: ") + name);
  }

  Matrix result;
  result.cols = static_cast<int>(info->shape[0]);
  result.rows = rows;
  result.type = info->type;
  // Row stride mirrors the kernels: Q8_0 has no QuantKind mapping (its fused
  // kernels stand alone), F32 is dense, and the K-quants share one helper.
  std::size_t row_bytes = 0;
  if (result.type == gguf::TensorType::F32) {
    row_bytes = static_cast<std::size_t>(result.cols) * sizeof(float);
  } else if (result.type == gguf::TensorType::Q8_0) {
    row_bytes = static_cast<std::size_t>(result.cols / cpu::kQ8BlockElements) *
                cpu::kQ8BlockBytes;
  } else {
    cpu::QuantKind kind;
    if (!to_quant_kind(result.type, kind)) {
      throw std::runtime_error(std::string("unsupported type for ") + name +
                               ": " + gguf::type_name(result.type));
    }
    row_bytes = cpu::quant_row_bytes(kind, result.cols);
  }
  result.data = file_.tensor_data(*info) +
                static_cast<std::size_t>(row_begin) * row_bytes;

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
  if (expected < 0 ||
      info->elements != static_cast<std::uint64_t>(expected)) {
    throw std::runtime_error(std::string("wrong length for ") + name);
  }
  return reinterpret_cast<const float*>(file_.tensor_data(*info));
}

void Qwen2::matvec(const Matrix& weight, const float* input,
                   float* output, const cpu::Q8Quantized* quantized) const {
  const std::uint64_t t0 = clock_ns();
  // Decode is memory-bound and synchronisation-sensitive, so it uses the
  // decode pool rather than one worker per logical processor. Sizing each call
  // to the matrix was measured and gave no gain over a flat count, so the
  // simpler rule stands.
  const int workers =
      threads_ > 0 ? threads_ : cpu::resolve_decode_threads();
  const cpu::GemvConfig config{workers, &decode_pool()};
  if (weight.type == gguf::TensorType::Q8_0) {
    if (vnni_ && quantized != nullptr && quantized->values != nullptr &&
        quantized->scales != nullptr) {
      cpu::Q8BatchArgs args;
      args.data = weight.data;
      args.rows = weight.rows;
      args.cols = weight.cols;
      args.input = input;
      args.tokens = 1;
      args.out = output;
      args.quantized = quantized->values;
      args.quantized_scales = quantized->scales;
      cpu::gemv_q8_batch_vnni(args, config);
      phase_matvec_ns_ += clock_ns() - t0;
      return;
    }
    cpu::Q8GemvArgs args;
    args.data = weight.data;
    args.rows = weight.rows;
    args.cols = weight.cols;
    args.vector = input;
    args.out = output;
    kernels().gemv_q8(args, config);
    phase_matvec_ns_ += clock_ns() - t0;
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
    phase_matvec_ns_ += clock_ns() - t0;
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
    phase_matvec_ns_ += clock_ns() - t0;
    return;
  }
  throw std::runtime_error("matvec: unsupported weight type");
}

void Qwen2::batch_matvec(const Matrix& weight, const float* input,
                         float* output, int tokens,
                         const cpu::Q8Quantized* quantized) const {
  if (tokens <= 0) {
    return;
  }
  const std::uint64_t t0 = clock_ns();
  // Prompt processing is compute-bound, so it takes every logical processor
  // rather than the smaller decode pool (see prefill_threads()).
  const int workers = prefill_threads();
  const cpu::GemvConfig config{workers, &prefill_pool()};

  // Batch offload: one H2D (activations) + one dispatch + one D2H per mat.
  // Q8_0 rides the device today (F32 twin unwritten, K-quants stay CPU by
  // construction of wants_gpu_batch); any backend hiccup falls through to
  // the exact CPU kernels below.
  if (weight.type == gguf::TensorType::Q8_0 &&
      wants_gpu_batch(tokens, weight.type) && gpu::use_gpu()) {
    const std::size_t row_bytes =
        static_cast<std::size_t>(weight.cols / cpu::kQ8BlockElements) *
        cpu::kQ8BlockBytes;
    gpu::GemmQ8Result dispatched = gpu::gemm_q8(
        weight.data, static_cast<std::size_t>(weight.rows) * row_bytes,
        weight.rows, weight.cols, row_bytes, input, output, tokens);
    if (dispatched.ok) {
      phase_gpu_ns_ +=
          dispatched.alloc_ns + dispatched.h2d_ns + dispatched.submit_ns +
          dispatched.d2h_ns;
      phase_gpu_bytes_ +=
          static_cast<std::uint64_t>(tokens) *
              (gpu::hidden_bytes(weight.cols) +
               gpu::logits_bytes(weight.rows));
      return;
    }
  }

  if (weight.type == gguf::TensorType::Q8_0) {
    if (vnni_ && quantized != nullptr && quantized->values != nullptr &&
        quantized->scales != nullptr) {
      cpu::Q8BatchArgs args;
      args.data = weight.data;
      args.rows = weight.rows;
      args.cols = weight.cols;
      args.input = input;
      args.tokens = tokens;
      args.out = output;
      args.quantized = quantized->values;
      args.quantized_scales = quantized->scales;
      cpu::gemv_q8_batch_vnni(args, config);
      phase_matvec_ns_ += clock_ns() - t0;
      return;
    }
    cpu::Q8BatchArgs args;
    args.data = weight.data;
    args.rows = weight.rows;
    args.cols = weight.cols;
    args.input = input;
    args.tokens = tokens;
    args.out = output;
    kernels().gemv_q8_batch(args, config);
    phase_matvec_ns_ += clock_ns() - t0;
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
    // Integer path for prefill-size batches: tokens == 1 keeps the float
    // path (quantizer costs more than it saves, measured 0.8x), and the
    // scalar oracle keeps the exact path (fidelity must price kq8, never
    // assume it) — same `clean` idiom as the GPU/vNNI gates.
    if (kq8_ && !scalar_kernels_ && tokens > 1 &&
        (kind == cpu::QuantKind::Q4_K || kind == cpu::QuantKind::Q6_K)) {
      cpu::gemv_dequant_batch_kq8_avx2(args, config);
    } else {
      kernels().gemv_dequant_batch(args, config);
    }
    phase_matvec_ns_ += clock_ns() - t0;
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
    phase_matvec_ns_ += clock_ns() - t0;
    return;
  }
  throw std::runtime_error("batch_matvec: unsupported weight type");
}

cpu::Q8Quantized Qwen2::quantize_activations(const float* input, int tokens,
                                             int cols) {
  cpu::Q8Quantized result;
  if (!vnni_ || input == nullptr || tokens <= 0 || cols <= 0 ||
      cols % cpu::kQ8BlockElements != 0) {
    return result;
  }
  const std::uint64_t t0 = clock_ns();
  const std::size_t values = static_cast<std::size_t>(tokens) *
                             static_cast<std::size_t>(cols);
  const std::size_t scales =
      static_cast<std::size_t>(tokens) *
      static_cast<std::size_t>(cols / cpu::kQ8BlockElements);
  if (q8_values_.size() < values) {
    q8_values_.resize(values);
  }
  if (q8_scales_.size() < scales) {
    q8_scales_.resize(scales);
  }
  cpu::quantize_q8_batch(input, tokens, cols, q8_values_.data(),
                         q8_scales_.data());
  result.values = q8_values_.data();
  result.scales = q8_scales_.data();
  phase_matvec_ns_ += clock_ns() - t0;
  return result;
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

void Qwen2::reset_phase_timers() const {
  phase_matvec_ns_ = 0;
  phase_attn_ns_ = 0;
  phase_norm_ns_ = 0;
  phase_gpu_ns_ = 0;
  phase_gpu_bytes_ = 0;
}

void Qwen2::rms_norm(float* out, const float* in, const float* weight) const {
  const std::uint64_t t0 = clock_ns();
  const int dim = config_.embedding_length;
  const auto features = cpu::detect().features;
  if (!scalar_kernels_ && features.avx2 && features.fma) {
    cpu::rms_norm_avx2(out, in, weight, dim, config_.rms_epsilon);
    phase_norm_ns_ += clock_ns() - t0;
    return;
  }
  cpu::rms_norm_scalar(out, in, weight, dim, config_.rms_epsilon);
  phase_norm_ns_ += clock_ns() - t0;
}

void Qwen2::apply_rope(float* values, int heads, int position) const {
  const std::uint64_t t0 = clock_ns();
  const int half = config_.head_dim / 2;
  const std::size_t base =
      static_cast<std::size_t>(position) * static_cast<std::size_t>(half);
  const auto features = cpu::detect().features;
  const bool vectorise = !scalar_kernels_ && features.avx2 && features.fma;
  if (config_.rope_interleaved) {
    // Llama-style consecutive pairs; the duplicated tables hold each
    // coefficient twice so one vector load covers four pairs.
    const float* cosines = rope_cos_dup_.data() + base * 2;
    const float* sines = rope_sin_dup_.data() + base * 2;
    if (vectorise) {
      cpu::apply_rope_interleaved_avx2(values, heads, config_.head_dim,
                                       position, cosines, sines);
      phase_norm_ns_ += clock_ns() - t0;
      return;
    }
    cpu::apply_rope_interleaved_scalar(values, heads, config_.head_dim,
                                       position, cosines, sines);
    phase_norm_ns_ += clock_ns() - t0;
    return;
  }
  const float* cosines = rope_cos_.data() + base;
  const float* sines = rope_sin_.data() + base;
  if (vectorise) {
    cpu::apply_rope_avx2(values, heads, config_.head_dim, position, cosines,
                         sines);
    phase_norm_ns_ += clock_ns() - t0;
    return;
  }
  cpu::apply_rope_scalar(values, heads, config_.head_dim, position, cosines,
                         sines);
  phase_norm_ns_ += clock_ns() - t0;
}

void Qwen2::silu_gated(float* gate, const float* up, int count) const {
  const std::uint64_t t0 = clock_ns();
  const auto features = cpu::detect().features;
  if (!scalar_kernels_ && features.avx2 && features.fma) {
    cpu::silu_gated_avx2(gate, up, count);
  } else {
    cpu::silu_gated_scalar(gate, up, count);
  }
  phase_norm_ns_ += clock_ns() - t0;
}

void Qwen2::attention_head(int sequence, int layer, int position, int head,
                           const float* query, float* out,
                           float* scores) const {
  const int head_dim = config_.head_dim;
  const int kv_dim = config_.kv_dim();
  const int group = config_.head_count / config_.head_count_kv;
  const int kv_head = head / group;
  const std::size_t kv_head_offset =
      static_cast<std::size_t>(kv_head) * head_dim;
  const std::size_t layer_base =
      (static_cast<std::size_t>(sequence) * config_.block_count +
       static_cast<std::size_t>(layer)) *
      context_ * kv_dim;

  cpu::AttentionArgs args;
  args.query = query;
  if (kv_half_) {
    args.half = true;
    args.keys_half = k_cache_half_.data() + layer_base + kv_head_offset;
    args.values_half = v_cache_half_.data() + layer_base + kv_head_offset;
  } else if (kv_q8_) {
    args.q8 = true;
    args.keys_q8 = k_cache_q8_.data() + layer_base + kv_head_offset;
    args.values_q8 = v_cache_q8_.data() + layer_base + kv_head_offset;
    args.keys_q8_scales = k_cache_q8_scales_.data() +
                          (layer_base + kv_head_offset) / 32;
    args.values_q8_scales = v_cache_q8_scales_.data() +
                            (layer_base + kv_head_offset) / 32;
  } else {
    args.keys = k_cache_.data() + layer_base + kv_head_offset;
    args.values = v_cache_.data() + layer_base + kv_head_offset;
  }
  args.kv_stride = kv_dim;
  args.span = position + 1;
  args.head_dim = head_dim;
  args.scale = 1.0F / std::sqrt(static_cast<float>(head_dim));
  args.scores = scores;
  args.out = out;

  // Static sparse subset (f32 KV only, never the Scalar oracle): gather the
  // kept rows into thread-local scratch and run the same kernel over them.
  // The span fits check keeps the exact dense path bit-identical below it.
  if (sparse_.enabled && !kv_half_ && !kv_q8_ &&
      attention_mode_ != AttentionMode::Scalar &&
      args.span > sparse_.sink + sparse_.window) {
    const std::vector<int> kept = build_sparse_index(args.span, sparse_);
    thread_local std::vector<float> sparse_keys;
    thread_local std::vector<float> sparse_values;
    const std::size_t kept_count = kept.size();
    sparse_keys.resize(kept_count * static_cast<std::size_t>(head_dim));
    sparse_values.resize(kept_count * static_cast<std::size_t>(head_dim));
    for (std::size_t kept_index = 0; kept_index < kept_count; ++kept_index) {
      const float* key_row =
          args.keys + static_cast<std::size_t>(kept[kept_index]) * kv_dim;
      const float* value_row =
          args.values + static_cast<std::size_t>(kept[kept_index]) * kv_dim;
      std::memcpy(sparse_keys.data() + kept_index * head_dim, key_row,
                  sizeof(float) * static_cast<std::size_t>(head_dim));
      std::memcpy(sparse_values.data() + kept_index * head_dim, value_row,
                  sizeof(float) * static_cast<std::size_t>(head_dim));
    }
    args.keys = sparse_keys.data();
    args.values = sparse_values.data();
    args.kv_stride = head_dim;
    args.span = static_cast<int>(kept_count);
  }

  if (attention_mode_ == AttentionMode::Scalar) {
    cpu::attention_head_scalar(args);
  } else {
    kernels().attention_head(args);
  }
}

bool Qwen2::uses_gqa(int position) const {
  const int threads = threads_ > 0 ? threads_ : cpu::resolve_decode_threads();
  return threads > 1 &&
         (attention_mode_ == AttentionMode::Gqa ||
          (attention_mode_ == AttentionMode::Auto && position >= 256));
}

bool Qwen2::uses_sparse(int position) const {
  return sparse_.enabled && !kv_half_ && !kv_q8_ &&
         attention_mode_ != AttentionMode::Scalar &&
         position + 1 > sparse_.sink + sparse_.window;
}

void Qwen2::attention(int sequence, int layer, int position,
                      const float* queries, float* out_buffer) {
  const std::uint64_t t0 = clock_ns();
  const int heads = config_.head_count;
  const int head_dim = config_.head_dim;
  const int threads = threads_ > 0 ? threads_ : cpu::resolve_decode_threads();

  // Static sparse takes the per-head loop (with the subset applied inside
  // attention_head) before the GQA schedule: one index list per query shared
  // across the group is future work, and the grouped kernels stay dense.
  if (uses_sparse(position)) {
    core::ThreadPool& pool = decode_pool();
    if (threads <= 1) {
      for (int head = 0; head < heads; ++head) {
        attention_head(sequence, layer, position, head,
                       queries + static_cast<std::size_t>(head) * head_dim,
                       out_buffer + static_cast<std::size_t>(head) * head_dim,
                       scores_.data());
      }
      phase_attn_ns_ += clock_ns() - t0;
      return;
    }
    pool.parallel_for(heads, 1, [&](int head) {
      attention_head(sequence, layer, position, head,
                     queries + static_cast<std::size_t>(head) * head_dim,
                     out_buffer + static_cast<std::size_t>(head) * head_dim,
                     scores_.data() +
                         static_cast<std::size_t>(head) * context_);
    });
    phase_attn_ns_ += clock_ns() - t0;
    return;
  }

  if (uses_gqa(position)) {
    attention_gqa(sequence, layer, position, queries, out_buffer);
    phase_attn_ns_ += clock_ns() - t0;
    return;
  }

  // At short context the pool round trip costs more than the work, so keep the
  // heads inline until there is enough history to pay for it.
  if (threads <= 1 || position < 64) {
    for (int head = 0; head < heads; ++head) {
      attention_head(sequence, layer, position, head,
                     queries + static_cast<std::size_t>(head) * head_dim,
                     out_buffer + static_cast<std::size_t>(head) * head_dim,
                     scores_.data());
    }
    phase_attn_ns_ += clock_ns() - t0;
    return;
  }

  core::ThreadPool& pool = decode_pool();
  pool.parallel_for(heads, 1, [&](int head) {
    attention_head(sequence, layer, position, head,
                   queries + static_cast<std::size_t>(head) * head_dim,
                   out_buffer + static_cast<std::size_t>(head) * head_dim,
                   scores_.data() +
                       static_cast<std::size_t>(head) * context_);
  });
  phase_attn_ns_ += clock_ns() - t0;
}

void Qwen2::attention_gqa(int sequence, int layer, int position,
                          const float* queries, float* out) {
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
      (static_cast<std::size_t>(sequence) * config_.block_count +
       static_cast<std::size_t>(layer)) *
      context_ * kv_dim;

  core::ThreadPool& pool = decode_pool();
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
    if (kv_half_) {
      args.half = true;
      args.keys_half =
          k_cache_half_.data() + layer_base + head_offset + row_offset;
      args.values_half =
          v_cache_half_.data() + layer_base + head_offset + row_offset;
    } else if (kv_q8_) {
      args.q8 = true;
      args.keys_q8 =
          k_cache_q8_.data() + layer_base + head_offset + row_offset;
      args.values_q8 =
          v_cache_q8_.data() + layer_base + head_offset + row_offset;
      args.keys_q8_scales = k_cache_q8_scales_.data() +
                            (layer_base + head_offset + row_offset) / 32;
      args.values_q8_scales = v_cache_q8_scales_.data() +
                              (layer_base + head_offset + row_offset) / 32;
    } else {
      args.keys = k_cache_.data() + layer_base + head_offset + row_offset;
      args.values = v_cache_.data() + layer_base + head_offset + row_offset;
    }
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
    const std::size_t base =
        static_cast<std::size_t>(kv_head) *
        static_cast<std::size_t>(actual_chunks) *
        static_cast<std::size_t>(group);
    merge_gqa_chunks(group_maxima_.data(), group_sums_.data(),
                     group_acc_.data(), base, actual_chunks, group, group_head,
                     head_dim, reduction_acc_.data(),
                     out + static_cast<std::size_t>(head) * head_dim);
  }
}

void Qwen2::merge_gqa_chunks(const float* maxima, const float* sums,
                             const float* accumulators, std::size_t base,
                             int chunks, int group, int group_head,
                             int head_dim, float* reduction, float* out) {
  bool initialized = false;
  float maximum = -INFINITY;
  float sum = 0.0F;
  for (int index = 0; index < head_dim; ++index) {
    reduction[index] = 0.0F;
  }

  for (int chunk = 0; chunk < chunks; ++chunk) {
    const std::size_t state =
        base +
        static_cast<std::size_t>(chunk) * static_cast<std::size_t>(group) +
        static_cast<std::size_t>(group_head);
    const float chunk_max = maxima[state];
    const float chunk_sum = sums[state];
    if (!(chunk_sum > 0.0F)) {
      continue;
    }
    const float* chunk_acc =
        accumulators + state * static_cast<std::size_t>(head_dim);
    if (!initialized) {
      maximum = chunk_max;
      sum = chunk_sum;
      for (int index = 0; index < head_dim; ++index) {
        reduction[index] = chunk_acc[index];
      }
      initialized = true;
      continue;
    }

    const float merged_max = std::max(maximum, chunk_max);
    const float alpha = std::exp(maximum - merged_max);
    const float beta = std::exp(chunk_max - merged_max);
    sum = sum * alpha + chunk_sum * beta;
    for (int index = 0; index < head_dim; ++index) {
      reduction[index] = reduction[index] * alpha + chunk_acc[index] * beta;
    }
    maximum = merged_max;
  }

  if (initialized && sum > 0.0F) {
    const float inverse = 1.0F / sum;
    for (int index = 0; index < head_dim; ++index) {
      out[index] = reduction[index] * inverse;
    }
  } else {
    for (int index = 0; index < head_dim; ++index) {
      out[index] = 0.0F;
    }
  }
}

void Qwen2::attention_batch(int sequence, int layer, int start_position,
                             int tokens) {
  const std::uint64_t t0 = clock_ns();
  const int heads = config_.head_count;
  const int head_dim = config_.head_dim;
  const int dim = config_.embedding_length;

  core::ThreadPool& pool = prefill_pool();
  const int workers = prefill_threads();
  // Short prompts leave workers idle when parallelising over tokens alone
  // (4 tokens on 6 workers). Split to token x head so every worker has work;
  // each task gets its own score slice. Long prompts keep the token path so
  // the task count stays small.
  if (tokens >= workers) {
    pool.parallel_for(tokens, 1, [&](int token) {
      const int position = start_position + token;
      const float* queries =
          batch_q_.data() + static_cast<std::size_t>(token) * dim;
      float* out = batch_attn_.data() + static_cast<std::size_t>(token) * dim;
      for (int head = 0; head < heads; ++head) {
        float* scores = scores_.data() +
                        (static_cast<std::size_t>(token) * heads + head) *
                            context_;
        attention_head(sequence, layer, position, head,
                       queries + static_cast<std::size_t>(head) * head_dim,
                       out + static_cast<std::size_t>(head) * head_dim, scores);
      }
    });
    phase_attn_ns_ += clock_ns() - t0;
    return;
  }
  pool.parallel_for(tokens * heads, 1, [&](int task) {
    const int token = task / heads;
    const int head = task % heads;
    const int position = start_position + token;
    const float* queries =
        batch_q_.data() + static_cast<std::size_t>(token) * dim;
    float* out = batch_attn_.data() + static_cast<std::size_t>(token) * dim;
    float* scores = scores_.data() +
                    (static_cast<std::size_t>(token) * heads + head) *
                        context_;
    attention_head(sequence, layer, position, head,
                   queries + static_cast<std::size_t>(head) * head_dim,
                   out + static_cast<std::size_t>(head) * head_dim, scores);
  });
  phase_attn_ns_ += clock_ns() - t0;
}

const float* Qwen2::forward_batch(const int* tokens, const int* positions,
                                  int count, const int* sequences) {
  if (tokens == nullptr || positions == nullptr || sequences == nullptr ||
      count <= 0) {
    throw std::runtime_error("forward_batch: invalid batch");
  }
  const int dim = config_.embedding_length;
  const int kv_dim = config_.kv_dim();
  const int ffn = config_.feed_forward_length;
  const int heads = config_.head_count;
  const int head_dim = config_.head_dim;
  const int capacity =
      static_cast<int>(batch_x_.size() / static_cast<std::size_t>(dim));
  if (count > capacity) {
    throw std::runtime_error("forward_batch: batch exceeds the prefill rows");
  }
  for (int index = 0; index < count; ++index) {
    if (tokens[index] < 0 || tokens[index] >= config_.vocab_size) {
      throw std::runtime_error("forward_batch: token outside the vocabulary");
    }
    if (positions[index] < 0 || positions[index] >= context_) {
      throw std::runtime_error("forward_batch: position outside the context");
    }
    if (sequences[index] < 0 || sequences[index] >= sequence_count_) {
      throw std::runtime_error("forward_batch: sequence outside the slots");
    }
  }

  for (int index = 0; index < count; ++index) {
    dequant_row(token_embd_, tokens[index],
                batch_x_.data() + static_cast<std::size_t>(index) * dim);
  }

  for (int index = 0; index < config_.block_count; ++index) {
    const Layer& layer = layers_[static_cast<std::size_t>(index)];

    for (int token = 0; token < count; ++token) {
      rms_norm(batch_h_.data() + static_cast<std::size_t>(token) * dim,
               batch_x_.data() + static_cast<std::size_t>(token) * dim,
               layer.attn_norm);
    }
    const cpu::Q8Quantized q_attn_in =
        vnni_ ? quantize_activations(batch_h_.data(), count, dim)
              : cpu::Q8Quantized{};
    const cpu::Q8Quantized* q_attn_in_ptr = vnni_ ? &q_attn_in : nullptr;
    batch_matvec(layer.wq, batch_h_.data(), batch_q_.data(), count,
                 q_attn_in_ptr);
    batch_matvec(layer.wk, batch_h_.data(), batch_k_.data(), count,
                 q_attn_in_ptr);
    batch_matvec(layer.wv, batch_h_.data(), batch_v_.data(), count,
                 q_attn_in_ptr);

    for (int token = 0; token < count; ++token) {
      const int position = positions[token];
      const int sequence = sequences[token];
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
      apply_rope(q, heads, position);
      apply_rope(k, config_.head_count_kv, position);

      const std::size_t slot =
          (static_cast<std::size_t>(sequence) * config_.block_count +
           static_cast<std::size_t>(index)) *
              context_ * kv_dim +
          static_cast<std::size_t>(position) * kv_dim;
      if (kv_half_) {
        std::uint16_t* k_out = k_cache_half_.data() + slot;
        std::uint16_t* v_out = v_cache_half_.data() + slot;
        for (int axis = 0; axis < kv_dim; ++axis) {
          k_out[axis] = cpu::float_to_half(k[axis]);
          v_out[axis] = cpu::float_to_half(v[axis]);
        }
      } else if (kv_q8_) {
        store_kv_q8(k, v, kv_dim, slot, k_cache_q8_.data(),
                    v_cache_q8_.data(), k_cache_q8_scales_.data(),
                    v_cache_q8_scales_.data());
      } else {
        std::memcpy(k_cache_.data() + slot, k,
                    sizeof(float) * static_cast<std::size_t>(kv_dim));
        std::memcpy(v_cache_.data() + slot, v,
                    sizeof(float) * static_cast<std::size_t>(kv_dim));
      }
    }

    core::ThreadPool& attn_pool = prefill_pool();
    const int attn_workers = prefill_threads();
    {
      const std::uint64_t ta0 = clock_ns();
      if (count >= attn_workers) {
        attn_pool.parallel_for(count, 1, [&](int token) {
          const int position = positions[token];
          const int sequence = sequences[token];
          const float* q =
              batch_q_.data() + static_cast<std::size_t>(token) * dim;
          float* out =
              batch_attn_.data() + static_cast<std::size_t>(token) * dim;
          for (int head = 0; head < heads; ++head) {
            float* scores = scores_.data() +
                            (static_cast<std::size_t>(token) * heads + head) *
                                context_;
            attention_head(sequence, index, position, head,
                           q + static_cast<std::size_t>(head) * head_dim,
                           out + static_cast<std::size_t>(head) * head_dim,
                           scores);
          }
        });
      } else {
        attn_pool.parallel_for(count * heads, 1, [&](int task) {
          const int token = task / heads;
          const int head = task % heads;
          const int position = positions[token];
          const int sequence = sequences[token];
          const float* q =
              batch_q_.data() + static_cast<std::size_t>(token) * dim;
          float* out =
              batch_attn_.data() + static_cast<std::size_t>(token) * dim;
          float* scores = scores_.data() +
                          (static_cast<std::size_t>(token) * heads + head) *
                              context_;
          attention_head(sequence, index, position, head,
                         q + static_cast<std::size_t>(head) * head_dim,
                         out + static_cast<std::size_t>(head) * head_dim,
                         scores);
        });
      }
      phase_attn_ns_ += clock_ns() - ta0;
    }

    const cpu::Q8Quantized q_attn_out =
        vnni_ ? quantize_activations(batch_attn_.data(), count, dim)
              : cpu::Q8Quantized{};
    batch_matvec(layer.wo, batch_attn_.data(), batch_h_.data(), count,
                 vnni_ ? &q_attn_out : nullptr);
    for (int token = 0; token < count; ++token) {
      float* x = batch_x_.data() + static_cast<std::size_t>(token) * dim;
      const float* residual =
          batch_h_.data() + static_cast<std::size_t>(token) * dim;
      for (int axis = 0; axis < dim; ++axis) {
        x[axis] += residual[axis];
      }
    }

    for (int token = 0; token < count; ++token) {
      rms_norm(batch_h_.data() + static_cast<std::size_t>(token) * dim,
               batch_x_.data() + static_cast<std::size_t>(token) * dim,
               layer.ffn_norm);
    }
    const cpu::Q8Quantized q_ffn =
        vnni_ ? quantize_activations(batch_h_.data(), count, dim)
              : cpu::Q8Quantized{};
    batch_matvec(layer.w_gate, batch_h_.data(), batch_gate_.data(), count,
                 vnni_ ? &q_ffn : nullptr);
    batch_matvec(layer.w_up, batch_h_.data(), batch_up_.data(), count,
                 vnni_ ? &q_ffn : nullptr);
    const std::size_t activation_elements =
        static_cast<std::size_t>(count) * static_cast<std::size_t>(ffn);
    silu_gated(batch_gate_.data(), batch_up_.data(),
               static_cast<int>(activation_elements));

    const cpu::Q8Quantized q_down =
        vnni_ ? quantize_activations(batch_gate_.data(), count, ffn)
              : cpu::Q8Quantized{};
    batch_matvec(layer.w_down, batch_gate_.data(), batch_h_.data(), count,
                 vnni_ ? &q_down : nullptr);
    for (int token = 0; token < count; ++token) {
      float* x = batch_x_.data() + static_cast<std::size_t>(token) * dim;
      const float* residual =
          batch_h_.data() + static_cast<std::size_t>(token) * dim;
      for (int axis = 0; axis < dim; ++axis) {
        x[axis] += residual[axis];
      }
    }
  }

  for (int token = 0; token < count; ++token) {
    rms_norm(batch_h_.data() + static_cast<std::size_t>(token) * dim,
             batch_x_.data() + static_cast<std::size_t>(token) * dim,
             output_norm_);
  }
  const cpu::Q8Quantized q_out =
      vnni_ ? quantize_activations(batch_h_.data(), count, dim)
            : cpu::Q8Quantized{};
  batch_matvec(output_, batch_h_.data(), logits_batch_.data(), count,
               vnni_ ? &q_out : nullptr);
  return logits_batch_.data();
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
  const int ffn = config_.feed_forward_length;

  dequant_row(token_embd_, token, x_.data());

  for (int index = 0; index < config_.block_count; ++index) {
    const Layer& layer = layers_[static_cast<std::size_t>(index)];

    rms_norm(h_.data(), x_.data(), layer.attn_norm);
    const cpu::Q8Quantized q_attn_in =
        vnni_ ? quantize_activations(h_.data(), 1, dim) : cpu::Q8Quantized{};
    const cpu::Q8Quantized* q_attn_in_ptr = vnni_ ? &q_attn_in : nullptr;
    matvec(layer.wq, h_.data(), q_.data(), q_attn_in_ptr);
    matvec(layer.wk, h_.data(), k_.data(), q_attn_in_ptr);
    matvec(layer.wv, h_.data(), v_.data(), q_attn_in_ptr);
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
    if (kv_half_) {
      std::uint16_t* k_out = k_cache_half_.data() + slot;
      std::uint16_t* v_out = v_cache_half_.data() + slot;
      for (int axis = 0; axis < kv_dim; ++axis) {
        k_out[axis] = cpu::float_to_half(k_[static_cast<std::size_t>(axis)]);
        v_out[axis] = cpu::float_to_half(v_[static_cast<std::size_t>(axis)]);
      }
    } else if (kv_q8_) {
      store_kv_q8(k_.data(), v_.data(), kv_dim, slot, k_cache_q8_.data(),
                  v_cache_q8_.data(), k_cache_q8_scales_.data(),
                  v_cache_q8_scales_.data());
    } else {
      std::memcpy(k_cache_.data() + slot, k_.data(),
                  sizeof(float) * static_cast<std::size_t>(kv_dim));
      std::memcpy(v_cache_.data() + slot, v_.data(),
                  sizeof(float) * static_cast<std::size_t>(kv_dim));
    }

    attention(0, index, position, q_.data(), attn_.data());

    const cpu::Q8Quantized q_attn_out =
        vnni_ ? quantize_activations(attn_.data(), 1, dim) : cpu::Q8Quantized{};
    matvec(layer.wo, attn_.data(), h_.data(),
           vnni_ ? &q_attn_out : nullptr);
    for (int axis = 0; axis < dim; ++axis) {
      x_[static_cast<std::size_t>(axis)] += h_[axis];
    }

    rms_norm(h_.data(), x_.data(), layer.ffn_norm);
    const cpu::Q8Quantized q_ffn =
        vnni_ ? quantize_activations(h_.data(), 1, dim) : cpu::Q8Quantized{};
    const cpu::Q8Quantized* q_ffn_ptr = vnni_ ? &q_ffn : nullptr;
    matvec(layer.w_gate, h_.data(), gate_.data(), q_ffn_ptr);
    matvec(layer.w_up, h_.data(), up_.data(), q_ffn_ptr);
    silu_gated(gate_.data(), up_.data(), ffn);
    const cpu::Q8Quantized q_down =
        vnni_ ? quantize_activations(gate_.data(), 1, ffn) : cpu::Q8Quantized{};
    matvec(layer.w_down, gate_.data(), h_.data(),
           vnni_ ? &q_down : nullptr);
    for (int axis = 0; axis < dim; ++axis) {
      x_[static_cast<std::size_t>(axis)] += h_[axis];
    }
  }

  rms_norm(h_.data(), x_.data(), output_norm_);
  return output_logits(h_.data());
}

const float* Qwen2::prefill(const int* tokens, int count, int start_position,
                            int sequence) {
  return prefill_run(tokens, count, start_position, nullptr, sequence);
}

const float* Qwen2::prefill_run(const int* tokens, int count, int start_position,
                                float* all_logits, int sequence) {
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
  if (all_logits != nullptr && count > capacity) {
    throw std::runtime_error("prefill_run: batched logits need one chunk");
  }

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

      const cpu::Q8Quantized q_attn_in =
          vnni_ ? quantize_activations(batch_h_.data(), chunk, dim)
                : cpu::Q8Quantized{};
      const cpu::Q8Quantized* q_attn_in_ptr = vnni_ ? &q_attn_in : nullptr;
      batch_matvec(layer.wq, batch_h_.data(), batch_q_.data(), chunk,
                   q_attn_in_ptr);
      batch_matvec(layer.wk, batch_h_.data(), batch_k_.data(), chunk,
                   q_attn_in_ptr);
      batch_matvec(layer.wv, batch_h_.data(), batch_v_.data(), chunk,
                   q_attn_in_ptr);

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
            (static_cast<std::size_t>(sequence) * config_.block_count +
             static_cast<std::size_t>(index)) *
                context_ * kv_dim +
            static_cast<std::size_t>(position) * kv_dim;
        if (kv_half_) {
          std::uint16_t* k_out = k_cache_half_.data() + slot;
          std::uint16_t* v_out = v_cache_half_.data() + slot;
          for (int axis = 0; axis < kv_dim; ++axis) {
            k_out[axis] = cpu::float_to_half(k[axis]);
            v_out[axis] = cpu::float_to_half(v[axis]);
          }
        } else if (kv_q8_) {
          store_kv_q8(k, v, kv_dim, slot, k_cache_q8_.data(),
                      v_cache_q8_.data(), k_cache_q8_scales_.data(),
                      v_cache_q8_scales_.data());
        } else {
          std::memcpy(k_cache_.data() + slot, k,
                      sizeof(float) * static_cast<std::size_t>(kv_dim));
          std::memcpy(v_cache_.data() + slot, v,
                      sizeof(float) * static_cast<std::size_t>(kv_dim));
        }
      }

      // Every key and value in this chunk is in the cache before attention
      // runs, so each token sees exactly the causal prefix it should, and the
      // tokens can be attended in parallel.
      attention_batch(sequence, index, chunk_start, chunk);

      const cpu::Q8Quantized q_attn_out =
          vnni_ ? quantize_activations(batch_attn_.data(), chunk, dim)
                : cpu::Q8Quantized{};
      batch_matvec(layer.wo, batch_attn_.data(), batch_h_.data(), chunk,
                   vnni_ ? &q_attn_out : nullptr);
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

      const cpu::Q8Quantized q_ffn =
          vnni_ ? quantize_activations(batch_h_.data(), chunk, dim)
                : cpu::Q8Quantized{};
      const cpu::Q8Quantized* q_ffn_ptr = vnni_ ? &q_ffn : nullptr;
      batch_matvec(layer.w_gate, batch_h_.data(), batch_gate_.data(), chunk,
                   q_ffn_ptr);
      batch_matvec(layer.w_up, batch_h_.data(), batch_up_.data(), chunk,
                   q_ffn_ptr);
      const std::size_t activation_elements =
          static_cast<std::size_t>(chunk) * static_cast<std::size_t>(ffn);
      silu_gated(batch_gate_.data(), batch_up_.data(),
                 static_cast<int>(activation_elements));

      const cpu::Q8Quantized q_down =
          vnni_ ? quantize_activations(batch_gate_.data(), chunk, ffn)
                : cpu::Q8Quantized{};
      batch_matvec(layer.w_down, batch_gate_.data(), batch_h_.data(), chunk,
                   vnni_ ? &q_down : nullptr);
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

  if (all_logits != nullptr) {
    for (int token = 0; token < last_rows; ++token) {
      rms_norm(batch_h_.data() + static_cast<std::size_t>(token) * dim,
               batch_x_.data() + static_cast<std::size_t>(token) * dim,
               output_norm_);
    }
    const cpu::Q8Quantized q_out =
        vnni_ ? quantize_activations(batch_h_.data(), last_rows, dim)
              : cpu::Q8Quantized{};
    batch_matvec(output_, batch_h_.data(), all_logits, last_rows,
                 vnni_ ? &q_out : nullptr);
    const std::size_t last_row =
        static_cast<std::size_t>(last_rows - 1) * config_.vocab_size;
    std::memcpy(logits_.data(), all_logits + last_row,
                sizeof(float) * static_cast<std::size_t>(config_.vocab_size));
    return logits_.data();
  }

  // Only the final prompt token needs logits to seed decoding.
  const std::size_t last = static_cast<std::size_t>(last_rows - 1) * dim;
  std::memcpy(x_.data(), batch_x_.data() + last,
              sizeof(float) * static_cast<std::size_t>(dim));
  rms_norm(h_.data(), x_.data(), output_norm_);
  return output_logits(h_.data());
}

int Qwen2::verify_draft(const int* tokens, int count, int start_position,
                         int* predicted) {
  if (tokens == nullptr || predicted == nullptr) {
    throw std::runtime_error("verify_draft: invalid draft");
  }
  // Single verification path: the logits form computes the same batched pass,
  // and argmax over its rows is exactly what this entry point always did.
  const float* rows = verify_draft_logits(tokens, count, start_position);
  for (int token = 0; token < count; ++token) {
    predicted[token] = argmax(
        rows + static_cast<std::size_t>(token) * config_.vocab_size,
        config_.vocab_size);
  }
  return count;
}

const float* Qwen2::verify_draft_logits(const int* tokens, int count,
                                        int start_position) {
  if (tokens == nullptr || count <= 0 || count > kSpeculativeBatch) {
    throw std::runtime_error("verify_draft_logits: invalid draft");
  }
  const std::size_t required =
      static_cast<std::size_t>(count) *
      static_cast<std::size_t>(config_.vocab_size);
  if (verify_logits_.size() < required) {
    throw std::runtime_error("verify_draft_logits: logits scratch is too small");
  }

  prefill_run(tokens, count, start_position, verify_logits_.data(), 0);
  return verify_logits_.data();
}

void Qwen2::reset_cache() {
  if (kv_half_) {
    std::fill(k_cache_half_.begin(), k_cache_half_.end(), 0);
    std::fill(v_cache_half_.begin(), v_cache_half_.end(), 0);
  } else if (kv_q8_) {
    std::fill(k_cache_q8_.begin(), k_cache_q8_.end(), 0);
    std::fill(v_cache_q8_.begin(), v_cache_q8_.end(), 0);
    std::fill(k_cache_q8_scales_.begin(), k_cache_q8_scales_.end(), 0.0F);
    std::fill(v_cache_q8_scales_.begin(), v_cache_q8_scales_.end(), 0.0F);
  } else {
    std::fill(k_cache_.begin(), k_cache_.end(), 0.0F);
    std::fill(v_cache_.begin(), v_cache_.end(), 0.0F);
  }
}

int Qwen2::argmax(const float* values, int count) {
  if (values == nullptr || count <= 0) {
    throw std::invalid_argument("argmax: null values or empty count");
  }
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
