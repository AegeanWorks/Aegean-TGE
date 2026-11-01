#pragma once

#include <memory>
#include <string>
#include <vector>

#include "tge/core/thread_pool.hpp"
#include "tge/models/gguf.hpp"

namespace tge::models {

/// How decode attention is scheduled.
///
/// `Auto` uses grouped-query attention over position chunks once the history is
/// long enough to pay for the reduction, and per-head attention below that.
/// `Scalar` is the sequential oracle the fidelity harness compares against.
enum class AttentionMode { Auto, Scalar, Head, Gqa };

struct Qwen2Config {
  int block_count = 0;
  int embedding_length = 0;
  int feed_forward_length = 0;
  int head_count = 0;
  int head_count_kv = 0;
  int head_dim = 0;
  int context_length = 0;
  int vocab_size = 0;
  float rms_epsilon = 1e-6F;
  float rope_theta = 10000.0F;
  /// End-of-sequence and end-of-turn ids from the tokenizer metadata. -1 when
  /// the file does not name one.
  int eos_token_id = -1;
  int eot_token_id = -1;
  /// Every id that ends generation: the file's eos/eot, any `eos_token_ids`
  /// array, and for Qwen the end-of-text token, which is also its BOS.
  std::vector<int> end_token_ids;

  int kv_dim() const { return head_count_kv * head_dim; }
  bool is_end_token(int token) const {
    if (token < 0) {
      return false;
    }
    for (const int id : end_token_ids) {
      if (id == token) {
        return true;
      }
    }
    return false;
  }
};

/// Reads the decoder configuration from a model file. Throws when it is
/// incomplete. Shared by the loader and the memory planner, which needs the
/// geometry without touching weights.
Qwen2Config read_config(const gguf::File& file);

/// A Qwen2-family decoder (Qwen2.5 uses the same graph as Llama with QKV biases
/// and NeoX-style rotary embedding).
///
/// This exists to produce a real tokens-per-second number on real weights. It is
/// deliberately narrow: one token at a time, greedy sampling, no tokenizer and
/// no batching. Prompt tokens are fed through the same single-token path.
class Qwen2 {
 public:
  /// `context_limit` caps the KV cache allocation. 0 selects the default; the
  /// result is always the smaller of that and the model's own context length.
  ///
  /// `eager_vocabulary` materialises the metadata arrays during the load. The
  /// model body never needs them, but a caller that will build a tokenizer can
  /// have it reuse them instead of parsing the file a second time.
  static std::unique_ptr<Qwen2> load(const std::string& path,
                                     std::string& error,
                                     int context_limit = 0,
                                     bool eager_vocabulary = false);

  const Qwen2Config& config() const { return config_; }
  int max_context() const { return context_; }
  int vocab_size() const { return config_.vocab_size; }

  /// The mapped model file, so a caller can build a tokenizer from the same
  /// vocabulary without reopening it.
  const gguf::File& file() const { return file_; }

  /// Worker threads used by every matrix-vector call. 0 selects the default.
  /// Each model owns its own decode and prefill pools, so concurrent sessions
  /// never publish into a shared worker set.
  void set_threads(int threads);

  /// Attention scheduling. See `AttentionMode`.
  void set_attention_mode(AttentionMode mode) { attention_mode_ = mode; }
  AttentionMode attention_mode() const { return attention_mode_; }

  /// Advances the KV cache by one token and returns `vocab_size` logits. The
  /// pointer is owned by the model and valid until the next call.
  const float* forward(int token, int position);

  /// Processes a prompt in one pass and returns the logits for its last token.
  ///
  /// Each weight is read once per chunk instead of once per token, which is the
  /// difference between prompt processing at tens and at hundreds of tokens per
  /// second. Attention itself stays per-token.
  const float* prefill(const int* tokens, int count, int start_position);

  void reset_cache();

  static int argmax(const float* values, int count);

 private:
  struct Matrix {
    const std::uint8_t* data = nullptr;
    int rows = 0;
    int cols = 0;
    gguf::TensorType type = gguf::TensorType::Unknown;
  };

  struct Layer {
    const float* attn_norm = nullptr;
    const float* ffn_norm = nullptr;
    const float* q_bias = nullptr;
    const float* k_bias = nullptr;
    const float* v_bias = nullptr;
    Matrix wq;
    Matrix wk;
    Matrix wv;
    Matrix wo;
    Matrix w_gate;
    Matrix w_up;
    Matrix w_down;
  };

  Qwen2() = default;

  Matrix matrix(const char* name) const;
  const float* vector(const char* name, int expected) const;
  void matvec(const Matrix& weight, const float* input, float* output) const;
  void batch_matvec(const Matrix& weight, const float* input, float* output,
                    int tokens) const;
  void dequant_row(const Matrix& weight, int row, float* output) const;

  void rms_norm(float* out, const float* in, const float* weight) const;
  void apply_rope(float* values, int heads, int position) const;
  /// Lazy per-session pools; the decode and prefill sizes differ.
  core::ThreadPool& decode_pool() const;
  core::ThreadPool& prefill_pool() const;
  /// All query heads for one decode token, spread over the pool.
  void attention(int layer, int position, const float* queries, float* out);
  /// Grouped-query attention over position chunks, for long decode histories.
  void attention_gqa(int layer, int position, const float* queries, float* out);
  /// Attention for every token in a prefill chunk, spread over the pool. Every
  /// key and value in the chunk is cached before this runs.
  void attention_batch(int layer, int start_position, int tokens);
  /// One query head against its key/value cache. `scores` holds `position + 1`
  /// values and must not overlap another task's slice.
  void attention_head(int layer, int position, int head, const float* query,
                      float* out, float* scores) const;

  gguf::File file_;
  Qwen2Config config_;
  int context_ = 0;
  int threads_ = 0;
  AttentionMode attention_mode_ = AttentionMode::Auto;
  mutable std::unique_ptr<core::ThreadPool> decode_pool_;
  mutable std::unique_ptr<core::ThreadPool> prefill_pool_;

  Matrix token_embd_;
  Matrix output_;
  const float* output_norm_ = nullptr;
  std::vector<Layer> layers_;

  std::vector<float> k_cache_;
  std::vector<float> v_cache_;

  std::vector<float> x_;
  std::vector<float> h_;
  std::vector<float> q_;
  std::vector<float> k_;
  std::vector<float> v_;
  std::vector<float> attn_;
  std::vector<float> gate_;
  std::vector<float> up_;
  std::vector<float> logits_;
  std::vector<float> scores_;
  // Grouped-query attention scratch: per-chunk online-softmax state for every
  // query head in every KV group.
  std::vector<float> group_maxima_;
  std::vector<float> group_sums_;
  std::vector<float> group_acc_;
  std::vector<float> reduction_acc_;
  // Cosine and sine for every position and rotary half, built once at load.
  std::vector<float> rope_cos_;
  std::vector<float> rope_sin_;

  // Prefill workspace, sized for one chunk of prompt tokens.
  std::vector<float> batch_x_;
  std::vector<float> batch_h_;
  std::vector<float> batch_q_;
  std::vector<float> batch_k_;
  std::vector<float> batch_v_;
  std::vector<float> batch_attn_;
  std::vector<float> batch_gate_;
  std::vector<float> batch_up_;
};

}  // namespace tge::models
