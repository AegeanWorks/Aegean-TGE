#pragma once

#include <memory>
#include <string>
#include <vector>

#include "tge/models/gguf.hpp"

namespace tge::models {

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

  int kv_dim() const { return head_count_kv * head_dim; }
};

/// A Qwen2-family decoder (Qwen2.5 uses the same graph as Llama with QKV biases
/// and NeoX-style rotary embedding).
///
/// This exists to produce a real tokens-per-second number on real weights. It is
/// deliberately narrow: one token at a time, greedy sampling, no tokenizer and
/// no batching. Prompt tokens are fed through the same single-token path.
class Qwen2 {
 public:
  static std::unique_ptr<Qwen2> load(const std::string& path,
                                     std::string& error);

  const Qwen2Config& config() const { return config_; }
  int max_context() const { return context_; }
  int vocab_size() const { return config_.vocab_size; }

  /// Worker threads used by every matrix-vector call. 0 selects the default.
  void set_threads(int threads) { threads_ = threads; }

  /// Advances the KV cache by one token and returns `vocab_size` logits. The
  /// pointer is owned by the model and valid until the next call.
  const float* forward(int token, int position);

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
  void dequant_row(const Matrix& weight, int row, float* output) const;

  void rms_norm(float* out, const float* in, const float* weight) const;
  void apply_rope(float* values, int heads, int position) const;
  void attention(int layer, int position);

  gguf::File file_;
  Qwen2Config config_;
  int context_ = 0;
  int threads_ = 0;

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
  std::vector<float> rope_freq_;
};

}  // namespace tge::models
