#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "tge/core/thread_pool.hpp"
#include "tge/models/gguf.hpp"

namespace tge::cpu {
struct Q8Quantized;
}

namespace tge::models {

/// How decode attention is scheduled.
///
/// `Auto` uses grouped-query attention over position chunks once the history is
/// long enough to pay for the reduction, and per-head attention below that.
/// `Scalar` is the sequential oracle the fidelity harness compares against.
enum class AttentionMode { Auto, Scalar, Head, Gqa };

/// Static sparse selection: sink + strided middle + window.
///
/// LSA-shaped (§3.2 kept-set geometry: sink + selected long-range + window)
/// but content-blind: the middle keeps every `stride`-th position instead of
/// scored buckets. Opt-in; the dense path is bit-identical whenever the span
/// fits in sink + window, the KV cache is not f32, or the mode is Scalar.
struct SparseConfig {
  bool enabled = false;
  int sink = 4;     ///< First positions always kept (paper Table 2: nsink = 4).
  int window = 512; ///< Most-recent positions always kept (paper: nw = 512).
  int stride = 4;   ///< Every stride-th position kept in between (must be >= 2).
};

/// Sorted unique kept positions in [0, span). Returns the full 0..span-1
/// range (dense) when the span fits in sink + window, so the caller keeps
/// the exact dense path instead of a gathered copy.
std::vector<int> build_sparse_index(int span, const SparseConfig& config);

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
  /// Rotary embedding layout. NeoX pairs the first half of each head with
  /// the second half (Qwen and friends); interleaved pairs consecutive
  /// elements (Llama and friends, ggml NORM). Selected from the model
  /// architecture at load; the wrong layout reads as fluent-looking garbage,
  /// so a new architecture must declare its side here, never silently.
  bool rope_interleaved = false;
  /// YaRN magnitude scale (`rope.scaling.attn_factor`, Phi-3's 1.19024):
  /// llama multiplies every cos/sin table entry by it, sharpening attention
  /// by the square. 1.0 when the file does not name the key, which leaves
  /// every existing model bit-identical.
  float rope_attn_factor = 1.0F;
  /// Original training context for long/short YaRN factor selection (0 =
  /// fall back to `context_length`, mirroring the reference default).
  int rope_orig_context = 0;
  /// YaRN factor tables (`rope_factors_short/long.weight`) are present at
  /// file level. Like the reference, the loader applies per-block factor
  /// tensors when a file carries them and treats absent ones as all-ones;
  /// the file-level tables on the Phi export are vestigial there, so the
  /// flag stays informational: short/long selection only matters past the
  /// original context length, and every shipped file agrees below it.
  bool has_yarn_factors = false;
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
                                      bool eager_vocabulary = false,
                                      bool kv_half = false,
                                      bool kv_q8 = false,
                                      int sequences = 1);

  const Qwen2Config& config() const { return config_; }
  int max_context() const { return context_; }
  int vocab_size() const { return config_.vocab_size; }

  /// Tokens one speculative verification pass can check. Five are proposed by
  /// the drafter and one pads the batched kernel's six-token group.
  static constexpr int kSpeculativeBatch = 6;

  /// The mapped model file, so a caller can build a tokenizer from the same
  /// vocabulary without reopening it.
  const gguf::File& file() const { return file_; }

  /// Worker threads used by every matrix-vector call. 0 selects the default.
  /// Each model owns its own decode and prefill pools, so concurrent sessions
  /// never publish into a shared worker set.
  void set_threads(int threads);

  /// Prefill worker override for the phase-specific plan. 0 (default) means
  /// every logical core: prefill is compute-bound. Pass an explicit count to
  /// reproduce the same-thread comparison (e.g. match --threads for a fair
  /// fight against another engine's -t N, or isolate the split in an A/B).
  void set_prefill_threads(int threads);

  /// Decode-worker CPU pinning. Unpinned (default) lets the OS place workers
  /// everywhere: on the measured box this decodes faster with lower run
  /// variance than pinning, singly and under sessions contention (see the
  /// log). Pass --pin-threads 1 to restore P-core pinning.
  void set_pin_threads(bool pin);

  /// Attention scheduling. See `AttentionMode`.
  void set_attention_mode(AttentionMode mode) { attention_mode_ = mode; }
  AttentionMode attention_mode() const { return attention_mode_; }

  /// Static sparse selection (sink + strided middle + window). Disabled by
  /// default; when enabled, the per-head decode/prefill paths attend only the
  /// kept positions once the span exceeds sink + window. Requires stride >= 2
  /// (throws otherwise) and applies to the f32 KV cache only: half/int8 and
  /// Scalar mode keep the exact dense path.
  void set_sparse_config(const SparseConfig& config);
  SparseConfig sparse_config() const { return sparse_; }

  /// Force the scalar oracle for rms_norm, RoPE and SiLU even where AVX2 is
  /// available. The fidelity harness sets this on the reference arm so one
  /// number prices the whole vectorisation program, not just attention.
  /// Production paths leave it off.
  void set_scalar_kernels(bool enabled) { scalar_kernels_ = enabled; }

    /// Use the AVX-VNNI Q8_0 batch kernel, quantizing each distinct activation
    /// once per layer and reusing it across the matrices that share it.
    void set_vnni(bool enabled) { vnni_ = enabled; }
    bool vnni() const { return vnni_; }

    /// Use the integer Q4_K/Q6_K x Q8_K batch kernel (per-row Q8 activation
    /// quant + maddubs dots) for prefill-size batches. Decode (tokens == 1)
    /// always keeps the float path. Benchmark-only until the fidelity price
    /// lands; see measurement-log §54.
    void set_kq8(bool enabled) { kq8_ = enabled; }
    bool kq8() const { return kq8_; }

  /// True when the KV cache is stored as IEEE half.
  bool kv_half() const { return kv_half_; }

  /// True when the KV cache is stored as int8 blocks (one float scale per 32
  /// values). Halves the half-cache footprint; the fidelity harness prices
  /// the quantization the same way it prices `--kv f16`.
  bool kv_q8() const { return kv_q8_; }

  /// GPU offload request (`--gpu`, default off). True only *wishes* for the
  /// GPU: dispatch additionally requires the backend ready (`gpu::use_gpu`)
  /// and the static winners (`wants_gpu_*` below). Without an SDK build the
  /// backend reports Unavailable and the request is a silent no-op.
  void set_use_gpu(bool enabled) { use_gpu_requested_ = enabled; }
  bool use_gpu_requested() const { return use_gpu_requested_; }

  /// Placement wishes. Backend readiness is NOT included; the dispatcher
  /// conjoins `gpu::use_gpu()` at the call site, so these stay unit-testable
  /// without a GPU. Batch: tokens >= 8 on Q8_0/F32 with clean flags (no
  /// VNNI activations, no scalar-oracle arm, schedule not forced Scalar).
  /// GQA: grouped schedule already holds for the span, dense path only (no
  /// sparse subset, no scalar arm). Everything else keeps the CPU kernels.
  bool wants_gpu_batch(int tokens, gguf::TensorType type) const;
  bool wants_gpu_gqa(int span) const;
  /// Decode-output wish: the single-token 151936-wide projection. Same
  /// conjunctions as batch minus the token floor.
  bool wants_gpu_output(gguf::TensorType type) const;

  /// Independent KV slots the model can decode at once.
  int sequence_count() const { return sequence_count_; }

  /// Advances the KV cache by one token and returns `vocab_size` logits. The
  /// pointer is owned by the model and valid until the next call.
  const float* forward(int token, int position);

  /// Decodes one token per sequence in a single weight pass and returns
  /// `count * vocab_size` logits, row `i` for sequence `sequences[i]`.
  const float* forward_batch(const int* tokens, const int* positions, int count,
                             const int* sequences);

  /// Longest leading-token run of [tokens, tokens+count) across valid
  /// same-geometry multi-prefix table slots. No side effects; 0 when the
  /// table is empty or the KV form is not f32. Lets a batch admission skip
  /// fully-cached prompt rows after priming them (see below).
  int shared_prefix_len(const int* tokens, int count) const;

  /// Memcpy per-layer K/V rows [0..count) from the best table slot into
  /// `sequence`'s cache rows. Succeeds iff the best run covers the whole
  /// [0..count) prompt with count >= 16 and the KV form is f32; on success
  /// the caller may skip recomputing rows [0..count-1) and compute only a
  /// trailing row for its logits. Returns false with no effect otherwise.
  bool prime_cache_from_table(int sequence, const int* tokens, int count);

  /// Processes a prompt in one pass and returns the logits for its last token.
  ///
  /// Each weight is read once per chunk instead of once per token, which is the
  /// difference between prompt processing at tens and at hundreds of tokens per
  /// second. Attention itself stays per-token.
  const float* prefill(const int* tokens, int count, int start_position,
                       int sequence = 0);

  /// Runs up to `kSpeculativeBatch` draft tokens at `start_position` in one
  /// batched pass and writes the greedy argmax after each position to
  /// `predicted`. The draft tokens' key/value rows are committed to the cache;
  /// a rejected suffix is simply overwritten by later tokens.
  int verify_draft(const int* tokens, int count, int start_position,
                   int* predicted);

  /// Same batched pass as `verify_draft`, but returns all `count` positions'
  /// logits (`count` x `vocab_size` floats, owned by the model, valid until
  /// the next call) instead of argmax predictions. Powers
  /// distribution-preserving speculative sampling, which needs the full
  /// target distribution at each draft position.
  const float* verify_draft_logits(const int* tokens, int count,
                                   int start_position);

  void reset_cache();

  static int argmax(const float* values, int count);

  /// Phase timers: wall nanoseconds accumulated inside matvec/batch_matvec,
  /// attention, and norms/RoPE/SiLU. Zero arithmetic change; the CLI resets
  /// before a measured region and prints the shares. All counters are
  /// mutated from const methods, so they are mutable.
  void reset_phase_timers() const;
  std::uint64_t phase_matvec_ns() const { return phase_matvec_ns_; }
  std::uint64_t phase_attn_ns() const { return phase_attn_ns_; }
  std::uint64_t phase_norm_ns() const { return phase_norm_ns_; }
  /// GPU offload time (submit + transfers) and bytes moved, accumulated
  /// since the last reset. Zero whenever the CPU path served the token.
  std::uint64_t phase_gpu_ns() const { return phase_gpu_ns_; }
  std::uint64_t phase_gpu_bytes() const { return phase_gpu_bytes_; }

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
  /// Row-range view over one tensor's rows: the same bytes, the same type,
  /// `rows` rows starting at `row_begin`. Fused tensors (`attn_qkv`,
  /// fused `ffn_up`) map onto the split matrices through this; every kernel
  /// consumes the view exactly like a standalone matrix because K-quant
  /// super-blocks never cross a row. Throws on an out-of-range slice.
  Matrix matrix_slice(const char* name, int row_begin, int rows) const;
  const float* vector(const char* name, int expected) const;
  void matvec(const Matrix& weight, const float* input, float* output,
              const cpu::Q8Quantized* quantized = nullptr) const;
  void batch_matvec(const Matrix& weight, const float* input, float* output,
                    int tokens, const cpu::Q8Quantized* quantized = nullptr,
                    bool float_only = false) const;
  /// Final vocabulary projection for one decoded token: the CPU rms_norm
  /// into h_ stays at the call site; this tries the single-sync GPU GEMV
  /// (one H2D + one D2H) when the static winners hold and the backend is
  /// ready, and runs the exact existing CPU matvec otherwise. F32 weights
  /// stay CPU until their GPU twin lands; only Q8_0 rides the device today.
  const float* output_logits(float* hidden);
  void dequant_row(const Matrix& weight, int row, float* output) const;

  /// Quantizes one activation batch for the VNNI kernels. The scratch is owned
  /// by the model, so the result is valid until the next call.
  cpu::Q8Quantized quantize_activations(const float* input, int tokens, int cols);

  /// Shared prefill body. `all_logits`, when non-null, receives one row of
  /// logits per token in the final chunk; the public prefill passes null.
  const float* prefill_run(const int* tokens, int count, int start_position,
                           float* all_logits, int sequence);

  void rms_norm(float* out, const float* in, const float* weight) const;
  void apply_rope(float* values, int heads, int position) const;
  void silu_gated(float* gate, const float* up, int count) const;
  /// Lazy per-session pools; the decode and prefill sizes differ.
  /// Decode is bandwidth-bound and stays on a few pinned P-core threads;
  /// prefill is compute-bound and takes every logical core, unpinned.
  core::ThreadPool& decode_pool() const;
  core::ThreadPool& prefill_pool() const;
  /// Worker count for prefill work (batch GEMM, chunk attention).
  int prefill_threads() const;
  /// All query heads for one decode token, spread over the pool.
  void attention(int sequence, int layer, int position, const float* queries,
                 float* out);
  /// Grouped-query attention over position chunks, for long decode histories.
  void attention_gqa(int sequence, int layer, int position,
                     const float* queries, float* out);
  /// Attention for every token in a prefill chunk, spread over the pool. Every
  /// key and value in the chunk is cached before this runs.
  void attention_batch(int sequence, int layer, int start_position, int tokens);
  /// One query head against its key/value cache. `scores` holds `position + 1`
  /// values and must not overlap another task's slice.
  void attention_head(int sequence, int layer, int position, int head,
                      const float* query, float* out, float* scores) const;
  /// True when this position takes the grouped-query schedule in decode and
  /// prefill alike; `threads <= 1` keeps the per-head path.
  bool uses_gqa(int position) const;
  /// True when the static sparse path applies at this position: enabled,
  /// f32 KV, non-oracle mode, and the span exceeds sink + window. Checked
  /// before `uses_gqa`, so sparse takes the per-head loop at long context;
  /// GQA keeps the exact dense path whenever sparse does not apply.
  bool uses_sparse(int position) const;
  /// Merges one query head's per-chunk online-softmax states.
  static void merge_gqa_chunks(const float* maxima, const float* sums,
                               const float* accumulators, std::size_t base,
                               int chunks, int group, int group_head,
                               int head_dim, float* reduction, float* out);

  gguf::File file_;
  Qwen2Config config_;
  int context_ = 0;
  int threads_ = 0;
  int prefill_threads_override_ = 0;
  bool pin_threads_ = false;
  int sequence_count_ = 1;
  AttentionMode attention_mode_ = AttentionMode::Auto;
  SparseConfig sparse_;
    bool scalar_kernels_ = false;
    bool vnni_ = false;
    bool kq8_ = true;
  bool kv_half_ = false;
  bool kv_q8_ = false;
  bool use_gpu_requested_ = false;
  mutable std::unique_ptr<core::ThreadPool> decode_pool_;
  mutable std::unique_ptr<core::ThreadPool> prefill_pool_;
  mutable std::uint64_t phase_matvec_ns_ = 0;
  mutable std::uint64_t phase_attn_ns_ = 0;
  mutable std::uint64_t phase_norm_ns_ = 0;
  mutable std::uint64_t phase_gpu_ns_ = 0;
  mutable std::uint64_t phase_gpu_bytes_ = 0;

  Matrix token_embd_;
  Matrix output_;
  const float* output_norm_ = nullptr;
  std::vector<Layer> layers_;
  // Shared zero QKV bias for files that omit bias tensors (e.g. Llama-3.2):
  // layers point here instead of faulting at load.
  std::vector<float> zero_bias_;

  std::vector<float> k_cache_;
  std::vector<float> v_cache_;
  std::vector<std::uint16_t> k_cache_half_;
  std::vector<std::uint16_t> v_cache_half_;
  // Int8 KV form: values parallel the float cache element for element, with
  // one float scale per 32 values in matching order.
  std::vector<std::int8_t> k_cache_q8_;
  std::vector<std::int8_t> v_cache_q8_;
  std::vector<float> k_cache_q8_scales_;
  std::vector<float> v_cache_q8_scales_;

  std::vector<float> x_;
  std::vector<float> h_;
  // VNNI activation scratch: one batch of rows plus its per-block scales.
  std::vector<std::int8_t> q8_values_;
  std::vector<float> q8_scales_;
  std::vector<float> q_;
  std::vector<float> k_;
  std::vector<float> v_;
  std::vector<float> attn_;
  std::vector<float> gate_;
  std::vector<float> up_;
  std::vector<float> logits_;
  std::vector<float> verify_logits_;
  std::vector<float> logits_batch_;
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
  // Interleaved-layout copies with every coefficient doubled
  // ([c0,c0,c1,c1,...]), so the AVX2 kernel loads one contiguous vector per
  // 8 values instead of gathering pairs. Built alongside the tables above.
  std::vector<float> rope_cos_dup_;
  std::vector<float> rope_sin_dup_;

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
