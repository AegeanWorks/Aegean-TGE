#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>

#include "tge/core/thread_pool.hpp"

namespace tge::cpu {

/// Addressing contract for every AVX2 kernel below: model and activation
/// buffers are only 16-byte aligned (`operator new`), never 32. Every lane
/// loads and stores explicitly unaligned (`_mm256_loadu_*`); forming a
/// `__m256*` over a float buffer and indexing it emits aligned VMOVAPS and
/// faults (see measurement-log §28i). The sole exception is a function-local
/// `alignas(32)` spill, which is 32-byte aligned by construction.

/// Row-major single-precision GEMM: C[M,N] = A[M,K] * B[K,N].
struct GemmArgs {
  int m = 0;
  int n = 0;
  int k = 0;
  const float* a = nullptr;
  int lda = 0;
  const float* b = nullptr;
  int ldb = 0;
  float* c = nullptr;
  int ldc = 0;
  // Fused clamp epilogue (IDEA-1539, default off): when true, the store
  // path clamps to [vmin, vmax]. Additive only; existing callers keep the
  // exact current behavior.
  bool do_clamp = false;
  float vmin = 0.0F;
  float vmax = 0.0F;
};

struct GemmConfig {
  /// Worker threads. 0 selects the engine default; 1 forces the calling thread.
  int threads = 0;
  /// If non-null, use this pool instead of the process-wide fallback. A model
  /// passes its own per-session pool so concurrent sessions never share one.
  core::ThreadPool* pool = nullptr;
};

/// Every GEMM implementation has this signature, so dispatch is a function
/// pointer and callers state their threading intent explicitly.
using GemmFn = void (*)(const GemmArgs& args, const GemmConfig& config);

void gemm_avx2(const GemmArgs& args, const GemmConfig& config);

// Benchmark controls for attributing the GEMM rewrite. Dispatch selects
// neither; `gemm_avx2` is the shipped kernel.
/// Packing without cache blocking.
void gemm_avx2_unblocked(const GemmArgs& args, const GemmConfig& config);
/// Neither packing nor blocking: the pre-rewrite kernel.
void gemm_avx2_unpacked(const GemmArgs& args, const GemmConfig& config);

/// Matrix-vector product: out[rows] = matrix[rows,cols] * vector[cols].
///
/// This, not GEMM, is the shape of autoregressive decoding: one token advances
/// one activation vector through every weight matrix. The matrix is streamed
/// once per call and never reused within it, so this is a pure bandwidth
/// problem and is tuned as one.
struct GemvArgs {
  int rows = 0;
  int cols = 0;
  const float* matrix = nullptr;
  int ldm = 0;
  const float* vector = nullptr;
  float* out = nullptr;
};

struct GemvConfig {
  int threads = 0;
  /// If non-null, use this pool instead of the process-wide fallback.
  core::ThreadPool* pool = nullptr;
};

using GemvFn = void (*)(const GemvArgs& args, const GemvConfig& config);

void gemv_avx2(const GemvArgs& args, const GemvConfig& config);

/// Q8_0 matrix-vector product: out[rows] = W[rows,cols] * vector[cols], where W
/// is stored as GGUF Q8_0 blocks (fp16 scale + 32 int8 per 34-byte block).
///
/// This is the decode path for real models. Eight bits per weight instead of
/// thirty-two is the difference between four bytes and one byte crossing the
/// bus per weight per token, and decode is bandwidth-bound.
struct Q8GemvArgs {
  const std::uint8_t* data = nullptr;
  int rows = 0;
  int cols = 0;
  const float* vector = nullptr;
  float* out = nullptr;
};

using Q8GemvFn = void (*)(const Q8GemvArgs& args, const GemvConfig& config);

void gemv_q8_0_avx2(const Q8GemvArgs& args, const GemvConfig& config);
void gemv_q8_0_scalar(const Q8GemvArgs& args, const GemvConfig& config);

/// Batched Q8_0 projection: out[tokens][rows] = input[tokens][cols] * W[rows][cols]^T.
///
/// This is prompt processing, and it is why prefill is not the same problem as
/// decode. Decoding one token reads every weight; processing a prompt of P
/// tokens one token at a time reads every weight P times. Reusing one weight
/// row across a group of tokens turns that into a single pass, which is the
/// entire difference between tens and hundreds of tokens per second.
struct Q8BatchArgs {
  const std::uint8_t* data = nullptr;  // [rows][cols], Q8_0 blocks
  int rows = 0;
  int cols = 0;
  const float* input = nullptr;        // [tokens][cols], row stride = cols
  int tokens = 0;
  float* out = nullptr;                // [tokens][rows], row stride = rows
  // Optional activation rows already quantized to Q8_0. The VNNI kernel uses
  // these instead of quantizing `input` per call, so a batch of matrices that
  // share one activation (q/k/v, gate/up) amortizes the conversion.
  const std::int8_t* quantized = nullptr;        // [tokens][cols]
  const float* quantized_scales = nullptr;       // [tokens][cols / 32]
};

using Q8BatchFn = void (*)(const Q8BatchArgs& args, const GemvConfig& config);

/// One batch of activation rows already quantized to Q8_0, ready to reuse
/// across the matrices that share it.
struct Q8Quantized {
  const std::int8_t* values = nullptr;  // [tokens][cols]
  const float* scales = nullptr;        // [tokens][cols / 32]
};

void gemv_q8_batch_avx2(const Q8BatchArgs& args, const GemvConfig& config);
void gemv_q8_batch_scalar(const Q8BatchArgs& args, const GemvConfig& config);

/// Quantizes activation rows exactly as `gemv_q8_batch_vnni` does, so a caller
/// can reuse one conversion across several matrices.
void quantize_q8_batch(const float* input, int tokens, int cols,
                       std::int8_t* values, float* scales);

/// AVX-VNNI variant. Reachable from the benchmark harness and, with `--vnni`,
/// from the model; dispatch elsewhere keeps the float kernel.
void gemv_q8_batch_vnni(const Q8BatchArgs& args, const GemvConfig& config);

/// Quantized formats with a super-block layout. Q8_0 has its own fused kernels;
/// these three share one dequantize-then-dot path because their per-row decode
/// is cheap enough to run once and reuse across a token group.
enum class QuantKind : int {
  Q8_0 = 0,
  Q4_K = 1,
  Q5_K = 2,
  Q6_K = 3,
  Q4_0 = 4,
  Q5_0 = 5,
  Q5_1 = 6,
};

std::size_t quant_row_bytes(QuantKind kind, int cols);

/// Expands one row of quantized weights into floats. Written once per row and
/// reused by every token in the group, so the cost is amortized.
void dequantize_row(QuantKind kind, const std::uint8_t* source, float* destination,
                    int cols);

/// Vectorised dequantisation for the formats that are measured hot. Falls back
/// to the scalar path for the rest, and must stay bit-identical to it - see
/// `q4k_avx2_matches_scalar_exactly` in the test suite.
void dequantize_row_avx2(QuantKind kind, const std::uint8_t* source,
                         float* destination, int cols);

struct DequantBatchArgs {
  const std::uint8_t* data = nullptr;
  QuantKind kind = QuantKind::Q4_K;
  int rows = 0;
  int cols = 0;
  const float* input = nullptr;  // [tokens][cols]
  int tokens = 0;
  float* out = nullptr;          // [tokens][rows]
};

using DequantBatchFn = void (*)(const DequantBatchArgs& args,
                                const GemvConfig& config);

void gemv_dequant_batch_avx2(const DequantBatchArgs& args,
                              const GemvConfig& config);
void gemv_dequant_batch_scalar(const DequantBatchArgs& args,
                               const GemvConfig& config);

/// Integer Q4_K/Q6_K x Q8_K batch prototype (bench-only, behind
/// `bench batch --kernel kq8`; see measurement-log §57). Same banding
/// as gemv_dequant_batch_avx2; activations are quantized to Q8_K once
/// per call and every dot runs the integer maddubs path.
void gemv_dequant_batch_kq8_avx2(const DequantBatchArgs& args,
                                 const GemvConfig& config);

// Single-row integer Q4_K/Q6_K x Q8_K decode prototype (bench-only, behind
// `bench batch --kernel kq8d`). Same dots and scalar quantizer as the batch
// entry, but the activation row lives in a thread-local buffer (never
// heap-allocated per call) and output needs no token loop.
void gemv_dequant_kq8_row_avx2(const DequantBatchArgs& args,
                               const GemvConfig& config);

/// One query head's attention against its key/value cache.
///
/// The kernel vectorises the query-key dot products, the softmax exponential
/// (shared Cephes-style exp, see exp_avx2.hpp) and the weighted value sum.
/// The scalar path is the oracle; the harness prices the approximation.
struct AttentionArgs {
  const float* query = nullptr;   // head_dim values
  const float* keys = nullptr;    // span rows, stride kv_stride
  const float* values = nullptr;  // span rows, stride kv_stride
  // When `half`, read `keys_half`/`values_half` instead of the float rows.
  const std::uint16_t* keys_half = nullptr;
  const std::uint16_t* values_half = nullptr;
  bool half = false;
  // When `q8`, read int8 `keys_q8`/`values_q8` with one float scale per 32
  // values instead. Scales parallel the values with stride kv_stride / 32,
  // so a head slice starting at element offset O uses scale offset O / 32.
  const std::int8_t* keys_q8 = nullptr;
  const std::int8_t* values_q8 = nullptr;
  const float* keys_q8_scales = nullptr;
  const float* values_q8_scales = nullptr;
  bool q8 = false;
  int kv_stride = 0;
  int span = 0;
  int head_dim = 0;
  float scale = 1.0F;
  float* scores = nullptr;  // span values, overwritten with the softmax
  float* out = nullptr;     // head_dim values
};

using AttentionHeadFn = void (*)(const AttentionArgs& args);

/// Sequential reference. This is the oracle the vectorised path is compared
/// against, so its summation order must not change.
void attention_head_scalar(const AttentionArgs& args);
void attention_head_avx2(const AttentionArgs& args);

/// One KV group's attention over one position chunk, with an online softmax.
///
/// Grouped-query attention has several query heads sharing each KV head. Each
/// query head keeps an online (running maximum, running sum, running value sum)
/// state over the chunk. The caller merges the chunks, which is what lets
/// decode split positions across the pool as well as heads.
struct AttentionGroupArgs {
  const float* queries = nullptr;   // group_heads * head_dim
  const float* keys = nullptr;      // count rows, stride kv_stride
  const float* values = nullptr;    // count rows, stride kv_stride
  const std::uint16_t* keys_half = nullptr;
  const std::uint16_t* values_half = nullptr;
  bool half = false;
  // Int8 KV form; same scale layout contract as AttentionArgs above.
  const std::int8_t* keys_q8 = nullptr;
  const std::int8_t* values_q8 = nullptr;
  const float* keys_q8_scales = nullptr;
  const float* values_q8_scales = nullptr;
  bool q8 = false;
  int kv_stride = 0;
  int count = 0;
  int head_dim = 0;
  int group_heads = 0;
  float scale = 1.0F;
  float* maxima = nullptr;        // group_heads
  float* sums = nullptr;          // group_heads
  float* accumulators = nullptr;  // group_heads * head_dim
};

using AttentionGroupFn = void (*)(const AttentionGroupArgs& args);

void attention_group_scalar(const AttentionGroupArgs& args);
void attention_group_avx2(const AttentionGroupArgs& args);

/// Bytes occupied by one Q8_0 block: 2-byte scale plus 32 signed bytes.
inline constexpr std::size_t kQ8BlockBytes = 34;
inline constexpr int kQ8BlockElements = 32;

/// Converts one IEEE half to float. Inline so the scalar and vector paths agree
/// bit for bit on scale decoding without a call per block.
inline float half_to_float(std::uint16_t value) {
  const std::uint32_t sign = static_cast<std::uint32_t>(value & 0x8000u) << 16;
  std::uint32_t exponent = (value >> 10) & 0x1Fu;
  std::uint32_t mantissa = value & 0x3FFu;

  std::uint32_t bits = 0;
  if (exponent == 0) {
    if (mantissa == 0) {
      bits = sign;
    } else {
      // Subnormal: renormalise into a float exponent.
      int shift = 0;
      while ((mantissa & 0x400u) == 0) {
        mantissa <<= 1;
        ++shift;
      }
      mantissa &= 0x3FFu;
      bits = sign | (static_cast<std::uint32_t>(127 - 15 - shift + 1) << 23) |
             (mantissa << 13);
    }
  } else if (exponent == 31) {
    bits = sign | 0x7F800000u | (mantissa << 13);
  } else {
    bits = sign | ((exponent - 15 + 127) << 23) | (mantissa << 13);
  }

  float result = 0.0F;
  std::memcpy(&result, &bits, sizeof(result));
  return result;
}

/// Converts one float to IEEE half, round to nearest even. Used by the f16 KV
/// store; keeps the conversion identical for the scalar and vector paths.
inline std::uint16_t float_to_half(float value) {
  std::uint32_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  const std::uint32_t sign = (bits >> 16) & 0x8000u;
  const std::uint32_t exponent = (bits >> 23) & 0xFFu;
  const std::uint32_t mantissa = bits & 0x7FFFFFu;
  if (exponent == 0xFFu) {
    return static_cast<std::uint16_t>(
        sign | 0x7C00u | (mantissa != 0 ? 0x200u : 0u));
  }
  const int unbiased = static_cast<int>(exponent) - 127;
  if (unbiased < -24) {
    return static_cast<std::uint16_t>(sign);
  }
  if (unbiased < -14) {
    const std::uint32_t shift = static_cast<std::uint32_t>(-unbiased - 14);
    const std::uint32_t significand = 0x800000u | mantissa;
    const std::uint32_t half = significand >> (shift + 13);
    const std::uint32_t round = (significand >> (shift + 12)) & 1u;
    return static_cast<std::uint16_t>(sign | (half + round));
  }
  if (unbiased > 15) {
    return static_cast<std::uint16_t>(sign | 0x7C00u);
  }
  const std::uint32_t half =
      (static_cast<std::uint32_t>(unbiased + 15) << 10) | (mantissa >> 13);
  const std::uint32_t round = (mantissa >> 12) & 1u;
  return static_cast<std::uint16_t>(sign | (half + round));
}

/// Streaming kernels used to measure the memory bandwidth ceiling.
using CopyStreamFn = void (*)(float* dst, const float* src, std::size_t count);
using SumStreamFn = float (*)(const float* src, std::size_t count);

void copy_stream_avx2(float* dst, const float* src, std::size_t count);
float sum_stream_avx2(const float* src, std::size_t count);

/// Portable fallbacks, used when AVX2 is unavailable so a measurement can
/// never require an instruction the CPU does not have.
void copy_stream_scalar(float* dst, const float* src, std::size_t count);
float sum_stream_scalar(const float* src, std::size_t count);

/// RMS normalisation over one row: out[i] = in[i] / sqrt(mean(in^2) + eps) * weight[i].
/// The scalar path is the oracle; the AVX2 path changes summation order and
/// uses vectorised sqrt/divide, so it is priced by the fidelity harness, not
/// assumed bit-identical.
void rms_norm_scalar(float* out, const float* in, const float* weight, int dim,
                     float eps);
void rms_norm_avx2(float* out, const float* in, const float* weight, int dim,
                   float eps);

/// Rotary position embedding (Qwen NeoX layout: first half vs second half).
/// Tables hold one cos/sin pair per position per half-dim element.
void apply_rope_scalar(float* values, int heads, int head_dim, int position,
                       const float* cos_table, const float* sin_table);
void apply_rope_avx2(float* values, int heads, int head_dim, int position,
                     const float* cos_table, const float* sin_table);

/// Rotary position embedding (Llama interleaved layout: consecutive pairs).
/// Mirrors ggml's GGML_ROPE_TYPE_NORM exactly: pair k rotates by the k-th
/// table entry. `cos_table`/`sin_table` must be the DUPLICATED tables
/// ([c0,c0,c1,c1,...]), one contiguous vector per 8 values.
void apply_rope_interleaved_scalar(float* values, int heads, int head_dim,
                                   int position, const float* cos_table,
                                   const float* sin_table);
void apply_rope_interleaved_avx2(float* values, int heads, int head_dim,
                                 int position, const float* cos_table,
                                 const float* sin_table);

/// Gated SiLU: gate[i] = silu(gate[i]) * up[i], silu(x) = x / (1 + exp(-x)).
/// The AVX2 path uses a vectorised exp approximation (Cephes-style, ~1e-6
/// relative); the scalar path is the oracle and the fidelity harness prices
/// the difference.
void silu_gated_scalar(float* gate, const float* up, int count);
void silu_gated_avx2(float* gate, const float* up, int count);

/// Softmax numerator pass over `logits`: out[i] = exp((logits[i] - max) * invT)
/// with max found in index order, sum accumulated in order; returns the sum.
/// The AVX2 path uses the shared Cephes exp (exp_avx2.hpp, ~1e-6 relative);
/// the scalar path is the oracle. Sampled draws shift slightly, so the gate
/// is a chi-square test, not bit-identity (greedy never touches exp).
float softmax_exp_scalar(const float* logits, int count, float temperature,
                         float* out);
float softmax_exp_avx2(const float* logits, int count, float temperature,
                       float* out);

/// Turns a requested thread count into a usable one.
int resolve_thread_count(int requested);

/// Thread count for memory-bound decode work.
///
/// Deliberately below the logical processor count. Measured on this 6P+4E part:
/// 4-6 workers sustain about 33 tokens/s while 16 workers fall to 22, because
/// per-call synchronisation and E-core contention cost more than the extra
/// bandwidth returns. Prefill is compute-bound and keeps the full count.
int resolve_decode_threads();

}  // namespace tge::cpu
