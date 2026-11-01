#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>

namespace tge::cpu {

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
};

struct GemmConfig {
  /// Worker threads. 0 selects the engine default; 1 forces the calling thread.
  int threads = 0;
};

/// Every GEMM implementation has this signature, so dispatch is a function
/// pointer and callers state their threading intent explicitly.
using GemmFn = void (*)(const GemmArgs& args, const GemmConfig& config);

void gemm_avx2(const GemmArgs& args, const GemmConfig& config);

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
};

using Q8BatchFn = void (*)(const Q8BatchArgs& args, const GemvConfig& config);

void gemv_q8_batch_avx2(const Q8BatchArgs& args, const GemvConfig& config);
void gemv_q8_batch_scalar(const Q8BatchArgs& args, const GemvConfig& config);

/// AVX-VNNI variant, reachable from the benchmark harness. Dispatch keeps the
/// float kernel: this one wins in isolation but loses end to end.
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
};

std::size_t quant_row_bytes(QuantKind kind, int cols);

/// Expands one row of quantized weights into floats. Written once per row and
/// reused by every token in the group, so the cost is amortized.
void dequantize_row(QuantKind kind, const std::uint8_t* source, float* destination,
                    int cols);

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

/// Streaming kernels used to measure the memory bandwidth ceiling.
using CopyStreamFn = void (*)(float* dst, const float* src, std::size_t count);
using SumStreamFn = float (*)(const float* src, std::size_t count);

void copy_stream_avx2(float* dst, const float* src, std::size_t count);
float sum_stream_avx2(const float* src, std::size_t count);

/// Portable fallbacks, used when AVX2 is unavailable so a measurement can
/// never require an instruction the CPU does not have.
void copy_stream_scalar(float* dst, const float* src, std::size_t count);
float sum_stream_scalar(const float* src, std::size_t count);

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
