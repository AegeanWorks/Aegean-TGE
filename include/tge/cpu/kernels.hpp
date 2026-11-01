#pragma once

#include <cstddef>

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

}  // namespace tge::cpu
