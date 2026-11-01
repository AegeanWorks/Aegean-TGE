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

/// Streaming kernels used to measure the memory bandwidth ceiling. These are
/// diagnostics, not part of the inference path.
void copy_stream_avx2(float* dst, const float* src, std::size_t count);
float sum_stream_avx2(const float* src, std::size_t count);

/// Turns a requested thread count into a usable one.
int resolve_thread_count(int requested);

}  // namespace tge::cpu
