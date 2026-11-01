#include "tge/core/dispatch.hpp"

#include <algorithm>

#include "tge/core/thread_pool.hpp"
#include "tge/tensor/ops.hpp"

namespace tge {
namespace {

KernelSet build(const cpu::Features& features) {
  KernelSet set;
  set.features = features;

  const bool avx2 = features.avx2 && features.fma;
  if (avx2) {
    set.gemm_f32 = &cpu::gemm_avx2;
    set.gemm_name = "avx2-fma";
    set.gemv_f32 = &cpu::gemv_avx2;
    set.gemv_name = "avx2-fma";
    set.copy_stream = &cpu::copy_stream_avx2;
    set.sum_stream = &cpu::sum_stream_avx2;
    set.stream_name = "avx2";
  } else {
    set.gemm_f32 = &ops::gemm_ref;
    set.gemm_name = "reference";
    set.gemv_f32 = &ops::gemv_ref;
    set.gemv_name = "reference";
    set.copy_stream = &cpu::copy_stream_scalar;
    set.sum_stream = &cpu::sum_stream_scalar;
    set.stream_name = "scalar";
  }

  return set;
}

}  // namespace

KernelSet select_kernels(const cpu::Features& features) {
  return build(features);
}

const KernelSet& kernels() {
  static const KernelSet set = build(cpu::detect().features);
  return set;
}

}  // namespace tge

namespace tge::cpu {

int resolve_thread_count(int requested) {
  if (requested > 0) {
    return std::min(requested, 1024);
  }
  return core::default_thread_count();
}

}  // namespace tge::cpu
