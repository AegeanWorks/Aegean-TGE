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
    set.gemv_q8 = &cpu::gemv_q8_0_avx2;
    set.gemv_q8_name = "avx2-fma";
    set.gemv_q8_batch = &cpu::gemv_q8_batch_avx2;
    set.gemv_q8_batch_name = "avx2-fma";
    set.gemv_dequant_batch = &cpu::gemv_dequant_batch_avx2;
    set.gemv_dequant_name = "avx2-fma";
    set.attention_head = &cpu::attention_head_avx2;
    set.attention_name = "avx2-fma";
    set.attention_group = &cpu::attention_group_avx2;
    set.copy_stream = &cpu::copy_stream_avx2;
    set.sum_stream = &cpu::sum_stream_avx2;
    set.stream_name = "avx2";
  } else {
    set.gemm_f32 = &ops::gemm_ref;
    set.gemm_name = "reference";
    set.gemv_f32 = &ops::gemv_ref;
    set.gemv_name = "reference";
    set.gemv_q8 = &cpu::gemv_q8_0_scalar;
    set.gemv_q8_name = "scalar";
    set.gemv_q8_batch = &cpu::gemv_q8_batch_scalar;
    set.gemv_q8_batch_name = "scalar";
    set.gemv_dequant_batch = &cpu::gemv_dequant_batch_scalar;
    set.gemv_dequant_name = "scalar";
    set.attention_head = &cpu::attention_head_scalar;
    set.attention_name = "scalar";
    set.attention_group = &cpu::attention_group_scalar;
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

int resolve_decode_threads() {
  static const int threads = [] {
    const CpuInfo info = detect();
    // Decode is unpinned by default (§28ab). Physical cores first; P-cores
    // stay the fallback for topologies without a physical count. Two extra
    // workers beyond the physical count now win after the priority changes:
    // 12 beats 10 in 40/48 interleaved pairs (+4.9% / +6.1% medians,
    // 2026-11-21 log). Capped at the logical count so 8-core machines do not
    // oversubscribe; `--threads` still overrides either value.
    if (info.topology.physical_cores > 0) {
      const int physical = static_cast<int>(info.topology.physical_cores);
      const int logical = info.topology.logical_cores > 0
                              ? static_cast<int>(info.topology.logical_cores)
                              : physical;
      return std::min(logical, physical + 2);
    }
    if (info.topology.performance_cores > 0) {
      return static_cast<int>(info.topology.performance_cores);
    }
    return core::default_thread_count();
  }();
  return threads;
}

}  // namespace tge::cpu
