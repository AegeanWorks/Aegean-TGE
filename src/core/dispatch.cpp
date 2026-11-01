#include "tge/core/dispatch.hpp"

#include "tge/tensor/ops.hpp"

namespace tge {
namespace {

KernelSet build(const cpu::Features& features) {
  KernelSet set;
  set.features = features;

  if (features.avx2 && features.fma) {
    set.gemm_f32 = &cpu::gemm_avx2;
    set.gemm_name = "avx2-fma";
  } else {
    set.gemm_f32 = &ops::gemm_ref;
    set.gemm_name = "reference";
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
