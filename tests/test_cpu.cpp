#include <string>

#include "harness.hpp"
#include "tge/core/cpu.hpp"
#include "tge/core/dispatch.hpp"

TGE_TEST(cpu_reports_identity_and_topology) {
  const tge::cpu::CpuInfo info = tge::cpu::detect();
  TGE_CHECK(!info.vendor.empty());
  TGE_CHECK(!info.brand.empty());
  TGE_CHECK(info.topology.physical_cores >= 1);
  TGE_CHECK(info.topology.logical_cores >= info.topology.physical_cores);
  TGE_CHECK(info.topology.numa_nodes >= 1);
}

TGE_TEST(cpu_feature_implications_hold) {
  const tge::cpu::Features features = tge::cpu::detect().features;
  if (features.avx2) {
    TGE_CHECK(features.avx);
  }
  if (features.avx) {
    TGE_CHECK(features.sse2);
    TGE_CHECK(features.os_avx);
  }
  if (features.avx512f) {
    TGE_CHECK(features.avx2);
    TGE_CHECK(features.os_avx512);
  }
  if (features.avx_vnni) {
    TGE_CHECK(features.avx2);
  }
  if (features.amx_int8) {
    TGE_CHECK(features.amx_tile);
  }
}

TGE_TEST(cpu_best_isa_is_known) {
  const tge::cpu::CpuInfo info = tge::cpu::detect();
  const std::string isa = info.best_isa();
  TGE_CHECK(!isa.empty());
  TGE_CHECK(isa == "avx2" || isa == "avx" || isa == "sse4.2" ||
            isa == "sse2" || isa == "scalar" || isa == "avx-vnni" ||
            isa == "avx512" || isa == "avx512-vnni" || isa == "amx");
}

TGE_TEST(dispatch_selects_a_callable_kernel) {
  const tge::cpu::Features features = tge::cpu::detect().features;
  const tge::KernelSet set = tge::select_kernels(features);
  TGE_REQUIRE(set.gemm_f32 != nullptr);
  TGE_CHECK(!set.gemm_name.empty());
  if (features.avx2 && features.fma) {
    TGE_CHECK(set.gemm_name == "avx2-fma");
  } else {
    TGE_CHECK(set.gemm_name == "reference");
  }
}
