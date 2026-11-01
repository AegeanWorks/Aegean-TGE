#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace tge::cpu {

/// Instruction-set features reported by CPUID, filtered by what the operating
/// system has actually enabled in XCR0. A feature is true only when both the
/// silicon and the OS support it, so callers may dispatch on it directly.
struct Features {
  bool sse2 = false;
  bool sse3 = false;
  bool ssse3 = false;
  bool sse41 = false;
  bool sse42 = false;
  bool avx = false;
  bool avx2 = false;
  bool fma = false;
  bool f16c = false;
  bool avx_vnni = false;
  bool avx512f = false;
  bool avx512dq = false;
  bool avx512bw = false;
  bool avx512vl = false;
  bool avx512vnni = false;
  bool amx_tile = false;
  bool amx_int8 = false;
  bool os_avx = false;
  bool os_avx512 = false;
};

/// One processor efficiency class. Hybrid parts report two classes; a
/// homogeneous part reports a single class.
struct CoreClass {
  std::uint32_t efficiency_class = 0;
  std::uint32_t physical_cores = 0;
  std::uint32_t logical_cores = 0;
};

struct Topology {
  std::uint32_t physical_cores = 0;
  std::uint32_t logical_cores = 0;
  std::uint32_t performance_cores = 0;
  std::uint32_t efficient_cores = 0;
  std::uint32_t l1d_bytes = 0;
  std::uint32_t l1i_bytes = 0;
  std::uint32_t l2_bytes = 0;
  std::uint32_t l3_bytes = 0;
  std::uint32_t numa_nodes = 0;
  std::vector<CoreClass> classes;
};

struct CpuInfo {
  std::string vendor;
  std::string brand;
  std::uint32_t family = 0;
  std::uint32_t model = 0;
  std::uint32_t stepping = 0;
  std::uint32_t max_leaf = 0;
  Features features;
  Topology topology;

  /// Highest instruction set that may safely be executed on this machine.
  const char* best_isa() const;
};

CpuInfo detect();

std::string describe(const CpuInfo& info);
std::string to_json(const CpuInfo& info);

}  // namespace tge::cpu
