#include "tge/core/cpu.hpp"

#include <algorithm>
#include <cctype>
#include <cstring>
#include <sstream>
#include <string>
#include <vector>

#if defined(_MSC_VER)
#include <intrin.h>
#else
#include <cpuid.h>
#endif

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <thread>
#endif

namespace tge::cpu {
namespace {

struct Regs {
  std::uint32_t eax = 0;
  std::uint32_t ebx = 0;
  std::uint32_t ecx = 0;
  std::uint32_t edx = 0;
};

Regs cpuid(std::uint32_t leaf, std::uint32_t subleaf = 0) noexcept {
  Regs regs;
#if defined(_MSC_VER)
  int values[4];
  __cpuidex(values, static_cast<int>(leaf), static_cast<int>(subleaf));
  regs.eax = static_cast<std::uint32_t>(values[0]);
  regs.ebx = static_cast<std::uint32_t>(values[1]);
  regs.ecx = static_cast<std::uint32_t>(values[2]);
  regs.edx = static_cast<std::uint32_t>(values[3]);
#else
  __cpuid_count(leaf, subleaf, regs.eax, regs.ebx, regs.ecx, regs.edx);
#endif
  return regs;
}

bool bit(std::uint32_t value, unsigned index) noexcept {
  return ((value >> index) & 1u) != 0;
}

std::uint64_t read_xcr0() noexcept {
#if defined(_MSC_VER)
  return _xgetbv(0);
#else
  std::uint32_t low = 0;
  std::uint32_t high = 0;
  __asm__ __volatile__("xgetbv" : "=a"(low), "=d"(high) : "c"(0));
  return (static_cast<std::uint64_t>(high) << 32) | low;
#endif
}

void trim(std::string& text) {
  const auto not_space = [](unsigned char ch) { return std::isspace(ch) == 0; };
  text.erase(text.begin(),
             std::find_if(text.begin(), text.end(), not_space));
  text.erase(std::find_if(text.rbegin(), text.rend(), not_space).base(),
             text.end());
}

std::uint32_t popcount_mask(std::uintptr_t mask) noexcept {
  std::uint32_t count = 0;
  while (mask != 0) {
    mask &= (mask - 1);
    ++count;
  }
  return count;
}

void detect_topology(Topology& topology) {
#if defined(_WIN32)
  DWORD length = 0;
  GetLogicalProcessorInformationEx(RelationAll, nullptr, &length);
  if (length != 0) {
    std::vector<unsigned char> buffer(length);
    if (GetLogicalProcessorInformationEx(
            RelationAll,
            reinterpret_cast<PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX>(
                buffer.data()),
            &length)) {
      std::size_t offset = 0;
      while (offset + sizeof(SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX) <=
             length) {
        const auto* entry =
            reinterpret_cast<const SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX*>(
                buffer.data() + offset);
        if (entry->Size == 0) {
          break;
        }
        switch (entry->Relationship) {
          case RelationProcessorCore: {
            std::uint32_t logical = 0;
            for (WORD group = 0; group < entry->Processor.GroupCount; ++group) {
              logical += popcount_mask(entry->Processor.GroupMask[group].Mask);
            }
            if (logical == 0) {
              logical = 1;
            }
            const std::uint32_t efficiency =
                entry->Processor.EfficiencyClass;
            auto match = std::find_if(
                topology.classes.begin(), topology.classes.end(),
                [efficiency](const CoreClass& candidate) {
                  return candidate.efficiency_class == efficiency;
                });
            if (match == topology.classes.end()) {
              topology.classes.push_back(
                  CoreClass{efficiency, 1, logical});
            } else {
              match->physical_cores += 1;
              match->logical_cores += logical;
            }
            topology.physical_cores += 1;
            topology.logical_cores += logical;
            break;
          }
          case RelationCache: {
            const CACHE_RELATIONSHIP& cache = entry->Cache;
            const std::uint32_t cache_size =
                static_cast<std::uint32_t>(cache.CacheSize);
            if (cache.Level == 1 && cache.Type == CacheData) {
              topology.l1d_bytes += cache_size;
            } else if (cache.Level == 1 && cache.Type == CacheInstruction) {
              topology.l1i_bytes += cache_size;
            } else if (cache.Level == 2) {
              topology.l2_bytes = std::max(topology.l2_bytes, cache_size);
            } else if (cache.Level == 3) {
              topology.l3_bytes = std::max(topology.l3_bytes, cache_size);
            }
            break;
          }
          case RelationNumaNode:
            topology.numa_nodes += 1;
            break;
          default:
            break;
        }
        offset += entry->Size;
      }
    }
  }

  if (topology.physical_cores == 0) {
    SYSTEM_INFO info;
    GetSystemInfo(&info);
    topology.logical_cores = info.dwNumberOfProcessors;
    topology.physical_cores = info.dwNumberOfProcessors;
  }
#else
  topology.logical_cores = std::thread::hardware_concurrency();
  topology.physical_cores = topology.logical_cores;
#endif

  if (topology.numa_nodes == 0) {
    topology.numa_nodes = 1;
  }

  if (!topology.classes.empty()) {
    std::uint32_t fastest = topology.classes.front().efficiency_class;
    for (const CoreClass& core_class : topology.classes) {
      fastest = std::max(fastest, core_class.efficiency_class);
    }
    for (const CoreClass& core_class : topology.classes) {
      if (core_class.efficiency_class == fastest) {
        topology.performance_cores += core_class.physical_cores;
      } else {
        topology.efficient_cores += core_class.physical_cores;
      }
    }
    std::sort(topology.classes.begin(), topology.classes.end(),
              [](const CoreClass& left, const CoreClass& right) {
                return left.efficiency_class < right.efficiency_class;
              });
  }
}

std::string escape(const std::string& text) {
  std::string out;
  out.reserve(text.size());
  for (char ch : text) {
    if (ch == '"' || ch == '\\') {
      out.push_back('\\');
    }
    out.push_back(ch);
  }
  return out;
}

}  // namespace

const char* CpuInfo::best_isa() const {
  if (features.amx_int8) return "amx";
  if (features.avx512vnni) return "avx512-vnni";
  if (features.avx512f) return "avx512";
  if (features.avx_vnni) return "avx-vnni";
  if (features.avx2) return "avx2";
  if (features.avx) return "avx";
  if (features.sse42) return "sse4.2";
  if (features.sse2) return "sse2";
  return "scalar";
}

CpuInfo detect() {
  CpuInfo info;

  const Regs leaf0 = cpuid(0);
  info.max_leaf = leaf0.eax;

  char vendor[13] = {};
  std::memcpy(vendor + 0, &leaf0.ebx, 4);
  std::memcpy(vendor + 4, &leaf0.edx, 4);
  std::memcpy(vendor + 8, &leaf0.ecx, 4);
  info.vendor = vendor;

  const Regs leaf1 = cpuid(1);
  const std::uint32_t base_family = (leaf1.eax >> 8) & 0xFu;
  const std::uint32_t base_model = (leaf1.eax >> 4) & 0xFu;
  const std::uint32_t ext_family = (leaf1.eax >> 20) & 0xFFu;
  const std::uint32_t ext_model = (leaf1.eax >> 16) & 0xFu;
  info.stepping = leaf1.eax & 0xFu;
  info.family = base_family == 0xFu ? base_family + ext_family : base_family;
  info.model = (base_family == 0x6u || base_family == 0xFu)
                   ? base_model + (ext_model << 4)
                   : base_model;

  info.features.sse2 = bit(leaf1.edx, 26);
  info.features.sse3 = bit(leaf1.ecx, 0);
  info.features.ssse3 = bit(leaf1.ecx, 9);
  info.features.sse41 = bit(leaf1.ecx, 19);
  info.features.sse42 = bit(leaf1.ecx, 20);
  info.features.fma = bit(leaf1.ecx, 12);
  info.features.f16c = bit(leaf1.ecx, 29);
  info.features.avx = bit(leaf1.ecx, 28);
  const bool osxsave = bit(leaf1.ecx, 27);

  if (info.max_leaf >= 7) {
    const Regs leaf7 = cpuid(7, 0);
    info.features.avx2 = bit(leaf7.ebx, 5);
    info.features.avx512f = bit(leaf7.ebx, 16);
    info.features.avx512dq = bit(leaf7.ebx, 17);
    info.features.avx512bw = bit(leaf7.ebx, 30);
    info.features.avx512vl = bit(leaf7.ebx, 31);
    info.features.avx512vnni = bit(leaf7.ecx, 11);

    const Regs leaf7_sub1 = cpuid(7, 1);
    info.features.avx_vnni = bit(leaf7_sub1.eax, 4);
    info.features.amx_tile = bit(leaf7_sub1.edx, 24);
    info.features.amx_int8 = bit(leaf7_sub1.edx, 25);
  }

  if (osxsave) {
    const std::uint64_t xcr0 = read_xcr0();
    const bool xmm_state = (xcr0 & 0x2u) != 0;
    const bool ymm_state = (xcr0 & 0x4u) != 0;
    const bool zmm_state = (xcr0 & 0xE0u) == 0xE0u;
    info.features.os_avx = xmm_state && ymm_state;
    info.features.os_avx512 = info.features.os_avx && zmm_state;
  }

  // Never report a vector feature the kernel will not preserve across a
  // context switch, regardless of what the silicon advertises.
  if (!info.features.os_avx) {
    info.features.avx = false;
    info.features.avx2 = false;
    info.features.fma = false;
    info.features.f16c = false;
    info.features.avx_vnni = false;
  }
  if (!info.features.os_avx512) {
    info.features.avx512f = false;
    info.features.avx512dq = false;
    info.features.avx512bw = false;
    info.features.avx512vl = false;
    info.features.avx512vnni = false;
    info.features.amx_tile = false;
    info.features.amx_int8 = false;
  }

  const Regs extended = cpuid(0x80000000u);
  if (extended.eax >= 0x80000004u) {
    char brand[49] = {};
    for (std::uint32_t index = 0; index < 3; ++index) {
      const Regs part = cpuid(0x80000002u + index);
      std::memcpy(brand + index * 16 + 0, &part.eax, 4);
      std::memcpy(brand + index * 16 + 4, &part.ebx, 4);
      std::memcpy(brand + index * 16 + 8, &part.ecx, 4);
      std::memcpy(brand + index * 16 + 12, &part.edx, 4);
    }
    info.brand = brand;
    trim(info.brand);
  }
  if (info.brand.empty()) {
    info.brand = info.vendor;
  }

  detect_topology(info.topology);
  return info;
}

std::string describe(const CpuInfo& info) {
  const Features& f = info.features;
  std::ostringstream out;
  out << info.brand << " [" << info.vendor << " family " << info.family
      << " model " << info.model << " stepping " << info.stepping << "]\n";
  out << "  best ISA      : " << info.best_isa() << "\n";
  out << "  cores         : " << info.topology.physical_cores << " physical / "
      << info.topology.logical_cores << " logical";
  if (info.topology.efficient_cores > 0) {
    out << " (" << info.topology.performance_cores << " performance, "
        << info.topology.efficient_cores << " efficient)";
  }
  out << "\n";
  out << "  caches        : L1D " << info.topology.l1d_bytes / 1024
      << " KiB, L2 " << info.topology.l2_bytes / 1024 << " KiB, L3 "
      << info.topology.l3_bytes / 1024 << " KiB\n";
  out << "  NUMA nodes    : " << info.topology.numa_nodes << "\n";
  out << "  instructions  :";
  const std::pair<const char*, bool> flags[] = {
      {"sse2", f.sse2},         {"sse3", f.sse3},
      {"ssse3", f.ssse3},       {"sse4.1", f.sse41},
      {"sse4.2", f.sse42},      {"avx", f.avx},
      {"avx2", f.avx2},         {"fma", f.fma},
      {"f16c", f.f16c},         {"avx-vnni", f.avx_vnni},
      {"avx512f", f.avx512f},   {"avx512vnni", f.avx512vnni},
      {"amx-int8", f.amx_int8}};
  for (const auto& [name, present] : flags) {
    if (present) {
      out << " " << name;
    }
  }
  out << "\n";
  return out.str();
}

std::string to_json(const CpuInfo& info) {
  const Features& f = info.features;
  std::ostringstream out;
  out << "{\n";
  out << "  \"vendor\": \"" << escape(info.vendor) << "\",\n";
  out << "  \"brand\": \"" << escape(info.brand) << "\",\n";
  out << "  \"family\": " << info.family << ",\n";
  out << "  \"model\": " << info.model << ",\n";
  out << "  \"stepping\": " << info.stepping << ",\n";
  out << "  \"best_isa\": \"" << info.best_isa() << "\",\n";
  out << "  \"physical_cores\": " << info.topology.physical_cores << ",\n";
  out << "  \"logical_cores\": " << info.topology.logical_cores << ",\n";
  out << "  \"performance_cores\": " << info.topology.performance_cores << ",\n";
  out << "  \"efficient_cores\": " << info.topology.efficient_cores << ",\n";
  out << "  \"l1d_bytes\": " << info.topology.l1d_bytes << ",\n";
  out << "  \"l2_bytes\": " << info.topology.l2_bytes << ",\n";
  out << "  \"l3_bytes\": " << info.topology.l3_bytes << ",\n";
  out << "  \"numa_nodes\": " << info.topology.numa_nodes << ",\n";
  out << "  \"features\": {";
  const std::pair<const char*, bool> flags[] = {
      {"sse2", f.sse2},         {"sse3", f.sse3},
      {"ssse3", f.ssse3},       {"sse4_1", f.sse41},
      {"sse4_2", f.sse42},      {"avx", f.avx},
      {"avx2", f.avx2},         {"fma", f.fma},
      {"f16c", f.f16c},         {"avx_vnni", f.avx_vnni},
      {"avx512f", f.avx512f},   {"avx512vnni", f.avx512vnni},
      {"amx_int8", f.amx_int8}};
  bool first = true;
  for (const auto& [name, present] : flags) {
    if (!first) {
      out << ", ";
    }
    out << "\"" << name << "\": " << (present ? "true" : "false");
    first = false;
  }
  out << "}\n";
  out << "}\n";
  return out.str();
}

}  // namespace tge::cpu
