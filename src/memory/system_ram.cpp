#include "tge/memory/system_ram.hpp"

#include <sstream>

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

namespace tge::memory {
namespace {

std::string to_gib(std::uint64_t bytes) {
  const double gib = static_cast<double>(bytes) / (1024.0 * 1024.0 * 1024.0);
  std::ostringstream out;
  out.setf(std::ios::fixed);
  out.precision(2);
  out << gib << " GiB";
  return out.str();
}

}  // namespace

SystemMemory query_system_memory() {
  SystemMemory memory;

#if defined(_WIN32)
  MEMORYSTATUSEX status{};
  status.dwLength = sizeof(status);
  if (GlobalMemoryStatusEx(&status)) {
    memory.total_physical = status.ullTotalPhys;
    memory.available_physical = status.ullAvailPhys;
    memory.total_commit = status.ullTotalPageFile;
    memory.available_commit = status.ullAvailPageFile;
    memory.load_percent = status.dwMemoryLoad;
  }

  unsigned long long installed_kilobytes = 0;
  if (GetPhysicallyInstalledSystemMemory(&installed_kilobytes)) {
    memory.installed_physical = installed_kilobytes * 1024ull;
  }

  SYSTEM_INFO info{};
  GetSystemInfo(&info);
  memory.page_size = info.dwPageSize;
  memory.allocation_granularity = info.dwAllocationGranularity;
#endif

  if (memory.installed_physical == 0) {
    memory.installed_physical = memory.total_physical;
  }
  return memory;
}

bool refresh(SystemMemory& memory) {
#if defined(_WIN32)
  MEMORYSTATUSEX status{};
  status.dwLength = sizeof(status);
  if (!GlobalMemoryStatusEx(&status)) {
    return false;
  }
  memory.available_physical = status.ullAvailPhys;
  memory.available_commit = status.ullAvailPageFile;
  memory.load_percent = status.dwMemoryLoad;
  return true;
#else
  (void)memory;
  return false;
#endif
}

std::string describe(const SystemMemory& memory) {
  std::ostringstream out;
  out << "  installed     : " << to_gib(memory.installed_physical) << "\n";
  out << "  total visible : " << to_gib(memory.total_physical) << "\n";
  out << "  available     : " << to_gib(memory.available_physical) << " ("
      << memory.load_percent << "% in use)\n";
  out << "  commit limit  : " << to_gib(memory.total_commit) << " (pagefile"
      << ", diagnostics only)\n";
  out << "  page size     : " << memory.page_size << " bytes\n";
  return out.str();
}

}  // namespace tge::memory
