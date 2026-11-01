#pragma once

#include <cstdint>
#include <string>

namespace tge::memory {

/// Physical and commit memory as Windows reports it. Commit memory includes the
/// pagefile and is reported for diagnostics only: it is never used to plan
/// inference placement, because pagefile-backed execution is not a supported
/// memory tier.
struct SystemMemory {
  std::uint64_t total_physical = 0;
  std::uint64_t available_physical = 0;
  std::uint64_t installed_physical = 0;
  std::uint64_t total_commit = 0;
  std::uint64_t available_commit = 0;
  std::uint64_t page_size = 0;
  std::uint64_t allocation_granularity = 0;
  std::uint32_t load_percent = 0;
};

SystemMemory query_system_memory();

/// Re-reads the volatile fields (available physical, commit, load) in place.
/// Returns false when the query fails, leaving the input untouched.
bool refresh(SystemMemory& memory);

std::string describe(const SystemMemory& memory);

}  // namespace tge::memory
