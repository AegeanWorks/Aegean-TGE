#include "tge/memory/budget.hpp"

#include <algorithm>
#include <sstream>

#include "tge/memory/system_ram.hpp"

namespace tge::memory {
namespace {

std::string format_gib(std::uint64_t bytes) {
  std::ostringstream out;
  out.setf(std::ios::fixed);
  out.precision(2);
  out << (static_cast<double>(bytes) / (1024.0 * 1024.0 * 1024.0)) << " GiB";
  return out.str();
}

double share_of(const BudgetPolicy& policy, int pool) {
  switch (pool) {
    case 0:
      return policy.weights_share;
    case 1:
      return policy.kv_share;
    case 2:
      return policy.scratch_share;
    default:
      return policy.sessions_share;
  }
}

}  // namespace

std::uint64_t BudgetSnapshot::reserved_total() const {
  std::uint64_t total = 0;
  for (const PoolState& pool : pools) {
    total += pool.used;
  }
  return total;
}

std::uint64_t BudgetSnapshot::available_total() const { return usable; }

MemoryBudget::MemoryBudget(BudgetPolicy policy) : policy_(policy) { refresh(); }

void MemoryBudget::refresh() {
  std::lock_guard<std::mutex> lock(mutex_);
  const SystemMemory memory = query_system_memory();
  state_.total_physical = memory.total_physical;
  state_.available_physical = memory.available_physical;
  plan_locked();
}

void MemoryBudget::plan_locked() {
  const std::uint64_t total = state_.total_physical;
  const std::uint64_t available = state_.available_physical;

  std::uint64_t os_reserve = static_cast<std::uint64_t>(
      static_cast<double>(total) * policy_.os_reserve_fraction);
  os_reserve = std::max(os_reserve, policy_.os_reserve_floor);
  os_reserve = std::min(os_reserve, total);

  const std::uint64_t headroom = total - os_reserve;
  const std::uint64_t claimed = static_cast<std::uint64_t>(
      static_cast<double>(available) * policy_.usable_fraction);

  state_.reserved_for_os = os_reserve;
  state_.usable = std::min(headroom, claimed);

  double weight_sum = 0.0;
  for (int pool = 0; pool < kPoolCount; ++pool) {
    weight_sum += std::max(0.0, share_of(policy_, pool));
  }
  if (weight_sum <= 0.0) {
    weight_sum = 1.0;
  }

  for (int pool = 0; pool < kPoolCount; ++pool) {
    const double share = std::max(0.0, share_of(policy_, pool)) / weight_sum;
    state_.pools[pool].limit =
        static_cast<std::uint64_t>(static_cast<double>(state_.usable) * share);
  }
}

bool MemoryBudget::try_reserve(Pool pool, std::uint64_t bytes) {
  std::lock_guard<std::mutex> lock(mutex_);
  PoolState& state = state_.pools[static_cast<int>(pool)];
  const std::uint64_t free_bytes = state.limit - std::min(state.limit, state.used);
  if (bytes > free_bytes) {
    return false;
  }
  state.used += bytes;
  return true;
}

void MemoryBudget::release(Pool pool, std::uint64_t bytes) {
  std::lock_guard<std::mutex> lock(mutex_);
  PoolState& state = state_.pools[static_cast<int>(pool)];
  state.used = bytes >= state.used ? 0 : state.used - bytes;
}

std::uint64_t MemoryBudget::available(Pool pool) const {
  std::lock_guard<std::mutex> lock(mutex_);
  const PoolState& state = state_.pools[static_cast<int>(pool)];
  return state.limit - std::min(state.limit, state.used);
}

bool MemoryBudget::admit(std::uint64_t bytes) const {
  std::lock_guard<std::mutex> lock(mutex_);
  const std::uint64_t used = state_.reserved_total();
  return bytes <= state_.usable - std::min(state_.usable, used);
}

BudgetSnapshot MemoryBudget::snapshot() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return state_;
}

BudgetPolicy MemoryBudget::policy() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return policy_;
}

const char* pool_name(Pool pool) {
  switch (pool) {
    case Pool::Weights:
      return "weights";
    case Pool::KvCache:
      return "kv_cache";
    case Pool::Scratch:
      return "scratch";
    case Pool::Sessions:
      return "sessions";
  }
  return "unknown";
}

std::string describe(const BudgetSnapshot& snapshot) {
  std::ostringstream out;
  out << "  physical      : " << format_gib(snapshot.total_physical)
      << " total, " << format_gib(snapshot.available_physical)
      << " available\n";
  out << "  os reserve    : " << format_gib(snapshot.reserved_for_os) << "\n";
  out << "  usable budget : " << format_gib(snapshot.usable) << "\n";
  for (int pool = 0; pool < kPoolCount; ++pool) {
    const std::string name = pool_name(static_cast<Pool>(pool));
    out << "  " << name;
    for (std::size_t pad = name.size(); pad < 14; ++pad) {
      out << ' ';
    }
    out << ": limit " << format_gib(snapshot.pools[pool].limit) << ", used "
        << format_gib(snapshot.pools[pool].used) << "\n";
  }
  return out.str();
}

}  // namespace tge::memory
