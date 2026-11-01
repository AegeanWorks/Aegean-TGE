#pragma once

#include <cstdint>
#include <mutex>
#include <string>

namespace tge::memory {

/// Independent buckets so that one workload cannot starve another. KV cache is
/// deliberately separate from weights: growing a context must never be able to
/// evict model weights that are still in use.
enum class Pool : int { Weights = 0, KvCache = 1, Scratch = 2, Sessions = 3 };

inline constexpr int kPoolCount = 4;

struct BudgetPolicy {
  /// Fraction of installed RAM kept for Windows and other applications.
  double os_reserve_fraction = 0.25;
  /// Absolute lower bound on the OS reserve, applied even on small machines.
  std::uint64_t os_reserve_floor = 2ull * 1024 * 1024 * 1024;
  /// Fraction of *currently available* RAM the engine is willing to claim.
  double usable_fraction = 0.90;

  double weights_share = 0.62;
  double kv_share = 0.23;
  double scratch_share = 0.08;
  double sessions_share = 0.07;
};

struct PoolState {
  std::uint64_t limit = 0;
  std::uint64_t used = 0;
};

struct BudgetSnapshot {
  std::uint64_t total_physical = 0;
  std::uint64_t available_physical = 0;
  std::uint64_t reserved_for_os = 0;
  std::uint64_t usable = 0;
  PoolState pools[kPoolCount];

  std::uint64_t reserved_total() const;
  std::uint64_t available_total() const;
};

/// Tracks what the engine has committed per pool and refuses admissions that
/// would cross a pool limit or the overall usable budget.
class MemoryBudget {
 public:
  explicit MemoryBudget(BudgetPolicy policy = {});

  /// Re-reads system memory and re-plans limits. Usage already recorded is
  /// preserved; a shrinking budget can therefore drive a pool negative, which
  /// `available` reports as zero.
  void refresh();

  bool try_reserve(Pool pool, std::uint64_t bytes);
  void release(Pool pool, std::uint64_t bytes);

  std::uint64_t available(Pool pool) const;
  bool admit(std::uint64_t bytes) const;

  BudgetSnapshot snapshot() const;
  BudgetPolicy policy() const;

 private:
  void plan_locked();

  mutable std::mutex mutex_;
  BudgetPolicy policy_;
  BudgetSnapshot state_;
};

const char* pool_name(Pool pool);
std::string describe(const BudgetSnapshot& snapshot);

}  // namespace tge::memory
