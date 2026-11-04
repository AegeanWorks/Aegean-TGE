#pragma once

// Portable IO advice helpers for IDEA-1586. Pure accounting + range math,
// no OS calls; the gguf.cpp prefetch path keeps its behavior until the
// wiring step reads these helpers. Deterministic, portable C++20.

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <utility>

namespace tge::memory {

enum class Advise { Sequential, Random, WillNeed, DontNeed };

struct AlignedRange {
  std::uintptr_t addr = 0;
  std::size_t len = 0;
};

inline AlignedRange align_range(std::uintptr_t addr, std::size_t len,
                                std::size_t page = 4096) {
  if (page == 0 || (page & (page - 1)) != 0) {
    throw std::runtime_error("io_advice: page must be a power of two");
  }
  if (len == 0) return {addr & ~(page - 1), 0};
  const std::uintptr_t base = addr & ~(page - 1);
  const std::uintptr_t end = (addr + len + page - 1) & ~(page - 1);
  return {base, end - base};
}

class PrefetchBudget {
 public:
  explicit PrefetchBudget(std::size_t bytes_allowed)
      : allowed_(bytes_allowed), used_(0) {}

  std::size_t allowed() const { return allowed_; }
  std::size_t used() const { return used_; }

  /// Reserves n bytes. Returns false (no throw) when over budget.
  bool try_reserve(std::size_t n) {
    if (n > allowed_ || used_ + n > allowed_) return false;
    used_ += n;
    return true;
  }

  void release(std::size_t n) {
    if (n > used_) throw std::runtime_error("io_advice: release over used");
    used_ -= n;
  }

 private:
  std::size_t allowed_;
  std::size_t used_;
};

}  // namespace tge::memory
