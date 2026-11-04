#include "tge/core/thread_pool.hpp"

#include <algorithm>
#include <map>
#include <memory>
#include <stdexcept>
#include <utility>

#if defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86))
#include <intrin.h>
#endif

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

namespace tge::core {
namespace {

/// Pause attempts before a thread parks. Decode makes ~169 back-to-back
/// parallel calls per token, far closer together than a park/wake round trip,
/// so a worker that spins through the gap avoids the handoff entirely.
constexpr int kSpinAttempts = 2048;

inline void cpu_relax() {
#if defined(__GNUC__) && (defined(__x86_64__) || defined(__i386__))
  __builtin_ia32_pause();
#elif defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86))
  _mm_pause();
#else
  std::this_thread::yield();
#endif
}

#if defined(_WIN32)
/// Logical indices (group 0) of the fastest efficiency class. Single-group
/// assumption holds for the 16-logical dev machine; larger machines keep the
/// default OS placement rather than a wrong mask. Computed once under the
/// function-local static guard, so concurrently starting workers cannot race
/// on the cache.
std::vector<int> p_core_logicals() {
  static const std::vector<int> cached = [] {
    std::vector<int> out;
    DWORD length = 0;
    GetLogicalProcessorInformationEx(RelationAll, nullptr, &length);
    if (length == 0) {
      return out;
    }
    std::vector<unsigned char> buffer(length);
    if (!GetLogicalProcessorInformationEx(
            RelationAll,
            reinterpret_cast<PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX>(
                buffer.data()),
            &length)) {
      return out;
    }
  struct CoreEntry {
    DWORD efficiency = 0;
    std::vector<int> processors;
  };
  std::vector<CoreEntry> cores;
  std::size_t offset = 0;
  int logical = 0;
  while (offset + sizeof(SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX) <= length) {
    const auto* entry =
        reinterpret_cast<const SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX*>(
            buffer.data() + offset);
    if (entry->Size == 0) {
      break;
    }
    if (entry->Relationship == RelationProcessorCore) {
      CoreEntry core;
      core.efficiency = entry->Processor.EfficiencyClass;
      for (WORD group = 0; group < entry->Processor.GroupCount; ++group) {
        const KAFFINITY mask = entry->Processor.GroupMask[group].Mask;
        // Only group 0 is pinnable with a plain affinity mask here.
        if (entry->Processor.GroupMask[group].Group != 0) {
          continue;
        }
        for (int bit = 0; bit < 64; ++bit) {
          if ((mask >> bit) & 1) {
            core.processors.push_back(logical + bit);
          }
        }
      }
      // Fallback when masks are unavailable: count the core once.
      if (core.processors.empty()) {
        core.processors.push_back(logical);
      }
      logical += static_cast<int>(core.processors.size());
      cores.push_back(std::move(core));
    }
    offset += entry->Size;
  }
  DWORD fastest = 0;
  for (const auto& core : cores) {
    fastest = std::max(fastest, core.efficiency);
  }
  for (const auto& core : cores) {
    if (core.efficiency == fastest) {
      out.insert(out.end(), core.processors.begin(),
                 core.processors.end());
    }
  }
    return out;
  }();
  return cached;
}

inline void pin_worker_to_p_core(const std::vector<int>& p_cores,
                                 int worker_index) {
  if (p_cores.empty()) {
    return;
  }
  // Reserve logical 0 for the calling thread / OS; workers take the rest
  // round-robin so 5 decode workers land on 5 distinct P-core threads.
  const int pick = p_cores[static_cast<std::size_t>(
      (worker_index + 1) % static_cast<int>(p_cores.size()))];
  if (pick < 0 || pick >= 64) {
    return;
  }
  const DWORD_PTR mask = static_cast<DWORD_PTR>(1ULL << pick);
  SetThreadAffinityMask(GetCurrentThread(), mask);
}
#else
inline void pin_worker_to_p_core(int /*worker_index*/) {}
#endif

}  // namespace

ThreadPool::ThreadPool(int threads, bool pin) {
  size_ = std::max(1, threads);
  workers_.reserve(static_cast<std::size_t>(size_ - 1));
  // Resolve the P-core list once, here on the constructing thread, and hand
  // each worker its own copy. Fifteen workers start within microseconds of
  // each other; letting all of them be the first to touch a lazily-initialized
  // detection cache is a thundering herd on a static-init guard. Eager init
  // keeps workers off that path entirely.
  const std::vector<int> p_cores = pin ? p_core_logicals() : std::vector<int>{};
  for (int index = 0; index < size_ - 1; ++index) {
    workers_.emplace_back([this, index, pin, p_cores] {
      if (pin) {
        pin_worker_to_p_core(p_cores, index);
      }
      worker();
    });
  }
}

ThreadPool::~ThreadPool() {
  {
    std::lock_guard<std::mutex> lock(mutex_);
    stop_.store(true, std::memory_order_release);
  }
  wake_.notify_all();
  for (std::thread& worker_thread : workers_) {
    if (worker_thread.joinable()) {
      worker_thread.join();
    }
  }
}

void ThreadPool::claim_chunks(const std::function<void(int)>& body, int count,
                              int grain) {
  for (;;) {
    if (failed_.load(std::memory_order_relaxed)) {
      return;  // Another thread already banked a failure; drain, don't pile on.
    }
    const int start = next_.fetch_add(grain, std::memory_order_relaxed);
    if (start >= count) {
      return;
    }
    const int end = std::min(start + grain, count);
    for (int index = start; index < end; ++index) {
      try {
        body(index);
      } catch (...) {
        std::lock_guard<std::mutex> lock(exception_mutex_);
        if (exception_ == nullptr) {
          exception_ = std::current_exception();
        }
        failed_.store(true, std::memory_order_relaxed);
        return;
      }
    }
  }
}

void ThreadPool::parallel_for(int count, int grain,
                              const std::function<void(int)>& body) {
  if (count <= 0) {
    return;
  }
  grain = std::max(1, grain);
  bool expected = false;
  if (!active_.compare_exchange_strong(expected, true,
                                       std::memory_order_acq_rel)) {
    throw std::runtime_error("thread_pool: concurrent parallel_for on one pool");
  }
  struct ActiveGuard {
    ThreadPool* pool;
    ~ActiveGuard() { pool->active_.store(false, std::memory_order_release); }
  };
  ActiveGuard guard{this};
  {
    std::lock_guard<std::mutex> lock(exception_mutex_);
    exception_ = nullptr;
  }
  failed_.store(false, std::memory_order_relaxed);
  if (size_ == 1 || count <= grain) {
    for (int index = 0; index < count; ++index) {
      body(index);
    }
    return;
  }

  body_ = &body;
  count_ = count;
  grain_ = grain;
  completed_.store(0, std::memory_order_relaxed);
  next_.store(0, std::memory_order_relaxed);
  const std::size_t workers = workers_.size();
  // Publish the work. Every field above is visible to a worker that observes
  // this generation through the acquire load below.
  generation_.fetch_add(1, std::memory_order_seq_cst);

  // A spinning worker needs no signal. Only pay for the notify when someone is
  // actually parked. This store/load pair races with a worker's outgoing
  // store/load pair, so both sides are sequentially consistent: otherwise both
  // can read stale values on x86 and a parked worker misses the generation.
  if (sleepers_.load(std::memory_order_seq_cst) > 0) {
    std::lock_guard<std::mutex> lock(mutex_);
    wake_.notify_all();
  }

  claim_chunks(body, count, grain);

  // Spin for the workers, then park if this call is long enough to make the
  // spin wasteful. `main_waiting_` pairs with the worker's completion check and
  // is sequentially consistent for the same reason.
  int attempts = 0;
  while (completed_.load(std::memory_order_seq_cst) != workers) {
    if (++attempts < kSpinAttempts) {
      cpu_relax();
      continue;
    }
    std::unique_lock<std::mutex> lock(mutex_);
    main_waiting_.store(1, std::memory_order_seq_cst);
    finished_.wait(lock, [this, workers] {
      return completed_.load(std::memory_order_seq_cst) == workers;
    });
    main_waiting_.store(0, std::memory_order_seq_cst);
    break;
  }
  body_ = nullptr;
  std::exception_ptr failure;
  {
    std::lock_guard<std::mutex> lock(exception_mutex_);
    failure = exception_;
  }
  if (failure != nullptr) {
    std::rethrow_exception(failure);
  }
}

void ThreadPool::worker() {
  std::uint64_t seen = 0;

  for (;;) {
    int attempts = 0;
    while (generation_.load(std::memory_order_acquire) == seen &&
           !stop_.load(std::memory_order_acquire)) {
      if (++attempts < kSpinAttempts) {
        cpu_relax();
        continue;
      }
      std::unique_lock<std::mutex> lock(mutex_);
      // Register as asleep before the final predicate check. Both this count
      // and the publisher's read are sequentially consistent, so either the
      // publisher sees the sleeper or the sleeper sees the generation.
      sleepers_.fetch_add(1, std::memory_order_seq_cst);
      wake_.wait(lock, [this, seen] {
        return generation_.load(std::memory_order_seq_cst) != seen ||
               stop_.load(std::memory_order_acquire);
      });
      sleepers_.fetch_sub(1, std::memory_order_acq_rel);
      break;
    }

    if (stop_.load(std::memory_order_acquire)) {
      return;
    }

    const std::uint64_t current = generation_.load(std::memory_order_acquire);
    if (current == seen) {
      continue;  // spurious wake-up
    }
    seen = current;

    const std::function<void(int)>* body = body_;
    const int count = count_;
    const int grain = grain_;
    if (body != nullptr) {
      claim_chunks(*body, count, grain);
    }

    completed_.fetch_add(1, std::memory_order_seq_cst);
    if (main_waiting_.load(std::memory_order_seq_cst) != 0) {
      std::lock_guard<std::mutex> lock(mutex_);
      finished_.notify_all();
    }
  }
}

ThreadPool& acquire_pool(int threads) {
  // Fallback registry for the standalone benchmarks and tests. A model owns its
  // own pools and passes them through the kernel configs, so concurrent
  // sessions never reach this shared registry.
  static std::mutex registry_mutex;
  static std::map<int, std::unique_ptr<ThreadPool>> pools;

  const int size = std::max(1, threads);
  std::lock_guard<std::mutex> lock(registry_mutex);
  auto entry = pools.find(size);
  if (entry == pools.end()) {
    entry = pools.emplace(size, std::make_unique<ThreadPool>(size)).first;
  }
  return *entry->second;
}

int default_thread_count() {
  const unsigned available = std::thread::hardware_concurrency();
  return available == 0 ? 1 : static_cast<int>(available);
}

}  // namespace tge::core
