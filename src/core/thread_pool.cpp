#include "tge/core/thread_pool.hpp"

#include <algorithm>
#include <map>
#include <memory>
#include <utility>

#if defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86))
#include <intrin.h>
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

}  // namespace

ThreadPool::ThreadPool(int threads) {
  size_ = std::max(1, threads);
  workers_.reserve(static_cast<std::size_t>(size_ - 1));
  for (int index = 0; index < size_ - 1; ++index) {
    workers_.emplace_back([this] { worker(); });
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
    const int start = next_.fetch_add(grain, std::memory_order_relaxed);
    if (start >= count) {
      return;
    }
    const int end = std::min(start + grain, count);
    for (int index = start; index < end; ++index) {
      body(index);
    }
  }
}

void ThreadPool::parallel_for(int count, int grain,
                              const std::function<void(int)>& body) {
  if (count <= 0) {
    return;
  }
  grain = std::max(1, grain);
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
