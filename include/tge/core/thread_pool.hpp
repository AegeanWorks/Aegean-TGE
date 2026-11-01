#pragma once

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

namespace tge::core {

/// Fixed-size worker pool with a blocking `parallel_for`.
///
/// Work is claimed dynamically rather than partitioned up front. On a hybrid
/// part that matters: performance and efficiency cores differ in throughput by
/// a wide margin, and a static split makes the slowest worker decide when the
/// whole call finishes.
class ThreadPool {
 public:
  /// `threads` counts the calling thread. ThreadPool(1) spawns no workers and
  /// runs every parallel_for body inline.
  explicit ThreadPool(int threads);
  ~ThreadPool();

  ThreadPool(const ThreadPool&) = delete;
  ThreadPool& operator=(const ThreadPool&) = delete;

  int size() const { return size_; }

  /// Invokes `body(index)` for each index in [0, count), in chunks of `grain`.
  /// Returns when every chunk has completed. Calls are serialized.
  void parallel_for(int count, int grain, const std::function<void(int)>& body);

 private:
  void worker();
  void claim_chunks(const std::function<void(int)>& body, int count, int grain);

  int size_ = 1;
  std::vector<std::thread> workers_;

  std::mutex mutex_;
  std::condition_variable wake_;
  std::condition_variable finished_;

  const std::function<void(int)>* body_ = nullptr;
  int count_ = 0;
  int grain_ = 1;
  std::size_t completed_ = 0;
  std::uint64_t generation_ = 0;
  bool stop_ = false;

  std::atomic<int> next_{0};
};

/// Returns a process-lifetime pool for the requested thread count, creating it
/// on first use. Pools are cached so repeated benchmark iterations do not pay
/// thread creation cost.
ThreadPool& acquire_pool(int threads);

/// Logical processors usable by this process.
int default_thread_count();

}  // namespace tge::core
