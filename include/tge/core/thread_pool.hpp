#pragma once

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <exception>
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
  /// runs every parallel_for body inline. When `pin` is true, workers pin
  /// to distinct P-core threads (decode: latency-sensitive, bandwidth-bound,
  /// wants no E-core stragglers or migration). When false, placement stays
  /// with the OS (prefill: compute-bound throughput, wants every core).
  explicit ThreadPool(int threads, bool pin = true);
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
  std::atomic<std::size_t> completed_{0};
  std::atomic<std::uint64_t> generation_{0};
  std::atomic<bool> stop_{false};

  /// Single-flight guard: parallel_for publishes into the shared slots
  /// above, so a concurrent or nested call would corrupt the in-flight
  /// call. The second caller gets an exception instead of silent
  /// corruption (the "Calls are serialized" contract, now enforced).
  std::atomic<bool> active_{false};
  /// First failure banked by any thread; rethrown by the caller after
  /// the join so a throwing body neither terminates the process nor
  /// leaves the pool unusable.
  std::mutex exception_mutex_;
  std::exception_ptr exception_;
  std::atomic<bool> failed_{false};

  /// Workers parked on `wake_`, and whether the caller is parked on
  /// `finished_`. The publisher only pays for a condition-variable notify when
  /// one of them is actually asleep; the common case is a spin handoff.
  std::atomic<int> sleepers_{0};
  std::atomic<int> main_waiting_{0};

  std::atomic<int> next_{0};
};

/// Returns the process-lifetime fallback pool for the requested thread count.
/// Models own their pools and pass them through the kernel configs, so this
/// registry is for the standalone benchmarks and tests.
ThreadPool& acquire_pool(int threads);

/// Logical processors usable by this process.
int default_thread_count();

}  // namespace tge::core
