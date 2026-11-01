#include "tge/core/thread_pool.hpp"

#include <algorithm>
#include <map>
#include <memory>
#include <utility>

namespace tge::core {

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
    stop_ = true;
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

  {
    std::lock_guard<std::mutex> lock(mutex_);
    body_ = &body;
    count_ = count;
    grain_ = grain;
    completed_ = 0;
    next_.store(0, std::memory_order_relaxed);
    ++generation_;
  }
  wake_.notify_all();

  claim_chunks(body, count, grain);

  std::unique_lock<std::mutex> lock(mutex_);
  finished_.wait(lock, [this] { return completed_ == workers_.size(); });
  body_ = nullptr;
}

void ThreadPool::worker() {
  std::uint64_t seen = 0;
  std::unique_lock<std::mutex> lock(mutex_);

  for (;;) {
    wake_.wait(lock, [this, seen] {
      return stop_ || generation_ != seen;
    });
    if (stop_) {
      return;
    }
    seen = generation_;
    const std::function<void(int)>* body = body_;
    const int count = count_;
    const int grain = grain_;
    lock.unlock();

    if (body != nullptr) {
      claim_chunks(*body, count, grain);
    }

    lock.lock();
    ++completed_;
    finished_.notify_one();
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
