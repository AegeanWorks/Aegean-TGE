#include <atomic>

#include "harness.hpp"
#include "tge/core/thread_pool.hpp"
#include "tge/cpu/kernels.hpp"

TGE_TEST(thread_pool_runs_every_index_once) {
  tge::core::ThreadPool pool(4);
  constexpr int count = 1000;

  std::atomic<int> hits[count];
  for (std::atomic<int>& hit : hits) {
    hit.store(0);
  }

  pool.parallel_for(count, 7, [&hits](int index) {
    hits[index].fetch_add(1, std::memory_order_relaxed);
  });

  int singles = 0;
  for (const std::atomic<int>& hit : hits) {
    if (hit.load() == 1) {
      ++singles;
    }
  }
  TGE_CHECK(singles == count);
}

TGE_TEST(thread_pool_single_thread_runs_inline) {
  tge::core::ThreadPool pool(1);
  int total = 0;
  pool.parallel_for(10, 1, [&total](int index) { total += index; });
  TGE_CHECK(total == 45);
}

TGE_TEST(thread_pool_handles_empty_ranges) {
  tge::core::ThreadPool pool(4);
  int calls = 0;
  pool.parallel_for(0, 1, [&calls](int) { ++calls; });
  TGE_CHECK(calls == 0);
}

TGE_TEST(thread_pool_repeats_without_leaking_work) {
  tge::core::ThreadPool pool(4);
  std::atomic<long long> total{0};
  for (int round = 0; round < 20; ++round) {
    pool.parallel_for(100, 3, [&total](int index) {
      total.fetch_add(index, std::memory_order_relaxed);
    });
  }
  TGE_CHECK(total.load() == 20LL * 4950LL);
}

TGE_TEST(resolve_thread_count_is_bounded) {
  TGE_CHECK(tge::cpu::resolve_thread_count(1) == 1);
  TGE_CHECK(tge::cpu::resolve_thread_count(4) == 4);
  TGE_CHECK(tge::cpu::resolve_thread_count(100000) == 1024);
  TGE_CHECK(tge::cpu::resolve_thread_count(0) >= 1);
}
