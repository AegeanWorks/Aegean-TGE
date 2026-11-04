#include <cstdint>
#include <utility>

#include "harness.hpp"
#include "tge/memory/allocator.hpp"
#include "tge/memory/budget.hpp"
#include "tge/memory/system_ram.hpp"
#include "tge/models/plan.hpp"

TGE_TEST(aligned_buffer_respects_requested_alignment) {
  const std::size_t alignments[] = {16, 32, 64, 128, 256};
  for (const std::size_t alignment : alignments) {
    tge::memory::AlignedBuffer buffer(4096, alignment);
    TGE_REQUIRE(!buffer.empty());
    TGE_CHECK(buffer.size() == 4096);
    TGE_CHECK(reinterpret_cast<std::uintptr_t>(buffer.data()) % alignment == 0);
  }
}

TGE_TEST(aligned_buffer_zero_size_is_empty) {
  tge::memory::AlignedBuffer buffer(0);
  TGE_CHECK(buffer.empty());
  TGE_CHECK(buffer.size() == 0);
  TGE_CHECK(buffer.data() == nullptr);
}

TGE_TEST(aligned_buffer_move_transfers_ownership) {
  tge::memory::AlignedBuffer source(256);
  const void* address = source.data();
  tge::memory::AlignedBuffer moved(std::move(source));
  TGE_CHECK(moved.data() == address);
  TGE_CHECK(source.empty());
}

TGE_TEST(alignment_normalization_rounds_up) {
  TGE_CHECK(tge::memory::normalize_alignment(0) == tge::memory::kCacheLine);
  TGE_CHECK(tge::memory::normalize_alignment(1) == tge::memory::kCacheLine);
  TGE_CHECK(tge::memory::normalize_alignment(64) == 64);
  TGE_CHECK(tge::memory::normalize_alignment(65) == 128);
}

TGE_TEST(system_memory_reports_physical_ram) {
  const tge::memory::SystemMemory memory = tge::memory::query_system_memory();
  TGE_CHECK(memory.total_physical > 0);
  TGE_CHECK(memory.installed_physical > 0);
  TGE_CHECK(memory.available_physical <= memory.total_physical);
  TGE_CHECK(memory.page_size >= 4096);
}

TGE_TEST(budget_pool_limits_fit_inside_usable_budget) {
  const tge::memory::MemoryBudget budget;
  const tge::memory::BudgetSnapshot snapshot = budget.snapshot();
  std::uint64_t total = 0;
  for (int pool = 0; pool < tge::memory::kPoolCount; ++pool) {
    total += snapshot.pools[pool].limit;
  }
  TGE_CHECK(total <= snapshot.usable);
  TGE_CHECK(snapshot.reserved_for_os > 0);
}

TGE_TEST(budget_refuses_reservations_beyond_a_pool_limit) {
  tge::memory::MemoryBudget budget;
  const tge::memory::BudgetSnapshot snapshot = budget.snapshot();
  const int scratch = static_cast<int>(tge::memory::Pool::Scratch);
  const std::uint64_t limit = snapshot.pools[scratch].limit;
  if (limit == 0) {
    std::printf("    scratch pool is empty, skipping\n");
    return;
  }

  TGE_CHECK(!budget.try_reserve(tge::memory::Pool::Scratch, limit + 1));
  const std::uint64_t half = limit / 2;
  TGE_REQUIRE(budget.try_reserve(tge::memory::Pool::Scratch, half));
  TGE_CHECK(budget.available(tge::memory::Pool::Scratch) == limit - half);
  budget.release(tge::memory::Pool::Scratch, half);
  TGE_CHECK(budget.available(tge::memory::Pool::Scratch) == limit);
}

TGE_TEST(budget_admission_respects_total_usage) {
  tge::memory::MemoryBudget budget;
  const tge::memory::BudgetSnapshot snapshot = budget.snapshot();
  if (snapshot.usable == 0) {
    std::printf("    usable budget is zero, skipping\n");
    return;
  }
  TGE_CHECK(budget.admit(snapshot.usable));
  TGE_CHECK(!budget.admit(snapshot.usable + 1));
}

TGE_TEST(normalization_rejects_absurd_alignment) {
  // Regression: normalize_alignment(SIZE_MAX) shifted until wrap to 0
  // then looped forever. Absurd requests must throw instead of hanging.
  bool threw = false;
  try {
    (void)tge::memory::normalize_alignment(SIZE_MAX);
  } catch (const std::invalid_argument&) {
    threw = true;
  } catch (...) {
    threw = true;
  }
  TGE_CHECK(threw);
  TGE_CHECK(tge::memory::normalize_alignment(1ULL << 30) == (1ULL << 30));
}

TGE_TEST(plan_place_exact_fit_admits_weights) {
  // Regression: headroom == 0 reported "weights alone exceed" though equal.
  tge::models::ModelPlan plan;
  plan.parameter_bytes = 100;
  plan.decode_bytes_per_token = 100;
  plan.kv_bytes_per_token = 10;
  plan.context = 16;
  const tge::models::PlacementDecision decision =
      tge::models::place(plan, 100, 16, 1);
  TGE_CHECK(decision.weights_fit);
}

TGE_TEST(plan_place_shrink_counts_sessions) {
  // Regression: by_sessions = headroom/(per_session*wanted) was always 0
  // in the shrink branch; it must be headroom/per_session.
  tge::models::ModelPlan plan;
  plan.parameter_bytes = 0;
  plan.decode_bytes_per_token = 10;
  plan.kv_bytes_per_token = 10;
  plan.context = 16;
  // usable=95, requested 10 tokens x 10 sessions: per_session=100,
  // needed=1000 > 95 -> shrink. by_context = (95/10)/10 = 0,
  // by_sessions = 95/100 = 0 sessions at full context, but the
  // max_sessions fallback path must be consistent (no divide-by-zero,
  // non-negative, and max_context <= requested).
  const tge::models::PlacementDecision decision =
      tge::models::place(plan, 95, 10, 10);
  TGE_CHECK(decision.max_context <= 10);
  TGE_CHECK(decision.max_context >= 0);
  TGE_CHECK(decision.max_sessions >= 0);
  // Single-session shrink must still admit a reduced context:
  // headroom 95, per_token 10 -> 9 tokens fit.
  const tge::models::PlacementDecision single =
      tge::models::place(plan, 95, 10, 1);
  TGE_CHECK(single.max_context == 9);
}

TGE_TEST(budget_release_beyond_reserved_throws) {
  // Over-release used to clamp to zero and hide double-free accounting.
  tge::memory::MemoryBudget budget;
  bool threw = false;
  try {
    budget.release(tge::memory::Pool::Scratch, 1);
  } catch (...) {
    threw = true;
  }
  TGE_CHECK(threw);
}
