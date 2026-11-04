#include <cstdint>

#include "harness.hpp"
#include "tge/memory/io_advice.hpp"

TGE_TEST(io_advice_align_range_page) {
  auto r = tge::memory::align_range(0x1003, 100, 4096);
  TGE_CHECK(r.addr == 0x1000);
  TGE_CHECK(r.len == 4096);
  auto empty = tge::memory::align_range(0x2000, 0, 4096);
  TGE_CHECK(empty.len == 0);
  bool threw = false;
  try {
    tge::memory::align_range(0x1000, 100, 3000);
  } catch (const std::runtime_error&) {
    threw = true;
  }
  TGE_CHECK(threw);
}

TGE_TEST(io_advice_budget_reserve_release) {
  tge::memory::PrefetchBudget budget(1024);
  TGE_CHECK(budget.try_reserve(512));
  TGE_CHECK(budget.used() == 512);
  TGE_CHECK(!budget.try_reserve(600));
  TGE_CHECK(budget.used() == 512);
  TGE_CHECK(budget.try_reserve(512));
  TGE_CHECK(!budget.try_reserve(1));
  budget.release(1024);
  TGE_CHECK(budget.used() == 0);
  bool threw = false;
  try {
    budget.release(1);
  } catch (const std::runtime_error&) {
    threw = true;
  }
  TGE_CHECK(threw);
}
