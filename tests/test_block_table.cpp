#include <stdexcept>

#include "harness.hpp"
#include "tge/models/block_table.hpp"

namespace {

using tge::models::BlockTable;

TGE_TEST(block_table_alloc_free_reuse) {
  BlockTable table(4, 16);
  TGE_CHECK(table.free_count() == 4);
  const int a = table.allocate();
  const int b = table.allocate();
  TGE_CHECK(a != b);
  TGE_CHECK(table.free_count() == 2);
  table.release(a);
  TGE_CHECK(table.free_count() == 3);
  const int c = table.allocate();
  TGE_CHECK(c == a);
  TGE_CHECK(table.refcount(c) == 1);
}

TGE_TEST(block_table_full_throws) {
  BlockTable table(1, 16);
  table.allocate();
  bool threw = false;
  try {
    table.allocate();
  } catch (const std::runtime_error&) {
    threw = true;
  }
  TGE_CHECK(threw);
}

TGE_TEST(block_table_slot_for) {
  auto s0 = BlockTable::slot_for(0, 16);
  TGE_CHECK(s0.first == 0 && s0.second == 0);
  auto s1 = BlockTable::slot_for(17, 16);
  TGE_CHECK(s1.first == 1 && s1.second == 1);
  auto s2 = BlockTable::slot_for(64, 16);
  TGE_CHECK(s2.first == 4 && s2.second == 0);
  bool threw = false;
  try {
    BlockTable::slot_for(-1, 16);
  } catch (const std::runtime_error&) {
    threw = true;
  }
  TGE_CHECK(threw);
}

TGE_TEST(block_table_share_cow) {
  BlockTable table(2, 16);
  const int a = table.allocate();
  table.share(a);
  TGE_CHECK(table.refcount(a) == 2);
  table.release(a);
  TGE_CHECK(table.refcount(a) == 1);
  TGE_CHECK(table.free_count() == 1);
  table.release(a);
  TGE_CHECK(table.free_count() == 2);
}

TGE_TEST(block_table_prefix_hash) {
  const int prefix_a[] = {1, 2, 3, 4};
  const int prefix_b[] = {1, 2, 3, 5};
  const std::uint64_t ha1 = BlockTable::hash_tokens(prefix_a, 4);
  const std::uint64_t ha2 = BlockTable::hash_tokens(prefix_a, 4);
  const std::uint64_t hb = BlockTable::hash_tokens(prefix_b, 4);
  TGE_CHECK(ha1 == ha2);
  TGE_CHECK(ha1 != hb);
  BlockTable table(4, 16);
  TGE_CHECK(table.find_prefix(ha1) == -1);
  const int blk = table.allocate();
  table.remember_prefix(ha1, blk);
  TGE_CHECK(table.find_prefix(ha1) == blk);
  TGE_CHECK(table.find_prefix(hb) == -1);
}

TGE_TEST(block_table_bad_ids_throw) {
  BlockTable table(2, 16);
  bool threw = false;
  try {
    table.share(7);
  } catch (const std::runtime_error&) {
    threw = true;
  }
  TGE_CHECK(threw);
  threw = false;
  try {
    table.release(0);
  } catch (const std::runtime_error&) {
    threw = true;
  }
  TGE_CHECK(threw);
}

}  // namespace
