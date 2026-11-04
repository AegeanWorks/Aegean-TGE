#include <stdexcept>

#include "harness.hpp"
#include "tge/models/block_table.hpp"
#include "tge/models/kv_block_layout.hpp"

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

TGE_TEST(kv_layout_k_round_trip_identity) {
  const int rows = 4, dim = 8;
  std::vector<float> src(static_cast<std::size_t>(rows * dim));
  for (std::size_t i = 0; i < src.size(); ++i) src[i] = static_cast<float>(i) * 0.25F + 1.0F;
  auto col = tge::models::reorder_k_block_to_column_major(src.data(), rows, dim);
  TGE_CHECK(col.size() == src.size());
  // Column-major spot check: col[d * rows + r] == src[r * dim + d].
  TGE_CHECK(col[static_cast<std::size_t>(1 * rows + 2)] == src[static_cast<std::size_t>(2 * dim + 1)]);
  auto back = tge::models::reorder_k_block_to_row_major(col.data(), rows, dim);
  TGE_CHECK(back.size() == src.size());
  for (std::size_t i = 0; i < src.size(); ++i) TGE_CHECK(back[i] == src[i]);
}

TGE_TEST(kv_layout_v_row_major_identity) {
  const int rows = 3, dim = 5;
  std::vector<float> src(static_cast<std::size_t>(rows * dim), 2.5F);
  src[0] = -1.0F;
  auto out = tge::models::reorder_v_block_row_major(src.data(), rows, dim);
  TGE_CHECK(out.size() == src.size());
  for (std::size_t i = 0; i < src.size(); ++i) TGE_CHECK(out[i] == src[i]);
}

TGE_TEST(kv_layout_gather_blocks_order) {
  const int dim = 2;
  std::vector<std::vector<float>> blocks = {{1.0F, 2.0F}, {3.0F, 4.0F}, {5.0F, 6.0F}};
  const int table[] = {2, 0};
  auto out = tge::models::gather_blocks_row_major(blocks, table, 2, dim);
  TGE_CHECK(out.size() == 4);
  TGE_CHECK(out[0] == 5.0F && out[1] == 6.0F && out[2] == 1.0F && out[3] == 2.0F);
  bool threw = false;
  const int bad[] = {9};
  try {
    tge::models::gather_blocks_row_major(blocks, bad, 1, dim);
  } catch (const std::runtime_error&) {
    threw = true;
  }
  TGE_CHECK(threw);
}

}  // namespace
