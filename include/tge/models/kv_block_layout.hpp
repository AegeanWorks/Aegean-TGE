#pragma once

// KV block layout helpers for IDEA-1585 (K column-major / V row-major per
// block, bit-neutral reorder). Pure functions, no engine state; the Qwen2
// cache keeps its layout until the wiring step. Round-trip must be exact.

#include <cstddef>
#include <stdexcept>
#include <vector>

namespace tge::models {

// Reorders one K block from row-major [rows][dim] to column-major
// [dim][rows] (contiguous columns). Exact reorder, no arithmetic.
inline std::vector<float> reorder_k_block_to_column_major(const float* src,
                                                          int rows, int dim) {
  if (rows <= 0 || dim <= 0 || src == nullptr) {
    throw std::runtime_error("kv_layout: bad k reorder arguments");
  }
  std::vector<float> dst(static_cast<std::size_t>(rows) * static_cast<std::size_t>(dim));
  for (int r = 0; r < rows; ++r) {
    for (int d = 0; d < dim; ++d) {
      dst[static_cast<std::size_t>(d) * static_cast<std::size_t>(rows) +
          static_cast<std::size_t>(r)] =
          src[static_cast<std::size_t>(r) * static_cast<std::size_t>(dim) +
              static_cast<std::size_t>(d)];
    }
  }
  return dst;
}

// Inverse of the above: column-major [dim][rows] back to row-major.
inline std::vector<float> reorder_k_block_to_row_major(const float* src,
                                                       int rows, int dim) {
  if (rows <= 0 || dim <= 0 || src == nullptr) {
    throw std::runtime_error("kv_layout: bad k inverse arguments");
  }
  std::vector<float> dst(static_cast<std::size_t>(rows) * static_cast<std::size_t>(dim));
  for (int r = 0; r < rows; ++r) {
    for (int d = 0; d < dim; ++d) {
      dst[static_cast<std::size_t>(r) * static_cast<std::size_t>(dim) +
          static_cast<std::size_t>(d)] =
          src[static_cast<std::size_t>(d) * static_cast<std::size_t>(rows) +
              static_cast<std::size_t>(r)];
    }
  }
  return dst;
}

// V row-major per block is the dense order already: identity copy that
// documents the contract and gives the wiring step a single call site.
inline std::vector<float> reorder_v_block_row_major(const float* src, int rows,
                                                    int dim) {
  if (rows <= 0 || dim <= 0 || src == nullptr) {
    throw std::runtime_error("kv_layout: bad v reorder arguments");
  }
  const std::size_t n = static_cast<std::size_t>(rows) * static_cast<std::size_t>(dim);
  return std::vector<float>(src, src + n);
}

// Gathers rows from physical blocks in logical order into one contiguous
// span: out = concat(blocks[table[i]][:]). Used by the future slot-mapping
// gather; here it is pure and tested without the cache.
inline std::vector<float> gather_blocks_row_major(
    const std::vector<std::vector<float>>& blocks, const int* table, int count,
    int dim) {
  if (count < 0 || dim <= 0 || (count > 0 && table == nullptr)) {
    throw std::runtime_error("kv_layout: bad gather arguments");
  }
  std::vector<float> out;
  out.reserve(static_cast<std::size_t>(count) * static_cast<std::size_t>(dim));
  for (int i = 0; i < count; ++i) {
    const int b = table[i];
    if (b < 0 || b >= static_cast<int>(blocks.size())) {
      throw std::runtime_error("kv_layout: table block out of range");
    }
    const auto& blk = blocks[static_cast<std::size_t>(b)];
    if (blk.size() % static_cast<std::size_t>(dim) != 0) {
      throw std::runtime_error("kv_layout: block size not a multiple of dim");
    }
    out.insert(out.end(), blk.begin(), blk.end());
  }
  return out;
}

}  // namespace tge::models
