#pragma once

// Block-table foundation for IDEA-1517 (paged KV + prefix hash) and
// IDEA-1585 (slot mapping + KV-split scheduler).
//
// Design-only in this step: pure logical mapping + prefix-hash + refcount.
// No engine wiring; `Qwen2` keeps its contiguous cache until the wiring step.
// All misuse fails loud (`std::runtime_error`), matching the forward_batch
// contract. Deterministic, single-threaded, portable C++20.

#include <cstdint>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>

namespace tge::models {

inline constexpr int kBlockTokens = 16;

class BlockTable {
 public:
  explicit BlockTable(int blocks, int block_tokens = kBlockTokens)
      : block_tokens_(block_tokens), refcount_(static_cast<std::size_t>(blocks), 0) {
    if (blocks <= 0) {
      throw std::runtime_error("BlockTable: blocks must be positive");
    }
    if (block_tokens <= 0) {
      throw std::runtime_error("BlockTable: block_tokens must be positive");
    }
    for (int i = blocks - 1; i >= 0; --i) {
      free_.push_back(i);
    }
  }

  int block_tokens() const { return block_tokens_; }
  int capacity() const { return static_cast<int>(refcount_.size()); }
  int free_count() const { return static_cast<int>(free_.size()); }

  /// Pops a physical block. Throws when the table is full.
  int allocate() {
    if (free_.empty()) {
      throw std::runtime_error("BlockTable: table full");
    }
    const int block = free_.back();
    free_.pop_back();
    refcount_[static_cast<std::size_t>(block)] = 1;
    return block;
  }

  /// Shares a live block (copy-on-write prefix sharing). Throws on bad id.
  void share(int block) {
    live(block);
    ++refcount_[static_cast<std::size_t>(block)];
  }

  /// Releases one reference; returns the block to the free list at zero.
  void release(int block) {
    live(block);
    std::size_t i = static_cast<std::size_t>(block);
    if (refcount_[i] <= 0) {
      throw std::runtime_error("BlockTable: release of free block");
    }
    if (--refcount_[i] == 0) {
      free_.push_back(block);
      prefix_owner_.erase(prefix_of_block_.count(block) ? prefix_of_block_[block]
                                                        : kNoHash);
    }
  }

  int refcount(int block) const {
    live(block);
    return refcount_[static_cast<std::size_t>(block)];
  }

  /// Logical position -> (block_index, offset). Pure arithmetic, no state.
  static std::pair<int, int> slot_for(int logical_pos, int block_tokens) {
    if (logical_pos < 0 || block_tokens <= 0) {
      throw std::runtime_error("BlockTable: bad slot_for arguments");
    }
    return {logical_pos / block_tokens, logical_pos % block_tokens};
  }

  /// FNV-1a over little-endian int tokens. Deterministic across runs.
  static std::uint64_t hash_tokens(const int* tokens, int count) {
    if (count < 0 || (count > 0 && tokens == nullptr)) {
      throw std::runtime_error("BlockTable: bad hash arguments");
    }
    std::uint64_t hash = 1469598103934665603ULL;
    for (int i = 0; i < count; ++i) {
      std::uint32_t word = static_cast<std::uint32_t>(tokens[i]);
      for (int b = 0; b < 4; ++b) {
        hash ^= static_cast<std::uint64_t>((word >> (b * 8)) & 0xFFu);
        hash *= 1099511628211ULL;
      }
    }
    return hash;
  }

  /// Records hash -> physical block for prefix-hit lookup.
  void remember_prefix(std::uint64_t hash, int block) {
    live(block);
    prefix_owner_[hash] = block;
    prefix_of_block_[block] = hash;
  }

  /// Returns the physical block for a prefix hash, or -1 on miss.
  int find_prefix(std::uint64_t hash) const {
    auto it = prefix_owner_.find(hash);
    return it == prefix_owner_.end() ? -1 : it->second;
  }

 private:
  static constexpr std::uint64_t kNoHash = 0xFFFFFFFFFFFFFFFFULL;

  void live(int block) const {
    if (block < 0 || block >= capacity()) {
      throw std::runtime_error("BlockTable: block id out of range");
    }
    if (refcount_[static_cast<std::size_t>(block)] <= 0) {
      throw std::runtime_error("BlockTable: block is not live");
    }
  }

  int block_tokens_;
  std::vector<int> refcount_;
  std::vector<int> free_;
  std::unordered_map<std::uint64_t, int> prefix_owner_;
  std::unordered_map<int, std::uint64_t> prefix_of_block_;
};

}  // namespace tge::models
