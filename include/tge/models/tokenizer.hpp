#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "tge/models/gguf.hpp"

namespace tge::models {

/// Converts between token ids and text.
///
/// Two vocabulary families are supported, selected by the file:
/// - GPT-2/BPE (`tokenizer.ggml.model` anything but "llama", or a merges
///   table present): byte-level pieces plus the merge table.
/// - SentencePiece ("llama" model with no merges table, e.g. Phi-3): pieces
///   with scores; encoding is the llama.cpp SPM-BPE loop (dummy U+2581
///   prefix, greedy highest-score pair merges over merge ranks), not a
///   Viterbi pass.
/// Decoding inverts the byte alphabet for BPE and maps U+2581 to space for
/// SPM. CONTROL pieces are omitted; USER_DEFINED pieces print.
class Tokenizer {
 public:
  static std::unique_ptr<Tokenizer> from_gguf(const gguf::File& file,
                                              std::string& error);

  /// Opens the file with full metadata. The model loader skips large arrays, so
  /// a caller that needs the vocabulary cannot reuse that File.
  ///
  /// When `cache_dir` is non-empty, a validated sidecar holding the complete
  /// tokenizer state is restored instead of rebuilding, and a miss rebuilds
  /// and rewrites it. The sidecar is keyed on path, size and mtime; any
  /// validation failure is a silent miss. Empty disables the cache.
  static std::unique_ptr<Tokenizer> from_path(
      const std::string& path, std::string& error,
      const std::string& cache_dir = std::string());

  std::string decode(const std::vector<int>& tokens) const;
  std::string decode_span(const int* tokens, int count) const;

  /// Byte-pair encoding of `text`.
  ///
  /// Pre-tokenization approximates the model family's split rules. ASCII text
  /// matches the reference tokenizer exactly; unusual Unicode punctuation can
  /// split differently, which changes the ids but not the decoded text.
  ///
  /// `match_special` matches CONTROL and USER_DEFINED tokens literally. A
  /// server should pass false for user content, so that content cannot forge a
  /// chat template or a tool call.
  std::vector<int> encode(const std::string& text,
                          bool match_special = true) const;

  /// True when the file declares the tiktoken-style `llama-bpe`
  /// pre-tokenizer (as opposed to the default Qwen2-style splitter).
  /// The only rule that differs today is digit runs: tiktoken groups up to
  /// three digits per piece, Qwen2 keeps one digit per piece.
  bool llama_bpe() const { return llama_bpe_digits_; }

  /// True when the file declares the SentencePiece family: a "llama" model
  /// with no merges table (Phi-3 and friends). Encoding runs the llama.cpp
  /// SPM-BPE merge loop over piece scores (merge ranks), not Viterbi.
  bool spm() const { return spm_; }

  int vocab_size() const { return static_cast<int>(pieces_.size()); }

 private:
  Tokenizer() = default;

  /// Flat open-addressing string map: one arena holding every key back to
  /// back plus a slot table carrying (hash, offset, length, value, state).
  /// `std::unordered_map` allocates a node and a string per entry, which is
  /// most of the tokenizer build cost on a 128k/280k-entry vocabulary; the
  /// arena lets the complete final state be serialized and restored verbatim.
  /// Semantics match the previous maps where it matters: insert keeps the
  /// first value, erase tombstones, lookups return the stored int.
  struct StringIndex {
    struct Slot {
      std::uint64_t hash = 0;
      std::uint32_t offset = 0;
      std::uint32_t length = 0;
      int value = 0;
      std::uint8_t state = 0;  ///< 0 empty, 1 occupied, 2 tombstone.
    };
    struct Entry {
      int second = 0;
    };
    class Iterator {
     public:
      Iterator() = default;
      explicit Iterator(const Slot* slot) : slot_(slot), entry_{slot->value} {}
      bool operator==(const Iterator& other) const { return slot_ == other.slot_; }
      bool operator!=(const Iterator& other) const { return slot_ != other.slot_; }
      const Entry* operator->() const { return &entry_; }

     private:
      const Slot* slot_ = nullptr;
      Entry entry_{};
    };

    std::string arena_;
    std::vector<Slot> slots_;
    std::size_t size_ = 0;
    std::size_t tombstones_ = 0;

    static std::uint64_t hash(const std::string& key) {
      std::uint64_t value = 1469598103934665603ULL;
      for (const char byte : key) {
        value ^= static_cast<unsigned char>(byte);
        value *= 1099511628211ULL;
      }
      return value;
    }

    Iterator end() const { return Iterator(); }

    Iterator find(const std::string& key) const {
      if (slots_.empty()) {
        return end();
      }
      const std::uint64_t value = hash(key);
      const std::size_t mask = slots_.size() - 1;
      std::size_t index = static_cast<std::size_t>(value) & mask;
      for (std::size_t probe = 0; probe <= mask; ++probe) {
        const Slot& slot = slots_[index];
        if (slot.state == 0) {
          return end();
        }
        if (slot.state == 1 && slot.hash == value &&
            slot.length == key.size() &&
            std::memcmp(arena_.data() + slot.offset, key.data(),
                        key.size()) == 0) {
          return Iterator(&slot);
        }
        index = (index + 1) & mask;
      }
      return end();
    }

    bool empty() const { return size_ == 0; }

    void reserve(std::size_t count) {
      std::size_t slots = 16;
      while (slots * 7 < count * 10) {
        slots <<= 1;
      }
      if (slots > slots_.size()) {
        rehash(slots);
      }
    }

    std::pair<Iterator, bool> emplace(const std::string& key, int value) {
      const Iterator found = find(key);
      if (found != end()) {
        return std::make_pair(found, false);
      }
      if (slots_.empty()) {
        reserve(16);
      } else if ((size_ + tombstones_ + 1) * 10 > slots_.size() * 7) {
        rehash(slots_.size() * 2);
      }
      const std::uint64_t value_hash = hash(key);
      const std::uint32_t offset = static_cast<std::uint32_t>(arena_.size());
      arena_.append(key);
      insert_slot(value_hash, offset, static_cast<std::uint32_t>(key.size()),
                  value);
      return std::make_pair(find(key), true);
    }

    std::size_t erase(const std::string& key) {
      if (slots_.empty()) {
        return 0;
      }
      const std::uint64_t value = hash(key);
      const std::size_t mask = slots_.size() - 1;
      std::size_t index = static_cast<std::size_t>(value) & mask;
      for (std::size_t probe = 0; probe <= mask; ++probe) {
        Slot& slot = slots_[index];
        if (slot.state == 0) {
          return 0;
        }
        if (slot.state == 1 && slot.hash == value &&
            slot.length == key.size() &&
            std::memcmp(arena_.data() + slot.offset, key.data(),
                        key.size()) == 0) {
          slot.state = 2;
          --size_;
          ++tombstones_;
          return 1;
        }
        index = (index + 1) & mask;
      }
      return 0;
    }

    void rehash(std::size_t slots) {
      std::vector<Slot> old;
      old.swap(slots_);
      slots_.assign(slots, Slot{});
      size_ = 0;
      tombstones_ = 0;
      for (const Slot& slot : old) {
        if (slot.state == 1) {
          insert_slot(slot.hash, slot.offset, slot.length, slot.value);
        }
      }
    }

    void insert_slot(std::uint64_t value, std::uint32_t offset,
                     std::uint32_t length, int stored) {
      const std::size_t mask = slots_.size() - 1;
      std::size_t index = static_cast<std::size_t>(value) & mask;
      std::size_t tombstone = slots_.size();
      while (slots_[index].state != 0) {
        if (slots_[index].state == 2 && tombstone == slots_.size()) {
          tombstone = index;
        }
        index = (index + 1) & mask;
      }
      if (tombstone != slots_.size()) {
        index = tombstone;
        --tombstones_;
      }
      Slot& slot = slots_[index];
      slot.hash = value;
      slot.offset = offset;
      slot.length = length;
      slot.value = stored;
      slot.state = 1;
      ++size_;
    }
  };

  /// Sidecar load/save used by the cached `from_path`. Both are silent on
  /// failure: the cache is disposable and a miss simply rebuilds.
  static bool load_cache(const std::string& file, std::uint64_t key,
                         std::unique_ptr<Tokenizer>& tokenizer);
  static void save_cache(const std::string& file, std::uint64_t key,
                         const Tokenizer& tokenizer);
  static bool cache_location(const std::string& model_path,
                             const std::string& cache_dir,
                             std::uint64_t& key, std::string& file);

  /// Parses a `<0xNN>` byte-fallback piece. Returns -1 for anything else.
  static int byte_piece_value(const std::string& piece);

  /// Byte-pair encodes a run with no control tokens in it.
  std::vector<int> encode_plain(const std::string& text) const;

  /// SentencePiece-encoding of a run with no control tokens in it:
  /// SentencePiece-BPE per llama.cpp (`llm_tokenizer_spm`) — dummy prefix,
  /// space-to-U+2581, then greedy highest-score pair merges. Scores are
  /// merge ranks, so a Viterbi pass would invert the objective.
  /// Special (control/user-defined) pieces never match here; they are only
  /// reachable through the literal control-token pass in `encode`.
  std::vector<int> encode_spm(const std::string& text) const;

  /// Arms SentencePiece mode on a freshly loaded vocabulary: reads the
  /// scores, records byte fallbacks, and removes byte and special pieces
  /// from the Viterbi lookup (bytes match by value, specials only by the
  /// literal control-token pass). Returns false with `error` set when the
  /// file cannot support the mode.
  static bool enable_spm(Tokenizer* tokenizer, const gguf::File& file,
                         std::string& error);

  /// Index into `control_tokens_` of the control token that starts at
  /// `position`, or -1. Control tokens are matched literally, before the
  /// pre-tokenizer, so a chat template typed as text stays one id.
  int match_control(const std::string& text, std::size_t position) const;

  std::vector<std::string> pieces_;
  std::array<std::int32_t, 512> byte_of_codepoint_{};
  std::array<std::uint32_t, 256> codepoint_of_byte_{};
  StringIndex vocab_;
  StringIndex merge_rank_;
  std::vector<std::pair<std::string, int>> control_tokens_;
  std::unordered_map<char, std::vector<int>> control_by_first_;
  /// One flag per vocabulary entry. Control tokens are matched in text but
  /// omitted from decoded output, as the reference tokenizer does.
  std::vector<std::uint8_t> control_;
  /// tiktoken-style digit grouping (see `llama_bpe()`); false keeps the
  /// Qwen2 single-digit splitting the engine originally shipped with.
  bool llama_bpe_digits_ = false;
  /// SentencePiece mode (see `spm()`): Viterbi over `scores_` instead of
  /// merges. The longest piece bounds the per-position lookup window.
  bool spm_ = false;
  std::vector<float> scores_;
  std::array<int, 256> byte_token_{};
};

}  // namespace tge::models
