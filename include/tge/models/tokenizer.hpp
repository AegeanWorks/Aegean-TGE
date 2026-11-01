#pragma once

#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "tge/models/gguf.hpp"

namespace tge::models {

/// Converts between token ids and text.
///
/// GGUF stores vocabulary entries in the GPT-2 byte-level representation: every
/// raw byte is remapped onto a printable code point so the vocabulary is valid
/// UTF-8. Decoding is a table lookup per character; encoding is that plus the
/// merge table.
class Tokenizer {
 public:
  static std::unique_ptr<Tokenizer> from_gguf(const gguf::File& file,
                                              std::string& error);

  /// Opens the file with full metadata. The model loader skips large arrays, so
  /// a caller that needs the vocabulary cannot reuse that File.
  static std::unique_ptr<Tokenizer> from_path(const std::string& path,
                                              std::string& error);

  std::string decode(const std::vector<int>& tokens) const;
  std::string decode_span(const int* tokens, int count) const;

  /// Byte-pair encoding of `text`.
  ///
  /// Pre-tokenization approximates the model family's split rules. ASCII text
  /// matches the reference tokenizer exactly; unusual Unicode punctuation can
  /// split differently, which changes the ids but not the decoded text.
  std::vector<int> encode(const std::string& text) const;

  int vocab_size() const { return static_cast<int>(pieces_.size()); }

 private:
  Tokenizer() = default;

  std::vector<std::string> pieces_;
  std::array<std::int32_t, 512> byte_of_codepoint_{};
  std::array<std::uint32_t, 256> codepoint_of_byte_{};
  std::unordered_map<std::string, int> vocab_;
  std::unordered_map<std::string, int> merge_rank_;
};

}  // namespace tge::models
