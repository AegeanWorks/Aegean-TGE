#pragma once

#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "tge/models/gguf.hpp"

namespace tge::models {

/// Turns token ids back into text.
///
/// GGUF stores vocabulary entries in the GPT-2 byte-level representation: every
/// raw byte is remapped onto a printable code point so the vocabulary is valid
/// UTF-8. Decoding is therefore a table lookup per character, not a BPE search.
class Tokenizer {
 public:
  static std::unique_ptr<Tokenizer> from_gguf(const gguf::File& file,
                                              std::string& error);

  std::string decode(const std::vector<int>& tokens) const;
  std::string decode_span(const int* tokens, int count) const;

  int vocab_size() const { return static_cast<int>(pieces_.size()); }

 private:
  Tokenizer() = default;

  std::vector<std::string> pieces_;
  std::array<std::int32_t, 512> byte_of_codepoint_{};
};

}  // namespace tge::models
