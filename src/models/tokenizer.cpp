#include "tge/models/tokenizer.hpp"

#include <cstddef>

namespace tge::models {
namespace {

/// Reads one UTF-8 code point and advances the cursor. Returns 0 at the end.
std::uint32_t next_codepoint(const std::string& text, std::size_t& position) {
  if (position >= text.size()) {
    return 0;
  }
  const auto lead = static_cast<unsigned char>(text[position]);
  int extra = 0;
  std::uint32_t value = 0;
  if (lead < 0x80) {
    ++position;
    return lead;
  }
  if ((lead & 0xE0) == 0xC0) {
    extra = 1;
    value = lead & 0x1Fu;
  } else if ((lead & 0xF0) == 0xE0) {
    extra = 2;
    value = lead & 0x0Fu;
  } else if ((lead & 0xF8) == 0xF0) {
    extra = 3;
    value = lead & 0x07u;
  } else {
    ++position;
    return 0xFFFD;
  }
  if (position + static_cast<std::size_t>(extra) >= text.size()) {
    position = text.size();
    return 0xFFFD;
  }
  for (int index = 1; index <= extra; ++index) {
    const auto continuation = static_cast<unsigned char>(text[position + index]);
    value = (value << 6) | (continuation & 0x3Fu);
  }
  position += static_cast<std::size_t>(extra) + 1;
  return value;
}

}  // namespace

std::unique_ptr<Tokenizer> Tokenizer::from_gguf(const gguf::File& file,
                                                std::string& error) {
  const gguf::Value* tokens = file.find("tokenizer.ggml.tokens");
  if (tokens == nullptr || tokens->type != gguf::ValueType::Array) {
    error = "model has no tokenizer.ggml.tokens array";
    return nullptr;
  }

  std::unique_ptr<Tokenizer> tokenizer(new Tokenizer());
  tokenizer->pieces_.reserve(tokens->elements.size());
  for (const gguf::Value& element : tokens->elements) {
    tokenizer->pieces_.push_back(element.string_value);
  }
  if (tokenizer->pieces_.empty()) {
    error = "tokenizer vocabulary is empty";
    return nullptr;
  }

  // The byte-level alphabet: printable bytes map to themselves, the rest are
  // shifted above 255 in a fixed order. Invert that so decoding is a lookup.
  tokenizer->byte_of_codepoint_.fill(-1);
  int next = 0;
  for (int byte = 0; byte < 256; ++byte) {
    const bool printable = (byte >= '!' && byte <= '~') ||
                           (byte >= 161 && byte <= 172) ||
                           (byte >= 174 && byte <= 255);
    const int code_point = printable ? byte : 256 + next++;
    if (code_point >= 0 && code_point < 512) {
      tokenizer->byte_of_codepoint_[static_cast<std::size_t>(code_point)] =
          byte;
    }
  }
  return tokenizer;
}

std::string Tokenizer::decode_span(const int* tokens, int count) const {
  std::string text;
  if (tokens == nullptr || count <= 0) {
    return text;
  }
  text.reserve(static_cast<std::size_t>(count) * 4);

  for (int index = 0; index < count; ++index) {
    const int id = tokens[index];
    if (id < 0 || id >= static_cast<int>(pieces_.size())) {
      continue;
    }
    const std::string& piece = pieces_[static_cast<std::size_t>(id)];
    std::size_t position = 0;
    while (position < piece.size()) {
      const std::uint32_t code_point = next_codepoint(piece, position);
      const int byte =
          code_point < byte_of_codepoint_.size()
              ? byte_of_codepoint_[static_cast<std::size_t>(code_point)]
              : -1;
      if (byte >= 0) {
        text.push_back(static_cast<char>(byte));
      }
    }
  }
  return text;
}

std::string Tokenizer::decode(const std::vector<int>& tokens) const {
  return decode_span(tokens.data(), static_cast<int>(tokens.size()));
}

}  // namespace tge::models
