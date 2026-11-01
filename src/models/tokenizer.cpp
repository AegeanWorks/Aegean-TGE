#include "tge/models/tokenizer.hpp"

#include <cstddef>
#include <limits>

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

void append_codepoint(std::string& out, std::uint32_t code_point) {
  if (code_point < 0x80) {
    out.push_back(static_cast<char>(code_point));
  } else if (code_point < 0x800) {
    out.push_back(static_cast<char>(0xC0 | (code_point >> 6)));
    out.push_back(static_cast<char>(0x80 | (code_point & 0x3F)));
  } else if (code_point < 0x10000) {
    out.push_back(static_cast<char>(0xE0 | (code_point >> 12)));
    out.push_back(static_cast<char>(0x80 | ((code_point >> 6) & 0x3F)));
    out.push_back(static_cast<char>(0x80 | (code_point & 0x3F)));
  } else {
    out.push_back(static_cast<char>(0xF0 | (code_point >> 18)));
    out.push_back(static_cast<char>(0x80 | ((code_point >> 12) & 0x3F)));
    out.push_back(static_cast<char>(0x80 | ((code_point >> 6) & 0x3F)));
    out.push_back(static_cast<char>(0x80 | (code_point & 0x3F)));
  }
}

bool is_ascii_letter(std::uint32_t code_point) {
  return (code_point >= 'a' && code_point <= 'z') ||
         (code_point >= 'A' && code_point <= 'Z');
}

bool is_ascii_digit(std::uint32_t code_point) {
  return code_point >= '0' && code_point <= '9';
}

bool is_space(std::uint32_t code_point) {
  return code_point == ' ' || code_point == '\t' || code_point == '\n' ||
         code_point == '\r' || code_point == 0x0B || code_point == 0x0C ||
         code_point == 0xA0 || code_point == 0x2028 || code_point == 0x2029 ||
         code_point == 0x3000;
}

bool is_punctuation(std::uint32_t code_point) {
  if (code_point < 0x21 || code_point > 0x7E) {
    return false;
  }
  return !is_ascii_letter(code_point) && !is_ascii_digit(code_point);
}

/// Anything outside ASCII that is not whitespace is treated as a letter. That
/// covers accented Latin and CJK correctly; exotic punctuation is folded into
/// the letter run instead of being split, which changes the ids but not the
/// decoded text.
bool is_letter(std::uint32_t code_point) {
  if (code_point < 0x80) {
    return is_ascii_letter(code_point);
  }
  return !is_space(code_point);
}

/// Length of the contraction at `position`, or 0 when there is none.
std::size_t contraction_length(const std::string& text, std::size_t position) {
  if (position >= text.size() || text[position] != '\'') {
    return 0;
  }
  const auto at = [&text](std::size_t index) -> char {
    if (index >= text.size()) {
      return '\0';
    }
    const char value = text[index];
    return (value >= 'A' && value <= 'Z')
               ? static_cast<char>(value - 'A' + 'a')
               : value;
  };
  if (at(position + 1) == 's' || at(position + 1) == 't') {
    return 2;
  }
  if (at(position + 1) == 'm' || at(position + 1) == 'd') {
    return 2;
  }
  if (at(position + 1) == 'r' && at(position + 2) == 'e') {
    return 3;
  }
  if (at(position + 1) == 'v' && at(position + 2) == 'e') {
    return 3;
  }
  if (at(position + 1) == 'l' && at(position + 2) == 'l') {
    return 3;
  }
  return 0;
}

/// Splits text into the pieces the merge step runs over.
std::vector<std::string> split_pre_tokens(const std::string& text) {
  std::vector<std::string> pieces;
  std::size_t position = 0;
  const std::size_t size = text.size();

  const auto peek = [&text, size](std::size_t index, std::size_t& length) {
    if (index >= size) {
      length = 0;
      return std::uint32_t{0};
    }
    std::size_t cursor = index;
    const std::uint32_t value = next_codepoint(text, cursor);
    length = cursor - index;
    return value;
  };

  while (position < size) {
    const std::size_t start = position;

    const std::size_t contraction = contraction_length(text, position);
    if (contraction != 0) {
      position += contraction;
      pieces.push_back(text.substr(start, position - start));
      continue;
    }

    std::size_t length = 0;
    std::uint32_t code_point = peek(position, length);

    if (is_space(code_point) && code_point != ' ' &&
        code_point != '\t') {
      // Newlines stand alone, with any spaces that precede them.
      while (position < size) {
        code_point = peek(position, length);
        if (!is_space(code_point)) {
          break;
        }
        position += length;
      }
      pieces.push_back(text.substr(start, position - start));
      continue;
    }

    if (code_point == ' ') {
      // Count the run. A single trailing space binds to the following word; any
      // extra spaces are their own pieces, matching the reference splitter.
      std::size_t run = 0;
      while (position + run < size && text[position + run] == ' ') {
        ++run;
      }
      std::size_t after = 0;
      const std::uint32_t following = peek(position + run, after);
      if (following != 0 && is_letter(following)) {
        if (run > 1) {
          // Emit the excess spaces, then let the next round take " " + word.
          pieces.push_back(text.substr(start, run - 1));
          position += run - 1;
          continue;
        }
        position += 1;  // consume the single space with the word
        while (position < size) {
          code_point = peek(position, length);
          if (!is_letter(code_point)) {
            break;
          }
          position += length;
        }
        pieces.push_back(text.substr(start, position - start));
        continue;
      }
      position += run;
      pieces.push_back(text.substr(start, run));
      continue;
    }

    if (is_letter(code_point)) {
      while (position < size) {
        code_point = peek(position, length);
        if (!is_letter(code_point)) {
          break;
        }
        position += length;
      }
      pieces.push_back(text.substr(start, position - start));
      continue;
    }

    if (is_ascii_digit(code_point)) {
      position += length;
      pieces.push_back(text.substr(start, position - start));
      continue;
    }

    // Punctuation run, plus any newlines immediately after it.
    while (position < size) {
      code_point = peek(position, length);
      if (!is_punctuation(code_point)) {
        break;
      }
      position += length;
    }
    while (position < size) {
      code_point = peek(position, length);
      if (code_point != '\r' && code_point != '\n') {
        break;
      }
      position += length;
    }
    if (position == start) {
      // Unknown byte, such as an invalid UTF-8 sequence: consume one unit.
      position += length == 0 ? 1 : length;
    }
    pieces.push_back(text.substr(start, position - start));
  }

  return pieces;
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
    tokenizer->codepoint_of_byte_[static_cast<std::size_t>(byte)] =
        static_cast<std::uint32_t>(code_point);
    if (code_point >= 0 && code_point < 512) {
      tokenizer->byte_of_codepoint_[static_cast<std::size_t>(code_point)] =
          byte;
    }
  }

  tokenizer->vocab_.reserve(tokenizer->pieces_.size() * 2);
  for (std::size_t index = 0; index < tokenizer->pieces_.size(); ++index) {
    tokenizer->vocab_.emplace(tokenizer->pieces_[index],
                              static_cast<int>(index));
  }

  // Merge ranks. GGUF stores each rule as "left right"; the separator is a
  // literal space, which the mapped alphabet never produces inside a piece.
  if (const gguf::Value* merges = file.find("tokenizer.ggml.merges");
      merges != nullptr && merges->type == gguf::ValueType::Array) {
    tokenizer->merge_rank_.reserve(merges->elements.size() * 2);
    for (std::size_t index = 0; index < merges->elements.size(); ++index) {
      tokenizer->merge_rank_.emplace(merges->elements[index].string_value,
                                     static_cast<int>(index));
    }
  }

  return tokenizer;
}

std::unique_ptr<Tokenizer> Tokenizer::from_path(const std::string& path,
                                                std::string& error) {
  try {
    // Open with full metadata: the model loader skips large arrays, so its File
    // has no vocabulary in it.
    const gguf::File file = gguf::File::open(path);
    return from_gguf(file, error);
  } catch (const std::exception& exception) {
    error = exception.what();
    return nullptr;
  }
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

std::vector<int> Tokenizer::encode(const std::string& text) const {
  std::vector<int> ids;
  if (text.empty()) {
    return ids;
  }

  for (const std::string& piece : split_pre_tokens(text)) {
    // Map raw bytes onto the byte-level alphabet, then split into one symbol per
    // code point: BPE starts from single characters.
    std::string encoded;
    encoded.reserve(piece.size() * 2);
    for (const char raw : piece) {
      append_codepoint(encoded,
                       codepoint_of_byte_[static_cast<unsigned char>(raw)]);
    }

    std::vector<std::string> symbols;
    symbols.reserve(encoded.size());
    {
      std::size_t position = 0;
      while (position < encoded.size()) {
        const std::size_t begin = position;
        next_codepoint(encoded, position);
        symbols.push_back(encoded.substr(begin, position - begin));
      }
    }

    // Repeatedly apply the lowest-ranked adjacent merge.
    while (symbols.size() > 1) {
      int best_rank = std::numeric_limits<int>::max();
      std::size_t best_index = 0;
      for (std::size_t index = 0; index + 1 < symbols.size(); ++index) {
        const auto found =
            merge_rank_.find(symbols[index] + " " + symbols[index + 1]);
        if (found != merge_rank_.end() && found->second < best_rank) {
          best_rank = found->second;
          best_index = index;
        }
      }
      if (best_rank == std::numeric_limits<int>::max()) {
        break;
      }
      symbols[best_index] += symbols[best_index + 1];
      symbols.erase(symbols.begin() + static_cast<std::ptrdiff_t>(best_index) + 1);
    }

    for (const std::string& symbol : symbols) {
      const auto found = vocab_.find(symbol);
      if (found != vocab_.end()) {
        ids.push_back(found->second);
        continue;
      }
      // Unknown multi-byte symbol: fall back to its individual characters.
      std::size_t position = 0;
      while (position < symbol.size()) {
        const std::size_t begin = position;
        next_codepoint(symbol, position);
        const auto single = vocab_.find(symbol.substr(begin, position - begin));
        if (single != vocab_.end()) {
          ids.push_back(single->second);
        }
      }
    }
  }

  return ids;
}

}  // namespace tge::models
