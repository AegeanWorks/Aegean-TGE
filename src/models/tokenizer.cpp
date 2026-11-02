#include "tge/models/tokenizer.hpp"

#include <algorithm>
#include <cstddef>
#include <limits>
#include <map>
#include <queue>
#include <utility>

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

void append_codepoint(std::string& out, std::uint32_t code_point) {  if (code_point < 0x80) {
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

/// Splits text into the pieces the merge step runs over. With
/// `group_digits` (tiktoken `llama-bpe`), ASCII digit runs split greedily
/// left-to-right in groups of at most three, matching `\p{N}{1,3}`; without
/// it every digit stands alone, matching the Qwen2 splitter this engine
/// verified first. Everything else is shared between the two.
std::vector<std::string> split_pre_tokens(const std::string& text,
                                          bool group_digits) {
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
      if (following != 0 &&
          (is_letter(following) || is_punctuation(following))) {
        if (run > 1) {
          // Emit the excess spaces, then let the next round take " " + word or
          // " " + punctuation run.
          pieces.push_back(text.substr(start, run - 1));
          position += run - 1;
          continue;
        }
        position += 1;  // consume the single space
        if (is_letter(following)) {
          while (position < size) {
            code_point = peek(position, length);
            if (!is_letter(code_point)) {
              break;
            }
            position += length;
          }
        } else {
          // An optional space leads a punctuation run in the reference pattern,
          // so ` =` and ` +` stay single pieces.
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
      if (group_digits) {
        // Tiktoken digit run: up to three ASCII digits, greedy from the left.
        // Merges cannot cross piece boundaries, so without this the groups
        // in the vocabulary ("199", "123") are unreachable and every id
        // after the first digit diverges from the reference.
        int count = 0;
        while (count < 3 && position < size) {
          std::size_t next_length = 0;
          const std::uint32_t next = peek(position, next_length);
          if (!is_ascii_digit(next)) {
            break;
          }
          position += next_length;
          ++count;
        }
      } else {
        position += length;
      }
      pieces.push_back(text.substr(start, position - start));
      continue;
    }

    // Punctuation run. One non-letter, non-digit character may lead the
    // following letter run: `[i`, `.com` and `/path` are single pieces in the
    // reference splitter.
    int punctuation_count = 0;
    while (position < size) {
      code_point = peek(position, length);
      if (!is_punctuation(code_point)) {
        break;
      }
      position += length;
      ++punctuation_count;
    }
    if (punctuation_count == 1) {
      const std::uint32_t following = peek(position, length);
      if (following != 0 && is_letter(following)) {
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
    }
    // Newlines immediately after a punctuation run stay with it.
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

int Tokenizer::byte_piece_value(const std::string& piece) {
  // Byte fallbacks are exactly six ASCII characters: `<0xNN>`.
  if (piece.size() != 6 || piece[0] != '<' || piece[1] != '0' ||
      (piece[2] != 'x' && piece[2] != 'X') || piece[5] != '>') {
    return -1;
  }
  const auto nibble = [](char digit) -> int {
    if (digit >= '0' && digit <= '9') {
      return digit - '0';
    }
    if (digit >= 'a' && digit <= 'f') {
      return digit - 'a' + 10;
    }
    if (digit >= 'A' && digit <= 'F') {
      return digit - 'A' + 10;
    }
    return -1;
  };
  const int high = nibble(piece[3]);
  const int low = nibble(piece[4]);
  if (high < 0 || low < 0) {
    return -1;
  }
  return high * 16 + low;
}

bool Tokenizer::enable_spm(Tokenizer* tokenizer, const gguf::File& file,
                           std::string& error) {
  const gguf::Value* scores = file.find("tokenizer.ggml.scores");
  if (scores == nullptr || scores->type != gguf::ValueType::Array ||
      scores->elements.size() != tokenizer->pieces_.size()) {
    error = "SPM vocabulary has no per-piece tokenizer.ggml.scores array";
    return false;
  }
  tokenizer->scores_.reserve(scores->elements.size());
  for (const gguf::Value& element : scores->elements) {
    tokenizer->scores_.push_back(static_cast<float>(element.as_f64()));
  }
  tokenizer->byte_token_.fill(-1);
  for (std::size_t index = 0; index < tokenizer->pieces_.size(); ++index) {
    const int byte = byte_piece_value(tokenizer->pieces_[index]);
    if (byte < 0) {
      continue;
    }
    tokenizer->byte_token_[static_cast<std::size_t>(byte)] =
        static_cast<int>(index);
    // A byte piece must never win a literal-text lookup: "<0x0A>" typed as
    // text is six characters, not a newline. Bytes match by value below.
    tokenizer->vocab_.erase(tokenizer->pieces_[index]);
  }
  // Special pieces stay reachable only through the literal control-token
  // pass; letting the merge loop match them would let user content forge a
  // turn boundary or a tool call.
  for (const auto& control : tokenizer->control_tokens_) {
    tokenizer->vocab_.erase(control.first);
  }
  if (tokenizer->vocab_.empty()) {
    error = "SPM vocabulary has no encodable pieces";
    return false;
  }
  tokenizer->spm_ = true;
  return true;
}

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

  // CONTROL and USER_DEFINED tokens are matched literally during encoding. The
  // GGUF token-type table marks them (3 is CONTROL, 4 is USER_DEFINED); without
  // it, text keeps its byte-pair behaviour.
  if (const gguf::Value* types = file.find("tokenizer.ggml.token_type");
      types != nullptr && types->type == gguf::ValueType::Array) {
    constexpr std::uint64_t kControl = 3;
    constexpr std::uint64_t kUserDefined = 4;
    tokenizer->control_.assign(tokenizer->pieces_.size(), 0);
    for (std::size_t index = 0;
         index < types->elements.size() && index < tokenizer->pieces_.size();
         ++index) {
      const std::uint64_t type = types->elements[index].as_u64();
      if (type == kControl) {
        tokenizer->control_[index] = 1;
        tokenizer->control_tokens_.emplace_back(
            tokenizer->pieces_[index], static_cast<int>(index));
      } else if (type == kUserDefined) {
        tokenizer->control_tokens_.emplace_back(
            tokenizer->pieces_[index], static_cast<int>(index));
      }
    }
  }
  std::sort(tokenizer->control_tokens_.begin(),
            tokenizer->control_tokens_.end(),
            [](const auto& left, const auto& right) {
              return left.first.size() > right.first.size();
            });
    for (std::size_t index = 0; index < tokenizer->control_tokens_.size();
         ++index) {
      const std::string& piece = tokenizer->control_tokens_[index].first;
      if (!piece.empty()) {
        tokenizer->control_by_first_[piece.front()].push_back(
            static_cast<int>(index));
      }
    }

    // The file declares its splitter: "llama-bpe" is the tiktoken family
    // (digit runs group in threes); anything else keeps the Qwen2 splitter
    // this engine verified first. Unknown values deliberately fall back to
    // Qwen2 rather than failing the load.
    if (const gguf::Value* pre = file.find("tokenizer.ggml.pre");
        pre != nullptr && pre->type == gguf::ValueType::String) {
      tokenizer->llama_bpe_digits_ = (pre->string_value == "llama-bpe");
    }

    // A "llama"-model vocabulary with no merges table is SentencePiece, not
    // byte-pair: pieces carry scores and the best-scoring path wins. The
    // gpt2/BPE files (Qwen, Llama-3.2) keep the merge path bit-identical.
    bool has_merges = false;
    if (const gguf::Value* merges = file.find("tokenizer.ggml.merges");
        merges != nullptr && merges->type == gguf::ValueType::Array &&
        !merges->elements.empty()) {
      has_merges = true;
    }
    const std::string model_name = file.get_string("tokenizer.ggml.model", "");
    if (model_name == "llama" && !has_merges) {
      if (!Tokenizer::enable_spm(tokenizer.get(), file, error)) {
        return nullptr;
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
    // Control tokens are structural, not text. Printing them would put
    // `<|im_end|>` in the middle of the answer.
    if (static_cast<std::size_t>(id) < control_.size() &&
        control_[static_cast<std::size_t>(id)] != 0) {
      continue;
    }
    const std::string& piece = pieces_[static_cast<std::size_t>(id)];
    if (spm_) {
      // SentencePiece byte fallbacks have no mapped single character; every
      // other piece is raw UTF-8, and U+2581 reads back as a space.
      // GPT-2 byte remapping does not apply to this family.
      const int byte = byte_piece_value(piece);
      if (byte >= 0) {
        text.push_back(static_cast<char>(byte));
        continue;
      }
      for (std::size_t offset = 0; offset < piece.size();) {
        static constexpr char kWordJoiner[3] = {'\xE2', '\x96', '\x81'};
        if (offset + 3 <= piece.size() &&
            piece.compare(offset, 3, kWordJoiner, 3) == 0) {
          text.push_back(' ');
          offset += 3;
        } else {
          text.push_back(piece[offset]);
          ++offset;
        }
      }
      continue;
    }
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

std::vector<int> Tokenizer::encode_plain(const std::string& text) const {
  std::vector<int> ids;
  if (text.empty()) {
    return ids;
  }
  if (spm_) {
    return encode_spm(text);
  }

  for (const std::string& piece : split_pre_tokens(text, llama_bpe_digits_)) {
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

std::vector<int> Tokenizer::encode_spm(const std::string& text) const {
  // SentencePiece-BPE (llama.cpp `llm_tokenizer_spm`): pieces are raw UTF-8
  // with U+2581 for spaces, and the per-piece score is a merge rank, not a
  // log-probability — so the best path is found by greedily merging the
  // highest-scoring adjacent pair, never by Viterbi. (Viterbi over these
  // scores inverts the objective: the 0.0-scored byte fallbacks would beat
  // every real piece.) Normalization is the standard dummy prefix plus
  // space-to-U+2581.
  static constexpr char kWordJoiner[3] = {'\xE2', '\x96', '\x81'};
  std::string normalized;
  normalized.reserve(text.size() + 3);
  normalized.append(kWordJoiner, 3);
  for (const char raw : text) {
    if (raw == ' ') {
      normalized.append(kWordJoiner, 3);
    } else {
      normalized.push_back(raw);
    }
  }

  // Split into UTF-8 characters; each becomes one merge symbol.
  struct Symbol {
    int prev = -1;
    int next = -1;
    std::size_t start = 0;
    std::size_t length = 0;
  };
  std::vector<Symbol> symbols;
  {
    std::size_t position = 0;
    int index = 0;
    while (position < normalized.size()) {
      const std::size_t begin = position;
      const std::uint32_t code_point = next_codepoint(normalized, position);
      if (code_point == 0 || position <= begin) {
        break;  // Cannot happen: position always advances on valid input.
      }
      Symbol symbol;
      symbol.start = begin;
      symbol.length = position - begin;
      symbol.prev = index - 1;
      symbol.next = -1;
      if (index > 0) {
        symbols[static_cast<std::size_t>(index) - 1].next = index;
      }
      symbols.push_back(symbol);
      ++index;
    }
  }

  struct Bigram {
    int left = -1;
    int right = -1;
    float score = 0.0F;
    std::size_t size = 0;
  };
  struct BestFirst {
    bool operator()(const Bigram& left, const Bigram& right) const {
      // Highest score wins; ties break toward the leftmost pair.
      return left.score < right.score ||
             (left.score == right.score && left.left > right.left);
    }
  };
  std::priority_queue<Bigram, std::vector<Bigram>, BestFirst> work;
  // Split records for symbols that never became a piece: resegment replays
  // the merge that built them, or falls back to bytes.
  std::map<std::string, std::pair<int, int>> rev_merge;

  const auto symbol_text = [&normalized, &symbols](int index) {
    const Symbol& symbol = symbols[static_cast<std::size_t>(index)];
    return normalized.substr(symbol.start, symbol.length);
  };
  const auto try_add_bigram = [&](int left, int right) {
    if (left < 0 || right < 0) {
      return;
    }
    const std::string text =
        symbol_text(left) + symbol_text(right);
    const auto found = vocab_.find(text);
    if (found == vocab_.end()) {
      return;
    }
    Bigram bigram;
    bigram.left = left;
    bigram.right = right;
    bigram.score = scores_[static_cast<std::size_t>(found->second)];
    bigram.size = text.size();
    work.push(bigram);
    rev_merge[text] = std::make_pair(left, right);
  };
  for (int index = 1; index < static_cast<int>(symbols.size()); ++index) {
    try_add_bigram(index - 1, index);
  }

  while (!work.empty()) {
    const Bigram bigram = work.top();
    work.pop();
    Symbol& left = symbols[static_cast<std::size_t>(bigram.left)];
    Symbol& right = symbols[static_cast<std::size_t>(bigram.right)];
    if (left.length == 0 || right.length == 0 ||
        left.length + right.length != bigram.size) {
      continue;  // Stale entry: one side already merged elsewhere.
    }
    left.length += right.length;
    right.length = 0;
    left.next = right.next;
    if (right.next >= 0) {
      symbols[static_cast<std::size_t>(right.next)].prev = bigram.left;
    }
    try_add_bigram(left.prev, bigram.left);
    try_add_bigram(bigram.left, left.next);
  }

  // Emit each surviving symbol: a vocab piece when it is one, otherwise a
  // replay of its recorded split down to pieces, otherwise its bytes.
  std::vector<int> ids;
  std::vector<int> todo;
  for (int index = 0; index >= 0;) {
    const Symbol& symbol = symbols[static_cast<std::size_t>(index)];
    const int next = symbol.next;
    if (symbol.length > 0) {
      todo.push_back(index);
      while (!todo.empty()) {
        const int current = todo.back();
        todo.pop_back();
        const Symbol& part = symbols[static_cast<std::size_t>(current)];
        const std::string part_text =
            normalized.substr(part.start, part.length);
        const auto found = vocab_.find(part_text);
        if (found != vocab_.end()) {
          ids.push_back(found->second);
          continue;
        }
        const auto split = rev_merge.find(part_text);
        if (split == rev_merge.end()) {
          for (std::size_t offset = 0; offset < part.length; ++offset) {
            const int byte_id = byte_token_[static_cast<unsigned char>(
                normalized[part.start + offset])];
            if (byte_id >= 0) {
              ids.push_back(byte_id);
            }
          }
          continue;
        }
        todo.push_back(split->second.second);
        todo.push_back(split->second.first);
      }
    }
    index = next;
  }
  return ids;
}

int Tokenizer::match_control(const std::string& text,
                             std::size_t position) const {
  const auto bucket = control_by_first_.find(text[position]);
  if (bucket == control_by_first_.end()) {
    return -1;
  }
  for (const int index : bucket->second) {
    const std::string& piece =
        control_tokens_[static_cast<std::size_t>(index)].first;
    if (text.compare(position, piece.size(), piece) == 0) {
      return index;
    }
  }
  return -1;
}

std::vector<int> Tokenizer::encode(const std::string& text,
                                   bool match_special) const {
  std::vector<int> ids;
  if (text.empty()) {
    return ids;
  }
  if (!match_special) {
    return encode_plain(text);
  }

  // A control token is one id even when it is typed as text, so a chat
  // template can be written as a prompt. Everything between control tokens
  // goes through the ordinary byte-pair path.
  std::size_t plain_start = 0;
  std::size_t position = 0;
  while (position < text.size()) {
    const int match = match_control(text, position);
    if (match < 0) {
      ++position;
      continue;
    }
    if (position > plain_start) {
      const std::vector<int> plain =
          encode_plain(text.substr(plain_start, position - plain_start));
      ids.insert(ids.end(), plain.begin(), plain.end());
    }
    const auto& control = control_tokens_[static_cast<std::size_t>(match)];
    ids.push_back(control.second);
    position += control.first.size();
    plain_start = position;
  }
  if (plain_start < text.size()) {
    const std::vector<int> plain = encode_plain(text.substr(plain_start));
    ids.insert(ids.end(), plain.begin(), plain.end());
  }
  return ids;
}

}  // namespace tge::models
