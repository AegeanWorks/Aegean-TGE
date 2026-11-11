#include "tge/models/tokenizer.hpp"
#include <algorithm>

#include <cmath>
#include <cstddef>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <limits>
#include <map>
#include <queue>
#include <stdexcept>
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

// Lazy-heap BPE merge state for encode_plain. The old encoder rescanned the
// whole symbol list for the best pair on every merge (O(n^2) rank lookups);
// long unbroken pre-tokens (code, repeated text) spent hundreds of
// milliseconds there. Symbols stay in a linked list; candidates are ranked
// by (merge rank, left index), which reproduces the old leftmost-first tie
// break exactly. Small pieces keep the plain rescan: no heap allocation
// below the threshold.
struct BpeSymbol {
  std::size_t start = 0;
  std::size_t length = 0;
  int prev = -1;
  int next = -1;
  std::uint32_t version = 0;
  bool alive = true;
};

struct BpeCandidate {
  int rank = 0;
  int left = 0;
  int right = 0;
  std::uint32_t left_version = 0;
  std::uint32_t right_version = 0;
};

/// `priority_queue` puts the greatest first; "later" means higher rank or,
/// on a tie, a further-left index, so the top is the old scan's choice.
struct BpeCandidateIsLater {
  bool operator()(const BpeCandidate& left, const BpeCandidate& right) const {
    if (left.rank != right.rank) {
      return left.rank > right.rank;
    }
    return left.left > right.left;
  }
};

constexpr std::size_t kBpeHeapThreshold = 32;

/// Fixed 24-byte slot image so the sidecar stores the slot table as one raw
/// block and restores it with one read plus a single validation pass. The
/// padding is explicit and zeroed, so the image is deterministic across
/// builds; bump the sidecar version if this layout ever changes.
struct SerializedSlot {
  std::uint64_t hash = 0;
  std::uint32_t offset = 0;
  std::uint32_t length = 0;
  std::int32_t value = 0;
  std::uint8_t state = 0;
  std::uint8_t padding[3] = {0, 0, 0};
};
static_assert(sizeof(SerializedSlot) == 24, "sidecar slot layout changed");

/// Little-endian append-only buffer for the tokenizer state sidecar. Building
/// the whole image in memory keeps the write a single call; the cache is
/// disposable, so a torn file is caught by the magic/size/index validation on
/// the next load and simply rebuilt.
class ByteWriter {
 public:
  void u8(std::uint8_t value) { bytes_.push_back(value); }
  void u32(std::uint32_t value) {
    for (int shift = 0; shift < 32; shift += 8) {
      bytes_.push_back(static_cast<std::uint8_t>((value >> shift) & 0xFFu));
    }
  }
  void u64(std::uint64_t value) {
    for (int shift = 0; shift < 64; shift += 8) {
      bytes_.push_back(static_cast<std::uint8_t>((value >> shift) & 0xFFu));
    }
  }
  void raw(const void* data, std::size_t size) {
    const auto* bytes = static_cast<const std::uint8_t*>(data);
    bytes_.insert(bytes_.end(), bytes, bytes + size);
  }
  void str(const std::string& value) {
    u32(static_cast<std::uint32_t>(value.size()));
    raw(value.data(), value.size());
  }
  const std::vector<std::uint8_t>& bytes() const { return bytes_; }

 private:
  std::vector<std::uint8_t> bytes_;
};

/// Bounds-checked reader for `ByteWriter` images. Every accessor returns false
/// on truncation, and the caller treats any failure as a cache miss.
class ByteReader {
 public:
  ByteReader(const std::uint8_t* data, std::size_t size)
      : data_(data), size_(size) {}

  bool u8(std::uint8_t& value) {
    if (remaining() < 1) {
      return false;
    }
    value = data_[position_++];
    return true;
  }
  bool u32(std::uint32_t& value) {
    if (remaining() < 4) {
      return false;
    }
    value = 0;
    for (int shift = 0; shift < 32; shift += 8) {
      value |= static_cast<std::uint32_t>(data_[position_++]) << shift;
    }
    return true;
  }
  bool u64(std::uint64_t& value) {
    if (remaining() < 8) {
      return false;
    }
    value = 0;
    for (int shift = 0; shift < 64; shift += 8) {
      value |= static_cast<std::uint64_t>(data_[position_++]) << shift;
    }
    return true;
  }
  bool raw(void* out, std::size_t size) {
    if (remaining() < size) {
      return false;
    }
    std::memcpy(out, data_ + position_, size);
    position_ += size;
    return true;
  }
  bool str(std::string& value, std::size_t limit) {
    std::uint32_t length = 0;
    if (!u32(length) || length > limit || remaining() < length) {
      return false;
    }
    value.assign(reinterpret_cast<const char*>(data_ + position_), length);
    position_ += length;
    return true;
  }

  std::size_t remaining() const { return size_ - position_; }

 private:
  const std::uint8_t* data_ = nullptr;
  std::size_t size_ = 0;
  std::size_t position_ = 0;
};

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
  for (const float score : tokenizer->scores_) {
    // A NaN/inf rank silently breaks the best-first merge order (NaN loses
    // every comparison); the file is corrupt, fail the load instead.
    if (!std::isfinite(score)) {
      error = "SPM vocabulary has a non-finite tokenizer.ggml.scores entry";
      return false;
    }
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

  tokenizer->vocab_.reserve(tokenizer->pieces_.size());
  for (std::size_t index = 0; index < tokenizer->pieces_.size(); ++index) {
    tokenizer->vocab_.emplace(tokenizer->pieces_[index],
                              static_cast<int>(index));
  }

  // Merge ranks. GGUF stores each rule as "left right"; the separator is a
  // literal space, which the mapped alphabet never produces inside a piece.
  if (const gguf::Value* merges = file.find("tokenizer.ggml.merges");
      merges != nullptr && merges->type == gguf::ValueType::Array) {
    tokenizer->merge_rank_.reserve(merges->elements.size());
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
    // BPE twin of the SPM erasure in enable_spm: specials are matched
    // literally via control_tokens_, never via merges. Leaving them in
    // vocab_ lets merge rules fuse user text into a CONTROL/USER_DEFINED
    // id even with match_special=false, forging turn boundaries/tool calls.
    for (const auto& control : tokenizer->control_tokens_) {
      tokenizer->vocab_.erase(control.first);
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

bool Tokenizer::cache_location(const std::string& model_path,
                               const std::string& cache_dir,
                               std::uint64_t& key, std::string& file) {
  std::error_code error;
  const std::uintmax_t size = std::filesystem::file_size(model_path, error);
  if (error) {
    return false;
  }
  const auto modified = std::filesystem::last_write_time(model_path, error);
  if (error) {
    return false;
  }
  std::uint64_t value = StringIndex::hash(model_path);
  value ^= static_cast<std::uint64_t>(size) * 0x9E3779B97F4A7C15ULL;
  value ^= static_cast<std::uint64_t>(modified.time_since_epoch().count()) *
           0xC2B2AE3D27D4EB4FULL;
  key = value;
  char name[32] = {};
  std::snprintf(name, sizeof(name), "%016llx.bin",
                static_cast<unsigned long long>(value));
  file = (std::filesystem::path(cache_dir) / name).string();
  return true;
}

void Tokenizer::save_cache(const std::string& file, std::uint64_t key,
                           const Tokenizer& tokenizer) {
  constexpr std::uint32_t kMagic = 0x53544754u;  // "TGTS"
  constexpr std::uint32_t kVersion = 3;
  ByteWriter writer;
  writer.u32(kMagic);
  writer.u32(kVersion);
  writer.u64(key);
  writer.u32(static_cast<std::uint32_t>(tokenizer.pieces_.size()));
  for (const std::string& piece : tokenizer.pieces_) {
    writer.str(piece);
  }
  writer.raw(tokenizer.byte_of_codepoint_.data(),
             sizeof(std::int32_t) * tokenizer.byte_of_codepoint_.size());
  writer.raw(tokenizer.codepoint_of_byte_.data(),
             sizeof(std::uint32_t) * tokenizer.codepoint_of_byte_.size());
  const auto write_index = [&writer](const StringIndex& index) {
    writer.u32(static_cast<std::uint32_t>(index.arena_.size()));
    writer.raw(index.arena_.data(), index.arena_.size());
    writer.u32(static_cast<std::uint32_t>(index.slots_.size()));
    std::vector<SerializedSlot> slots(index.slots_.size());
    for (std::size_t slot_index = 0; slot_index < index.slots_.size();
         ++slot_index) {
      const StringIndex::Slot& slot = index.slots_[slot_index];
      SerializedSlot& stored = slots[slot_index];
      stored.hash = slot.hash;
      stored.offset = slot.offset;
      stored.length = slot.length;
      stored.value = slot.value;
      stored.state = slot.state;
    }
    writer.raw(slots.data(), slots.size() * sizeof(SerializedSlot));
    writer.u64(static_cast<std::uint64_t>(index.size_));
    writer.u64(static_cast<std::uint64_t>(index.tombstones_));
  };
  write_index(tokenizer.vocab_);
  write_index(tokenizer.merge_rank_);
  writer.u32(static_cast<std::uint32_t>(tokenizer.control_.size()));
  writer.raw(tokenizer.control_.data(), tokenizer.control_.size());
  writer.u32(static_cast<std::uint32_t>(tokenizer.control_tokens_.size()));
  for (const auto& entry : tokenizer.control_tokens_) {
    writer.str(entry.first);
    writer.u32(static_cast<std::uint32_t>(entry.second));
  }
  writer.u32(static_cast<std::uint32_t>(tokenizer.control_by_first_.size()));
  for (const auto& entry : tokenizer.control_by_first_) {
    writer.u8(static_cast<std::uint8_t>(entry.first));
    writer.u32(static_cast<std::uint32_t>(entry.second.size()));
    for (const int id : entry.second) {
      writer.u32(static_cast<std::uint32_t>(id));
    }
  }
  writer.u32(tokenizer.llama_bpe_digits_ ? 1u : 0u);
  writer.u32(tokenizer.spm_ ? 1u : 0u);
  writer.u32(static_cast<std::uint32_t>(tokenizer.scores_.size()));
  writer.raw(tokenizer.scores_.data(),
             sizeof(float) * tokenizer.scores_.size());
  writer.raw(tokenizer.byte_token_.data(),
             sizeof(int) * tokenizer.byte_token_.size());

  std::error_code error;
  const std::filesystem::path target(file);
  std::filesystem::create_directories(target.parent_path(), error);
  const std::string temporary = file + ".tmp";
  {
    std::ofstream stream(temporary, std::ios::binary | std::ios::trunc);
    if (!stream) {
      return;
    }
    stream.write(reinterpret_cast<const char*>(writer.bytes().data()),
                 static_cast<std::streamsize>(writer.bytes().size()));
    stream.flush();
    if (!stream) {
      stream.close();
      std::filesystem::remove(temporary, error);
      return;
    }
  }
  std::filesystem::remove(target, error);
  std::filesystem::rename(temporary, target, error);
  if (error) {
    std::filesystem::remove(temporary, error);
  }
}

bool Tokenizer::load_cache(const std::string& file, std::uint64_t key,
                           std::unique_ptr<Tokenizer>& tokenizer) {
  constexpr std::uint32_t kMagic = 0x53544754u;  // "TGTS"
  constexpr std::uint32_t kVersion = 3;
  constexpr std::uint32_t kMaxPieces = 2000000;
  constexpr std::uint32_t kMaxArena = 64u * 1024u * 1024u;
  constexpr std::uint32_t kMaxSlots = 4000000;
  constexpr std::uint32_t kMaxScores = 2000000;
  constexpr std::uint32_t kMaxString = 1u * 1024u * 1024u;

  std::error_code error;
  const std::uintmax_t size = std::filesystem::file_size(file, error);
  if (error || size == 0 || size > (256ull * 1024ull * 1024ull)) {
    return false;
  }
  std::vector<std::uint8_t> data(static_cast<std::size_t>(size));
  {
    std::ifstream stream(file, std::ios::binary);
    if (!stream ||
        !stream.read(reinterpret_cast<char*>(data.data()),
                     static_cast<std::streamsize>(data.size()))) {
      return false;
    }
  }

  ByteReader reader(data.data(), data.size());
  std::uint32_t magic = 0;
  std::uint32_t version = 0;
  std::uint64_t stored_key = 0;
  if (!reader.u32(magic) || magic != kMagic || !reader.u32(version) ||
      version != kVersion || !reader.u64(stored_key) ||
      stored_key != key) {
    return false;
  }

  std::unique_ptr<Tokenizer> restored(new Tokenizer());
  std::uint32_t piece_count = 0;
  if (!reader.u32(piece_count) || piece_count == 0 ||
      piece_count > kMaxPieces) {
    return false;
  }
  restored->pieces_.reserve(piece_count);
  for (std::uint32_t index = 0; index < piece_count; ++index) {
    std::string piece;
    if (!reader.str(piece, kMaxString)) {
      return false;
    }
    restored->pieces_.push_back(std::move(piece));
  }
  if (!reader.raw(restored->byte_of_codepoint_.data(),
                  sizeof(std::int32_t) *
                      restored->byte_of_codepoint_.size()) ||
      !reader.raw(restored->codepoint_of_byte_.data(),
                  sizeof(std::uint32_t) *
                      restored->codepoint_of_byte_.size())) {
    return false;
  }

  const auto read_index = [&reader](StringIndex& index) {
    std::uint32_t arena_length = 0;
    if (!reader.u32(arena_length) || arena_length > kMaxArena ||
        reader.remaining() < arena_length) {
      return false;
    }
    index.arena_.resize(arena_length);
    if (arena_length > 0 && !reader.raw(index.arena_.data(), arena_length)) {
      return false;
    }
    std::uint32_t slot_count = 0;
    if (!reader.u32(slot_count) || slot_count > kMaxSlots) {
      return false;
    }
    if (slot_count != 0 && (slot_count & (slot_count - 1)) != 0) {
      return false;
    }
    static_assert(sizeof(StringIndex::Slot) == sizeof(SerializedSlot),
                  "slot layout must match the sidecar image");
    if (reader.remaining() / sizeof(SerializedSlot) < slot_count) {
      return false;
    }
    index.slots_.resize(slot_count);
    if (slot_count > 0 &&
        !reader.raw(index.slots_.data(),
                    index.slots_.size() * sizeof(StringIndex::Slot))) {
      return false;
    }
    std::size_t occupied = 0;
    for (std::uint32_t slot_index = 0; slot_index < slot_count; ++slot_index) {
      const StringIndex::Slot& slot = index.slots_[slot_index];
      if (slot.state > 2) {
        return false;
      }
      if (slot.state == 1) {
        if (static_cast<std::uint64_t>(slot.offset) + slot.length >
            arena_length) {
          return false;
        }
        ++occupied;
      }
    }
    std::uint64_t stored_size = 0;
    std::uint64_t stored_tombstones = 0;
    if (!reader.u64(stored_size) || !reader.u64(stored_tombstones) ||
        stored_size > slot_count || stored_tombstones > slot_count ||
        stored_size != occupied) {
      return false;
    }
    index.size_ = static_cast<std::size_t>(stored_size);
    index.tombstones_ = static_cast<std::size_t>(stored_tombstones);
    return true;
  };
  if (!read_index(restored->vocab_) ||
      !read_index(restored->merge_rank_)) {
    return false;
  }

  std::uint32_t control_count = 0;
  if (!reader.u32(control_count) || control_count > piece_count ||
      (control_count != 0 && control_count != piece_count)) {
    return false;
  }
  restored->control_.resize(control_count);
  if (control_count > 0 &&
      !reader.raw(restored->control_.data(), control_count)) {
    return false;
  }
  std::uint32_t control_token_count = 0;
  if (!reader.u32(control_token_count) || control_token_count > piece_count ||
      (control_token_count > 0 && control_count != piece_count)) {
    return false;
  }
  restored->control_tokens_.reserve(control_token_count);
  for (std::uint32_t index = 0; index < control_token_count; ++index) {
    std::string piece;
    std::uint32_t id = 0;
    if (!reader.str(piece, kMaxString) || !reader.u32(id) || id >= piece_count) {
      return false;
    }
    restored->control_tokens_.emplace_back(std::move(piece),
                                           static_cast<int>(id));
  }
  std::uint32_t by_first_count = 0;
  if (!reader.u32(by_first_count) || by_first_count > control_token_count) {
    return false;
  }
  std::array<bool, 256> seen_first{};
  for (std::uint32_t index = 0; index < by_first_count; ++index) {
    std::uint8_t first = 0;
    std::uint32_t count = 0;
    if (!reader.u8(first) || !reader.u32(count) || count == 0 ||
        count > control_token_count || seen_first[first]) {
      return false;
    }
    seen_first[first] = true;
    std::vector<int> ids;
    ids.reserve(count);
    for (std::uint32_t id_index = 0; id_index < count; ++id_index) {
      std::uint32_t id = 0;
      if (!reader.u32(id) || id >= control_token_count) {
        return false;
      }
      ids.push_back(static_cast<int>(id));
    }
    restored->control_by_first_.emplace(static_cast<char>(first),
                                        std::move(ids));
  }
  std::uint32_t llama_bpe_digits = 0;
  std::uint32_t spm = 0;
  if (!reader.u32(llama_bpe_digits) || llama_bpe_digits > 1 ||
      !reader.u32(spm) || spm > 1) {
    return false;
  }
  restored->llama_bpe_digits_ = llama_bpe_digits != 0;
  restored->spm_ = spm != 0;
  std::uint32_t score_count = 0;
  if (!reader.u32(score_count) || score_count > kMaxScores) {
    return false;
  }
  restored->scores_.resize(score_count);
  if (score_count > 0 &&
      !reader.raw(restored->scores_.data(), sizeof(float) * score_count)) {
    return false;
  }
  if (!reader.raw(restored->byte_token_.data(),
                  sizeof(int) * restored->byte_token_.size())) {
    return false;
  }
  for (const int id : restored->byte_token_) {
    if (id < -1 || id >= static_cast<int>(piece_count)) {
      return false;
    }
  }
  for (const float score : restored->scores_) {
    if (!std::isfinite(score)) {
      return false;
    }
  }
  if (reader.remaining() != 0) {
    return false;
  }
  tokenizer = std::move(restored);
  return true;
}

std::unique_ptr<Tokenizer> Tokenizer::from_path(const std::string& path,
                                                std::string& error,
                                                const std::string& cache_dir) {
  try {
    // Open with full metadata: the model loader skips large arrays, so its File
    // has no vocabulary in it.
    if (!cache_dir.empty()) {
      std::uint64_t key = 0;
      std::string file;
      if (cache_location(path, cache_dir, key, file)) {
        std::unique_ptr<Tokenizer> cached;
        if (load_cache(file, key, cached)) {
          return cached;
        }
        const gguf::File file_data = gguf::File::open(path);
        std::unique_ptr<Tokenizer> tokenizer = from_gguf(file_data, error);
        if (tokenizer != nullptr) {
          save_cache(file, key, *tokenizer);
        }
        return tokenizer;
      }
    }
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

    std::vector<BpeSymbol> symbols;
    symbols.reserve(encoded.size());
    {
      std::size_t position = 0;
      int index = 0;
      while (position < encoded.size()) {
        const std::size_t begin = position;
        next_codepoint(encoded, position);
        BpeSymbol symbol;
        symbol.start = begin;
        symbol.length = position - begin;
        symbol.prev = index - 1;
        if (index > 0) {
          symbols[static_cast<std::size_t>(index) - 1].next = index;
        }
        symbols.push_back(symbol);
        ++index;
      }
    }

    // One reusable key: rank lookup builds "left right" per candidate without
    // allocating, reproducing the old concatenation exactly.
    std::string key;
    const auto rank_of = [&](int left, int right) {
      const BpeSymbol& first = symbols[static_cast<std::size_t>(left)];
      const BpeSymbol& second = symbols[static_cast<std::size_t>(right)];
      key.assign(encoded, first.start, first.length);
      key.push_back(' ');
      key.append(encoded, second.start, second.length);
      const auto found = merge_rank_.find(key);
      return found != merge_rank_.end() ? found->second : -1;
    };
    const auto merge_at = [&](int left, int right) {
      BpeSymbol& first = symbols[static_cast<std::size_t>(left)];
      const BpeSymbol& second = symbols[static_cast<std::size_t>(right)];
      first.length += second.length;
      first.next = second.next;
      if (first.next >= 0) {
        symbols[static_cast<std::size_t>(first.next)].prev = left;
      }
      symbols[static_cast<std::size_t>(right)].alive = false;
      ++first.version;
    };

    if (symbols.size() > kBpeHeapThreshold) {
      std::priority_queue<BpeCandidate, std::vector<BpeCandidate>,
                          BpeCandidateIsLater>
          work;
      const auto push_pair = [&](int left, int right) {
        if (left < 0 || right < 0) {
          return;
        }
        const int rank = rank_of(left, right);
        if (rank < 0) {
          return;
        }
        BpeCandidate candidate;
        candidate.rank = rank;
        candidate.left = left;
        candidate.right = right;
        candidate.left_version =
            symbols[static_cast<std::size_t>(left)].version;
        candidate.right_version =
            symbols[static_cast<std::size_t>(right)].version;
        work.push(candidate);
      };
      for (int index = 1; index < static_cast<int>(symbols.size()); ++index) {
        push_pair(index - 1, index);
      }
      while (!work.empty()) {
        const BpeCandidate candidate = work.top();
        work.pop();
        const BpeSymbol& first = symbols[static_cast<std::size_t>(candidate.left)];
        const BpeSymbol& second =
            symbols[static_cast<std::size_t>(candidate.right)];
        if (!first.alive || !second.alive ||
            first.version != candidate.left_version ||
            second.version != candidate.right_version ||
            first.next != candidate.right) {
          continue;
        }
        const int before = first.prev;
        const int after = second.next;
        merge_at(candidate.left, candidate.right);
        push_pair(before, candidate.left);
        push_pair(candidate.left, after);
      }
    } else {
      bool merged = true;
      while (merged) {
        merged = false;
        int best_rank = std::numeric_limits<int>::max();
        int best_left = -1;
        for (int index = 0; index >= 0;
             index = symbols[static_cast<std::size_t>(index)].next) {
          const int next = symbols[static_cast<std::size_t>(index)].next;
          if (next < 0) {
            break;
          }
          const int rank = rank_of(index, next);
          if (rank >= 0 && rank < best_rank) {
            best_rank = rank;
            best_left = index;
          }
        }
        if (best_left >= 0) {
          merge_at(best_left,
                   symbols[static_cast<std::size_t>(best_left)].next);
          merged = true;
        }
      }
    }

    for (int index = 0; index >= 0;
         index = symbols[static_cast<std::size_t>(index)].next) {
      const BpeSymbol& symbol = symbols[static_cast<std::size_t>(index)];
      const std::string piece_text = encoded.substr(symbol.start, symbol.length);
      const auto found = vocab_.find(piece_text);
      if (found != vocab_.end()) {
        ids.push_back(found->second);
        continue;
      }
      // Unknown multi-byte symbol: fall back to its individual characters.
      // A character with no vocabulary entry is a corrupt-vocabulary error,
      // not skippable text: dropping it silently rewrites the user's prompt.
      // (Full Qwen/Llama vocabs cover every byte, so this fires only on
      // truncated fixtures.)
      std::size_t position = 0;
      while (position < piece_text.size()) {
        const std::size_t begin = position;
        next_codepoint(piece_text, position);
        const auto single = vocab_.find(piece_text.substr(begin, position - begin));
        if (single == vocab_.end()) {
          throw std::invalid_argument("encode_plain: text has no vocabulary entry");
        }
        ids.push_back(single->second);
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
            // Same contract as the BPE path: an unencodable byte is a
            // corrupt vocabulary, not skippable text.
            if (byte_id < 0) {
              throw std::invalid_argument(
                  "encode_spm: byte has no vocabulary entry");
            }
            ids.push_back(byte_id);
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
