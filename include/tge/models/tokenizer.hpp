#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
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
///   with scores, tokenized by Viterbi over the best-scoring path.
/// Decoding is a table lookup per piece in both modes.
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
  /// with no merges table (Phi-3 and friends). Encoding runs Viterbi over
  /// piece scores instead of byte-pair merges.
  bool spm() const { return spm_; }

  int vocab_size() const { return static_cast<int>(pieces_.size()); }

 private:
  Tokenizer() = default;

  /// Byte-pair encodes a run with no control tokens in it.
  std::vector<int> encode_plain(const std::string& text) const;

  /// SentencePiece-encoding of a run with no control tokens in it:
  /// SentencePiece-BPE per llama.cpp (`llm_tokenizer_spm`) — dummy prefix,
  /// space-to-U+2581, then greedy highest-score pair merges. Scores are
  /// merge ranks, so a Viterbi pass would invert the objective.
  /// Special (control/user-defined) pieces never match here; they are only
  /// reachable through the literal control-token pass in `encode`.
  std::vector<int> encode_spm(const std::string& text) const;

  /// Parses a `<0xNN>` byte-fallback piece. Returns -1 for anything else.
  static int byte_piece_value(const std::string& piece);

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
  std::unordered_map<std::string, int> vocab_;
  std::unordered_map<std::string, int> merge_rank_;
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
