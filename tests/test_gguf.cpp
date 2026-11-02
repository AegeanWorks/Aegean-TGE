#include <cstdint>
#include <cstring>
#include <exception>
#include <filesystem>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

#include "harness.hpp"
#include "tge/models/gguf.hpp"
#include "tge/models/qwen2.hpp"
#include "tge/models/tokenizer.hpp"

namespace {

void put_u32(std::vector<std::uint8_t>& out, std::uint32_t value) {
  for (int index = 0; index < 4; ++index) {
    out.push_back(static_cast<std::uint8_t>(value >> (8 * index)));
  }
}

void put_u64(std::vector<std::uint8_t>& out, std::uint64_t value) {
  for (int index = 0; index < 8; ++index) {
    out.push_back(static_cast<std::uint8_t>(value >> (8 * index)));
  }
}

void put_string(std::vector<std::uint8_t>& out, const std::string& text) {
  put_u64(out, text.size());
  out.insert(out.end(), text.begin(), text.end());
}

/// Writes bytes to a uniquely named file in the system temp directory and
/// removes it on destruction, so a failing assertion cannot leave litter.
class TempFile {
 public:
  TempFile(const std::vector<std::uint8_t>& bytes, const char* name) {
    path_ = std::filesystem::temp_directory_path() / name;
    std::ofstream stream(path_, std::ios::binary | std::ios::trunc);
    stream.write(reinterpret_cast<const char*>(bytes.data()),
                 static_cast<std::streamsize>(bytes.size()));
  }
  ~TempFile() {
    std::error_code ignored;
    std::filesystem::remove(path_, ignored);
  }

  TempFile(const TempFile&) = delete;
  TempFile& operator=(const TempFile&) = delete;

  std::string path() const { return path_.string(); }

 private:
  std::filesystem::path path_;
};

/// Minimal valid file: three metadata keys and one Q8_0 tensor of 32x2.
std::vector<std::uint8_t> build_model() {
  std::vector<std::uint8_t> out;
  put_u32(out, 0x46554747u);  // "GGUF"
  put_u32(out, 3);
  put_u64(out, 1);  // tensors
  put_u64(out, 3);  // metadata entries

  put_string(out, "general.architecture");
  put_u32(out, 8);  // string
  put_string(out, "test");

  put_string(out, "test.block_count");
  put_u32(out, 4);  // u32
  put_u32(out, 4);

  put_string(out, "general.alignment");
  put_u32(out, 4);
  put_u32(out, 32);

  put_string(out, "w");
  put_u32(out, 2);   // rank
  put_u64(out, 32);  // ne0
  put_u64(out, 2);   // ne1
  put_u32(out, 8);   // Q8_0
  put_u64(out, 0);   // offset

  while (out.size() % 32 != 0) {
    out.push_back(0);
  }
  for (int index = 0; index < 68; ++index) {  // 2 rows x 34 bytes
    out.push_back(static_cast<std::uint8_t>(index));
  }
  return out;
}

bool open_throws(const std::vector<std::uint8_t>& bytes, const char* name) {
  const TempFile temp(bytes, name);
  try {
    const tge::gguf::File file = tge::gguf::File::open(temp.path());
    (void)file;
  } catch (const std::exception&) {
    return true;
  }
  return false;
}

}  // namespace

TGE_TEST(gguf_parses_a_minimal_file) {
  const TempFile temp(build_model(), "tge_gguf_ok.gguf");
  const tge::gguf::File file = tge::gguf::File::open(temp.path());

  TGE_CHECK(file.version() == 3);
  TGE_CHECK(file.alignment() == 32);
  TGE_CHECK(file.metadata().size() == 3);
  TGE_CHECK(file.get_string("general.architecture", "") == "test");
  TGE_CHECK(file.get_u64("test.block_count", 0) == 4);

  TGE_REQUIRE(file.tensors().size() == 1);
  const tge::gguf::TensorInfo* info = file.tensor("w");
  TGE_REQUIRE(info != nullptr);
  TGE_CHECK(info->shape.size() == 2);
  TGE_CHECK(info->shape[0] == 32);
  TGE_CHECK(info->shape[1] == 2);
  TGE_CHECK(info->type == tge::gguf::TensorType::Q8_0);
  TGE_CHECK(info->elements == 64);
  TGE_CHECK(info->bytes == 68);
  TGE_CHECK(file.tensor_data(*info) != nullptr);
}

TGE_TEST(gguf_rejects_a_bad_magic) {
  std::vector<std::uint8_t> bytes = build_model();
  bytes[0] = 'X';
  TGE_CHECK(open_throws(bytes, "tge_gguf_magic.gguf"));
}

TGE_TEST(gguf_rejects_a_truncated_file) {
  std::vector<std::uint8_t> bytes = build_model();
  bytes.resize(bytes.size() - 10);
  TGE_CHECK(open_throws(bytes, "tge_gguf_trunc.gguf"));
}

TGE_TEST(gguf_rejects_an_implausible_array_length) {
  // A corrupt length field must be rejected before anything is allocated for
  // it, not discovered when the first element read runs off the end.
  std::vector<std::uint8_t> out;
  put_u32(out, 0x46554747u);
  put_u32(out, 3);
  put_u64(out, 0);  // tensors
  put_u64(out, 1);  // metadata entries
  put_string(out, "x");
  put_u32(out, 9);  // array
  put_u32(out, 0);  // u8 elements
  put_u64(out, 1000000000ull);
  TGE_CHECK(open_throws(out, "tge_gguf_array.gguf"));
}

TGE_TEST(gguf_rejects_an_unsupported_version) {
  std::vector<std::uint8_t> bytes = build_model();
  bytes[4] = 99;  // version low byte
  TGE_CHECK(open_throws(bytes, "tge_gguf_version.gguf"));
}

TGE_TEST(tokenizer_encodes_with_merges) {
  // A miniature byte-level vocabulary: the characters of "hello", the bang, and
  // one merged piece built by four rules.
  std::vector<std::uint8_t> out;
  put_u32(out, 0x46554747u);
  put_u32(out, 3);
  put_u64(out, 0);  // tensors
  put_u64(out, 2);  // metadata entries

  put_string(out, "tokenizer.ggml.tokens");
  put_u32(out, 9);
  put_u32(out, 8);
  put_u64(out, 6);
  for (const char* piece : {"h", "e", "l", "o", "!", "hello"}) {
    put_string(out, piece);
  }

  put_string(out, "tokenizer.ggml.merges");
  put_u32(out, 9);
  put_u32(out, 8);
  put_u64(out, 4);
  for (const char* rule : {"h e", "he l", "hel l", "hell o"}) {
    put_string(out, rule);
  }

  const TempFile temp(out, "tge_gguf_merges.gguf");
  const tge::gguf::File file = tge::gguf::File::open(temp.path());

  std::string error;
  const std::unique_ptr<tge::models::Tokenizer> tokenizer =
      tge::models::Tokenizer::from_gguf(file, error);
  TGE_REQUIRE(tokenizer != nullptr);

  // The merge chain must collapse "hello" to the single learned piece, and the
  // splitter must keep "!" as its own token.
  const std::vector<int> hello = tokenizer->encode("hello");
  TGE_REQUIRE(hello.size() == 1);
  TGE_CHECK(hello[0] == 5);

  const std::vector<int> shouted = tokenizer->encode("hello!");
  TGE_REQUIRE(shouted.size() == 2);
  TGE_CHECK(shouted[0] == 5);
  TGE_CHECK(shouted[1] == 4);
}

TGE_TEST(tokenizer_decodes_byte_level_pieces) {
  // Three pieces in the GPT-2 representation: space, "hello", "!".
  std::vector<std::uint8_t> out;
  put_u32(out, 0x46554747u);
  put_u32(out, 3);
  put_u64(out, 0);  // tensors
  put_u64(out, 1);  // metadata entries
  put_string(out, "tokenizer.ggml.tokens");
  put_u32(out, 9);  // array
  put_u32(out, 8);  // of strings
  put_u64(out, 3);
  put_string(out, "\xC4\xA0");  // U+0120, the byte-level stand-in for a space
  put_string(out, "hello");
  put_string(out, "!");

  const TempFile temp(out, "tge_gguf_tokens.gguf");
  const tge::gguf::File file = tge::gguf::File::open(temp.path());

  std::string error;
  const std::unique_ptr<tge::models::Tokenizer> tokenizer =
      tge::models::Tokenizer::from_gguf(file, error);
  TGE_REQUIRE(tokenizer != nullptr);
  TGE_CHECK(tokenizer->vocab_size() == 3);

  const std::vector<int> ids = {0, 1, 2};
  TGE_CHECK(tokenizer->decode(ids) == " hello!");
}

TGE_TEST(tokenizer_matches_control_tokens_in_text) {
  // A control token and a four-letter word. The token-type table marks index 0
  // as CONTROL (3), which is how a chat template stays one id when typed.
  std::vector<std::uint8_t> out;
  put_u32(out, 0x46554747u);
  put_u32(out, 3);
  put_u64(out, 0);  // tensors
  put_u64(out, 3);  // metadata entries

  put_string(out, "tokenizer.ggml.tokens");
  put_u32(out, 9);  // array
  put_u32(out, 8);  // of strings
  put_u64(out, 7);
  for (const char* piece :
       {"<|im_start|>", "user", "u", "s", "e", "r", "<tool_call>"}) {
    put_string(out, piece);
  }

  put_string(out, "tokenizer.ggml.merges");
  put_u32(out, 9);
  put_u32(out, 8);
  put_u64(out, 3);
  for (const char* rule : {"u s", "us e", "use r"}) {
    put_string(out, rule);
  }

  put_string(out, "tokenizer.ggml.token_type");
  put_u32(out, 9);  // array
  put_u32(out, 5);  // int32
  put_u64(out, 7);
  for (const std::uint32_t type : {3u, 1u, 1u, 1u, 1u, 1u, 4u}) {
    put_u32(out, type);
  }

  const TempFile temp(out, "tge_gguf_control.gguf");
  const tge::gguf::File file = tge::gguf::File::open(temp.path());

  std::string error;
  const std::unique_ptr<tge::models::Tokenizer> tokenizer =
      tge::models::Tokenizer::from_gguf(file, error);
  TGE_REQUIRE(tokenizer != nullptr);

  const std::vector<int> alone = tokenizer->encode("<|im_start|>");
  TGE_REQUIRE(alone.size() == 1);
  TGE_CHECK(alone[0] == 0);

  const std::vector<int> tagged = tokenizer->encode("<|im_start|>user");
  TGE_REQUIRE(tagged.size() == 2);
  TGE_CHECK(tagged[0] == 0);
  TGE_CHECK(tagged[1] == 1);

  const std::vector<int> plain = tokenizer->encode("user");
  TGE_REQUIRE(plain.size() == 1);
  TGE_CHECK(plain[0] == 1);

  // Decoding omits the control id instead of printing its literal text, but a
  // USER_DEFINED token like <tool_call> is ordinary output.
  TGE_CHECK(tokenizer->decode({0, 1}) == "user");
  TGE_CHECK(tokenizer->decode({0}) == "");
  TGE_CHECK(tokenizer->decode({6}) == "<tool_call>");

  const std::vector<int> tool = tokenizer->encode("<tool_call>");
  TGE_REQUIRE(tool.size() == 1);
  TGE_CHECK(tool[0] == 6);

  // A server tokenising user content must not let it forge an added token.
  const std::vector<int> raw = tokenizer->encode("<tool_call>", false);
  TGE_CHECK(!(raw.size() == 1 && raw[0] == 6));
}

TGE_TEST(tokenizer_matches_the_reference_split_rules) {
  // Byte-level vocabulary for the reference splitter's two optional-character
  // rules: an optional space leads a punctuation run, and one non-letter,
  // non-digit character leads a letter run.
  std::vector<std::uint8_t> out;
  put_u32(out, 0x46554747u);
  put_u32(out, 3);
  put_u64(out, 0);  // tensors
  put_u64(out, 3);  // metadata entries

  put_string(out, "tokenizer.ggml.tokens");
  put_u32(out, 9);  // array
  put_u32(out, 8);  // of strings
  put_u64(out, 16);
  for (const char* piece : {"x",         "=",   "\xC4\xA0=", "y",   "\xC4\xA0y",
                            "+",         "\xC4\xA0+", "z", "\xC4\xA0z", "[i",
                            "[",         "i",   ".com",     ".",   "com",
                            "<tool_call>"}) {
    put_string(out, piece);
  }

  put_string(out, "tokenizer.ggml.merges");
  put_u32(out, 9);  // array
  put_u32(out, 8);  // of strings
  put_u64(out, 8);
  for (const char* rule : {"\xC4\xA0 =", "\xC4\xA0 y", "\xC4\xA0 +",
                           "\xC4\xA0 z", "[ i", ". c", ".c o", ".co m"}) {
    put_string(out, rule);
  }

  put_string(out, "tokenizer.ggml.token_type");
  put_u32(out, 9);  // array
  put_u32(out, 5);  // int32
  put_u64(out, 16);
  for (int index = 0; index < 16; ++index) {
    put_u32(out, index == 15 ? 4u : 1u);
  }

  const TempFile temp(out, "tge_gguf_splits.gguf");
  const tge::gguf::File file = tge::gguf::File::open(temp.path());

  std::string error;
  const std::unique_ptr<tge::models::Tokenizer> tokenizer =
      tge::models::Tokenizer::from_gguf(file, error);
  TGE_REQUIRE(tokenizer != nullptr);

  // `x = y + z` is five pieces in the reference: x, " =", " y", " +", " z".
  const std::vector<int> spaced = tokenizer->encode("x = y + z");
  TGE_REQUIRE(spaced.size() == 5);
  TGE_CHECK(spaced[0] == 0);
  TGE_CHECK(spaced[1] == 2);
  TGE_CHECK(spaced[2] == 4);
  TGE_CHECK(spaced[3] == 6);
  TGE_CHECK(spaced[4] == 8);

  // A single leading punctuation character joins the letter run after it.
  const std::vector<int> bracket = tokenizer->encode("[i");
  TGE_REQUIRE(bracket.size() == 1);
  TGE_CHECK(bracket[0] == 9);

  const std::vector<int> dotcom = tokenizer->encode(".com");
  TGE_REQUIRE(dotcom.size() == 1);
  TGE_CHECK(dotcom[0] == 12);
}

TGE_TEST(tokenizer_groups_digits_for_llama_bpe_only) {
  // tiktoken groups digit runs greedily left-to-right in threes ("12345"
  // splits "123"|"45", "1999" splits "199"|"9"); the Qwen2 splitter this
  // engine verified first keeps one digit per piece. The merge list below
  // can only reassemble groups the pre-tokenizer actually emits, so the
  // expected ids distinguish the two rules exactly.
  auto build = [](const char* pre) {
    std::vector<std::uint8_t> out;
    put_u32(out, 0x46554747u);
    put_u32(out, 3);
    put_u64(out, 0);  // tensors
    put_u64(out, 3);  // metadata entries
    put_string(out, "tokenizer.ggml.tokens");
    put_u32(out, 9);  // array
    put_u32(out, 8);  // of strings
    put_u64(out, 16);
    for (const char* piece :
         {"0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "12", "123",
          "45", "19", "199", "a"}) {
      put_string(out, piece);
    }
    put_string(out, "tokenizer.ggml.merges");
    put_u32(out, 9);  // array
    put_u32(out, 8);  // of strings
    put_u64(out, 5);
    for (const char* rule : {"1 2", "12 3", "4 5", "1 9", "19 9"}) {
      put_string(out, rule);
    }
    put_string(out, "tokenizer.ggml.pre");
    put_u32(out, 8);  // string
    put_string(out, pre);
    return out;
  };

  for (const char* pre : {"qwen2", "unknown-splitter"}) {
    const TempFile temp(build(pre), "tge_gguf_qwen_digits.gguf");
    const tge::gguf::File file = tge::gguf::File::open(temp.path());
    std::string error;
    const auto tokenizer = tge::models::Tokenizer::from_gguf(file, error);
    TGE_REQUIRE(tokenizer != nullptr);
    TGE_REQUIRE(!tokenizer->llama_bpe());
    const std::vector<int> split = tokenizer->encode("12345");
    TGE_REQUIRE(split == std::vector<int>({1, 2, 3, 4, 5}));
    const std::vector<int> mixed = tokenizer->encode("12a34");
    TGE_REQUIRE(mixed == std::vector<int>({1, 2, 15, 3, 4}));
  }

  {
    const TempFile temp(build("llama-bpe"), "tge_gguf_llama_digits.gguf");
    const tge::gguf::File file = tge::gguf::File::open(temp.path());
    std::string error;
    const auto tokenizer = tge::models::Tokenizer::from_gguf(file, error);
    TGE_REQUIRE(tokenizer != nullptr);
    TGE_REQUIRE(tokenizer->llama_bpe());
    // Greedy left-to-right triples: "123"|"45", "199"|"9".
    const std::vector<int> grouped = tokenizer->encode("12345");
    TGE_REQUIRE(grouped == std::vector<int>({11, 12}));
    const std::vector<int> grouped1999 = tokenizer->encode("1999");
    TGE_REQUIRE(grouped1999 == std::vector<int>({14, 9}));
    // A letter breaks the run: ["12"] reassembles, ["34"] has no merge.
    const std::vector<int> mixed = tokenizer->encode("12a34");
    TGE_REQUIRE(mixed == std::vector<int>({10, 15, 3, 4}));
  }
}

TGE_TEST(config_reads_end_token_ids) {  const auto put_uint = [](std::vector<std::uint8_t>& out, const char* key,
                           std::uint32_t value) {
    put_string(out, key);
    put_u32(out, 4);  // u32
    put_u32(out, value);
  };
  const auto put_float = [](std::vector<std::uint8_t>& out, const char* key,
                            float value) {
    put_string(out, key);
    put_u32(out, 6);  // f32;
    std::uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    put_u32(out, bits);
  };

  std::vector<std::uint8_t> out;
  put_u32(out, 0x46554747u);
  put_u32(out, 3);
  put_u64(out, 0);   // tensors
  put_u64(out, 13);  // metadata entries

  put_string(out, "general.architecture");
  put_u32(out, 8);
  put_string(out, "qwen2");
  put_uint(out, "qwen2.block_count", 1);
  put_uint(out, "qwen2.embedding_length", 8);
  put_uint(out, "qwen2.feed_forward_length", 16);
  put_uint(out, "qwen2.attention.head_count", 2);
  put_uint(out, "qwen2.attention.head_count_kv", 1);
  put_uint(out, "qwen2.context_length", 16);
  put_float(out, "qwen2.layer_norm_rms_epsilon", 1e-6F);
  put_float(out, "qwen2.rope.freq_base", 10000.0F);
  put_uint(out, "tokenizer.ggml.eos_token_id", 151645);
  put_uint(out, "tokenizer.ggml.eot_token_id", 151646);
  put_uint(out, "tokenizer.ggml.bos_token_id", 151643);
  put_string(out, "tokenizer.ggml.eos_token_ids");
  put_u32(out, 9);  // array
  put_u32(out, 4);  // u32
  put_u64(out, 2);
  put_u32(out, 151646);
  put_u32(out, 151647);

  const TempFile temp(out, "tge_gguf_config.gguf");
  const tge::gguf::File file = tge::gguf::File::open(temp.path());
  const tge::models::Qwen2Config config = tge::models::read_config(file);

  TGE_CHECK(config.eos_token_id == 151645);  TGE_CHECK(config.eot_token_id == 151646);
  TGE_CHECK(config.is_end_token(151643));  // Qwen end-of-text / BOS
  TGE_CHECK(config.is_end_token(151645));  // eos
  TGE_CHECK(config.is_end_token(151646));  // eot and eos_token_ids
  TGE_CHECK(config.is_end_token(151647));  // eos_token_ids
  TGE_CHECK(!config.is_end_token(0));
  TGE_CHECK(!config.is_end_token(-1));
}

namespace {

void put_phi_key_u32(std::vector<std::uint8_t>& out, const char* key,
                     std::uint32_t value) {
  put_string(out, key);
  put_u32(out, 4);  // u32
  put_u32(out, value);
}

void put_phi_key_f32(std::vector<std::uint8_t>& out, const char* key,
                     float value) {
  put_string(out, key);
  put_u32(out, 6);  // f32
  std::uint32_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  put_u32(out, bits);
}

/// Minimal phi3-style header: the epsilon lives under the `attention.` infix
/// and the rope rotates the whole head. `rope_dim` overrides the dimension
/// count so the partial-RoPE guard has something to reject. An `attn`
/// factor below zero omits the key (absent means 1.0); `scaling` null
/// omits the scaling-type key.
std::vector<std::uint8_t> build_phi_config(std::uint32_t rope_dim,
                                           float attn = -1.0F,
                                           const char* scaling = nullptr) {
  std::vector<std::uint8_t> out;
  put_u32(out, 0x46554747u);  // "GGUF"
  put_u32(out, 3);
  put_u64(out, 0);  // tensors
  std::uint64_t entries = 11;
  if (attn >= 0.0F) {
    ++entries;
  }
  if (scaling != nullptr) {
    ++entries;
  }
  put_u64(out, entries);  // metadata entries

  put_string(out, "general.architecture");
  put_u32(out, 8);
  put_string(out, "phi3");
  put_phi_key_u32(out, "phi3.block_count", 1);
  put_phi_key_u32(out, "phi3.embedding_length", 32);
  put_phi_key_u32(out, "phi3.feed_forward_length", 64);
  put_phi_key_u32(out, "phi3.attention.head_count", 2);
  put_phi_key_u32(out, "phi3.attention.head_count_kv", 1);
  put_phi_key_u32(out, "phi3.context_length", 16);
  put_phi_key_f32(out, "phi3.attention.layer_norm_rms_epsilon", 1e-5F);
  put_phi_key_f32(out, "phi3.rope.freq_base", 10000.0F);
  put_phi_key_u32(out, "phi3.rope.dimension_count", rope_dim);
  if (attn >= 0.0F) {
    put_phi_key_f32(out, "phi3.rope.scaling.attn_factor", attn);
  }
  if (scaling != nullptr) {
    put_string(out, "phi3.rope.scaling.type");
    put_u32(out, 8);  // string
    put_string(out, scaling);
  }
  put_phi_key_u32(out, "tokenizer.ggml.eos_token_id", 32000);
  return out;
}

bool read_phi_config_throws(const std::vector<std::uint8_t>& bytes,
                            const char* name) {
  const TempFile temp(bytes, name);
  try {
    const tge::gguf::File file = tge::gguf::File::open(temp.path());
    (void)tge::models::read_config(file);
  } catch (const std::exception&) {
    return true;
  }
  return false;
}

/// One tensor entry plus zero payload reservation: the caller appends the
/// payload bytes after the header is padded to the file alignment.
void put_tensor(std::vector<std::uint8_t>& out, const char* name,
                std::uint32_t rank, std::uint64_t ne0, std::uint64_t ne1,
                std::uint32_t type, std::uint64_t offset) {
  put_string(out, name);
  put_u32(out, rank);
  put_u64(out, ne0);
  if (rank > 1) {
    put_u64(out, ne1);
  }
  put_u32(out, type);
  put_u64(out, offset);
}

void put_tensor_f32(std::vector<std::uint8_t>& out, const char* name,
                    std::uint64_t elements, std::uint64_t offset) {
  put_string(out, name);
  put_u32(out, 1);  // rank
  put_u64(out, elements);
  put_u32(out, 0);  // F32
  put_u64(out, offset);
}

}  // namespace

TGE_TEST(config_prefers_the_attention_prefixed_epsilon) {
  // Phi-3.5 names it `phi3.attention.layer_norm_rms_epsilon` (1e-5); the
  // loader used to read only the unprefixed key and silently ran 1e-6.
  const TempFile temp(build_phi_config(16), "tge_gguf_phi_eps.gguf");
  const tge::gguf::File file = tge::gguf::File::open(temp.path());
  const tge::models::Qwen2Config config = tge::models::read_config(file);
  TGE_CHECK(config.rms_epsilon == 1e-5F);
  TGE_CHECK(config.rope_interleaved == false);  // phi3 is NeoX-family
  TGE_CHECK(config.has_yarn_factors == false);
}

TGE_TEST(config_rejects_a_partial_rope) {
  // Head width is 16 here (32 / 2 heads); a file rotating only 8 of them
  // would mis-rotate under the full-head tables, so it stops at the config.
  TGE_CHECK(read_phi_config_throws(build_phi_config(8),
                                   "tge_gguf_phi_rope.gguf"));
  TGE_CHECK(!read_phi_config_throws(build_phi_config(16),
                                    "tge_gguf_phi_rope_ok.gguf"));
}

TGE_TEST(config_reads_the_yarn_magnitude_scale) {
  // Absent key means 1.0: every existing model keeps its tables exactly.
  {
    const TempFile temp(build_phi_config(16), "tge_gguf_phi_yarn0.gguf");
    const tge::gguf::File file = tge::gguf::File::open(temp.path());
    const tge::models::Qwen2Config config = tge::models::read_config(file);
    TGE_CHECK(config.rope_attn_factor == 1.0F);
  }
  // Phi-3.5 names 1.19024; su/longrope and linear pass through, classic
  // YaRN blending stops loudly instead of running unblended.
  {
    const TempFile temp(build_phi_config(16, 1.19024F, "longrope"),
                        "tge_gguf_phi_yarn1.gguf");
    const tge::gguf::File file = tge::gguf::File::open(temp.path());
    const tge::models::Qwen2Config config = tge::models::read_config(file);
    TGE_CHECK(config.rope_attn_factor == 1.19024F);
  }
  TGE_CHECK(read_phi_config_throws(build_phi_config(16, 1.0F, "yarn"),
                                   "tge_gguf_phi_yarn2.gguf"));
}

TGE_TEST(phi_style_fused_tensors_load) {  // One-block phi3-shaped file: fused attn_qkv (dim + 2*kv_dim rows) and a
  // fused ffn_up (2*ffn rows), all tiny Q8_0 so the load is milliseconds.
  // dim = 32, heads = 2, kv heads = 1 -> head_dim 16, kv_dim 16.
  std::vector<std::uint8_t> out;
  put_u32(out, 0x46554747u);  // "GGUF"
  put_u32(out, 3);
  put_u64(out, 10);  // tensors
  put_u64(out, 11);  // metadata entries (same keys as build_phi_config(16))
  put_string(out, "general.architecture");
  put_u32(out, 8);
  put_string(out, "phi3");
  put_phi_key_u32(out, "phi3.block_count", 1);
  put_phi_key_u32(out, "phi3.embedding_length", 32);
  put_phi_key_u32(out, "phi3.feed_forward_length", 64);
  put_phi_key_u32(out, "phi3.attention.head_count", 2);
  put_phi_key_u32(out, "phi3.attention.head_count_kv", 1);
  put_phi_key_u32(out, "phi3.context_length", 16);
  put_phi_key_f32(out, "phi3.attention.layer_norm_rms_epsilon", 1e-5F);
  put_phi_key_f32(out, "phi3.rope.freq_base", 10000.0F);
  put_phi_key_u32(out, "phi3.rope.dimension_count", 16);
  put_phi_key_u32(out, "tokenizer.ggml.eos_token_id", 32000);

  // Q8_0 rows are 34 bytes per 32 columns; F32 rows are dense.
  struct Entry {
    const char* name;
    bool f32;
    std::uint64_t ne0;
    std::uint64_t ne1;
    std::uint64_t bytes;
  };
  const Entry entries[] = {
      {"token_embd.weight", false, 32, 64, 64 * 34},
      {"output.weight", false, 32, 64, 64 * 34},
      {"output_norm.weight", true, 32, 0, 32 * 4},
      {"blk.0.attn_norm.weight", true, 32, 0, 32 * 4},
      {"blk.0.ffn_norm.weight", true, 32, 0, 32 * 4},
      {"blk.0.attn_qkv.weight", false, 32, 64, 64 * 34},
      {"blk.0.attn_output.weight", false, 32, 32, 32 * 34},
      {"blk.0.ffn_up.weight", false, 32, 128, 128 * 34},
      {"blk.0.ffn_down.weight", false, 64, 32, 32 * 68},
      {"rope_factors_short.weight", true, 8, 0, 8 * 4},
  };
  std::uint64_t offset = 0;
  for (const Entry& entry : entries) {
    if (entry.f32) {
      put_tensor_f32(out, entry.name, entry.ne0, offset);
    } else {
      put_tensor(out, entry.name, 2, entry.ne0, entry.ne1, 8, offset);
    }
    offset += (entry.bytes + 31) / 32 * 32;
  }
  while (out.size() % 32 != 0) {
    out.push_back(0);
  }
  for (std::uint64_t index = 0; index < offset; ++index) {
    out.push_back(0);
  }

  const TempFile temp(out, "tge_gguf_phi_fused.gguf");
  std::string error;
  std::unique_ptr<tge::models::Qwen2> model =
      tge::models::Qwen2::load(temp.path(), error, 16);
  TGE_REQUIRE(model != nullptr);
  TGE_CHECK(error.empty());
  TGE_CHECK(model->config().rms_epsilon == 1e-5F);
  TGE_CHECK(model->config().has_yarn_factors == true);
  TGE_CHECK(model->config().vocab_size == 64);
  TGE_CHECK(model->max_context() == 16);
}

namespace {

/// A seven-piece SentencePiece vocabulary with hand-picked merge ranks:
/// 0 "a", 1 "b", 2 "ab", 3 U+2581, 4 U+2581+"a", 5 "<0x63>" (byte 'c'),
/// 6 "control" (CONTROL). No merges table: that absence is what selects
/// SPM mode for a "llama"-model file. Scores are merge ranks (higher merges
/// earlier), exactly as llama.cpp reads them.
std::vector<std::uint8_t> build_spm_vocab() {
  std::vector<std::uint8_t> out;
  put_u32(out, 0x46554747u);  // "GGUF"
  put_u32(out, 3);
  put_u64(out, 0);  // tensors
  put_u64(out, 4);  // metadata entries

  put_string(out, "tokenizer.ggml.tokens");
  put_u32(out, 9);  // array
  put_u32(out, 8);  // of strings
  put_u64(out, 7);
  put_string(out, "a");
  put_string(out, "b");
  put_string(out, "ab");
  put_string(out, "\xE2\x96\x81");  // U+2581
  put_string(out, "\xE2\x96\x81"
                  "a");
  put_string(out, "<0x63>");
  put_string(out, "control");

  put_string(out, "tokenizer.ggml.scores");
  put_u32(out, 9);  // array
  put_u32(out, 6);  // of f32
  put_u64(out, 7);
  for (const float score :
       {1.0F, 1.0F, 2.5F, 0.5F, 3.0F, -5.0F, 0.0F}) {
    std::uint32_t bits = 0;
    std::memcpy(&bits, &score, sizeof(bits));
    put_u32(out, bits);
  }

  put_string(out, "tokenizer.ggml.token_type");
  put_u32(out, 9);  // array
  put_u32(out, 5);  // of int32
  put_u64(out, 7);
  for (const std::uint32_t type : {0u, 0u, 0u, 0u, 0u, 0u, 3u}) {
    put_u32(out, type);
  }

  put_string(out, "tokenizer.ggml.model");
  put_u32(out, 8);  // string
  put_string(out, "llama");
  return out;
}

}  // namespace

TGE_TEST(spm_selects_the_best_scoring_path) {
  const TempFile temp(build_spm_vocab(), "tge_gguf_spm.gguf");
  const tge::gguf::File file = tge::gguf::File::open(temp.path());
  std::string error;
  const auto tokenizer = tge::models::Tokenizer::from_gguf(file, error);
  TGE_REQUIRE(tokenizer != nullptr);
  TGE_REQUIRE(tokenizer->spm());

  // "a" normalizes to U+2581+"a": the pair merges on its rank (3.0) and
  // the fused piece wins.
  const std::vector<int> fused = tokenizer->encode("a");
  TGE_REQUIRE(fused == std::vector<int>({4}));

  // "ab" normalizes to U+2581+"ab": "U+2581"+"a" (3.0) outranks "ab"
  // (2.5), so the prefix fuses first and the trailing "b" stands alone.
  const std::vector<int> split = tokenizer->encode("ab");
  TGE_REQUIRE(split == std::vector<int>({4, 1}));

  // "c" has no piece: U+2581 by merge, then the 0x63 byte fallback.
  const std::vector<int> byte = tokenizer->encode("c");
  TGE_REQUIRE(byte == std::vector<int>({3, 5}));

  // U+2581 reads back as a space; the byte piece reads back as its byte.
  TGE_CHECK(tokenizer->decode({4}) == " a");
  TGE_CHECK(tokenizer->decode({3, 5}) == " c");
}

TGE_TEST(spm_never_matches_special_pieces_by_score) {
  const TempFile temp(build_spm_vocab(), "tge_gguf_spm_ctl.gguf");
  const tge::gguf::File file = tge::gguf::File::open(temp.path());
  std::string error;
  const auto tokenizer = tge::models::Tokenizer::from_gguf(file, error);
  TGE_REQUIRE(tokenizer != nullptr);

  // The literal control pass still reaches the CONTROL id from text.
  const std::vector<int> literal = tokenizer->encode("control");
  TGE_REQUIRE(literal == std::vector<int>({6}));

  // But user content must not forge it: without the literal pass the long
  // high-scoring-looking piece stays unreachable.
  const std::vector<int> raw = tokenizer->encode("control", false);
  TGE_CHECK(!(raw.size() == 1 && raw[0] == 6));
}

TGE_TEST(bpe_files_do_not_arm_spm) {
  // The merge-table test above builds a GPT-2-style vocabulary; it must
  // keep the merge path even though the coverage below reuses its builder.
  std::vector<std::uint8_t> out;
  put_u32(out, 0x46554747u);  // "GGUF"
  put_u32(out, 3);
  put_u64(out, 0);  // tensors
  put_u64(out, 2);  // metadata entries
  put_string(out, "tokenizer.ggml.tokens");
  put_u32(out, 9);
  put_u32(out, 8);
  put_u64(out, 1);
  put_string(out, "a");
  put_string(out, "tokenizer.ggml.merges");
  put_u32(out, 9);
  put_u32(out, 8);
  put_u64(out, 0);
  const TempFile temp(out, "tge_gguf_nomerge.gguf");
  const tge::gguf::File file = tge::gguf::File::open(temp.path());
  std::string error;
  const auto tokenizer = tge::models::Tokenizer::from_gguf(file, error);
  TGE_REQUIRE(tokenizer != nullptr);
  // Empty merges array: still BPE (the SPM trigger needs a "llama" model
  // AND no merges at all).
  TGE_CHECK(!tokenizer->spm());
}
