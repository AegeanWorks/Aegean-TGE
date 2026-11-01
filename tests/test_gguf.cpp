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
  put_u64(out, 6);
  for (const char* piece : {"<|im_start|>", "user", "u", "s", "e", "r"}) {
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
  put_u64(out, 6);
  for (const std::uint32_t type : {3u, 1u, 1u, 1u, 1u, 1u}) {
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

  // Decoding omits the control id instead of printing its literal text.
  TGE_CHECK(tokenizer->decode({0, 1}) == "user");
  TGE_CHECK(tokenizer->decode({0}) == "");
}

TGE_TEST(config_reads_end_token_ids) {
  const auto put_uint = [](std::vector<std::uint8_t>& out, const char* key,
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
  put_u64(out, 11);  // metadata entries

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

  const TempFile temp(out, "tge_gguf_config.gguf");
  const tge::gguf::File file = tge::gguf::File::open(temp.path());
  const tge::models::Qwen2Config config = tge::models::read_config(file);

  TGE_CHECK(config.eos_token_id == 151645);
  TGE_CHECK(config.eot_token_id == 151646);
  TGE_CHECK(config.is_end_token(151645));
  TGE_CHECK(config.is_end_token(151646));
  TGE_CHECK(!config.is_end_token(0));
  TGE_CHECK(!config.is_end_token(-1));
}
