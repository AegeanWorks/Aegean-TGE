#include <cstdint>
#include <exception>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include "harness.hpp"
#include "tge/models/gguf.hpp"

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
