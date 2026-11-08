#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace tge::gguf {

enum class ValueType : std::uint32_t {
  Uint8 = 0,
  Int8 = 1,
  Uint16 = 2,
  Int16 = 3,
  Uint32 = 4,
  Int32 = 5,
  Float32 = 6,
  Bool = 7,
  String = 8,
  Array = 9,
  Uint64 = 10,
  Int64 = 11,
  Float64 = 12,
};

/// A metadata value. Array elements are stored inline; only the tags used by
/// model configuration are interpreted, everything else is preserved verbatim.
struct Value {
  ValueType type = ValueType::Uint32;
  std::uint64_t unsigned_value = 0;
  std::int64_t signed_value = 0;
  double float_value = 0.0;
  bool bool_value = false;
  std::string string_value;
  std::vector<Value> elements;

  std::uint64_t as_u64() const;
  double as_f64() const;
  bool as_bool() const;
  const std::string& as_string() const;
};

/// GGML element types this engine can name. The numeric values are the format's
/// own identifiers and must not be renumbered.
enum class TensorType : std::uint32_t {
  F32 = 0,
  F16 = 1,
  Q4_0 = 2,
  Q4_1 = 3,
  Q5_0 = 6,
  Q5_1 = 7,
  Q8_0 = 8,
  Q4_K = 12,
  Q5_K = 13,
  Q6_K = 14,
  Unknown = 0xFFFFFFFFu,
};

const char* type_name(TensorType type);
std::uint32_t type_block_size(TensorType type);
std::uint32_t type_block_bytes(TensorType type);
bool type_supported(TensorType type);

struct TensorInfo {
  std::string name;
  std::vector<std::uint64_t> shape;
  TensorType type = TensorType::Unknown;
  std::uint64_t offset = 0;
  std::uint64_t bytes = 0;
  std::uint64_t elements = 0;
};

/// Wrap-safe containment test for one tensor's byte range. Written as
/// subtractions so a hostile offset cannot wrap `data_offset + offset + bytes`
/// back under the file size.
inline bool tensor_range_fits(std::uint64_t data_offset, std::uint64_t offset,
                              std::uint64_t bytes, std::uint64_t size) {
  return data_offset <= size && offset <= size - data_offset &&
         bytes <= size - data_offset - offset;
}

/// A read-only view of a GGUF file mapped into this process.
///
/// Weights are never copied: `data()` points straight at the mapped bytes and
/// every tensor is addressed by offset. The OS page cache then decides what is
/// resident, which is the behaviour the memory architecture depends on.
class File {
 public:
  File() = default;
  ~File();
  File(const File&) = delete;
  File& operator=(const File&) = delete;
  File(File&& other) noexcept;
  File& operator=(File&& other) noexcept;

  /// `load_large_arrays = false` parses the metadata but skips arrays above a
  /// size threshold without materialising them. The model body needs none of
  /// them; the 300k vocabulary entries cost more to build than the weights cost
  /// to map. Callers that need them (the tokenizer) open their own File.
  static File open(const std::string& path, bool load_large_arrays = true);

  bool valid() const { return data_ != nullptr; }
  std::uint32_t version() const { return version_; }
  std::uint64_t alignment() const { return alignment_; }

  const std::unordered_map<std::string, Value>& metadata() const {
    return metadata_;
  }
  const std::vector<TensorInfo>& tensors() const { return tensors_; }

  const Value* find(const std::string& key) const;
  const TensorInfo* tensor(const std::string& name) const;

  std::uint64_t get_u64(const std::string& key, std::uint64_t fallback) const;
  double get_f64(const std::string& key, double fallback) const;
  std::string get_string(const std::string& key, const std::string& fallback) const;

  /// Pointer to the start of the tensor payload. Valid for the life of the File.
  const std::uint8_t* tensor_data(const TensorInfo& info) const;

  /// Touches every mapped page once so a cold first prefill does not pay the
  /// fault-in (43 s on the VM's storage for 676 MiB). The total bytes read are
  /// identical; this moves them out of the first token into an explicit step.
  void prefetch() const;

  std::uint64_t file_size() const { return size_; }

 private:
  void reset() noexcept;

  void* data_ = nullptr;
  void* map_handle_ = nullptr;
  void* file_handle_ = nullptr;
  std::uint64_t size_ = 0;
  std::uint32_t version_ = 0;
  std::uint64_t alignment_ = 32;
  std::uint64_t data_offset_ = 0;
  std::unordered_map<std::string, Value> metadata_;
  std::vector<TensorInfo> tensors_;
  std::unordered_map<std::string, std::size_t> tensor_index_;
};

}  // namespace tge::gguf
