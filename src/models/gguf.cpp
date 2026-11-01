#include "tge/models/gguf.hpp"

#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <utility>

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace tge::gguf {
namespace {

constexpr std::uint32_t kMagic = 0x46554747u;  // "GGUF"

class Cursor {
 public:
  Cursor(const std::uint8_t* begin, const std::uint8_t* end)
      : begin_(begin), position_(begin), end_(end) {}

  template <typename T>
  T read() {
    require(sizeof(T));
    T value{};
    std::memcpy(&value, position_, sizeof(T));
    position_ += sizeof(T);
    return value;
  }

  std::string read_string() {
    const std::uint64_t length = read<std::uint64_t>();
    require(length);
    std::string text(reinterpret_cast<const char*>(position_),
                     static_cast<std::size_t>(length));
    position_ += length;
    return text;
  }

  std::uint64_t offset() const {
    return static_cast<std::uint64_t>(position_ - begin_);
  }

  /// Bytes still available. Used to reject lengths that cannot possibly be
  /// honest before anything is allocated for them.
  std::uint64_t remaining() const {
    return static_cast<std::uint64_t>(end_ - position_);
  }

 private:
  void require(std::uint64_t bytes) const {
    // Compare against the remaining count, never by adding to the pointer: a
    // corrupt length field would otherwise overflow the pointer before the
    // comparison could fail.
    if (bytes > remaining()) {
      throw std::runtime_error("gguf: file truncated while reading");
    }
  }

  const std::uint8_t* begin_;
  const std::uint8_t* position_;
  const std::uint8_t* end_;
};

Value read_scalar(Cursor& cursor, ValueType type) {
  Value value;
  value.type = type;
  switch (type) {
    case ValueType::Uint8:
      value.unsigned_value = cursor.read<std::uint8_t>();
      break;
    case ValueType::Int8:
      value.signed_value = cursor.read<std::int8_t>();
      break;
    case ValueType::Uint16:
      value.unsigned_value = cursor.read<std::uint16_t>();
      break;
    case ValueType::Int16:
      value.signed_value = cursor.read<std::int16_t>();
      break;
    case ValueType::Uint32:
      value.unsigned_value = cursor.read<std::uint32_t>();
      break;
    case ValueType::Int32:
      value.signed_value = cursor.read<std::int32_t>();
      break;
    case ValueType::Float32:
      value.float_value = cursor.read<float>();
      break;
    case ValueType::Bool:
      value.bool_value = cursor.read<std::uint8_t>() != 0;
      break;
    case ValueType::String:
      value.string_value = cursor.read_string();
      break;
    case ValueType::Uint64:
      value.unsigned_value = cursor.read<std::uint64_t>();
      break;
    case ValueType::Int64:
      value.signed_value = cursor.read<std::int64_t>();
      break;
    case ValueType::Float64:
      value.float_value = cursor.read<double>();
      break;
    case ValueType::Array:
      throw std::runtime_error("gguf: nested arrays are not valid");
  }
  return value;
}

/// Smallest number of bytes one element of `type` can occupy. A string is at
/// least an 8-byte length prefix; every scalar is its own width.
std::uint64_t element_min_bytes(ValueType type) {
  switch (type) {
    case ValueType::Uint8:
    case ValueType::Int8:
    case ValueType::Bool:
      return 1;
    case ValueType::Uint16:
    case ValueType::Int16:
      return 2;
    case ValueType::Uint32:
    case ValueType::Int32:
    case ValueType::Float32:
      return 4;
    case ValueType::Uint64:
    case ValueType::Int64:
    case ValueType::Float64:
    case ValueType::String:
      return 8;
    case ValueType::Array:
      return 12;  // element type tag plus a length field
  }
  return 1;
}

Value read_value(Cursor& cursor) {
  const auto type = static_cast<ValueType>(cursor.read<std::uint32_t>());
  if (type != ValueType::Array) {
    return read_scalar(cursor, type);
  }

  Value value;
  value.type = ValueType::Array;
  const auto element_type = static_cast<ValueType>(cursor.read<std::uint32_t>());
  const std::uint64_t count = cursor.read<std::uint64_t>();

  // A count that cannot physically fit in the bytes that remain is corrupt.
  // Rejecting it here is what stops a malformed file from turning into a
  // multi-gigabyte allocation before the first read fails.
  const std::uint64_t minimum = element_min_bytes(element_type);
  if (count > cursor.remaining() / minimum) {
    throw std::runtime_error("gguf: array length exceeds the remaining file");
  }

  value.elements.reserve(
      static_cast<std::size_t>(std::min<std::uint64_t>(count, 4096)));
  for (std::uint64_t index = 0; index < count; ++index) {
    value.elements.push_back(read_scalar(cursor, element_type));
  }
  return value;
}

}  // namespace

const char* type_name(TensorType type) {
  switch (type) {
    case TensorType::F32:
      return "F32";
    case TensorType::F16:
      return "F16";
    case TensorType::Q4_0:
      return "Q4_0";
    case TensorType::Q4_1:
      return "Q4_1";
    case TensorType::Q5_0:
      return "Q5_0";
    case TensorType::Q5_1:
      return "Q5_1";
    case TensorType::Q8_0:
      return "Q8_0";
    case TensorType::Q4_K:
      return "Q4_K";
    case TensorType::Q5_K:
      return "Q5_K";
    case TensorType::Q6_K:
      return "Q6_K";
    case TensorType::Unknown:
      break;
  }
  return "unknown";
}

std::uint32_t type_block_size(TensorType type) {
  switch (type) {
    case TensorType::F32:
    case TensorType::F16:
      return 1;
    case TensorType::Q4_0:
    case TensorType::Q4_1:
    case TensorType::Q5_0:
    case TensorType::Q5_1:
    case TensorType::Q8_0:
      return 32;
    case TensorType::Q4_K:
    case TensorType::Q5_K:
    case TensorType::Q6_K:
      return 256;
    case TensorType::Unknown:
      break;
  }
  return 0;
}

std::uint32_t type_block_bytes(TensorType type) {
  switch (type) {
    case TensorType::F32:
      return 4;
    case TensorType::F16:
      return 2;
    case TensorType::Q4_0:
      return 18;
    case TensorType::Q4_1:
      return 20;
    case TensorType::Q5_0:
      return 22;
    case TensorType::Q5_1:
      return 24;
    case TensorType::Q8_0:
      return 34;
    case TensorType::Q4_K:
      return 144;
    case TensorType::Q5_K:
      return 176;
    case TensorType::Q6_K:
      return 210;
    case TensorType::Unknown:
      break;
  }
  return 0;
}

bool type_supported(TensorType type) {
  return type == TensorType::F32 || type == TensorType::F16 ||
         type == TensorType::Q8_0;
}

std::uint64_t Value::as_u64() const {
  switch (type) {
    case ValueType::Uint64:
    case ValueType::Uint32:
    case ValueType::Uint16:
    case ValueType::Uint8:
      return unsigned_value;
    case ValueType::Int64:
    case ValueType::Int32:
    case ValueType::Int16:
    case ValueType::Int8:
      return static_cast<std::uint64_t>(signed_value);
    case ValueType::Bool:
      return bool_value ? 1u : 0u;
    case ValueType::Float32:
    case ValueType::Float64:
      return static_cast<std::uint64_t>(float_value);
    default:
      return 0;
  }
}

double Value::as_f64() const {
  switch (type) {
    case ValueType::Float32:
    case ValueType::Float64:
      return float_value;
    case ValueType::Uint64:
    case ValueType::Uint32:
    case ValueType::Uint16:
    case ValueType::Uint8:
      return static_cast<double>(unsigned_value);
    case ValueType::Int64:
    case ValueType::Int32:
    case ValueType::Int16:
    case ValueType::Int8:
      return static_cast<double>(signed_value);
    case ValueType::Bool:
      return bool_value ? 1.0 : 0.0;
    default:
      return 0.0;
  }
}

bool Value::as_bool() const {
  if (type == ValueType::Bool) {
    return bool_value;
  }
  return as_u64() != 0;
}

const std::string& Value::as_string() const { return string_value; }

File::~File() { reset(); }

File::File(File&& other) noexcept { *this = std::move(other); }

File& File::operator=(File&& other) noexcept {
  if (this != &other) {
    reset();
    data_ = std::exchange(other.data_, nullptr);
    map_handle_ = std::exchange(other.map_handle_, nullptr);
    file_handle_ = std::exchange(other.file_handle_, nullptr);
    size_ = std::exchange(other.size_, 0);
    version_ = other.version_;
    alignment_ = other.alignment_;
    data_offset_ = other.data_offset_;
    metadata_ = std::move(other.metadata_);
    tensors_ = std::move(other.tensors_);
    tensor_index_ = std::move(other.tensor_index_);
  }
  return *this;
}

void File::reset() noexcept {
#if defined(_WIN32)
  if (data_ != nullptr) {
    UnmapViewOfFile(data_);
  }
  if (map_handle_ != nullptr) {
    CloseHandle(static_cast<HANDLE>(map_handle_));
  }
  if (file_handle_ != nullptr) {
    CloseHandle(static_cast<HANDLE>(file_handle_));
  }
#else
  if (data_ != nullptr) {
    munmap(data_, static_cast<std::size_t>(size_));
  }
  if (file_handle_ != nullptr) {
    close(static_cast<int>(reinterpret_cast<std::intptr_t>(file_handle_)));
  }
#endif
  data_ = nullptr;
  map_handle_ = nullptr;
  file_handle_ = nullptr;
  size_ = 0;
  metadata_.clear();
  tensors_.clear();
  tensor_index_.clear();
}

File File::open(const std::string& path) {
  File file;

#if defined(_WIN32)
  const int wide_length =
      MultiByteToWideChar(CP_UTF8, 0, path.c_str(), -1, nullptr, 0);
  if (wide_length <= 0) {
    throw std::runtime_error("gguf: invalid path");
  }
  std::wstring wide(static_cast<std::size_t>(wide_length), L'\0');
  MultiByteToWideChar(CP_UTF8, 0, path.c_str(), -1, wide.data(), wide_length);

  HANDLE handle = CreateFileW(wide.c_str(), GENERIC_READ, FILE_SHARE_READ,
                              nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL,
                              nullptr);
  if (handle == INVALID_HANDLE_VALUE) {
    throw std::runtime_error("gguf: cannot open " + path);
  }
  LARGE_INTEGER size{};
  if (!GetFileSizeEx(handle, &size)) {
    CloseHandle(handle);
    throw std::runtime_error("gguf: cannot size " + path);
  }
  HANDLE mapping = CreateFileMappingW(handle, nullptr, PAGE_READONLY, 0, 0, nullptr);
  if (mapping == nullptr) {
    CloseHandle(handle);
    throw std::runtime_error("gguf: cannot map " + path);
  }
  void* view = MapViewOfFile(mapping, FILE_MAP_READ, 0, 0, 0);
  if (view == nullptr) {
    CloseHandle(mapping);
    CloseHandle(handle);
    throw std::runtime_error("gguf: cannot map view of " + path);
  }
  file.file_handle_ = handle;
  file.map_handle_ = mapping;
  file.data_ = view;
  file.size_ = static_cast<std::uint64_t>(size.QuadPart);
#else
  const int descriptor = ::open(path.c_str(), O_RDONLY);
  if (descriptor < 0) {
    throw std::runtime_error("gguf: cannot open " + path);
  }
  struct stat status{};
  if (fstat(descriptor, &status) != 0) {
    ::close(descriptor);
    throw std::runtime_error("gguf: cannot size " + path);
  }
  void* view = mmap(nullptr, static_cast<std::size_t>(status.st_size),
                    PROT_READ, MAP_PRIVATE, descriptor, 0);
  if (view == MAP_FAILED) {
    ::close(descriptor);
    throw std::runtime_error("gguf: cannot map " + path);
  }
  file.file_handle_ = reinterpret_cast<void*>(static_cast<std::intptr_t>(descriptor));
  file.data_ = view;
  file.size_ = static_cast<std::uint64_t>(status.st_size);
#endif

  const auto* bytes = static_cast<const std::uint8_t*>(file.data_);
  Cursor cursor(bytes, bytes + file.size_);

  const std::uint32_t magic = cursor.read<std::uint32_t>();
  if (magic != kMagic) {
    throw std::runtime_error("gguf: not a GGUF file");
  }
  file.version_ = cursor.read<std::uint32_t>();
  if (file.version_ < 2 || file.version_ > 3) {
    throw std::runtime_error("gguf: unsupported version " +
                             std::to_string(file.version_));
  }

  const std::uint64_t tensor_count = cursor.read<std::uint64_t>();
  const std::uint64_t metadata_count = cursor.read<std::uint64_t>();

  for (std::uint64_t index = 0; index < metadata_count; ++index) {
    std::string key = cursor.read_string();
    Value value = read_value(cursor);
    file.metadata_.emplace(std::move(key), std::move(value));
  }

  // Alignment lives in the metadata, which is now parsed; default is 32.
  const auto alignment_entry = file.metadata_.find("general.alignment");
  if (alignment_entry != file.metadata_.end()) {
    const std::uint64_t candidate = alignment_entry->second.as_u64();
    if (candidate != 0 && (candidate & (candidate - 1)) == 0) {
      file.alignment_ = candidate;
    }
  }

  file.tensors_.reserve(static_cast<std::size_t>(tensor_count));
  for (std::uint64_t index = 0; index < tensor_count; ++index) {
    TensorInfo info;
    info.name = cursor.read_string();
    const std::uint32_t dimensions = cursor.read<std::uint32_t>();
    if (dimensions == 0 || dimensions > 8) {
      throw std::runtime_error("gguf: invalid tensor rank");
    }
    info.shape.resize(dimensions);
    for (std::uint32_t axis = 0; axis < dimensions; ++axis) {
      info.shape[axis] = cursor.read<std::uint64_t>();
    }
    info.type = static_cast<TensorType>(cursor.read<std::uint32_t>());
    info.offset = cursor.read<std::uint64_t>();

    info.elements = 1;
    for (const std::uint64_t extent : info.shape) {
      if (extent == 0) {
        throw std::runtime_error("gguf: zero-sized tensor " + info.name);
      }
      info.elements *= extent;
    }
    const std::uint32_t block = type_block_size(info.type);
    const std::uint32_t block_bytes = type_block_bytes(info.type);
    info.bytes = block == 0 ? 0 : (info.elements / block) * block_bytes;
    if (block == 0 || info.elements % block != 0) {
      throw std::runtime_error("gguf: unsupported type for tensor " + info.name);
    }
    file.tensors_.push_back(std::move(info));
  }

  const std::uint64_t align = file.alignment_;
  file.data_offset_ = (cursor.offset() + align - 1) / align * align;

  for (std::size_t index = 0; index < file.tensors_.size(); ++index) {
    const TensorInfo& info = file.tensors_[index];
    if (file.data_offset_ + info.offset + info.bytes > file.size_) {
      throw std::runtime_error("gguf: tensor extends past end of file: " +
                               info.name);
    }
    file.tensor_index_.emplace(info.name, index);
  }

  return file;
}

const Value* File::find(const std::string& key) const {
  const auto entry = metadata_.find(key);
  return entry == metadata_.end() ? nullptr : &entry->second;
}

const TensorInfo* File::tensor(const std::string& name) const {
  const auto entry = tensor_index_.find(name);
  if (entry == tensor_index_.end()) {
    return nullptr;
  }
  return &tensors_[entry->second];
}

std::uint64_t File::get_u64(const std::string& key, std::uint64_t fallback) const {
  const Value* value = find(key);
  return value == nullptr ? fallback : value->as_u64();
}

double File::get_f64(const std::string& key, double fallback) const {
  const Value* value = find(key);
  return value == nullptr ? fallback : value->as_f64();
}

std::string File::get_string(const std::string& key,
                             const std::string& fallback) const {
  const Value* value = find(key);
  if (value == nullptr || value->string_value.empty()) {
    return fallback;
  }
  return value->string_value;
}

const std::uint8_t* File::tensor_data(const TensorInfo& info) const {
  return static_cast<const std::uint8_t*>(data_) + data_offset_ + info.offset;
}

}  // namespace tge::gguf
