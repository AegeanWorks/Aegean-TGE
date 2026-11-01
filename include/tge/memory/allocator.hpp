#pragma once

#include <cstddef>

namespace tge::memory {

inline constexpr std::size_t kCacheLine = 64;
inline constexpr std::size_t kPageAlignment = 4096;

/// Owns one aligned allocation. Size zero yields an empty buffer rather than an
/// allocation, so callers do not need to special-case degenerate tensors.
class AlignedBuffer {
 public:
  AlignedBuffer() = default;
  explicit AlignedBuffer(std::size_t bytes, std::size_t alignment = kCacheLine);
  ~AlignedBuffer();

  AlignedBuffer(AlignedBuffer&& other) noexcept;
  AlignedBuffer& operator=(AlignedBuffer&& other) noexcept;
  AlignedBuffer(const AlignedBuffer&) = delete;
  AlignedBuffer& operator=(const AlignedBuffer&) = delete;

  void* data() noexcept { return data_; }
  const void* data() const noexcept { return data_; }
  float* f32() noexcept { return static_cast<float*>(data_); }
  const float* f32() const noexcept { return static_cast<const float*>(data_); }

  std::size_t size() const noexcept { return bytes_; }
  std::size_t alignment() const noexcept { return alignment_; }
  bool empty() const noexcept { return data_ == nullptr; }

  void reset() noexcept;

 private:
  void* data_ = nullptr;
  std::size_t bytes_ = 0;
  std::size_t alignment_ = kCacheLine;
};

/// Rounds an alignment request up to a supported power of two.
std::size_t normalize_alignment(std::size_t requested);

}  // namespace tge::memory
