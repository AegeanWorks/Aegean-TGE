#include "tge/memory/allocator.hpp"

#include <cstddef>
#include <cstdlib>
#include <new>
#include <stdexcept>
#include <utility>

#if defined(_WIN32)
#include <malloc.h>
#endif

namespace tge::memory {

std::size_t normalize_alignment(std::size_t requested) {
  if (requested <= kCacheLine) {
    return kCacheLine;
  }
  // Cap absurd requests: shifting past 2^30 would only serve a
  // multi-GB alignment that _aligned_malloc/posix_memalign cannot
  // honour, and shifting past 2^63 wraps to 0 and hangs below.
  constexpr std::size_t kMaxAlignment = std::size_t{1} << 30;
  if (requested > kMaxAlignment) {
    throw std::invalid_argument("normalize_alignment: request too large");
  }
  std::size_t alignment = kCacheLine;
  while (alignment < requested) {
    alignment <<= 1;
  }
  return alignment;
}

AlignedBuffer::AlignedBuffer(std::size_t bytes, std::size_t alignment)
    : alignment_(normalize_alignment(alignment)) {
  if (bytes == 0) {
    return;
  }

#if defined(_WIN32)
  void* allocated = _aligned_malloc(bytes, alignment_);
#else
  void* allocated = nullptr;
  if (posix_memalign(&allocated, alignment_, bytes) != 0) {
    allocated = nullptr;
  }
#endif

  if (allocated == nullptr) {
    throw std::bad_alloc();
  }
  data_ = allocated;
  bytes_ = bytes;
}

AlignedBuffer::~AlignedBuffer() { reset(); }

AlignedBuffer::AlignedBuffer(AlignedBuffer&& other) noexcept
    : data_(std::exchange(other.data_, nullptr)),
      bytes_(std::exchange(other.bytes_, 0)),
      alignment_(other.alignment_) {}

AlignedBuffer& AlignedBuffer::operator=(AlignedBuffer&& other) noexcept {
  if (this != &other) {
    reset();
    data_ = std::exchange(other.data_, nullptr);
    bytes_ = std::exchange(other.bytes_, 0);
    alignment_ = other.alignment_;
  }
  return *this;
}

void AlignedBuffer::reset() noexcept {
  if (data_ != nullptr) {
#if defined(_WIN32)
    _aligned_free(data_);
#else
    std::free(data_);
#endif
    data_ = nullptr;
  }
  bytes_ = 0;
}

}  // namespace tge::memory
