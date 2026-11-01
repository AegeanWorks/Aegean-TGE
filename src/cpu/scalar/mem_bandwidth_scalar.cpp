#include "tge/cpu/kernels.hpp"

#include <cstddef>

namespace tge::cpu {

void copy_stream_scalar(float* dst, const float* src, std::size_t count) {
  for (std::size_t index = 0; index < count; ++index) {
    dst[index] = src[index];
  }
}

float sum_stream_scalar(const float* src, std::size_t count) {
  float total = 0.0F;
  for (std::size_t index = 0; index < count; ++index) {
    total += src[index];
  }
  return total;
}

}  // namespace tge::cpu
