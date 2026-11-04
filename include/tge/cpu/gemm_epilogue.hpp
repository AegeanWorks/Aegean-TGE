#pragma once

// Fused clamp epilogue helpers for IDEA-1539. Pure functions used by the
// future microkernel store path; the kernel keeps its behavior until it
// reads GemmArgs::do_clamp. Scalar is the oracle, AVX2 must match bit for
// bit on finite inputs.

#include <cstddef>

namespace tge::cpu {

inline float clamp_scalar_one(float value, float vmin, float vmax) {
  if (value < vmin) return vmin;
  if (value > vmax) return vmax;
  return value;
}

inline void clamp_scalar(float* data, std::size_t count, float vmin,
                         float vmax) {
  if (data == nullptr) return;
  for (std::size_t i = 0; i < count; ++i) {
    data[i] = clamp_scalar_one(data[i], vmin, vmax);
  }
}

void clamp_avx2(float* data, std::size_t count, float vmin, float vmax);

}  // namespace tge::cpu
