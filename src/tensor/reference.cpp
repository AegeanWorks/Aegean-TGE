#include "tge/tensor/ops.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>

namespace tge::ops {

void gemm_ref(const cpu::GemmArgs& args, const cpu::GemmConfig& /*config*/) {
  for (int row = 0; row < args.m; ++row) {
    float* c_row = args.c + static_cast<std::size_t>(row) * args.ldc;
    for (int col = 0; col < args.n; ++col) {
      c_row[col] = 0.0F;
    }
    const float* a_row = args.a + static_cast<std::size_t>(row) * args.lda;
    for (int depth = 0; depth < args.k; ++depth) {
      const float scale = a_row[depth];
      if (scale == 0.0F) {
        continue;
      }
      const float* b_row = args.b + static_cast<std::size_t>(depth) * args.ldb;
      for (int col = 0; col < args.n; ++col) {
        c_row[col] += scale * b_row[col];
      }
    }
  }
}

void add_inplace(float* dst, const float* src, std::int64_t n) {
  for (std::int64_t index = 0; index < n; ++index) {
    dst[index] += src[index];
  }
}

void scale_inplace(float* dst, float factor, std::int64_t n) {
  for (std::int64_t index = 0; index < n; ++index) {
    dst[index] *= factor;
  }
}

float dot(const float* a, const float* b, std::int64_t n) {
  float total = 0.0F;
  for (std::int64_t index = 0; index < n; ++index) {
    total += a[index] * b[index];
  }
  return total;
}

float max_abs_diff(const float* a, const float* b, std::int64_t n) {
  float worst = 0.0F;
  for (std::int64_t index = 0; index < n; ++index) {
    worst = std::max(worst, std::fabs(a[index] - b[index]));
  }
  return worst;
}

}  // namespace tge::ops
