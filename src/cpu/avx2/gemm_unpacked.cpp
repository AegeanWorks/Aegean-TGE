#include "tge/cpu/kernels.hpp"

#include <algorithm>
#include <cstddef>
#include <cstring>

#include <immintrin.h>

namespace tge::cpu {
namespace {

// The kernel as it existed before the packing and blocking rewrite: a 6x16
// register tile that reads A and B in place, column tiles outside row tiles,
// no panels and no cache blocking. Kept as a benchmark control so the rewrite
// can be attributed; it is never selected by dispatch.
constexpr int kRows = 6;
constexpr int kCols = 16;

void microkernel(int k, const float* const* a_rows, const float* b, int ldb,
                 float* const* c_rows) {
  __m256 acc[kRows][2];
  for (int row = 0; row < kRows; ++row) {
    acc[row][0] = _mm256_setzero_ps();
    acc[row][1] = _mm256_setzero_ps();
  }

  for (int depth = 0; depth < k; ++depth) {
    const float* b_row = b + static_cast<std::size_t>(depth) * ldb;
    const __m256 b_low = _mm256_loadu_ps(b_row);
    const __m256 b_high = _mm256_loadu_ps(b_row + 8);
    for (int row = 0; row < kRows; ++row) {
      const __m256 a_value = _mm256_broadcast_ss(a_rows[row] + depth);
      acc[row][0] = _mm256_fmadd_ps(a_value, b_low, acc[row][0]);
      acc[row][1] = _mm256_fmadd_ps(a_value, b_high, acc[row][1]);
    }
  }

  for (int row = 0; row < kRows; ++row) {
    _mm256_storeu_ps(c_rows[row], acc[row][0]);
    _mm256_storeu_ps(c_rows[row] + 8, acc[row][1]);
  }
}

void scalar_region(int row_begin, int row_end, int col_begin, int col_end,
                   int k, const float* a, int lda, const float* b, int ldb,
                   float* c, int ldc) {
  for (int row = row_begin; row < row_end; ++row) {
    const float* a_row = a + static_cast<std::size_t>(row) * lda;
    float* c_row = c + static_cast<std::size_t>(row) * ldc;
    for (int depth = 0; depth < k; ++depth) {
      const float scale = a_row[depth];
      if (scale == 0.0F) {
        continue;
      }
      const float* b_row = b + static_cast<std::size_t>(depth) * ldb;
      for (int col = col_begin; col < col_end; ++col) {
        c_row[col] += scale * b_row[col];
      }
    }
  }
}

}  // namespace

void gemm_avx2_unpacked(const GemmArgs& args, const GemmConfig& /*config*/) {
  if (args.m <= 0 || args.n <= 0 || args.k <= 0) {
    return;
  }

  for (int row = 0; row < args.m; ++row) {
    std::memset(args.c + static_cast<std::size_t>(row) * args.ldc, 0,
                sizeof(float) * static_cast<std::size_t>(args.n));
  }

  const int row_tiles = args.m / kRows;
  const int col_tiles = args.n / kCols;

  for (int col_tile = 0; col_tile < col_tiles; ++col_tile) {
    const int col_base = col_tile * kCols;
    for (int row_tile = 0; row_tile < row_tiles; ++row_tile) {
      const int row_base = row_tile * kRows;
      const float* a_rows[kRows];
      float* c_rows[kRows];
      for (int row = 0; row < kRows; ++row) {
        a_rows[row] =
            args.a + static_cast<std::size_t>(row_base + row) * args.lda;
        c_rows[row] = args.c +
                      static_cast<std::size_t>(row_base + row) * args.ldc +
                      col_base;
      }
      microkernel(args.k, a_rows, args.b + col_base, args.ldb, c_rows);
    }
  }

  const int full_rows = row_tiles * kRows;
  const int full_cols = col_tiles * kCols;
  if (full_rows < args.m) {
    scalar_region(full_rows, args.m, 0, args.n, args.k, args.a, args.lda,
                  args.b, args.ldb, args.c, args.ldc);
  }
  if (full_cols < args.n) {
    scalar_region(0, full_rows, full_cols, args.n, args.k, args.a, args.lda,
                  args.b, args.ldb, args.c, args.ldc);
  }
}

}  // namespace tge::cpu
