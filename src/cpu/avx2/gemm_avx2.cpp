#include "tge/cpu/kernels.hpp"

#include <algorithm>
#include <cstddef>
#include <cstring>
#include <vector>

#include <immintrin.h>

#include "tge/core/thread_pool.hpp"
#include "tge/memory/allocator.hpp"

namespace tge::cpu {
namespace {

// Six rows by sixteen columns: twelve YMM accumulators plus two operand
// registers fit the sixteen architectural YMM registers without spilling.
constexpr int kRows = 6;
constexpr int kCols = 16;

// Packed A panels advance by 6 floats per depth step. Padding this to 8 was
// measured and rejected: it removes some cache-line splits on the broadcasts
// but costs 33 percent more panel traffic, and the traffic wins.
constexpr int kAStride = kRows;

// Blocking chosen so one packed A panel plus one packed B panel stay inside a
// 2 MiB L2: 144*256*4 = 144 KiB and 256*256*4 = 256 KiB.
constexpr int kMc = 144;
constexpr int kKc = 256;
constexpr int kNc = 256;

/// C[i,j] += sum_k A[i,k] * B[k,j] for one register tile. Both panels are
/// packed, so every operand load is contiguous. When `accumulate` is false the
/// existing C tile is known to be untouched and is overwritten instead of read,
/// which matters when the whole reduction fits in a single depth block.
inline void microkernel(int kc, const float* a_panel, const float* b_panel,
                        float* c, int ldc, int rows, int cols,
                        bool accumulate) {
  __m256 acc[kRows][2];
  for (int row = 0; row < kRows; ++row) {
    acc[row][0] = _mm256_setzero_ps();
    acc[row][1] = _mm256_setzero_ps();
  }

  for (int depth = 0; depth < kc; ++depth) {
    const __m256 b_low = _mm256_loadu_ps(b_panel);
    const __m256 b_high = _mm256_loadu_ps(b_panel + 8);
    for (int row = 0; row < kRows; ++row) {
      const __m256 a_value = _mm256_broadcast_ss(a_panel + row);
      acc[row][0] = _mm256_fmadd_ps(a_value, b_low, acc[row][0]);
      acc[row][1] = _mm256_fmadd_ps(a_value, b_high, acc[row][1]);
    }
    a_panel += kAStride;
    b_panel += kCols;
  }

  if (rows == kRows && cols == kCols) {
    for (int row = 0; row < kRows; ++row) {
      float* c_row = c + static_cast<std::size_t>(row) * ldc;
      if (accumulate) {
        _mm256_storeu_ps(c_row,
                         _mm256_add_ps(_mm256_loadu_ps(c_row), acc[row][0]));
        _mm256_storeu_ps(c_row + 8,
                         _mm256_add_ps(_mm256_loadu_ps(c_row + 8), acc[row][1]));
      } else {
        _mm256_storeu_ps(c_row, acc[row][0]);
        _mm256_storeu_ps(c_row + 8, acc[row][1]);
      }
    }
    return;
  }

  alignas(32) float tile[kRows][kCols];
  for (int row = 0; row < kRows; ++row) {
    _mm256_storeu_ps(tile[row], acc[row][0]);
    _mm256_storeu_ps(tile[row] + 8, acc[row][1]);
  }
  for (int row = 0; row < rows; ++row) {
    float* c_row = c + static_cast<std::size_t>(row) * ldc;
    for (int col = 0; col < cols; ++col) {
      if (accumulate) {
        c_row[col] += tile[row][col];
      } else {
        c_row[col] = tile[row][col];
      }
    }
  }
}

void pack_a(int mc, int kc, const float* a, int lda, float* packed) {
  float* out = packed;
  for (int row_block = 0; row_block < mc; row_block += kRows) {
    const int rows = std::min(kRows, mc - row_block);
    for (int depth = 0; depth < kc; ++depth) {
      for (int row = 0; row < kAStride; ++row) {
        *out++ = row < rows
                     ? a[static_cast<std::size_t>(row_block + row) * lda + depth]
                     : 0.0F;
      }
    }
  }
}

void pack_b(int kc, int nc, const float* b, int ldb, float* packed) {
  float* out = packed;
  for (int col_block = 0; col_block < nc; col_block += kCols) {
    const int cols = std::min(kCols, nc - col_block);
    for (int depth = 0; depth < kc; ++depth) {
      const float* b_row =
          b + static_cast<std::size_t>(depth) * ldb + col_block;
      for (int col = 0; col < kCols; ++col) {
        *out++ = col < cols ? b_row[col] : 0.0F;
      }
    }
  }
}

}  // namespace

int resolve_thread_count(int requested) {
  if (requested > 0) {
    return std::min(requested, 1024);
  }
  return core::default_thread_count();
}

void gemm_avx2(const GemmArgs& args, const GemmConfig& config) {
  if (args.m <= 0 || args.n <= 0 || args.k <= 0) {
    return;
  }

  const int threads = resolve_thread_count(config.threads);
  core::ThreadPool& pool = core::acquire_pool(threads);

  memory::AlignedBuffer b_panel;
  // One 64-byte aligned panel per row block, so broadcasts never straddle a
  // cache line and each worker owns its buffer outright.
  std::vector<memory::AlignedBuffer> a_panels(
      static_cast<std::size_t>((args.m + kMc - 1) / kMc));
  const int row_block_count = static_cast<int>(a_panels.size());

  for (int col_base = 0; col_base < args.n; col_base += kNc) {
    const int nc = std::min(kNc, args.n - col_base);
    const int col_blocks = (nc + kCols - 1) / kCols;

    for (int depth_base = 0; depth_base < args.k; depth_base += kKc) {
      const int kc = std::min(kKc, args.k - depth_base);
      const bool accumulate = depth_base > 0;

      const std::size_t b_needed =
          static_cast<std::size_t>(col_blocks) * kCols * kc;
      if (b_panel.size() < b_needed * sizeof(float)) {
        b_panel = memory::AlignedBuffer(b_needed * sizeof(float),
                                        memory::kCacheLine);
      }
      pack_b(kc, nc,
             args.b + static_cast<std::size_t>(depth_base) * args.ldb +
                 col_base,
             args.ldb, b_panel.f32());

      pool.parallel_for(row_block_count, 1, [&](int block) {
        const int row_base = block * kMc;
        const int mc = std::min(kMc, args.m - row_base);
        const int row_blocks = (mc + kRows - 1) / kRows;
        const std::size_t needed =
            static_cast<std::size_t>(row_blocks) * kAStride * kc;

        memory::AlignedBuffer& panel =
            a_panels[static_cast<std::size_t>(block)];
        if (panel.size() < needed * sizeof(float)) {
          panel = memory::AlignedBuffer(needed * sizeof(float),
                                        memory::kCacheLine);
        }
        float* a_panel = panel.f32();

        pack_a(mc, kc,
               args.a + static_cast<std::size_t>(row_base) * args.lda +
                   depth_base,
               args.lda, a_panel);

        for (int col_block = 0; col_block < col_blocks; ++col_block) {
          const int cols = std::min(kCols, nc - col_block * kCols);
          for (int rb = 0; rb < row_blocks; ++rb) {
            const int rows = std::min(kRows, mc - rb * kRows);
            microkernel(
                kc,
                a_panel + static_cast<std::size_t>(rb) * kAStride * kc,
                b_panel.f32() +
                    static_cast<std::size_t>(col_block) * kCols * kc,
                args.c +
                    static_cast<std::size_t>(row_base + rb * kRows) *
                        args.ldc +
                    col_base + col_block * kCols,
                args.ldc, rows, cols, accumulate);
          }
        }
      });
    }
  }
}

}  // namespace tge::cpu
