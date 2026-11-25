#include "tge/cpu/kernels.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <vector>

#include <immintrin.h>

#include "tge/core/thread_pool.hpp"

namespace tge::cpu {
namespace {

/// One output row of a Q8_0 matrix against a float activation vector.
///
/// Per 34-byte block: a half-precision scale, then 32 signed bytes. The int8
/// payload is widened to int32 and then to float so the multiply stays in the
/// vector unit; the scale is applied once per block rather than per element.
///
/// An integer-domain variant (quantize the activation too, then `maddubs` with
/// no widening at all) was implemented and measured: 31.0 tokens/s median
/// against 30.3-34.0 for this version, which is inside the noise band on this
/// machine. It was reverted because it costs exact token parity with llama.cpp
/// and buys nothing measurable. Decode here is limited by memory and per-call
/// synchronisation, not by the widening chain.
inline float dot_row_q8(const std::uint8_t* row, const float* vector,
                        int blocks) {
  __m256 total = _mm256_setzero_ps();

  for (int block = 0; block < blocks; ++block) {
    const std::uint8_t* entry = row + block * kQ8BlockBytes;
    std::uint16_t scale_bits = 0;
    std::memcpy(&scale_bits, entry, sizeof(scale_bits));
    const __m256 scale = _mm256_set1_ps(half_to_float(scale_bits));

    const auto* values = reinterpret_cast<const std::int8_t*>(entry + 2);
    const float* vector_block = vector + block * kQ8BlockElements;

    __m256 accumulator = _mm256_setzero_ps();
    for (int index = 0; index < kQ8BlockElements; index += 8) {
      const __m128i packed =
          _mm_loadl_epi64(reinterpret_cast<const __m128i*>(values + index));
      const __m256i widened = _mm256_cvtepi8_epi32(packed);
      const __m256 as_float = _mm256_cvtepi32_ps(widened);
      accumulator = _mm256_fmadd_ps(as_float,
                                    _mm256_loadu_ps(vector_block + index),
                                    accumulator);
    }
    total = _mm256_fmadd_ps(scale, accumulator, total);
  }

  const __m128 low = _mm256_castps256_ps128(total);
  const __m128 high = _mm256_extractf128_ps(total, 1);
  __m128 sum = _mm_add_ps(low, high);
  sum = _mm_add_ps(sum, _mm_movehl_ps(sum, sum));
  sum = _mm_add_ss(sum, _mm_shuffle_ps(sum, sum, 1));
  return _mm_cvtss_f32(sum);
}

}  // namespace

void gemv_q8_0_avx2(const Q8GemvArgs& args, const GemvConfig& config) {
  if (args.rows <= 0 || args.cols <= 0) {
    return;
  }
  if (args.cols % kQ8BlockElements != 0) {
    throw std::invalid_argument("gemv_q8_0: cols not a whole block");
  }

  const int threads = resolve_thread_count(config.threads);
  core::ThreadPool& pool =
      config.pool != nullptr ? *config.pool : core::acquire_pool(threads);

  const int blocks = args.cols / kQ8BlockElements;
  const std::size_t row_bytes =
      static_cast<std::size_t>(blocks) * kQ8BlockBytes;

  int band_rows = args.rows / (threads * 4);
  if (band_rows < 1) {
    band_rows = 1;
  }
  const int bands = (args.rows + band_rows - 1) / band_rows;

  pool.parallel_for(bands, 1, [&](int band) {
    const int begin = band * band_rows;
    const int end = std::min(args.rows, begin + band_rows);
    const std::uint8_t* row =
        args.data + static_cast<std::size_t>(begin) * row_bytes;
    for (int index = begin; index < end; ++index) {
      args.out[index] = dot_row_q8(row, args.vector, blocks);
      row += row_bytes;
    }
  });
}

void gemv_q8_0_group_avx2(const Q8GemvArgs* args, std::size_t count,
                          const GemvConfig& config) {
  if (count == 0) {
    return;
  }
  if (args == nullptr) {
    throw std::invalid_argument("gemv_q8_0_group: null arguments");
  }
  const int threads = resolve_thread_count(config.threads);
  core::ThreadPool& pool =
      config.pool != nullptr ? *config.pool : core::acquire_pool(threads);

  struct Task {
    const std::uint8_t* row;
    const float* vector;
    float* out;
    std::size_t row_bytes;
    int blocks;
    int begin;
    int end;
  };
  const std::size_t bands_per_matrix =
      static_cast<std::size_t>(threads) * 4;
  if (count > std::numeric_limits<std::size_t>::max() / bands_per_matrix) {
    throw std::length_error("gemv_q8_0_group: too many matrices");
  }
  const std::size_t expected = count * bands_per_matrix;
  constexpr std::size_t kStackTasks = 256;
  std::array<Task, kStackTasks> stack_tasks;
  std::vector<Task> heap_tasks;
  bool use_stack = expected <= kStackTasks;
  if (!use_stack) {
    heap_tasks.reserve(expected);
  }
  std::size_t task_count = 0;
  auto append_task = [&](const Task& task) mutable {
    if (use_stack && task_count < kStackTasks) {
      stack_tasks[task_count] = task;
      ++task_count;
      return;
    }
    if (use_stack) {
      heap_tasks.assign(stack_tasks.begin(),
                        stack_tasks.begin() + static_cast<std::ptrdiff_t>(task_count));
      heap_tasks.reserve(expected);
      use_stack = false;
    }
    heap_tasks.push_back(task);
    ++task_count;
  };
  for (std::size_t matrix = 0; matrix < count; ++matrix) {
    const Q8GemvArgs& item = args[matrix];
    if (item.rows <= 0 || item.cols <= 0) {
      continue;
    }
    if (item.cols % kQ8BlockElements != 0) {
      throw std::invalid_argument("gemv_q8_0_group: cols not a whole block");
    }
    if (item.data == nullptr || item.vector == nullptr || item.out == nullptr) {
      throw std::invalid_argument("gemv_q8_0_group: null buffer");
    }
    const int blocks = item.cols / kQ8BlockElements;
    const std::size_t row_bytes =
        static_cast<std::size_t>(blocks) * kQ8BlockBytes;
    int band_rows = item.rows / (threads * 4);
    if (band_rows < 1) {
      band_rows = 1;
    }
    for (int begin = 0; begin < item.rows; begin += band_rows) {
      const int end = std::min(item.rows, begin + band_rows);
      append_task(Task{item.data + static_cast<std::size_t>(begin) * row_bytes,
                       item.vector, item.out + begin, row_bytes, blocks,
                       begin, end});
    }
  }
  if (task_count == 0) {
    return;
  }
  if (task_count > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
    throw std::length_error("gemv_q8_0_group: too many bands");
  }
  const Task* tasks = use_stack ? stack_tasks.data() : heap_tasks.data();
  pool.parallel_for(static_cast<int>(task_count), 1, [&](int task_index) {
    const Task& task = tasks[static_cast<std::size_t>(task_index)];
    const std::uint8_t* row = task.row;
    float* out = task.out;
    for (int index = task.begin; index < task.end; ++index) {
      *out++ = dot_row_q8(row, task.vector, task.blocks);
      row += task.row_bytes;
    }
  });
}

}  // namespace tge::cpu
