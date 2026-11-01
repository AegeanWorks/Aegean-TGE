#pragma once

#include <cstdint>

#include "tge/cpu/kernels.hpp"

namespace tge::ops {

/// Portable reference implementations. Every optimized kernel is validated
/// against these, so they favor obviousness over speed. The reference GEMM
/// ignores `config` and always runs on the calling thread.
void gemm_ref(const cpu::GemmArgs& args, const cpu::GemmConfig& config);

/// Reference matrix-vector product, used as the correctness oracle for the
/// decode kernel.
void gemv_ref(const cpu::GemvArgs& args, const cpu::GemvConfig& config);

void add_inplace(float* dst, const float* src, std::int64_t n);
void scale_inplace(float* dst, float factor, std::int64_t n);
float dot(const float* a, const float* b, std::int64_t n);
float max_abs_diff(const float* a, const float* b, std::int64_t n);

}  // namespace tge::ops
