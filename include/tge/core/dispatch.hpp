#pragma once

#include <string>

#include "tge/core/cpu.hpp"
#include "tge/cpu/kernels.hpp"

namespace tge {

/// The concrete kernels chosen for this machine. Selection happens once, from
/// detected CPU features, and is never re-evaluated silently at runtime.
struct KernelSet {
  cpu::GemmFn gemm_f32 = nullptr;
  std::string gemm_name;
  cpu::Features features;
};

/// Chooses kernels for the detected feature set. Falls back to the portable
/// reference implementation when no optimized kernel is safe to execute.
KernelSet select_kernels(const cpu::Features& features);

/// Kernel set for the machine this process is running on (selected once).
const KernelSet& kernels();

}  // namespace tge
