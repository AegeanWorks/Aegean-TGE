// No-op GPU backend ([1/4] bring-up).
//
// Compiled in every configuration. When TGE_HAS_VULKAN=1 the real
// implementation in vulkan_context.cpp owns init()/shutdown()/status()/
// info()/use_gpu() instead and this TU contributes nothing (ODR-safe via
// the guard below). When 0, init() still *detects*: it probes for the
// Vulkan loader (vulkan-1.dll / libvulkan.so.1) with pure OS calls and no
// SDK headers, records what it found, and reports Unavailable so every
// caller keeps the proven CPU kernels behind the single fallback check.

#include "tge/gpu/backend.hpp"

#include <cstring>

#ifndef TGE_HAS_VULKAN
#define TGE_HAS_VULKAN 0
#endif

#if !TGE_HAS_VULKAN
#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <dlfcn.h>
#endif

namespace tge::gpu {
namespace {

GpuStatus g_status = GpuStatus::Disabled;
GpuInfo g_info;
bool g_loader_present = false;

void probe_loader() {
#if defined(_WIN32)
  HMODULE handle =
      LoadLibraryExW(L"vulkan-1.dll", nullptr, LOAD_LIBRARY_AS_DATAFILE);
  if (handle != nullptr) {
    g_loader_present = true;
    FreeLibrary(handle);
  }
#else
  void* handle = dlopen("libvulkan.so.1", RTLD_NOW | RTLD_LOCAL);
  if (handle != nullptr) {
    g_loader_present = true;
    dlclose(handle);
  }
#endif
}

}  // namespace

GpuStatus init() {
  if (g_status != GpuStatus::Disabled) {
    return g_status;
  }
  probe_loader();
  g_info = GpuInfo{};
  if (g_loader_present) {
    std::strncpy(g_info.name, "vulkan-1 loader present (compile-time off)",
                 sizeof(g_info.name) - 1);
  } else {
    std::strncpy(g_info.name, "no vulkan loader", sizeof(g_info.name) - 1);
  }
  g_info.available = false;
  g_status = GpuStatus::Unavailable;
  return g_status;
}

void shutdown() {
  g_status = GpuStatus::Disabled;
  g_info = GpuInfo{};
  g_loader_present = false;
}

GpuStatus status() { return g_status; }

const GpuInfo& info() { return g_info; }

bool use_gpu() { return g_status == GpuStatus::Ready; }

NoopResult run_noop() { return NoopResult{}; }

GemvQ8Result gemv_q8(const void*, std::size_t, int, int, std::size_t,
                     const float*, float*) {
  return GemvQ8Result{};
}

GemmQ8Result gemm_q8(const void*, std::size_t, int, int, std::size_t,
                     const float*, float*, int) {
  return GemmQ8Result{};
}

}  // namespace tge::gpu

#endif  // !TGE_HAS_VULKAN
