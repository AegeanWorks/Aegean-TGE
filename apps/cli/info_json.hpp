#pragma once

#include <string>
#include <string_view>

#include "tge/core/cpu.hpp"
#include "tge/core/dispatch.hpp"
#include "tge/gpu/backend.hpp"
#include "tge/memory/budget.hpp"
#include "tge/memory/system_ram.hpp"
#include "tge/models/qwen2.hpp"

namespace tge::cli {

inline std::string json_escape(std::string_view text) {
  std::string escaped;
  escaped.reserve(text.size() + 8);
  for (const char letter : text) {
    switch (letter) {
      case '"':
        escaped += "\\\"";
        break;
      case '\\':
        escaped += "\\\\";
        break;
      case '\n':
        escaped += "\\n";
        break;
      case '\r':
        escaped += "\\r";
        break;
      case '\t':
        escaped += "\\t";
        break;
      default:
        escaped.push_back(letter);
    }
  }
  return escaped;
}

inline const char* gpu_status_name(tge::gpu::GpuStatus status) {
  switch (status) {
    case tge::gpu::GpuStatus::Disabled:
      return "disabled";
    case tge::gpu::GpuStatus::Unavailable:
      return "unavailable";
    case tge::gpu::GpuStatus::Ready:
      return "ready";
  }
  return "unknown";
}

/// Complete `info --json` payload. The CPU object stays at the top level so
/// existing consumers keep their keys; every added section is a new object.
inline std::string info_json(const tge::cpu::CpuInfo& cpu,
                             const tge::memory::SystemMemory& memory,
                             const tge::memory::BudgetSnapshot& budget,
                             const tge::KernelSet& kernels,
                             tge::gpu::GpuStatus gpu_status,
                             const tge::gpu::GpuInfo& gpu_info,
                             int logical_threads, int decode_threads) {
  std::string out = tge::cpu::to_json(cpu);
  while (!out.empty() && (out.back() == '\n' || out.back() == ' ')) {
    out.pop_back();
  }
  if (!out.empty() && out.back() == '}') {
    out.pop_back();
  }

  out += ",\n  \"memory\": {";
  out += "\"total_physical\": " + std::to_string(memory.total_physical);
  out += ", \"available_physical\": " + std::to_string(memory.available_physical);
  out += ", \"installed_physical\": " + std::to_string(memory.installed_physical);
  out += ", \"total_commit\": " + std::to_string(memory.total_commit);
  out += ", \"available_commit\": " + std::to_string(memory.available_commit);
  out += ", \"page_size\": " + std::to_string(memory.page_size);
  out += ", \"load_percent\": " + std::to_string(memory.load_percent);
  out += "},\n";

  out += "  \"budget\": {";
  out += "\"total_physical\": " + std::to_string(budget.total_physical);
  out += ", \"available_physical\": " + std::to_string(budget.available_physical);
  out += ", \"reserved_for_os\": " + std::to_string(budget.reserved_for_os);
  out += ", \"usable\": " + std::to_string(budget.usable);
  out += ", \"pools\": [";
  for (int index = 0; index < tge::memory::kPoolCount; ++index) {
    if (index != 0) {
      out += ", ";
    }
    out += "{\"name\": \"";
    out += tge::memory::pool_name(static_cast<tge::memory::Pool>(index));
    out += "\", \"limit\": " + std::to_string(budget.pools[index].limit);
    out += ", \"used\": " + std::to_string(budget.pools[index].used) + "}";
  }
  out += "]},\n";

  out += "  \"kernels\": {";
  out += "\"gemm_f32\": \"" + json_escape(kernels.gemm_name) + "\"";
  out += ", \"gemv_f32\": \"" + json_escape(kernels.gemv_name) + "\"";
  out += ", \"attention\": \"" + json_escape(kernels.attention_name) + "\"";
  out += ", \"streaming\": \"" + json_escape(kernels.stream_name) + "\"";
  out += "},\n";

  out += "  \"threads\": {";
  out += "\"logical\": " + std::to_string(logical_threads);
  out += ", \"decode\": " + std::to_string(decode_threads);
  out += "},\n";

  out += "  \"gpu\": {";
  out += "\"status\": \"" + std::string(gpu_status_name(gpu_status)) + "\"";
  out += ", \"available\": ";
  out += gpu_info.available ? "true" : "false";
  out += ", \"device\": \"" + json_escape(gpu_info.name) + "\"";
  out += ", \"device_local_bytes\": " + std::to_string(gpu_info.device_local_bytes);
  out += ", \"host_visible_bytes\": " + std::to_string(gpu_info.host_visible_bytes);
  out += "},\n";

  out += "  \"auto_rules\": {\"kv_half_context\": " +
         std::to_string(tge::models::Qwen2::kAutoKvHalfContext) + "}\n";
  out += "}\n";
  return out;
}

}  // namespace tge::cli
