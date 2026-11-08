#include <string>

#include "cli_options.hpp"
#include "harness.hpp"
#include "info_json.hpp"

using tge::cli::apply_kv;
using tge::cli::command_usage;
using tge::cli::format_resolved;
using tge::cli::KvSelection;
using tge::cli::Preflight;
using tge::cli::validate_preflight;

TGE_TEST(cli_kv_last_flag_wins) {
  KvSelection selection;
  TGE_CHECK(apply_kv(selection, "auto"));
  TGE_CHECK(selection.auto_);
  TGE_CHECK(apply_kv(selection, "i8"));
  TGE_CHECK(selection.q8);
  TGE_CHECK(!selection.half);
  TGE_CHECK(!selection.auto_);

  TGE_CHECK(apply_kv(selection, "f16"));
  TGE_CHECK(selection.half);
  TGE_CHECK(!selection.q8);
  TGE_CHECK(!selection.auto_);

  TGE_CHECK(apply_kv(selection, "auto"));
  TGE_CHECK(selection.auto_);
  TGE_CHECK(!selection.half);

  TGE_CHECK(apply_kv(selection, "f32"));
  TGE_CHECK(!selection.half);
  TGE_CHECK(!selection.q8);
  TGE_CHECK(!selection.auto_);
  TGE_CHECK(std::string(tge::cli::kv_label(selection)) == "f32");
}

TGE_TEST(cli_kv_rejects_unknown_and_keeps_state) {
  KvSelection selection;
  TGE_CHECK(apply_kv(selection, "f16"));
  TGE_CHECK(!apply_kv(selection, "f64"));
  TGE_CHECK(selection.half);
  TGE_CHECK(!selection.auto_);
}

TGE_TEST(cli_preflight_rejects_impossible_combinations) {
  Preflight flags;
  TGE_CHECK(validate_preflight(flags).empty());

  flags.sparse = true;
  flags.kv_q8 = true;
  TGE_CHECK(validate_preflight(flags).find("--kv i8") != std::string::npos);

  flags = Preflight{};
  flags.sparse = true;
  flags.sparse_stride = 1;
  TGE_CHECK(validate_preflight(flags).find("--sparse-stride") !=
            std::string::npos);

  flags = Preflight{};
  flags.kv_half = true;
  flags.has_f16c = false;
  TGE_CHECK(validate_preflight(flags).find("F16C") != std::string::npos);

  flags = Preflight{};
  flags.vnni = true;
  flags.has_avx_vnni = false;
  TGE_CHECK(validate_preflight(flags).find("AVX-VNNI") != std::string::npos);

  flags = Preflight{};
  flags.sparse = true;
  flags.kv_half = true;
  flags.has_f16c = true;
  TGE_CHECK(validate_preflight(flags).empty());
}

TGE_TEST(cli_resolved_line_names_every_axis) {
  tge::cli::Resolved state;
  state.kv = "f16";
  state.kv_source = "auto";
  state.attention = "auto";
  state.decode_threads = 10;
  state.prefill_threads = 16;
  state.kernel = "kq8";
  state.gpu = "unavailable";
  state.speculation = "off";
  const std::string line = format_resolved(state);
  TGE_CHECK(line.find("kv=f16 (auto)") != std::string::npos);
  TGE_CHECK(line.find("decode_threads=10") != std::string::npos);
  TGE_CHECK(line.find("prefill_threads=16") != std::string::npos);
  TGE_CHECK(line.find("kernel=kq8") != std::string::npos);
  TGE_CHECK(line.find("gpu=unavailable") != std::string::npos);
  TGE_CHECK(line.find("speculation=off") != std::string::npos);
}

TGE_TEST(cli_command_usage_is_focused) {
  TGE_CHECK(command_usage("generate").find("--model") != std::string::npos);
  TGE_CHECK(command_usage("generate").find("bench") == std::string::npos);
  TGE_CHECK(command_usage("fidelity").find("--corpus") != std::string::npos);
  TGE_CHECK(command_usage("info").find("--json") != std::string::npos);
  TGE_CHECK(command_usage("nope").empty());
}

TGE_TEST(cli_info_json_has_every_section_and_escapes) {
  tge::cpu::CpuInfo cpu;
  cpu.vendor = "GenuineIntel";
  cpu.brand = "Test CPU";
  tge::memory::SystemMemory memory;
  memory.total_physical = 16ull * 1024 * 1024 * 1024;
  memory.available_physical = 8ull * 1024 * 1024 * 1024;
  tge::memory::BudgetSnapshot budget;
  budget.usable = 4ull * 1024 * 1024 * 1024;
  tge::KernelSet kernels;
  kernels.gemm_name = "avx2-fma";
  kernels.gemv_name = "avx2-fma";
  kernels.attention_name = "avx2-fma";
  kernels.stream_name = "avx2";
  tge::gpu::GpuInfo gpu_info;
  gpu_info.available = true;
  const std::string device = "GPU \"quoted\"";
  for (std::size_t index = 0; index < device.size(); ++index) {
    gpu_info.name[index] = device[index];
  }
  const std::string json = tge::cli::info_json(
      cpu, memory, budget, kernels, tge::gpu::GpuStatus::Unavailable, gpu_info,
      16, 10);

  TGE_CHECK(json.find("\"vendor\": \"GenuineIntel\"") != std::string::npos);
  TGE_CHECK(json.find("\"memory\"") != std::string::npos);
  TGE_CHECK(json.find("\"budget\"") != std::string::npos);
  TGE_CHECK(json.find("\"kernels\"") != std::string::npos);
  TGE_CHECK(json.find("\"threads\"") != std::string::npos);
  TGE_CHECK(json.find("\"gpu\"") != std::string::npos);
  TGE_CHECK(json.find("\"auto_rules\"") != std::string::npos);
  TGE_CHECK(json.find("GPU \\\"quoted\\\"") != std::string::npos);
  TGE_CHECK(json.find("\"kv_half_context\": " +
                      std::to_string(tge::models::Qwen2::kAutoKvHalfContext)) !=
            std::string::npos);
}
