#include <cmath>
#include <filesystem>
#include <fstream>
#include <string>

#include "cli_options.hpp"
#include "draft.hpp"
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

TGE_TEST(bench_help_reaches_each_subcommand) {
  // Regression (review-bench-subcommand-help-intercepted): generic bench help
  // promises `tge bench <name> --help` shows that benchmark's options, but the
  // blanket intercept printed generic usage. Each subcommand now has focused help.
  for (const char* name : {"gemm", "gemv", "mem", "batch", "sessions",
                           "continuous"}) {
    const std::string usage = tge::cli::bench_subcommand_usage(name);
    TGE_CHECK(!usage.empty());
    TGE_CHECK(usage.find(std::string("tge bench ") + name) != std::string::npos);
    TGE_CHECK(usage.find("usage:") != std::string::npos);
  }
  // Unknown/missing subcommand yields no focused help so dispatch falls back.
  TGE_CHECK(tge::cli::bench_subcommand_usage("nope").empty());
  TGE_CHECK(tge::cli::bench_subcommand_usage("").empty());
  // Focused gemm help carries gemm flags, not generate flags.
  const std::string gemm = tge::cli::bench_subcommand_usage("gemm");
  TGE_CHECK(gemm.find("--m N") != std::string::npos);
  TGE_CHECK(gemm.find("--model PATH") == std::string::npos);
}

TGE_TEST(positive_sum_guard_avoids_signed_overflow) {
  // Regression: prompt_tokens + tokens overflowed signed int for large CLI
  // values, wrapping negative and bypassing the context guard.
  TGE_CHECK(!tge::cli::positive_sum_exceeds(100, 200, 4096));
  TGE_CHECK(!tge::cli::positive_sum_exceeds(2000, 2096, 4096));
  TGE_CHECK(tge::cli::positive_sum_exceeds(2001, 2096, 4096));
  // The overflow case: both operands positive, sum wraps past INT_MAX.
  constexpr int kBig = 2000000000;
  TGE_CHECK(tge::cli::positive_sum_exceeds(kBig, kBig, 4096));
  TGE_CHECK(tge::cli::positive_sum_exceeds(kBig, kBig, 64));
  // The 64-row prefill guard uses the same helper.
  TGE_CHECK(tge::cli::positive_sum_exceeds(40, 30, 64));
  TGE_CHECK(!tge::cli::positive_sum_exceeds(40, 24, 64));
}

TGE_TEST(draft_fills_all_slots_on_five_deep_fallback) {
  // Regression (review-uninitialized-draft-slot, high): the greedy caller
  // declares int draft[6] and copies all six into the verifier batch, but a
  // 6-deep prompt_lookup_draft that succeeds on its 5-deep fallback wrote only
  // draft[0..4]. draft[5] was uninitialized stack memory. This history forces
  // that fallback (no 6-deep match, a 5-deep match), so a sentinel in draft[5]
  // must be overwritten by the fix.
  const std::vector<int> history = {0, 3, 3, 0, 0, 2, 3, 2, 3, 0, 3,
                                    2, 3, 0, 1, 2, 2, 1, 0, 1, 4};
  constexpr int kSentinel = -999;
  int draft[6] = {kSentinel, kSentinel, kSentinel,
                  kSentinel, kSentinel, kSentinel};
  TGE_CHECK(tge::cli::prompt_lookup_draft(history, draft, 6));
  for (int i = 0; i < 6; ++i) {
    TGE_CHECK(draft[i] != kSentinel);  // every slot written on success
  }
  // The tail filler uses the last real proposal.
  TGE_CHECK(draft[5] == draft[4]);
}

TGE_TEST(cli_kl_reports_infinite_divergence) {
  // Regression: the fidelity KL loop skipped any index where q <= 0, so a
  // candidate that put zero mass where the oracle has mass reported a small
  // finite KL instead of maximal divergence.
  const double ref[3] = {1.0, 1.0, 0.0};
  const double same[3] = {1.0, 1.0, 0.0};
  const double divergent[3] = {1.0, 0.0, 0.0};  // zero mass on oracle token 2
  // Identical -> zero KL.
  TGE_CHECK(tge::cli::kl_from_unnormalized(ref, same, 3) < 1e-12);
  // Divergent -> +infinity, not a finite number.
  const double kl = tge::cli::kl_from_unnormalized(ref, divergent, 3);
  TGE_CHECK(std::isinf(kl));
  TGE_CHECK(kl > 0.0);
  // A completely disjoint candidate also diverges.
  const double other[3] = {0.0, 0.0, 1.0};
  TGE_CHECK(std::isinf(tge::cli::kl_from_unnormalized(ref, other, 3)));
}

TGE_TEST(cli_throughput_guards_zero_seconds) {
  // Regression: a benchmark that finishes within one steady_clock tick
  // measures 0 seconds; report_row divided by it, printing inf/NaN GFLOPS.
  TGE_CHECK(tge::cli::throughput_gflops(2e9, 1.0) > 1.99);
  TGE_CHECK(tge::cli::throughput_gflops(2e9, 1.0) < 2.01);
  // Non-positive durations must yield a finite, non-negative rate. The bug
  // produced +inf at 0 seconds (printed as "inf" in the GFLOPS column).
  const double zero = tge::cli::throughput_gflops(2e9, 0.0);
  TGE_CHECK(std::isfinite(zero));
  TGE_CHECK(zero >= 0.0);
  const double negative = tge::cli::throughput_gflops(2e9, -1.0);
  TGE_CHECK(std::isfinite(negative));
  TGE_CHECK(negative >= 0.0);
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

TGE_TEST(cli_prompt_file_loads_text_and_rejects_bad_paths) {
  namespace fs = std::filesystem;
  const fs::path path = fs::temp_directory_path() / "tge_prompt_file_test.txt";
  const std::string body = "line one\nline two with \xC3\xA9 and no trailing newline";
  {
    std::ofstream out(path, std::ios::binary);
    out << body;
  }
  std::string text;
  TGE_CHECK(tge::cli::load_prompt_file(path.string(), text).empty());
  TGE_CHECK(text == body);

  std::error_code ignored;
  fs::remove(path, ignored);
  TGE_CHECK(!tge::cli::load_prompt_file(path.string(), text).empty());

  {
    std::ofstream out(path, std::ios::binary);
  }
  TGE_CHECK(tge::cli::load_prompt_file(path.string(), text).find("empty") !=
            std::string::npos);
  fs::remove(path, ignored);
}

TGE_TEST(cli_prompt_file_conflicts_with_other_prompt_sources) {
  TGE_CHECK(tge::cli::prompt_source_conflict(true, false, false).empty());
  TGE_CHECK(tge::cli::prompt_source_conflict(false, true, true).empty());
  TGE_CHECK(!tge::cli::prompt_source_conflict(true, true, false).empty());
  TGE_CHECK(!tge::cli::prompt_source_conflict(true, false, true).empty());
}

TGE_TEST(cli_generate_usage_names_prompt_file) {
  TGE_CHECK(command_usage("generate").find("--prompt-file") !=
            std::string::npos);
}
