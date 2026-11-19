#pragma once

#include <cmath>
#include <cstddef>
#include <fstream>
#include <iterator>
#include <limits>
#include <string>
#include <string_view>
#include <utility>

namespace tge::cli {

/// One `--kv` decision. Every valid value fully determines the triple, so the
/// last flag on the command line wins. The old parsers only cleared `auto_`
/// on `f32` in `generate`, which let an earlier `--kv auto` override a later
/// explicit tier.
struct KvSelection {
  bool half = false;
  bool q8 = false;
  bool auto_ = false;
};

inline bool apply_kv(KvSelection& selection, std::string_view value) {
  if (value == "f32") {
    selection = KvSelection{false, false, false};
    return true;
  }
  if (value == "f16") {
    selection = KvSelection{true, false, false};
    return true;
  }
  if (value == "i8") {
    selection = KvSelection{false, true, false};
    return true;
  }
  if (value == "auto") {
    selection = KvSelection{false, false, true};
    return true;
  }
  return false;
}

inline const char* kv_label(const KvSelection& selection) {
  if (selection.auto_) {
    return "auto";
  }
  if (selection.half) {
    return "f16";
  }
  if (selection.q8) {
    return "i8";
  }
  return "f32";
}

/// Reads a whole prompt file (`--prompt-file`) into `text`, so prompts longer
/// than the command-line limit can be used. Returns an empty string on success,
/// otherwise a message without the `<command>: ` prefix.
inline std::string load_prompt_file(const std::string& path,
                                    std::string& text) {
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    return "cannot open prompt file '" + path + "'";
  }
  std::string loaded((std::istreambuf_iterator<char>(in)),
                     std::istreambuf_iterator<char>());
  if (in.bad()) {
    return "cannot read prompt file '" + path + "'";
  }
  if (loaded.empty()) {
    return "prompt file '" + path + "' is empty";
  }
  text = std::move(loaded);
  return {};
}

/// True when the sum of two positive ints exceeds `limit`, computed without
/// forming the sum (which can overflow signed int for large CLI values). With
/// a, b > 0: a + b > limit  <=>  a > limit - b, and `limit - b` cannot overflow
/// when limit <= INT_MAX and b > 0. Pure so the overflow guard is testable.
inline bool positive_sum_exceeds(int a, int b, int limit) {
  return a > limit - b;
}

/// KL(reference || candidate) from unnormalized non-negative weights (already
/// exponent-shifted by the caller). A term where the oracle has mass (p > 0)
/// but the candidate has none (q <= 0) is +infinity: that is maximal divergence
/// and must be reported, not skipped -- skipping it let a kernel that put zero
/// mass where the oracle has mass report a small finite KL and pass the check.
/// Pure so the divergence treatment is unit-testable without a model.
inline double kl_from_unnormalized(const double* reference,
                                   const double* candidate, std::size_t count) {
  double ref_sum = 0.0;
  double cand_sum = 0.0;
  for (std::size_t i = 0; i < count; ++i) {
    ref_sum += reference[i];
    cand_sum += candidate[i];
  }
  if (!(ref_sum > 0.0)) {
    return 0.0;
  }
  double kl = 0.0;
  for (std::size_t i = 0; i < count; ++i) {
    const double p = reference[i] / ref_sum;
    if (p <= 0.0) {
      continue;
    }
    const double q = cand_sum > 0.0 ? candidate[i] / cand_sum : 0.0;
    if (q <= 0.0) {
      return std::numeric_limits<double>::infinity();
    }
    kl += p * std::log(p / q);
  }
  return kl;
}

/// Throughput column for the bench tables. A kernel that finishes within one
/// steady_clock tick measures as 0 seconds; dividing by that produced inf/NaN
/// in the GFLOPS column. Pure so the guard is unit-testable without a benchmark.
inline double throughput_gflops(double flops, double seconds) {
  return seconds > 0.0 ? flops / seconds / 1e9 : 0.0;
}

/// --prompt-file may not be combined with --prompt or --tokens. Returns an empty
/// string when the combination is valid, otherwise a message without the prefix.
inline std::string prompt_source_conflict(bool has_file, bool has_text, bool has_tokens) {
  if (has_file && (has_text || has_tokens)) {
    return "--prompt-file cannot be combined with --prompt or --tokens";
  }
  return {};
}

/// Pre-load checks shared by `generate` and `fidelity`. Returns an empty string
/// on success, otherwise a message without the `<command>: ` prefix. Keeping
/// this pure makes every combination unit-testable without a model.
struct Preflight {
  bool sparse = false;
  int sparse_window = 512;
  int sparse_stride = 4;
  bool kv_half = false;
  bool kv_q8 = false;
  bool vnni = false;
  bool gpu_on = false;
  bool has_f16c = true;
  bool has_avx_vnni = true;
};

inline std::string validate_preflight(const Preflight& flags) {
  if (flags.sparse && flags.kv_q8) {
    return "--sparse needs the f32|f16 KV cache (no --kv i8)";
  }
  if (flags.sparse && (flags.sparse_window < 1 || flags.sparse_stride < 2)) {
    return "--sparse-window must be >= 1 and --sparse-stride must be >= 2";
  }
  if (flags.kv_half && !flags.has_f16c) {
    return "--kv f16 needs F16C on this CPU";
  }
  if (flags.vnni && !flags.has_avx_vnni) {
    return "--vnni needs AVX-VNNI on this CPU";
  }
  return {};
}

/// The effective state the user/agent needs but the engine used to resolve
/// silently. Rendered as one append-only line so existing output anchors and
/// parsers keep working.
struct Resolved {
  std::string kv;
  std::string kv_source;
  std::string attention;
  int decode_threads = 0;
  int prefill_threads = 0;
  std::string kernel;
  std::string gpu;
  std::string speculation;
};

inline std::string format_resolved(const Resolved& state) {
  std::string line = "resolved : kv=" + state.kv;
  line += " (" + state.kv_source + ")";
  line += " attention=" + state.attention;
  line += " decode_threads=" + std::to_string(state.decode_threads);
  line += " prefill_threads=" + std::to_string(state.prefill_threads);
  line += " kernel=" + state.kernel;
  line += " gpu=" + state.gpu;
  line += " speculation=" + state.speculation;
  return line;
}

/// Focused help for one command. `--help` used to be treated as a value-taking
/// flag (`generate: '--help' needs a value`).
inline std::string command_usage(std::string_view command) {
  if (command == "generate") {
    return "usage: tge generate --model PATH [--prompt TEXT | --prompt-file PATH | --tokens IDS]\n"
           "                     [--n N] [--kv f32|f16|i8|auto]\n"
           "                     [--attention auto|scalar|head|gqa] [--gpu auto|off|on]\n"
           "                     [--kernel kq8|float] [--speculative] [--sparse]\n"
           "                     [--temp T] [--top-k N] [--top-p P] [--min-p P]\n"
           "                     [--repeat-penalty R] [--penalty-window N]\n"
           "                     [--threads N] [--prefill-threads N]\n";
  }
  if (command == "fidelity") {
    return "usage: tge fidelity --model PATH --corpus PATH\n"
           "                      [--context N] [--tokens N] [--kv f32|f16|i8|auto]\n"
           "                      [--kernel kq8|float] [--kq8d head|down|head,down]\n"
           "                      [--gpu auto|off|on] [--sparse]\n";
  }
  if (command == "bench") {
    return "usage: tge bench <gemm|gemv|mem|batch|sessions|continuous> [options]\n"
           "       run `tge bench <name> --help` for that benchmark's options\n";
  }
  if (command == "plan") {
    return "usage: tge plan --model PATH [--context N] [--sessions N] [--bandwidth GB/s]\n";
  }
  if (command == "gguf") {
    return "usage: tge gguf PATH\n";
  }
  if (command == "info") {
    return "usage: tge info [--json]\n";
  }
  return {};
}

/// Focused help for one `bench <name>` subcommand. The generic bench usage
/// promises `tge bench <name> --help` shows that benchmark's options, so the
/// promise must be real: dispatch routes --help with a subcommand here. Returns
/// an empty string for an unknown/missing subcommand so the caller falls back
/// to the generic usage. Pure so the help routing is unit-testable.
inline std::string bench_subcommand_usage(std::string_view subcommand) {
  if (subcommand == "gemm") {
    return "usage: tge bench gemm [--m N] [--n N] [--k N] [--repeat N] "
           "[--threads N]\n";
  }
  if (subcommand == "gemv") {
    return "usage: tge bench gemv [--rows N] [--cols N] [--repeat N] "
           "[--threads N]\n";
  }
  if (subcommand == "mem") {
    return "usage: tge bench mem [--size-mb N] [--repeat N] [--threads N]\n";
  }
  if (subcommand == "batch") {
    return "usage: tge bench batch [--rows N] [--cols N] [--tokens N] "
           "[--repeat N] [--format F] [--kernel K]\n";
  }
  if (subcommand == "sessions") {
    return "usage: tge bench sessions --model PATH [--sessions N] [--tokens N] "
           "[--threads N] [--context N] [--kv T] [--pin-threads 1|0]\n";
  }
  if (subcommand == "continuous") {
    return "usage: tge bench continuous --model PATH [--requests N] "
           "[--tokens N] [--threads N] [--sequences N|auto] [--context N] "
           "[--prompt-tokens N] [--prefill-threads N] [--vnni] [--kv T]\n";
  }
  return {};
}

}  // namespace tge::cli
