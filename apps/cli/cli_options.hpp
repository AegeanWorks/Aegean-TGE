#pragma once

#include <string>
#include <string_view>

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
    return "usage: tge generate --model PATH [--prompt TEXT | --tokens IDS]\n"
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

}  // namespace tge::cli
