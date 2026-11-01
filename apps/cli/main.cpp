#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "tge/core/cpu.hpp"
#include "tge/core/dispatch.hpp"
#include "tge/memory/allocator.hpp"
#include "tge/memory/budget.hpp"
#include "tge/memory/system_ram.hpp"
#include "tge/tensor/ops.hpp"

namespace {

using Clock = std::chrono::steady_clock;

void print_usage() {
  std::printf(
      "aegean-tge %s - native hybrid inference engine\n"
      "\n"
      "usage: tge <command> [options]\n"
      "\n"
      "commands:\n"
      "  info [--json]        CPU, memory and kernel selection report\n"
      "  bench [gemm] [opts]  matrix multiply throughput (default)\n"
      "        --m N            rows of A (default 1024)\n"
      "        --n N            columns of B (default 1024)\n"
      "        --k N            shared dimension (default 1024)\n"
      "        --repeat N       timed repetitions (default 5)\n"
      "        --threads N      worker threads (default: all logical)\n"
      "  bench mem [opts]     streaming memory bandwidth\n"
      "        --size-mb N      buffer size per direction (default 256)\n"
      "        --repeat N       timed repetitions (default 5)\n"
      "  version              print the engine version\n",
      TGE_VERSION);
}

void fill_deterministic(float* data, std::size_t count, std::uint32_t seed) {
  std::uint32_t state = seed | 1u;
  for (std::size_t index = 0; index < count; ++index) {
    state = state * 1664525u + 1013904223u;
    data[index] =
        (static_cast<float>((state >> 8) & 0xFFFFu) / 32768.0F) - 1.0F;
  }
}

double best_seconds(const tge::cpu::GemmFn kernel, const tge::cpu::GemmArgs& args,
                    int threads, int repeat) {
  const tge::cpu::GemmConfig config{threads};
  double best = 0.0;
  for (int iteration = 0; iteration < repeat; ++iteration) {
    const auto start = Clock::now();
    kernel(args, config);
    const auto finish = Clock::now();
    const double seconds = std::chrono::duration<double>(finish - start).count();
    if (best == 0.0 || seconds < best) {
      best = seconds;
    }
  }
  return best;
}

void report_row(const char* kernel, int threads, double seconds, double flops) {
  std::printf("  %-12s  %7d  %9.3f ms  %8.2f\n", kernel, threads,
              seconds * 1e3, flops / seconds / 1e9);
}

int command_bench(int m, int n, int k, int repeat, int threads) {
  if (m <= 0 || n <= 0 || k <= 0 || repeat <= 0) {
    std::fprintf(stderr, "bench: dimensions and repeat must be positive\n");
    return 2;
  }

  const int workers = tge::cpu::resolve_thread_count(threads);
  const tge::KernelSet& set = tge::kernels();

  const std::size_t a_count = static_cast<std::size_t>(m) * k;
  const std::size_t b_count = static_cast<std::size_t>(k) * n;
  const std::size_t c_count = static_cast<std::size_t>(m) * n;

  tge::memory::AlignedBuffer a(a_count * sizeof(float));
  tge::memory::AlignedBuffer b(b_count * sizeof(float));
  tge::memory::AlignedBuffer c_ref(c_count * sizeof(float));
  tge::memory::AlignedBuffer c_opt(c_count * sizeof(float));

  fill_deterministic(a.f32(), a_count, 12345u);
  fill_deterministic(b.f32(), b_count, 54321u);

  const tge::cpu::GemmArgs reference_args{m, n, k, a.f32(), k, b.f32(), n,
                                          c_ref.f32(), n};
  const tge::cpu::GemmArgs optimized_args{m, n, k, a.f32(), k, b.f32(), n,
                                          c_opt.f32(), n};
  const double flops = 2.0 * static_cast<double>(m) * n * k;
  const bool optimized = set.gemm_f32 != &tge::ops::gemm_ref;

  std::printf("gemm %dx%dx%d, best of %d\n", m, n, k, repeat);
  std::printf("  %-12s  %7s  %12s  %8s\n", "kernel", "threads", "time", "gflops");

  const double reference_seconds =
      best_seconds(&tge::ops::gemm_ref, reference_args, 1, repeat);
  report_row("reference", 1, reference_seconds, flops);

  double single_seconds = reference_seconds;
  if (optimized) {
    single_seconds =
        best_seconds(set.gemm_f32, optimized_args, 1, repeat);
    report_row(set.gemm_name.c_str(), 1, single_seconds, flops);
  }

  double parallel_seconds = single_seconds;
  if (optimized && workers > 1) {
    parallel_seconds =
        best_seconds(set.gemm_f32, optimized_args, workers, repeat);
    report_row(set.gemm_name.c_str(), workers, parallel_seconds, flops);
  }

  if (optimized) {
    const float worst = tge::ops::max_abs_diff(
        c_ref.f32(), c_opt.f32(), static_cast<std::int64_t>(c_count));
    std::printf("  max |reference - %s| = %.3e\n", set.gemm_name.c_str(),
                static_cast<double>(worst));
    if (!(worst < 1e-2f)) {
      std::fprintf(stderr, "bench: kernel disagrees with the reference\n");
      return 1;
    }
  }
  return 0;
}

int command_mem_bench(int size_mb, int repeat) {
  if (size_mb <= 0 || repeat <= 0) {
    std::fprintf(stderr, "bench mem: size and repeat must be positive\n");
    return 2;
  }

  const std::size_t count =
      static_cast<std::size_t>(size_mb) * 1024 * 1024 / sizeof(float);
  const std::size_t bytes = count * sizeof(float);

  tge::memory::AlignedBuffer source(bytes);
  tge::memory::AlignedBuffer target(bytes);
  fill_deterministic(source.f32(), count, 0xC0FFEEu);

  double copy_best = 0.0;
  double read_best = 0.0;
  float sink = 0.0F;

  for (int iteration = 0; iteration < repeat; ++iteration) {
    auto start = Clock::now();
    tge::cpu::copy_stream_avx2(target.f32(), source.f32(), count);
    auto finish = Clock::now();
    const double copy_seconds =
        std::chrono::duration<double>(finish - start).count();
    if (copy_best == 0.0 || copy_seconds < copy_best) {
      copy_best = copy_seconds;
    }

    start = Clock::now();
    sink += tge::cpu::sum_stream_avx2(source.f32(), count);
    finish = Clock::now();
    const double read_seconds =
        std::chrono::duration<double>(finish - start).count();
    if (read_best == 0.0 || read_seconds < read_best) {
      read_best = read_seconds;
    }
  }

  std::printf("memory bandwidth, %d MiB per buffer, best of %d\n", size_mb,
              repeat);
  std::printf("  copy (read+write)  %8.2f GB/s\n",
              (2.0 * static_cast<double>(bytes)) / copy_best / 1e9);
  std::printf("  read (sum)         %8.2f GB/s\n",
              static_cast<double>(bytes) / read_best / 1e9);
  std::printf("  check value %.1f\n", static_cast<double>(sink));
  return 0;
}

int command_info(bool json) {
  const tge::cpu::CpuInfo cpu = tge::cpu::detect();
  const tge::memory::SystemMemory memory = tge::memory::query_system_memory();
  const tge::KernelSet& kernels = tge::kernels();

  if (json) {
    std::printf("%s", tge::cpu::to_json(cpu).c_str());
    return 0;
  }

  std::printf("aegean-tge %s\n\n", TGE_VERSION);
  std::printf("CPU\n%s", tge::cpu::describe(cpu).c_str());
  std::printf("\nMEMORY\n%s", tge::memory::describe(memory).c_str());

  tge::memory::MemoryBudget budget;
  std::printf(
      "\nMEMORY BUDGET (policy: %.0f%% OS reserve, %.0f%% of available)\n%s",
      budget.policy().os_reserve_fraction * 100.0,
      budget.policy().usable_fraction * 100.0,
      tge::memory::describe(budget.snapshot()).c_str());

  std::printf("\nKERNELS\n");
  std::printf("  gemm f32      : %s\n", kernels.gemm_name.c_str());
  std::printf("  threads       : %d logical\n",
              tge::cpu::resolve_thread_count(0));
  return 0;
}

bool parse_int(const char* text, int& value) {
  char* end = nullptr;
  const long parsed = std::strtol(text, &end, 10);
  if (end == text || *end != '\0') {
    return false;
  }
  value = static_cast<int>(parsed);
  return true;
}

int run_gemm_bench(int argc, char** argv, int first) {
  int m = 1024;
  int n = 1024;
  int k = 1024;
  int repeat = 5;
  int threads = 0;

  for (int index = first; index < argc; index += 2) {
    if (index + 1 >= argc) {
      std::fprintf(stderr, "bench: '%s' needs a value\n", argv[index]);
      return 2;
    }
    const char* flag = argv[index];
    const char* value = argv[index + 1];
    bool ok = false;
    if (std::strcmp(flag, "--m") == 0) {
      ok = parse_int(value, m);
    } else if (std::strcmp(flag, "--n") == 0) {
      ok = parse_int(value, n);
    } else if (std::strcmp(flag, "--k") == 0) {
      ok = parse_int(value, k);
    } else if (std::strcmp(flag, "--repeat") == 0) {
      ok = parse_int(value, repeat);
    } else if (std::strcmp(flag, "--threads") == 0) {
      ok = parse_int(value, threads);
    } else {
      std::fprintf(stderr, "bench: unknown option '%s'\n", flag);
      return 2;
    }
    if (!ok) {
      std::fprintf(stderr, "bench: invalid value for '%s'\n", flag);
      return 2;
    }
  }
  return command_bench(m, n, k, repeat, threads);
}

int run_mem_bench(int argc, char** argv, int first) {
  int size_mb = 256;
  int repeat = 5;

  for (int index = first; index < argc; index += 2) {
    if (index + 1 >= argc) {
      std::fprintf(stderr, "bench mem: '%s' needs a value\n", argv[index]);
      return 2;
    }
    const char* flag = argv[index];
    const char* value = argv[index + 1];
    bool ok = false;
    if (std::strcmp(flag, "--size-mb") == 0) {
      ok = parse_int(value, size_mb);
    } else if (std::strcmp(flag, "--repeat") == 0) {
      ok = parse_int(value, repeat);
    } else {
      std::fprintf(stderr, "bench mem: unknown option '%s'\n", flag);
      return 2;
    }
    if (!ok) {
      std::fprintf(stderr, "bench mem: invalid value for '%s'\n", flag);
      return 2;
    }
  }
  return command_mem_bench(size_mb, repeat);
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    print_usage();
    return 2;
  }

  const std::string command = argv[1];
  if (command == "version" || command == "--version") {
    std::printf("%s\n", TGE_VERSION);
    return 0;
  }
  if (command == "help" || command == "--help" || command == "-h") {
    print_usage();
    return 0;
  }

  if (command == "info") {
    bool json = false;
    for (int index = 2; index < argc; ++index) {
      if (std::strcmp(argv[index], "--json") == 0) {
        json = true;
      } else {
        std::fprintf(stderr, "info: unknown option '%s'\n", argv[index]);
        return 2;
      }
    }
    return command_info(json);
  }

  if (command == "bench") {
    if (argc >= 3 && std::strcmp(argv[2], "mem") == 0) {
      return run_mem_bench(argc, argv, 3);
    }
    const int first = (argc >= 3 && std::strcmp(argv[2], "gemm") == 0) ? 3 : 2;
    return run_gemm_bench(argc, argv, first);
  }

  std::fprintf(stderr, "unknown command '%s'\n\n", command.c_str());
  print_usage();
  return 2;
}
