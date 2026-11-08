#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <fstream>
#include <memory>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <shellapi.h>
#endif

#include "tge/core/cpu.hpp"
#include "tge/core/dispatch.hpp"
#include "tge/core/thread_pool.hpp"
#include "tge/gpu/backend.hpp"
#include "tge/memory/allocator.hpp"
#include "tge/memory/budget.hpp"
#include "tge/memory/system_ram.hpp"
#include "tge/models/gguf.hpp"
#include "tge/models/plan.hpp"
#include "tge/models/qwen2.hpp"
#include "tge/models/sampling.hpp"
#include "tge/models/tokenizer.hpp"
#include "tge/tensor/ops.hpp"

namespace {

using Clock = std::chrono::steady_clock;

constexpr double kGiB = 1024.0 * 1024.0 * 1024.0;

void print_usage() {
  std::printf(
      "aegean-tge %s - native hybrid inference engine\n"
      "\n"
      "usage: tge <command> [options]\n"
      "\n"
      "commands:\n"
      "  info [--json]        CPU, memory and kernel selection report\n"
      "  bench gemm [opts]    matrix multiply throughput (default)\n"
      "        --m N            rows of A (default 1024)\n"
      "        --n N            columns of B (default 1024)\n"
      "        --k N            shared dimension (default 1024)\n"
      "        --repeat N       timed repetitions (default 5)\n"
      "        --threads N      worker threads (default: all logical)\n"
      "  bench gemv [opts]    decode-shaped matrix-vector throughput\n"
      "        --rows N         weight rows (default 4096)\n"
      "        --cols N         reduction width (default 4096)\n"
      "        --repeat N       timed repetitions (default 5)\n"
      "        --threads N      worker threads (default: all logical)\n"
      "  bench mem [opts]     streaming memory bandwidth\n"
      "        --size-mb N      buffer size per direction (default 256)\n"
      "        --repeat N       timed repetitions (default 5)\n"
      "        --threads N      worker threads (default: all logical)\n"
      "  bench batch [opts]   one prefill matrix-vector kernel, in isolation\n"
      "        --rows N --cols N --tokens N --repeat N\n"
      "        --format F       q8|q40|q4k|q5k|q6k|q50|q51 weight format (default q8)\n"
      "        --kernel K       vnni to route Q8 through the VNNI kernel,\n"
      "                       kq8 for the K-quant x Q8-activation integer prefill kernel,\n"
      "                       kq8d for its single-row decode variant (tokens=1)\n"
      "  gguf <path>          inspect a GGUF model file\n"
      "  plan [opts]          memory budget and decode ceiling, before loading\n"
      "        --model PATH --context N --sessions N --bandwidth GB/s\n"
      "  bench sessions [opts] concurrent decoding over shared weights\n"
      "        --model PATH --sessions N --tokens N --threads N --context N\n"
      "        --kv T           KV cache type: f32|f16|i8|auto (default f32)\n"
      "        --pin-threads 1|0  pin decode workers to P-cores (default 0)\n"
      "  bench continuous [opts] continuous batching over N KV slots\n"
      "        --model PATH --requests N --tokens N --threads N --sequences N\n"
      "        --prompt-tokens N   prompt length batched with the decodes\n"
      "        --prefill-threads N prefill workers (default: all logical)\n"
      "        --vnni | --kv f16   optional VNNI / half-KV trades\n"
      "  generate [opts]      greedy decode and token-rate measurement\n"
      "        --model PATH     GGUF model to load\n"
      "        --tokens \"a b c\"  prompt token ids\n"
      "        --prompt \"text\"   prompt as text, tokenized by the model\n"
      "        --n N            tokens to generate (default 32)\n"
      "        --threads N      worker threads (default: all logical)\n"
      "        --prefill-threads N prefill workers (default: all logical)\n"
      "        --pin-threads 1|0  pin decode workers to P-cores (default 0)\n"
      "        --context N      KV cache length (default 4096, capped by model)\n"
      "        --vocabulary M   auto|eager|lazy tokenizer parsing (default auto)\n"
      "        --temp T         sampling temperature; 0 is greedy (default 0)\n"
      "        --top-k N        keep the N best tokens (default off)\n"
      "        --top-p P        nucleus probability mass (default 1.0)\n"
      "        --min-p P        drop tokens below this fraction of the best\n"
      "        --repeat-penalty R  penalty for already seen tokens (default 1)\n"
      "        --penalty-window N  recent tokens the penalty sees; 0 is all (default 64)\n"
      "        --seed N         random seed for sampling\n"
      "        --attention A    auto|scalar|head|gqa attention (default auto)\n"
      "        --sparse         static sparse subset: sink(4)+stride+window\n"
      "        --sparse-window N  most-recent kept positions (default 512)\n"
      "        --sparse-stride N  every N-th middle position kept (default 4)\n"
      "        --speculative    prompt-lookup speculation: greedy +-penalty,\n"
      "                         temperature-only sampling (other cuts fall back)\n"
      "        --no-fuse        A/B gate: disable fused steady state (baseline\n"
      "                         correction-forward path, same binary)\n"
      "        --no-skip        A/B gate: disable SKIP-locked fast path\n"
      "                         (spans/bigram/gapped chain only)\n"
      "        --vnni           AVX-VNNI Q8 batch kernels (activation reused)\n"
      "        --kernel kq8     integer Q4_K/Q6_K x Q8_K prefill kernels (default on)\n"
      "        --kernel float   float dequant prefill kernels (default off)\n"
      "        --kq8d S         opt-in integer decode scopes: head|down|head,down\n"
      "                         (default off; lossy near-tie logits, see README)\n"
      "        --kv T           KV cache type: f32|f16|i8|auto (default f32)\n"
      "        --gpu M          auto|off|on GPU offload (default auto)\n"
      "        --prefetch       fault weights before first token, report ms\n"
      "        --print-ids      print generated token ids\n"
      "        --text           detokenize and print the generated text\n"
      "  fidelity [opts]      full-scalar oracle vs selected kernels, teacher-forced\n"
      "        --model PATH --corpus PATH\n"
      "        --vnni           compare against the VNNI batch kernels\n"
      "        --kernel kq8     compare against the integer Q4_K/Q6_K x Q8_K prefill kernel (default on)\n"
      "        --kernel float   compare against the float dequant prefill kernels (default off)\n"
      "        --kq8d S         compare against opt-in integer decode scopes:\n"
      "                         head|down|head,down (default off)\n"
      "        --kv T           KV cache type: f32|f16|i8|auto (default f32)\n"
      "        --gpu M          auto|off|on GPU offload (default auto)\n"
      "        --sparse         price the static sparse subset on the candidate\n"
      "        --sparse-window N  most-recent kept positions (default 512)\n"
      "        --sparse-stride N  every N-th middle position kept (default 4)\n"
      "        --context N      context used by both models (default 1024)\n"
      "        --tokens N       stop after N corpus tokens (default all)\n"
      "        --threads N      worker threads (default: all logical)\n"
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

double best_gemm_seconds(const tge::cpu::GemmFn kernel,
                         const tge::cpu::GemmArgs& args, int threads,
                         int repeat) {
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

double best_gemv_seconds(const tge::cpu::GemvFn kernel,
                         const tge::cpu::GemvArgs& args, int threads,
                         int repeat) {
  const tge::cpu::GemvConfig config{threads};
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

int command_gemm_bench(int m, int n, int k, int repeat, int threads,
                       const char* kernel) {
  if (m <= 0 || n <= 0 || k <= 0 || repeat <= 0) {
    std::fprintf(stderr, "bench gemm: dimensions and repeat must be positive\n");
    return 2;
  }

  const int workers = tge::cpu::resolve_thread_count(threads);
  const tge::KernelSet& set = tge::kernels();

  // Ablation controls: "packed" is the shipped kernel; the other two exist only
  // to attribute its speed.
  tge::cpu::GemmFn chosen = set.gemm_f32;
  const char* chosen_name = set.gemm_name.c_str();
  if (std::strcmp(kernel, "unpacked") == 0) {
    chosen = &tge::cpu::gemm_avx2_unpacked;
    chosen_name = "unpacked";
  } else if (std::strcmp(kernel, "unblocked") == 0) {
    chosen = &tge::cpu::gemm_avx2_unblocked;
    chosen_name = "packed-noblock";
  }

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
  std::printf("  %-12s  %7s  %12s  %8s\n", "kernel", "threads", "time",
              "gflops");

  const double reference_seconds =
      best_gemm_seconds(&tge::ops::gemm_ref, reference_args, 1, repeat);
  report_row("reference", 1, reference_seconds, flops);

  if (optimized) {
    report_row(chosen_name, 1,
               best_gemm_seconds(chosen, optimized_args, 1, repeat), flops);
    if (workers > 1) {
      report_row(chosen_name, workers,
                 best_gemm_seconds(chosen, optimized_args, workers, repeat),
                 flops);
    }

    const float worst = tge::ops::max_abs_diff(
        c_ref.f32(), c_opt.f32(), static_cast<std::int64_t>(c_count));
    std::printf("  max |reference - %s| = %.3e\n", set.gemm_name.c_str(),
                static_cast<double>(worst));
    if (!(worst < 1e-2f)) {
      std::fprintf(stderr, "bench gemm: kernel disagrees with the reference\n");
      return 1;
    }
  }
  return 0;
}

int command_gemv_bench(int rows, int cols, int repeat, int threads) {
  if (rows <= 0 || cols <= 0 || repeat <= 0) {
    std::fprintf(stderr, "bench gemv: dimensions and repeat must be positive\n");
    return 2;
  }

  const int workers = tge::cpu::resolve_thread_count(threads);
  const tge::KernelSet& set = tge::kernels();

  const std::size_t matrix_count = static_cast<std::size_t>(rows) * cols;
  const double matrix_bytes =
      static_cast<double>(matrix_count) * sizeof(float);

  tge::memory::AlignedBuffer matrix(matrix_count * sizeof(float));
  tge::memory::AlignedBuffer vector(static_cast<std::size_t>(cols) *
                                    sizeof(float));
  tge::memory::AlignedBuffer out_ref(static_cast<std::size_t>(rows) *
                                     sizeof(float));
  tge::memory::AlignedBuffer out_opt(static_cast<std::size_t>(rows) *
                                     sizeof(float));

  fill_deterministic(matrix.f32(), matrix_count, 0xA5A5A5u);
  fill_deterministic(vector.f32(), static_cast<std::size_t>(cols), 0x5A5A5Au);

  tge::cpu::GemvArgs reference_args;
  reference_args.rows = rows;
  reference_args.cols = cols;
  reference_args.matrix = matrix.f32();
  reference_args.ldm = cols;
  reference_args.vector = vector.f32();
  reference_args.out = out_ref.f32();

  tge::cpu::GemvArgs optimized_args = reference_args;
  optimized_args.out = out_opt.f32();

  tge::ops::gemv_ref(reference_args, tge::cpu::GemvConfig{1});

  std::printf("gemv %dx%d (%.1f MiB of weights), best of %d\n", rows, cols,
              matrix_bytes / (1024.0 * 1024.0), repeat);
  std::printf("  %-12s  %7s  %12s  %8s  %8s\n", "kernel", "threads", "time",
              "gflops", "GB/s");

  double best_gbs = 0.0;
  const int thread_points[2] = {1, workers};
  for (int point = 0; point < 2; ++point) {
    const int used = thread_points[point];
    if (point == 1 && used == 1) {
      continue;
    }
    const double seconds =
        best_gemv_seconds(set.gemv_f32, optimized_args, used, repeat);
    const double gbs = matrix_bytes / seconds / 1e9;
    best_gbs = std::max(best_gbs, gbs);
    std::printf("  %-12s  %7d  %9.3f ms  %8.2f  %8.2f\n",
                set.gemv_name.c_str(), used, seconds * 1e3,
                (2.0 * static_cast<double>(rows) * cols) / seconds / 1e9, gbs);
  }

  const float worst = tge::ops::max_abs_diff(
      out_ref.f32(), out_opt.f32(), static_cast<std::int64_t>(rows));
  std::printf("  max |reference - %s| = %.3e\n", set.gemv_name.c_str(),
              static_cast<double>(worst));
  if (!(worst <= 1e-4F * static_cast<float>(cols) + 1e-3F)) {
    std::fprintf(stderr, "bench gemv: kernel disagrees with the reference\n");
    return 1;
  }

  // Decode reads every weight once per token, so the matrix scan time is the
  // per-token floor for a model of this size.
  const double tokens_per_second = (best_gbs * 1e9) / kGiB;
  std::printf("  %.2f GB/s sustained; a 1 GiB model would top out near %.2f "
              "tokens/s\n",
              best_gbs, tokens_per_second);
  return 0;
}

int command_mem_bench(int size_mb, int repeat, int threads) {
  if (size_mb <= 0 || repeat <= 0) {
    std::fprintf(stderr, "bench mem: size and repeat must be positive\n");
    return 2;
  }

  const int workers = tge::cpu::resolve_thread_count(threads);
  tge::core::ThreadPool& pool = tge::core::acquire_pool(workers);
  const tge::KernelSet& set = tge::kernels();

  const std::size_t count =
      static_cast<std::size_t>(size_mb) * 1024 * 1024 / sizeof(float);
  const std::size_t bytes = count * sizeof(float);

  tge::memory::AlignedBuffer source(bytes);
  tge::memory::AlignedBuffer target(bytes);
  fill_deterministic(source.f32(), count, 0xC0FFEEu);

  const int chunks = std::max(1, workers * 4);
  const std::size_t per_chunk = (count + chunks - 1) / chunks;

  double copy_best = 0.0;
  double read_best = 0.0;
  std::vector<float> partials(static_cast<std::size_t>(chunks), 0.0F);

  for (int iteration = 0; iteration < repeat; ++iteration) {
    auto start = Clock::now();
    pool.parallel_for(chunks, 1, [&](int chunk) {
      const std::size_t begin = static_cast<std::size_t>(chunk) * per_chunk;
      if (begin >= count) {
        return;
      }
      const std::size_t length = std::min(per_chunk, count - begin);
      set.copy_stream(target.f32() + begin, source.f32() + begin, length);
    });
    auto finish = Clock::now();
    const double copy_seconds =
        std::chrono::duration<double>(finish - start).count();
    if (copy_best == 0.0 || copy_seconds < copy_best) {
      copy_best = copy_seconds;
    }

    start = Clock::now();
    pool.parallel_for(chunks, 1, [&](int chunk) {
      const std::size_t begin = static_cast<std::size_t>(chunk) * per_chunk;
      if (begin >= count) {
        return;
      }
      const std::size_t length = std::min(per_chunk, count - begin);
      partials[static_cast<std::size_t>(chunk)] =
          set.sum_stream(source.f32() + begin, length);
    });
    finish = Clock::now();
    const double read_seconds =
        std::chrono::duration<double>(finish - start).count();
    if (read_best == 0.0 || read_seconds < read_best) {
      read_best = read_seconds;
    }
  }

  float sink = 0.0F;
  for (const float partial : partials) {
    sink += partial;
  }

  std::printf("memory bandwidth, %d MiB per buffer, %d threads, best of %d\n",
              size_mb, workers, repeat);
  std::printf("  %-18s  %8s\n", "operation", "GB/s");
  std::printf("  %-18s  %8.2f\n", "copy (read+write)",
              (2.0 * static_cast<double>(bytes)) / copy_best / 1e9);
  std::printf("  %-18s  %8.2f\n", "read (sum)",
              static_cast<double>(bytes) / read_best / 1e9);
  std::printf("  kernel %s, check value %.1f\n", set.stream_name.c_str(),
              static_cast<double>(sink));
  return 0;
}

int command_gguf(const char* path) {
  try {
    const tge::gguf::File file = tge::gguf::File::open(path);

    std::printf("gguf v%u, %llu bytes, alignment %llu\n", file.version(),
                static_cast<unsigned long long>(file.file_size()),
                static_cast<unsigned long long>(file.alignment()));

    std::printf("\nmetadata (%zu keys)\n", file.metadata().size());
    std::vector<std::string> keys;
    keys.reserve(file.metadata().size());
    for (const auto& entry : file.metadata()) {
      keys.push_back(entry.first);
    }
    std::sort(keys.begin(), keys.end());
    for (const std::string& key : keys) {
      const tge::gguf::Value* value = file.find(key);
      if (value == nullptr) {
        continue;
      }
      if (value->type == tge::gguf::ValueType::String) {
        std::string text = value->as_string();
        if (text.size() > 60) {
          text = text.substr(0, 57) + "...";
        }
        std::printf("  %-44s = \"%s\"\n", key.c_str(), text.c_str());
      } else if (value->type == tge::gguf::ValueType::Array) {
        std::printf("  %-44s = array[%zu]\n", key.c_str(),
                    value->elements.size());
      } else if (value->type == tge::gguf::ValueType::Float32 ||
                 value->type == tge::gguf::ValueType::Float64) {
        std::printf("  %-44s = %g\n", key.c_str(), value->as_f64());
      } else {
        std::printf("  %-44s = %llu\n", key.c_str(),
                    static_cast<unsigned long long>(value->as_u64()));
      }
    }

    std::printf("\ntensors (%zu)\n", file.tensors().size());
    std::uint64_t total_bytes = 0;
    for (const tge::gguf::TensorInfo& info : file.tensors()) {
      total_bytes += info.bytes;
      std::printf("  %-40s ", info.name.c_str());
      for (std::size_t axis = 0; axis < info.shape.size(); ++axis) {
        std::printf("%s%llu", axis == 0 ? "" : "x",
                    static_cast<unsigned long long>(info.shape[axis]));
      }
      std::printf("  %-5s %8.2f MiB\n", tge::gguf::type_name(info.type),
                  static_cast<double>(info.bytes) / (1024.0 * 1024.0));
    }
    std::printf("  total %.2f MiB\n",
                static_cast<double>(total_bytes) / (1024.0 * 1024.0));
  } catch (const std::exception& error) {
    std::fprintf(stderr, "gguf: %s\n", error.what());
    return 1;
  }
  return 0;
}

/// Whether the vocabulary is parsed during the model load.
///
/// Auto picks per invocation: eager is cheaper when a tokenizer will be built
/// (the arrays are parsed once instead of twice), skipped is cheaper when it
/// will not.
enum class VocabularyMode { Auto, Eager, Lazy };

bool prompt_lookup_draft(const std::vector<int>& history, int* draft, int need);
bool skip_locked_draft(const std::vector<int>& history, int* draft, int need);

int command_generate(const std::string& path, const std::vector<int>& prompt_ids,
                     const std::string& prompt_text, int count, int threads,
                     bool print_ids,
                     const tge::models::SamplingParams& sampling,
                     int context_limit, bool print_text,
                     VocabularyMode vocabulary,
                     tge::models::AttentionMode attention_mode,
                     bool speculative, bool vnni, bool kq8, bool kq8d_head,
                     bool kq8d_down, bool kv_half, bool kv_q8,
                     bool prefetch, int prefill_threads, bool pin_threads,
                     tge::models::SparseConfig sparse,
                     const std::string& gpu_mode, bool disable_fuse = false,
                     bool disable_skip = false) {
  if (count <= 0) {
    std::fprintf(stderr, "generate: token count must be positive\n");
    return 2;
  }
  const tge::cpu::CpuInfo cpu = tge::cpu::detect();
  if (vnni && !cpu.features.avx_vnni) {
    std::fprintf(stderr, "generate: --vnni needs AVX-VNNI on this CPU\n");
    return 2;
  }
  if (kv_half && !cpu.features.f16c) {
    std::fprintf(stderr, "generate: --kv f16 needs F16C on this CPU\n");
    return 2;
  }
  if (kv_half && kv_q8) {
    std::fprintf(stderr, "generate: --kv takes one of f32|f16|i8\n");
    return 2;
  }
  // `--gpu on` requires a ready backend; fail here, before the load, instead
  // of silently benchmarking the CPU.
  if (gpu_mode == "on" &&
      tge::gpu::init() != tge::gpu::GpuStatus::Ready) {
    std::fprintf(stderr, "generate: --gpu on needs a ready GPU backend (%s)\n",
                 tge::gpu::info().name);
    return 2;
  }

  std::string error;
  // Whether a tokenizer will be built decides the cheaper load strategy: eager
  // costs ~65 ms up front but the tokenizer then reuses the parsed arrays
  // instead of parsing the file a second time.
  const bool needs_tokenizer = !prompt_text.empty() || print_text;
  const bool eager = vocabulary == VocabularyMode::Eager ||
                     (vocabulary == VocabularyMode::Auto && needs_tokenizer);
  const auto load_start = Clock::now();
  std::unique_ptr<tge::models::Qwen2> model = tge::models::Qwen2::load(
      path, error, context_limit, eager, kv_half, kv_q8);
  const auto load_end = Clock::now();
  if (!model) {
    std::fprintf(stderr, "generate: %s\n", error.c_str());
    return 1;
  }
  std::printf("load     : %.1f ms (%s vocabulary)\n",
              std::chrono::duration<double>(load_end - load_start).count() * 1e3,
              eager ? "eager" : "skipped");
  if (prefetch) {
    const auto prefetch_start = Clock::now();
    model->file().prefetch();
    const auto prefetch_end = Clock::now();
    std::printf("prefetch : %.1f ms (weights faulted before first token)\n",
                std::chrono::duration<double>(prefetch_end - prefetch_start)
                        .count() *
                    1e3);
  }
  if (context_limit > 0 && model->max_context() < context_limit) {
    std::printf("context  : %d requested, %d available in memory\n",
                context_limit, model->max_context());
  }
  model->set_threads(threads);
  model->set_prefill_threads(prefill_threads);
  model->set_pin_threads(pin_threads);
  model->set_attention_mode(attention_mode);
  // The verifier and a sequential decode must take the same attention path, or a
  // near tie can flip an accepted draft. Speculative runs pin the per-head
  // schedule; normal generation keeps auto's grouped schedule at long context.
  if (speculative && attention_mode != tge::models::AttentionMode::Scalar) {
    if (attention_mode == tge::models::AttentionMode::Gqa) {
      std::fprintf(
          stderr,
          "generate: --speculative uses the per-head attention schedule\n");
    }
    model->set_attention_mode(tge::models::AttentionMode::Head);
  }
  model->set_vnni(vnni);
  model->set_kq8(kq8);
  model->set_kq8d(kq8d_head, kq8d_down);
  if (sparse.enabled) {
    if (kv_q8) {
      std::fprintf(stderr,
                   "generate: --sparse needs the f32|f16 KV cache (no --kv i8)\n");
      return 2;
    }
    try {
      model->set_sparse_config(sparse);
    } catch (const std::exception& error) {
      std::fprintf(stderr, "generate: %s\n", error.what());
      return 2;
    }
    std::printf("sparse   : sink %d, window %d, stride %d\n", sparse.sink,
                sparse.window, sparse.stride);
  }
  // GPU request: auto tries the backend and silently keeps the CPU kernels
  // when it is unavailable; off forces the proven path.
  const tge::gpu::GpuStatus gpu_status = tge::gpu::init();
  model->set_use_gpu(gpu_mode != "off");
  std::printf("gpu        : %s (%s)\n", gpu_mode.c_str(),
              gpu_status == tge::gpu::GpuStatus::Ready
                  ? "ready, CPU fallback armed"
                  : tge::gpu::info().name);

  const tge::models::Qwen2Config& config = model->config();

  // Reuse the model's arrays when they were loaded, otherwise parse them from a
  // second mapping of the same file.
  const auto build_tokenizer = [&](std::string& tokenizer_error) {
    return eager ? tge::models::Tokenizer::from_gguf(model->file(), tokenizer_error)
                 : tge::models::Tokenizer::from_path(path, tokenizer_error);
  };

  std::vector<int> tokens = prompt_ids;
  if (tokens.empty() && !prompt_text.empty()) {
    std::string tokenizer_error;
    const auto tokenizer_start = Clock::now();
    const std::unique_ptr<tge::models::Tokenizer> tokenizer =
        build_tokenizer(tokenizer_error);
    const auto tokenizer_end = Clock::now();
    if (!tokenizer) {
      std::fprintf(stderr, "generate: %s\n", tokenizer_error.c_str());
      return 1;
    }
    std::printf("tokenizer: %.1f ms\n",
                std::chrono::duration<double>(tokenizer_end - tokenizer_start)
                        .count() *
                    1e3);
    tokens = tokenizer->encode(prompt_text);
    std::printf("prompt   : %zu tokens from text\n", tokens.size());
    if (print_ids) {
      std::printf("promptids:");
      for (const int token : tokens) {
        std::printf(" %d", token);
      }
      std::printf("\n");
    }
  }
  const std::vector<int>& prompt = tokens;

  if (prompt.empty()) {
    std::fprintf(stderr, "generate: at least one prompt token is required\n");
    return 2;
  }
  if (static_cast<int>(prompt.size()) + count + 1 > model->max_context()) {
    std::fprintf(stderr,
                 "generate: %zu prompt tokens + %d generated exceeds the %d "
                 "token context this build allocates\n",
                 prompt.size(), count, model->max_context());
    return 2;
  }
  for (const int token : prompt) {
    if (token < 0 || token >= model->vocab_size()) {
      std::fprintf(stderr, "generate: token %d is outside the vocabulary\n",
                   token);
      return 2;
    }
  }

  std::printf("model    : %s, %d layers, %d dim, %d/%d heads, vocab %d\n",
              path.c_str(), config.block_count, config.embedding_length,
              config.head_count, config.head_count_kv, config.vocab_size);

  model->reset_phase_timers();
  const auto prefill_start = Clock::now();
  const float* logits = model->prefill(prompt.data(),
                                       static_cast<int>(prompt.size()), 0);
  const auto prefill_end = Clock::now();
  const double prefill_seconds =
      std::chrono::duration<double>(prefill_end - prefill_start).count();

  std::printf("prefill  : %zu tokens in %.2f ms (%.1f tok/s)\n", prompt.size(),
              prefill_seconds * 1e3,
              static_cast<double>(prompt.size()) / prefill_seconds);
  {
    const double mv_ms =
        static_cast<double>(model->phase_matvec_ns()) / 1e6;
    const double at_ms = static_cast<double>(model->phase_attn_ns()) / 1e6;
    const double nx_ms = static_cast<double>(model->phase_norm_ns()) / 1e6;
    std::printf("phases   : prefill matvec %.1f ms, attn %.1f ms, norm %.1f ms\n",
                mv_ms, at_ms, nx_ms);
    if (model->phase_gpu_bytes() > 0) {
      std::printf("gpu      : prefill offload %.1f ms, %llu bytes moved\n",
                  static_cast<double>(model->phase_gpu_ns()) / 1e6,
                  static_cast<unsigned long long>(
                      model->phase_gpu_bytes()));
    }
  }
  model->reset_phase_timers();

  std::vector<int> generated;
  std::vector<int> history = prompt;
  std::vector<float> scores;
  std::vector<int> order;
  std::uint32_t random_state = sampling.seed != 0 ? sampling.seed : 0x9E3779B9u;

  const int vocab_size = model->vocab_size();
  const auto draw = [&]() {
    const int token = tge::models::sample_token(
        logits, vocab_size, sampling, history.data(),
        static_cast<int>(history.size()), scores, order, random_state);
    history.push_back(token);
    return token;
  };

  const bool no_penalty = (sampling.repeat_penalty <= 0.0F ||
                           sampling.repeat_penalty == 1.0F);
  // Greedy speculation works with or without the repeat penalty: argmax is
  // deterministic, so verifying penalized argmaxes keeps output identical.
  // (Sampled paths handle the penalty separately below.)
  const bool speculate = speculative && sampling.temperature <= 0.0F;
  // Sampled speculation needs the full target distribution: temperature-only
  // sampling, no top-k/p or min-p cuts. Repeat penalty is allowed: it is
  // history-deterministic, so the accept rule stays exact, and each position
  // below is penalized with the history as of that position.
  const bool speculate_sampled =
      speculative && sampling.temperature > 0.0F &&
      sampling.top_k <= 0 && sampling.top_p >= 1.0F &&
      sampling.min_p <= 0.0F;
  int position = static_cast<int>(prompt.size());
  long long spec_steps = 0;
  long long spec_accepted = 0;
  long long spec_output = 0;

  const auto decode_start = Clock::now();
  double fwd_seconds = 0.0;
  double samp_seconds = 0.0;
  // Stop where the model stops: an end-of-sequence or end-of-turn id ends the
  // turn instead of being fed back and answered as if it were a new prompt.
  // Fused speculative steady state: after a full-accept greedy step, the bonus
  // id is known but its successor logits were never materialized (its KV row
  // is NOT yet resident -- the next verify writes it as feed[0] along with 5
  // history-drafts in ONE 6-row sweep and post-judges all rows). This deletes
  // the baseline's 1-row correction forward. Full accept chains 1 sweep/step;
  // partial accept falls back to a 1-row forward exactly like the baseline.
  // Output identical by construction (argmax accept rule, same history, same
  // positions). Measured: locked repetition wins ~1.3-1.5x over the baseline
  // greedy path on Qwen2.5-0.5B and Llama-3.2-3B; non-locked prompts never
  // chain and behave exactly like the baseline. --no-fuse forces the baseline
  // path for same-binary A/B (Appendix A3); TGE_NO_FUSE=1 is the env fallback.
  bool fused = false;
  while (static_cast<int>(generated.size()) < count &&
         (generated.empty() || !config.is_end_token(generated.back()))) {
    if (!disable_fuse && fused && speculate && position + tge::models::Qwen2::kSpeculativeBatch <=
                                 model->max_context()) {
      int draft5[5];
      if (!prompt_lookup_draft(history, draft5, 5)) {
        // No drafts: the bonus KV was never written, so one 1-row forward of
        // the last token writes it and recovers current logits, emit one
        // plain token exactly like the single-token path, then resume
        // baseline mode. Costs 2 rows for 1 token on this rare exit only.
        const auto f0 = Clock::now();
        logits = model->forward(history.back(), position - 1);
        const int first = no_penalty
                              ? tge::models::Qwen2::argmax(logits, vocab_size)
                              : [&] {
                                  if (static_cast<int>(scores.size()) <
                                      vocab_size) {
                                    scores.assign(
                                        static_cast<std::size_t>(vocab_size),
                                        0.0F);
                                  }
                                  std::memcpy(
                                      scores.data(), logits,
                                      sizeof(float) *
                                          static_cast<std::size_t>(vocab_size));
                                  tge::models::apply_repeat_penalty(
                                      scores, vocab_size, sampling,
                                      history.data(),
                                      static_cast<int>(history.size()));
                                  return tge::models::Qwen2::argmax(
                                      scores.data(), vocab_size);
                                }();
        generated.push_back(first);
        history.push_back(first);
        if (!config.is_end_token(first) &&
            static_cast<int>(generated.size()) < count) {
          logits = model->forward(first, position);
        }
        const auto f1 = Clock::now();
        fwd_seconds += std::chrono::duration<double>(f1 - f0).count();
        position += 1;
        fused = false;
        if (config.is_end_token(first) ||
            static_cast<int>(generated.size()) >= count) {
          break;
        }
        continue;
      } else {
        int feed[tge::models::Qwen2::kSpeculativeBatch];
        feed[0] = history.back();
        for (int index = 0; index < 5; ++index) {
          feed[index + 1] = draft5[index];
        }
        int predicted[tge::models::Qwen2::kSpeculativeBatch];
        const auto v0 = Clock::now();
        // feed[0] (the bonus id) is written as a new KV row at vpos by this
        // verify along with the 5 drafts -- it was never resident, which is
        // exactly the correction forward this path deletes. One pass total.
        const int vpos = position - 1;
        if (no_penalty) {
          model->verify_draft(feed, tge::models::Qwen2::kSpeculativeBatch,
                              vpos, predicted);
        } else {
          const float* rows = model->verify_draft_logits(
              feed, tge::models::Qwen2::kSpeculativeBatch, vpos);
          if (static_cast<int>(scores.size()) < vocab_size) {
            scores.assign(static_cast<std::size_t>(vocab_size), 0.0F);
          }
          // Row i is penalized with the history as of feed[i]: history
          // already holds feed[0], so push feed[1..i] before each row.
          std::vector<int> widened = history;
          widened.reserve(history.size() + 6);
          for (int index = 0; index < 6; ++index) {
            if (index >= 1) {
              widened.push_back(feed[index]);
            }
            std::memcpy(scores.data(),
                        rows + static_cast<std::size_t>(index) * vocab_size,
                        sizeof(float) * static_cast<std::size_t>(vocab_size));
            tge::models::apply_repeat_penalty(
                scores, vocab_size, sampling, widened.data(),
                static_cast<int>(widened.size()));
            predicted[index] =
                tge::models::Qwen2::argmax(scores.data(), vocab_size);
          }
        }
        const auto v1 = Clock::now();
        fwd_seconds += std::chrono::duration<double>(v1 - v0).count();

        int accepted = 0;
        while (accepted < 5 && predicted[accepted] == feed[accepted + 1]) {
          ++accepted;
        }
        const int corrected = predicted[accepted];

        bool stopped = false;
        for (int index = 1; index <= accepted; ++index) {
          if (static_cast<int>(generated.size()) >= count) {
            stopped = true;
            break;
          }
          generated.push_back(feed[index]);
          history.push_back(feed[index]);
          if (config.is_end_token(feed[index])) {
            stopped = true;
            break;
          }
        }
        if (!stopped && static_cast<int>(generated.size()) < count) {
          generated.push_back(corrected);
          history.push_back(corrected);
          stopped = config.is_end_token(corrected);
        }

        ++spec_steps;
        spec_accepted += accepted;
        spec_output += accepted + (stopped ? 0 : 1);
        if (stopped || static_cast<int>(generated.size()) >= count) {
          break;
        }
        if (accepted == 5) {
          // Full accept: 5 drafts + bonus emitted, bonus at vpos + 6. Next
          // un-generated position is vpos + 7; next fused feed[0] (= bonus)
          // re-verifies at position - 1. Chain fused.
          position = vpos + 7;
          continue;
        }
        // Partial: correction forward exactly like the baseline. Corrected
        // sits at vpos + accepted + 1.
        const auto f0 = Clock::now();
        logits = model->forward(corrected, vpos + accepted + 1);
        const auto f1 = Clock::now();
        fwd_seconds += std::chrono::duration<double>(f1 - f0).count();
        position = vpos + accepted + 2;
        fused = false;
        continue;
      }
    }
    if (speculate && position + tge::models::Qwen2::kSpeculativeBatch <=
                         model->max_context()) {
      // Six-deep greedy draft: the verifier's whole batch carries drafts,
      // the padding slot is gone, and the five-deep form stays as the
      // fallback inside prompt_lookup_draft for short histories.
      int draft[6];
      bool first_ok = false;
      // Locked-repetition fast path first: period-1/period-2 cycles propose
      // along the cycle (verifier still decides, d0 pre-check unchanged);
      // falls back to the shipped span/bigram/gapped chain. Measured: locked
      // repetition accepts at the 6.00/step ceiling with identical ids;
      // non-locked prompts abstain and behave exactly like before.
      bool used_skip = false;
      if (!disable_skip && skip_locked_draft(history, draft, 6)) {
        used_skip = true;
        if (no_penalty) {
          first_ok =
              tge::models::Qwen2::argmax(logits, vocab_size) == draft[0];
        } else {
          if (static_cast<int>(scores.size()) < vocab_size) {
            scores.assign(static_cast<std::size_t>(vocab_size), 0.0F);
          }
          std::memcpy(scores.data(), logits,
                      sizeof(float) * static_cast<std::size_t>(vocab_size));
          tge::models::apply_repeat_penalty(
              scores, vocab_size, sampling, history.data(),
              static_cast<int>(history.size()));
          first_ok =
              tge::models::Qwen2::argmax(scores.data(), vocab_size) ==
              draft[0];
        }
      }
      if (!used_skip && prompt_lookup_draft(history, draft, 6)) {
        if (no_penalty) {
          first_ok =
              tge::models::Qwen2::argmax(logits, vocab_size) == draft[0];
        } else {
          // Penalized d0 check with the current history. argmax needs no
          // softmax (monotone), so this is copy + penalty + scan.
          if (static_cast<int>(scores.size()) < vocab_size) {
            scores.assign(static_cast<std::size_t>(vocab_size), 0.0F);
          }
          std::memcpy(scores.data(), logits,
                      sizeof(float) * static_cast<std::size_t>(vocab_size));
          tge::models::apply_repeat_penalty(
              scores, vocab_size, sampling, history.data(),
              static_cast<int>(history.size()));
          first_ok =
              tge::models::Qwen2::argmax(scores.data(), vocab_size) ==
              draft[0];
        }
      }
      if (first_ok) {
        int feed[tge::models::Qwen2::kSpeculativeBatch];
        for (int index = 0; index < 6; ++index) {
          feed[index] = draft[index];
        }
        int predicted[tge::models::Qwen2::kSpeculativeBatch];
        const auto v0 = Clock::now();
        if (no_penalty) {
          model->verify_draft(feed, tge::models::Qwen2::kSpeculativeBatch,
                              position, predicted);
        } else {
          // Penalized rows: row i predicts draft[i + 1], and the penalty
          // for that decision sees history + feed[0..i], exactly as
          // single-token decoding would. Widened history is built
          // incrementally; the single-token path would see the same window.
          const float* rows = model->verify_draft_logits(
              feed, tge::models::Qwen2::kSpeculativeBatch, position);
          if (static_cast<int>(scores.size()) < vocab_size) {
            scores.assign(static_cast<std::size_t>(vocab_size), 0.0F);
          }
          // Penalized rows: row i predicts draft[i + 1] with the history as
          // of that position (history + feed[0..i]), and row 5 predicts the
          // bonus token after the sixth draft. All six rows are penalized,
          // so the bonus keeps the exact single-token penalty semantics.
          std::vector<int> widened = history;
          widened.reserve(history.size() + 6);
          for (int index = 0; index < 6; ++index) {
            widened.push_back(feed[index]);
            std::memcpy(scores.data(),
                        rows +
                            static_cast<std::size_t>(index) * vocab_size,
                        sizeof(float) * static_cast<std::size_t>(vocab_size));
            tge::models::apply_repeat_penalty(
                scores, vocab_size, sampling, widened.data(),
                static_cast<int>(widened.size()));
            predicted[index] = tge::models::Qwen2::argmax(scores.data(),
                                                          vocab_size);
          }
        }
        const auto v1 = Clock::now();
        fwd_seconds += std::chrono::duration<double>(v1 - v0).count();

        int accepted = 1;
        while (accepted < 6 && predicted[accepted - 1] == draft[accepted]) {
          ++accepted;
        }
        const int corrected = predicted[accepted - 1];

        bool stopped = false;
        for (int index = 0; index < accepted; ++index) {
          if (static_cast<int>(generated.size()) >= count) {
            stopped = true;
            break;
          }
          generated.push_back(draft[index]);
          history.push_back(draft[index]);
          if (config.is_end_token(draft[index])) {
            stopped = true;
            break;
          }
        }
        if (!stopped && static_cast<int>(generated.size()) < count) {
          generated.push_back(corrected);
          history.push_back(corrected);
          stopped = config.is_end_token(corrected);
        }

        ++spec_steps;
        spec_accepted += accepted;
        spec_output += accepted + (stopped ? 0 : 1);
        if (!stopped && static_cast<int>(generated.size()) < count) {
          if (accepted == 6 && !disable_fuse) {
            // Full accept, bonus id known (KV not yet written): skip the 1-row
            // correction forward and chain into the fused steady state. Next
            // first (= bonus) is a known argmax, no logits needed.
            position += accepted + 1;
            fused = true;
            continue;
          }
          // Baseline (including --no-fuse full accepts): the 1-row correction
          // forward materializes the bonus logits for the next d0 pre-check.
          // Skipping it here would leave stale logits and diverge ids, which
          // is exactly what the fused path avoids by post-judging instead.
          const auto f0 = Clock::now();
          logits = model->forward(corrected, position + accepted);
          const auto f1 = Clock::now();
          fwd_seconds += std::chrono::duration<double>(f1 - f0).count();
          position += accepted + 1;
          fused = false;
          continue;
        }
        break;
      }
    }

    // Sampled speculation: same verifier, but each draft position goes
    // through the accept/reject rule against the full temperature-scaled
    // distribution, so the emitted tokens follow exactly the distribution
    // plain sampling would draw. No argmax pre-check: under sampling a
    // draft that differs from the argmax is still worth verifying.
    //
    // Positions: the current `logits` cover draft[0]; verify row i covers
    // draft[i + 1] (row i is the distribution after feed[i]), and row 4
    // feeds the bonus token. Draft[0] is judged before verifying, so a
    // first-position rejection skips the weight pass entirely.
    if (speculate_sampled && position + tge::models::Qwen2::kSpeculativeBatch <=
                                 model->max_context()) {
      // Sampled path keeps the historical five-deep draft shape: proposals
      // and verify rows are byte-identical to before this change, so the
      // distribution-preserving accept loop is untouched.
      int draft[5];
      if (prompt_lookup_draft(history, draft, 5)) {
        if (static_cast<int>(scores.size()) < vocab_size) {
          scores.assign(static_cast<std::size_t>(vocab_size), 0.0F);
        }
        // Penalized softmax of one row with the history as of this position:
        // the copy keeps the model's row intact, the penalty makes the
        // target exactly what plain sampling would draw, and the in-place
        // softmax is safe (each lane reads then writes its own element).
        auto penalized_softmax = [&](const float* row) {
          std::memcpy(scores.data(), row,
                      sizeof(float) * static_cast<std::size_t>(vocab_size));
          tge::models::apply_repeat_penalty(
              scores, vocab_size, sampling, history.data(),
              static_cast<int>(history.size()));
          tge::models::softmax_probabilities(scores.data(), vocab_size,
                                             sampling.temperature,
                                             scores.data());
        };
        const auto st0 = Clock::now();
        penalized_softmax(logits);
        tge::models::SpeculativeVerdict first =
            tge::models::speculative_accept(scores.data(), vocab_size,
                                            draft[0], random_state);
        int emitted = 0;
        int accepted_drafts = 0;
        bool stopped = false;
        if (static_cast<int>(generated.size()) >= count) {
          stopped = true;
        } else {
          generated.push_back(first.token);
          history.push_back(first.token);
          ++emitted;
          if (!first.accepted) {
            stopped = config.is_end_token(first.token);
          } else {
            ++accepted_drafts;
            stopped = config.is_end_token(first.token);
          }
        }

        if (!stopped && first.accepted &&
            static_cast<int>(generated.size()) < count) {
          int feed[tge::models::Qwen2::kSpeculativeBatch];
          for (int index = 0; index < 5; ++index) {
            feed[index] = draft[index];
          }
          feed[5] = 0;
          const auto v0 = Clock::now();
          const float* verify_logits = model->verify_draft_logits(
              feed, tge::models::Qwen2::kSpeculativeBatch, position);
          const auto v1 = Clock::now();
          fwd_seconds += std::chrono::duration<double>(v1 - v0).count();

          bool done = false;
          for (int index = 1; index < 5 && !done; ++index) {
            if (static_cast<int>(generated.size()) >= count) {
              stopped = true;
              break;
            }
            penalized_softmax(verify_logits +
                                  static_cast<std::size_t>(index - 1) *
                                      vocab_size);
            const tge::models::SpeculativeVerdict verdict =
                tge::models::speculative_accept(scores.data(), vocab_size,
                                                draft[index], random_state);
            generated.push_back(verdict.token);
            history.push_back(verdict.token);
            ++emitted;
            if (!verdict.accepted) {
              done = true;
            } else {
              ++accepted_drafts;
              if (config.is_end_token(verdict.token)) {
                stopped = true;
                done = true;
              }
            }
          }
          if (!done && !stopped &&
              static_cast<int>(generated.size()) < count) {
            penalized_softmax(verify_logits +
                              static_cast<std::size_t>(4) * vocab_size);
            const int bonus = tge::models::categorical_sample(
                scores.data(), vocab_size, random_state);
            generated.push_back(bonus);
            history.push_back(bonus);
            ++emitted;
            if (config.is_end_token(bonus)) {
              stopped = true;
            }
          }
        }
        const auto st1 = Clock::now();
        samp_seconds += std::chrono::duration<double>(st1 - st0).count();

        ++spec_steps;
        spec_accepted += accepted_drafts;
        spec_output += emitted;
        if (!stopped && emitted > 0 &&
            static_cast<int>(generated.size()) < count) {
          const auto f0 = Clock::now();
          logits =
              model->forward(generated.back(), position + emitted - 1);
          const auto f1 = Clock::now();
          fwd_seconds += std::chrono::duration<double>(f1 - f0).count();
          position += emitted;
          continue;
        }
        break;
      }
    }

    const auto s0 = Clock::now();
    const int token = draw();
    const auto s1 = Clock::now();
    samp_seconds += std::chrono::duration<double>(s1 - s0).count();
    generated.push_back(token);
    if (config.is_end_token(token) ||
        static_cast<int>(generated.size()) >= count) {
      break;
    }
    const auto f0 = Clock::now();
    logits = model->forward(token, position++);
    const auto f1 = Clock::now();
    fwd_seconds += std::chrono::duration<double>(f1 - f0).count();
  }
  const auto decode_end = Clock::now();

  if (config.is_end_token(generated.back())) {
    std::printf("stop     : end token %d after %zu tokens\n", generated.back(),
                generated.size());
  }
  const double decode_seconds =
      std::chrono::duration<double>(decode_end - decode_start).count();

  const int timed_tokens = static_cast<int>(generated.size()) - 1;
  if (timed_tokens > 0 && decode_seconds > 0.0) {
    std::printf("decode   : %d tokens in %.2f ms (%.2f tok/s)\n", timed_tokens,
                decode_seconds * 1e3, timed_tokens / decode_seconds);
    // Phase-0 breakdown: weight passes vs sampling. Attention/projection
    // split lives inside forward(); this names the sampling share so a
    // top-p/min-p regression shows up here first.
    const double fwd_ms = fwd_seconds * 1e3;
    const double samp_ms = samp_seconds * 1e3;
    const double samp_share =
        decode_seconds > 0.0 ? (samp_seconds / decode_seconds * 100.0) : 0.0;
    std::printf("timers   : forward %.2f ms, sample %.2f ms (%.1f%%)\n", fwd_ms,
                samp_ms, samp_share);
    const double mv_ms =
        static_cast<double>(model->phase_matvec_ns()) / 1e6;
    const double at_ms = static_cast<double>(model->phase_attn_ns()) / 1e6;
    const double nx_ms = static_cast<double>(model->phase_norm_ns()) / 1e6;
    const double ph_total = mv_ms + at_ms + nx_ms;
    std::printf("phases   : decode matvec %.1f ms (%.0f%%), attn %.1f ms "
                "(%.0f%%), norm %.1f ms (%.0f%%)\n",
                mv_ms, ph_total > 0.0 ? mv_ms / ph_total * 100.0 : 0.0, at_ms,
                ph_total > 0.0 ? at_ms / ph_total * 100.0 : 0.0, nx_ms,
                ph_total > 0.0 ? nx_ms / ph_total * 100.0 : 0.0);
    if (model->phase_gpu_bytes() > 0) {
      std::printf("gpu      : output offload %.1f ms, %llu bytes moved\n",
                  static_cast<double>(model->phase_gpu_ns()) / 1e6,
                  static_cast<unsigned long long>(
                      model->phase_gpu_bytes()));
    }
  }
  if (spec_steps > 0) {
    std::printf(
        "spec     : %lld steps, %lld draft tokens accepted (%.2f/step), %lld "
        "tokens emitted\n",
        spec_steps, spec_accepted,
        static_cast<double>(spec_accepted) / spec_steps, spec_output);
  }

  if (print_ids) {
    std::printf("ids      :");
    for (const int token : generated) {
      std::printf(" %d", token);
    }
    std::printf("\n");
  }

  if (print_text) {
    std::string tokenizer_error;
    std::unique_ptr<tge::models::Tokenizer> tokenizer =
        build_tokenizer(tokenizer_error);
    if (!tokenizer) {
      std::fprintf(stderr, "generate: %s\n", tokenizer_error.c_str());
    } else {
      std::printf("text     : %s\n", tokenizer->decode(generated).c_str());
    }
  }
  return 0;
}

int command_batch_bench(int rows, int cols, int tokens, int repeat,
                        bool use_vnni, bool use_kq8, bool use_kq8d,
                        int k_format) {
  if (rows <= 0 || cols <= 0 || tokens <= 0 || repeat <= 0) {
    std::fprintf(stderr, "bench batch: all dimensions must be positive\n");
    return 2;
  }
  if (cols % 32 != 0) {
    std::fprintf(stderr, "bench batch: cols must be a multiple of 32\n");
    return 2;
  }
  // k_format: 0 = Q8_0, 40 = Q4_0, 4 = Q4_K, 5 = Q5_K, 6 = Q6_K,
  // 50 = Q5_0, 51 = Q5_1.
  // Bench notes: inputs come from
  // AlignedBuffer (32-byte aligned), so kernels measured here never see the
  // 16-byte operator-new alignment of model buffers — see §28i.
  const bool use_k = k_format == 40 || k_format == 4 || k_format == 5 ||
                     k_format == 6 || k_format == 50 || k_format == 51;
  if ((k_format == 4 || k_format == 5 || k_format == 6) && cols % 256 != 0) {
    std::fprintf(stderr, "bench batch: Q4_K/Q5_K/Q6_K need cols to be a multiple of 256\n");
    return 2;
  }
  if ((k_format == 50 || k_format == 51) && cols % 32 != 0) {
    std::fprintf(stderr, "bench batch: Q5_0/Q5_1 need cols to be a multiple of 32\n");
    return 2;
  }

  const std::size_t row_bytes =
      k_format == 4 ? static_cast<std::size_t>(cols / 256) * 144
      : k_format == 5 ? static_cast<std::size_t>(cols / 256) * 176
      : k_format == 6 ? static_cast<std::size_t>(cols / 256) * 210
      : k_format == 40 ? static_cast<std::size_t>(cols / 32) * 18
      : k_format == 50 ? static_cast<std::size_t>(cols / 32) * 22
      : k_format == 51 ? static_cast<std::size_t>(cols / 32) * 24
                       : static_cast<std::size_t>(cols / 32) * 34;
  const std::size_t matrix_bytes =
      static_cast<std::size_t>(rows) * row_bytes;
  tge::memory::AlignedBuffer weights(matrix_bytes);
  std::uint32_t state = 0x51ED270Bu;
  for (std::size_t index = 0; index < matrix_bytes; ++index) {
    state = state * 1664525u + 1013904223u;
    static_cast<std::uint8_t*>(weights.data())[index] =
        static_cast<std::uint8_t>(state >> 24);
  }

  const std::size_t input_elements = static_cast<std::size_t>(tokens) * cols;
  const std::size_t output_elements = static_cast<std::size_t>(tokens) * rows;
  tge::memory::AlignedBuffer input(input_elements * sizeof(float));
  tge::memory::AlignedBuffer output(output_elements * sizeof(float));
  fill_deterministic(input.f32(), input_elements, 0xABCDEF01u);

  tge::cpu::Q8BatchArgs args;
  args.data = static_cast<const std::uint8_t*>(weights.data());
  args.rows = rows;
  args.cols = cols;
  args.input = input.f32();
  args.tokens = tokens;
  args.out = output.f32();

  const tge::cpu::GemvConfig config{0};
  const tge::KernelSet& set = tge::kernels();
  const double flops = 2.0 * static_cast<double>(rows) * cols * tokens;
  double best = 0.0;

  if (use_k) {
    tge::cpu::DequantBatchArgs args;
    args.data = static_cast<const std::uint8_t*>(weights.data());
    args.kind = k_format == 5   ? tge::cpu::QuantKind::Q5_K
                : k_format == 6 ? tge::cpu::QuantKind::Q6_K
                : k_format == 40 ? tge::cpu::QuantKind::Q4_0
                : k_format == 50 ? tge::cpu::QuantKind::Q5_0
                : k_format == 51 ? tge::cpu::QuantKind::Q5_1
                                 : tge::cpu::QuantKind::Q4_K;
    args.rows = rows;
    args.cols = cols;
    args.input = input.f32();
    args.tokens = tokens;
    args.out = output.f32();
    const tge::cpu::DequantBatchFn kernel =
        use_kq8d ? &tge::cpu::gemv_dequant_kq8_row_avx2
        : use_kq8 ? &tge::cpu::gemv_dequant_batch_kq8_avx2
                  : set.gemv_dequant_batch;
    for (int iteration = 0; iteration < repeat; ++iteration) {
      const auto start = Clock::now();
      kernel(args, config);
      const auto finish = Clock::now();
      const double seconds =
          std::chrono::duration<double>(finish - start).count();
      if (best == 0.0 || seconds < best) {
        best = seconds;
      }
    }
    std::printf("q%d%s batch %dx%d x %d tokens (%.1f MiB), best of %d\n",
                k_format == 50 || k_format == 51 ? 5
                : k_format == 40                 ? 4
                                                 : k_format,
                k_format == 40   ? "_0"
                : k_format == 50 ? "_0"
                : k_format == 51 ? "_1"
                                 : "_k",
                rows, cols, tokens,
                static_cast<double>(matrix_bytes) / (1024.0 * 1024.0), repeat);
    if (use_kq8) {
      std::printf("  kernel kq8 (integer Q4_K/Q6_K x Q8_K)\n");
    } else if (use_kq8d) {
      std::printf("  kernel kq8d (integer Q4_K/Q6_K x Q8_K, single row)\n");
    }
    std::printf("  %.3f ms  %.2f GFLOPS  %.2f GB/s\n", best * 1e3,
                flops / best / 1e9,
                static_cast<double>(matrix_bytes) / best / 1e9);
    return 0;
  }

  const tge::cpu::Q8BatchFn kernel =
      use_vnni ? &tge::cpu::gemv_q8_batch_vnni : set.gemv_q8_batch;
  best = 0.0;
  for (int iteration = 0; iteration < repeat; ++iteration) {
    const auto start = Clock::now();
    kernel(args, config);
    const auto finish = Clock::now();
    const double seconds = std::chrono::duration<double>(finish - start).count();
    if (best == 0.0 || seconds < best) {
      best = seconds;
    }
  }

  std::printf("q8 batch %dx%d x %d tokens (%.1f MiB of weights), %s, best of %d\n",
              rows, cols, tokens,
              static_cast<double>(matrix_bytes) / (1024.0 * 1024.0),
              use_vnni ? "vnni" : "float", repeat);
  std::printf("  %.3f ms  %.2f GFLOPS  %.2f GB/s\n", best * 1e3,
              flops / best / 1e9,
              static_cast<double>(matrix_bytes) / best / 1e9);
  return 0;
}

int command_plan(const std::string& path, int context, int sessions,
                 double bandwidth_gbs) {
  if (context <= 0 || sessions <= 0 || bandwidth_gbs <= 0.0) {
    std::fprintf(stderr,
                 "plan: --context, --sessions and --bandwidth must be positive\n");
    return 2;
  }
  std::string error;
  try {
    const auto start = Clock::now();
    // Metadata only: the tensor table states every size, so the whole budget is
    // known without reading weight data or committing memory.
    const tge::gguf::File file = tge::gguf::File::open(path, false);
    const tge::models::ModelPlan plan = tge::models::analyse(file, context);
    const auto finish = Clock::now();

    const tge::memory::MemoryBudget budget;
    const tge::memory::BudgetSnapshot snapshot = budget.snapshot();
    const tge::models::PlacementDecision decision = tge::models::place(
        plan, static_cast<std::int64_t>(snapshot.usable), plan.context, sessions);

    std::printf("plan     : analysed in %.2f ms\n",
                std::chrono::duration<double>(finish - start).count() * 1e3);
    std::printf("MODEL\n%s", tge::models::describe(plan).c_str());
    std::printf("\nPLACEMENT\n%s", tge::models::describe(decision).c_str());

    const double ceiling =
        tge::models::projected_tokens_per_second(plan, bandwidth_gbs * 1e9);
    std::printf("\nPROJECTION\n");
    std::printf("  decode ceiling   : %.1f t/s at %.0f GB/s sustained\n", ceiling,
                bandwidth_gbs);
    std::printf("  (bandwidth is the measured threaded read figure; a real "
                "decode also pays attention and sampling)\n");
  } catch (const std::exception& exception) {
    std::fprintf(stderr, "plan: %s\n", exception.what());
    return 1;
  }
  return 0;
}

int command_continuous_bench(const std::string& path, int requests, int tokens,
                             int threads, int sequences, int context,
                             int prompt_tokens, bool vnni, bool kv_half,
                             bool kv_q8, int prefill_threads) {
  if (requests <= 0 || tokens <= 0 || sequences <= 0 || prompt_tokens <= 0) {
    std::fprintf(stderr,
                 "bench continuous: requests, tokens, sequences and "
                 "prompt-tokens must be positive\n");
    return 2;
  }
  if (sequences > 64 || prompt_tokens + sequences > 64) {
    std::fprintf(stderr,
                 "bench continuous: prompt-tokens + sequences must fit the 64 "
                 "prefill rows\n");
    return 2;
  }
  const tge::cpu::CpuInfo cpu = tge::cpu::detect();
  if (vnni && !cpu.features.avx_vnni) {
    std::fprintf(stderr, "bench continuous: --vnni needs AVX-VNNI\n");
    return 2;
  }
  if (kv_half && !cpu.features.f16c) {
    std::fprintf(stderr, "bench continuous: --kv f16 needs F16C\n");
    return 2;
  }
  if (kv_half && kv_q8) {
    std::fprintf(stderr, "bench continuous: --kv takes one of f32|f16|i8\n");
    return 2;
  }
  std::string error;
  const auto load_start = Clock::now();
  std::unique_ptr<tge::models::Qwen2> model = tge::models::Qwen2::load(
      path, error, context, false, kv_half, kv_q8, sequences);
  if (!model) {
    std::fprintf(stderr, "bench continuous: %s\n", error.c_str());
    return 1;
  }
  model->set_threads(threads);
  model->set_prefill_threads(prefill_threads);
  model->set_vnni(vnni);
  const auto load_end = Clock::now();

  // Every request uses the same seed prompt, so identical transcripts are the
  // independence check for the per-slot KV cache.
  const int seed_token = 128;
  const int vocab = model->vocab_size();
  const std::vector<int> prompt(
      static_cast<std::size_t>(prompt_tokens), seed_token);

  struct Slot {
    bool active = false;
    int position = 0;
    std::vector<int> generated;
  };
  std::vector<Slot> slots(static_cast<std::size_t>(sequences));
  int admitted = 0;
  int finished = 0;
  long long decode_tokens = 0;
  std::vector<int> batch_tokens;
  std::vector<int> batch_positions;
  std::vector<int> batch_sequences;
  batch_tokens.reserve(static_cast<std::size_t>(sequences));
  batch_positions.reserve(static_cast<std::size_t>(sequences));
  batch_sequences.reserve(static_cast<std::size_t>(sequences));
  std::vector<std::vector<int>> transcripts(static_cast<std::size_t>(requests));

  const auto start = Clock::now();
  while (finished < requests) {
    batch_tokens.clear();
    batch_positions.clear();
    batch_sequences.clear();

    // The active decodes and one newly admitted request's prompt share a pass.
    for (int slot = 0; slot < sequences; ++slot) {
      const Slot& entry = slots[static_cast<std::size_t>(slot)];
      if (!entry.active) {
        continue;
      }
      batch_tokens.push_back(entry.generated.back());
      batch_positions.push_back(entry.position);
      batch_sequences.push_back(slot);
    }
    const int active_count = static_cast<int>(batch_tokens.size());

    int new_slot = -1;
    if (admitted < requests) {
      for (int slot = 0; slot < sequences; ++slot) {
        if (!slots[static_cast<std::size_t>(slot)].active) {
          new_slot = slot;
          break;
        }
      }
    }
    // Stage-2 shared-prefix skip: when the full admission prompt is already
    // cached, prime its K/V rows and admit only the trailing row (whose
    // logits produce the first generated token). Earlier rows' logits are
    // never read, and decode rows attend their own sequences only, so the
    // shortened batch is bit-identical for every row the caller reads.
    int prompt_skip = 0;
    if (new_slot >= 0 && prompt_tokens >= 16 &&
        model->shared_prefix_len(prompt.data(), prompt_tokens) >=
            prompt_tokens &&
        model->prime_cache_from_table(new_slot, prompt.data(),
                                      prompt_tokens)) {
      prompt_skip = prompt_tokens - 1;
    }
    if (new_slot >= 0) {
      for (int index = prompt_skip; index < prompt_tokens; ++index) {
        batch_tokens.push_back(prompt[static_cast<std::size_t>(index)]);
        batch_positions.push_back(index);
        batch_sequences.push_back(new_slot);
      }
    }
    if (batch_tokens.empty()) {
      break;
    }

    const float* logits = model->forward_batch(
        batch_tokens.data(), batch_positions.data(),
        static_cast<int>(batch_tokens.size()), batch_sequences.data());
    for (int index = 0; index < active_count; ++index) {
      const int next = tge::models::Qwen2::argmax(
          logits + static_cast<std::size_t>(index) * static_cast<std::size_t>(vocab),
          vocab);
      const int slot = batch_sequences[static_cast<std::size_t>(index)];
      Slot& entry = slots[static_cast<std::size_t>(slot)];
      entry.generated.push_back(next);
      ++entry.position;
      ++decode_tokens;
      if (static_cast<int>(entry.generated.size()) >= tokens ||
          model->config().is_end_token(next)) {
        transcripts[static_cast<std::size_t>(finished)] = entry.generated;
        ++finished;
        entry.active = false;
      }
    }
    if (new_slot >= 0) {
      const int row =
          active_count + (prompt_tokens - prompt_skip) - 1;
      const int first = tge::models::Qwen2::argmax(
          logits + static_cast<std::size_t>(row) * static_cast<std::size_t>(vocab),
          vocab);
      Slot& entry = slots[static_cast<std::size_t>(new_slot)];
      entry.active = true;
      entry.position = prompt_tokens;
      entry.generated.assign(1, first);
      ++admitted;
    }
  }
  const auto finish = Clock::now();

  const double seconds =
      std::chrono::duration<double>(finish - start).count();
  const double load_seconds =
      std::chrono::duration<double>(load_end - load_start).count();
  bool identical = true;
  for (int index = 1; index < requests; ++index) {
    if (transcripts[static_cast<std::size_t>(index)] != transcripts[0]) {
      identical = false;
    }
  }
  const double per_second =
      seconds > 0.0 ? static_cast<double>(decode_tokens) / seconds : 0.0;

  std::printf("continuous: %d requests, %d slots, %d tokens\n", requests,
              sequences, tokens);
  std::printf("load      : %.1f ms\n", load_seconds * 1e3);
  std::printf("wall      : %.3f s\n", seconds);
  std::printf("decode    : %lld tokens, %.2f t/s aggregate\n", decode_tokens,
              per_second);
  std::printf("state     : %s\n",
              identical ? "independent (all transcripts identical)"
                        : "LEAKED - transcripts differ");
  std::printf("first     :");
  for (const int id : transcripts[0]) {
    std::printf(" %d", id);
  }
  std::printf("\n");
  return identical ? 0 : 1;
}

int command_sessions_bench(const std::string& path, int sessions, int tokens,
                           int threads_per_session, int context,
                           bool pin_threads, bool kv_half, bool kv_q8) {
  if (sessions <= 0 || tokens <= 0) {
    std::fprintf(stderr,
                 "bench sessions: sessions and tokens must be positive\n");
    return 2;
  }
  if (kv_half && kv_q8) {
    std::fprintf(stderr, "bench sessions: --kv takes one of f32|f16|i8\n");
    return 2;
  }
  if (kv_half && !tge::cpu::detect().features.f16c) {
    std::fprintf(stderr, "bench sessions: --kv f16 needs F16C on this CPU\n");
    return 2;
  }

  std::vector<std::unique_ptr<tge::models::Qwen2>> models;
  models.reserve(static_cast<std::size_t>(sessions));

  const auto load_start = Clock::now();
  for (int index = 0; index < sessions; ++index) {
    std::string error;
    auto model = tge::models::Qwen2::load(path, error, context, false, kv_half,
                                          kv_q8);
    if (!model) {
      std::fprintf(stderr, "bench sessions: %s\n", error.c_str());
      return 1;
    }
    model->set_threads(threads_per_session);
    model->set_pin_threads(pin_threads);
    models.push_back(std::move(model));
  }
  const auto load_end = Clock::now();

  // Every session gets the same seed, so identical output is the independence
  // check: if any state leaked between them, the transcripts would diverge.
  const int seed_token = 128;
  std::vector<std::vector<int>> generated(static_cast<std::size_t>(sessions));
  std::vector<std::thread> workers;
  workers.reserve(static_cast<std::size_t>(sessions));
  std::atomic<int> ready{0};
  std::atomic<int> done{0};
  std::atomic<bool> go{false};

  for (int index = 0; index < sessions; ++index) {
    workers.emplace_back(
        [&models, &generated, &ready, &done, &go, index, tokens, seed_token]() {
          tge::models::Qwen2& model = *models[static_cast<std::size_t>(index)];
          std::vector<int>& out = generated[static_cast<std::size_t>(index)];
          out.reserve(static_cast<std::size_t>(tokens));

          // Warm the per-session pool and get the first logits outside the
          // timed region; pool creation is not decode throughput.
          const float* logits = model.forward(seed_token, 0);
          ready.fetch_add(1, std::memory_order_acq_rel);
          while (!go.load(std::memory_order_acquire)) {
            std::this_thread::yield();
          }

          for (int step = 0; step < tokens; ++step) {
            const int next =
                tge::models::Qwen2::argmax(logits, model.vocab_size());
            out.push_back(next);
            if (step + 1 < tokens) {
              logits = model.forward(next, step + 1);
            }
          }
          done.fetch_add(1, std::memory_order_acq_rel);
        });
  }

  while (ready.load(std::memory_order_acquire) < sessions) {
    std::this_thread::yield();
  }
  const auto start = Clock::now();
  go.store(true, std::memory_order_release);
  while (done.load(std::memory_order_acquire) < sessions) {
    std::this_thread::yield();
  }
  const auto finish = Clock::now();
  for (std::thread& worker : workers) {
    worker.join();
  }

  const double seconds = std::chrono::duration<double>(finish - start).count();
  const double load_seconds =
      std::chrono::duration<double>(load_end - load_start).count();

  bool identical = true;
  for (int index = 1; index < sessions; ++index) {
    if (generated[static_cast<std::size_t>(index)] !=
        generated[static_cast<std::size_t>(0)]) {
      identical = false;
    }
  }

  const int timed_tokens = tokens - 1;
  const double per_session =
      timed_tokens > 0 ? static_cast<double>(timed_tokens) / seconds : 0.0;
  const int shown_threads =
      threads_per_session > 0 ? threads_per_session
                              : tge::cpu::resolve_decode_threads();
  std::printf("sessions : %d x %d tokens, %d thread(s) each\n", sessions, tokens,
              shown_threads);
  std::printf("load     : %.1f ms for %d models\n", load_seconds * 1e3, sessions);
  std::printf("wall     : %.3f s\n", seconds);
  std::printf("decode   : %.2f t/s per session, %.2f t/s aggregate\n",
              per_session, per_session * sessions);
  std::printf("state    : %s\n",
              identical ? "independent (all transcripts identical)"
                        : "LEAKED - sessions disagree");
  return identical ? 0 : 1;
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
  std::printf("  gemv f32      : %s\n", kernels.gemv_name.c_str());
  std::printf("  attention     : %s\n", kernels.attention_name.c_str());
  std::printf("  streaming     : %s\n", kernels.stream_name.c_str());
  std::printf("  threads       : %d logical, %d for decode\n",
              tge::cpu::resolve_thread_count(0),
              tge::cpu::resolve_decode_threads());

  const tge::gpu::GpuStatus gpu_status = tge::gpu::init();
  std::printf("\nGPU\n");
  std::printf("  status        : %s\n",
              gpu_status == tge::gpu::GpuStatus::Ready
                  ? "ready"
                  : (gpu_status == tge::gpu::GpuStatus::Disabled
                         ? "disabled (backend off)"
                         : "unavailable (CPU fallback)"));
  std::printf("  device        : %s\n", tge::gpu::info().name);
  if (gpu_status == tge::gpu::GpuStatus::Ready) {
    const tge::gpu::NoopResult noop = tge::gpu::run_noop();
    std::printf("  roundtrip     : %s (submit %.1f us, readback %u)\n",
                noop.ok ? "ok" : "FAILED",
                static_cast<double>(noop.submit_ns) / 1e3, noop.readback);
  }
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

std::vector<int> parse_token_list(const char* text) {
  std::vector<int> tokens;
  const char* cursor = text;
  while (*cursor != '\0') {
    while (*cursor == ' ' || *cursor == ',' || *cursor == '\t') {
      ++cursor;
    }
    if (*cursor == '\0') {
      break;
    }
    char* end = nullptr;
    const long value = std::strtol(cursor, &end, 10);
    if (end == cursor) {
      break;
    }
    tokens.push_back(static_cast<int>(value));
    cursor = end;
  }
  return tokens;
}

bool parse_float(const char* text, float& value) {
  char* end = nullptr;
  const double parsed = std::strtod(text, &end);
  if (end == text || *end != '\0') {
    return false;
  }
  value = static_cast<float>(parsed);
  return true;
}

/// Bigram-chain fallback: when no exact span matches, chain the most
/// frequent successor of each token in history. draft[0] follows the last
/// history token; draft[i + 1] follows draft[i]. Proposes only observed
/// continuations (never invents a token); the verifier still decides.
/// One linear pass per chain step with a small distinct-successor set.
/// `need` is 6 for the greedy deep draft, 5 for the sampled path.
bool bigram_chain_draft(const std::vector<int>& history, int* draft, int need) {
  const int length = static_cast<int>(history.size());
  if (length < 2 || need < 1 || need > 6) {
    return false;
  }
  int prev = history.back();
  for (int index = 0; index < need; ++index) {
    int best = -1;
    int best_count = 0;
    // Small distinct-successor set: text bigrams fan out narrowly, so a
    // linear scan beats a hash table here and allocates nothing.
    int candidates[32] = {0};
    int tallies[32] = {0};
    int distinct = 0;
    for (int pos = 0; pos < length - 1; ++pos) {
      if (history[static_cast<std::size_t>(pos)] != prev) {
        continue;
      }
      const int cand = history[static_cast<std::size_t>(pos + 1)];
      int slot = 0;
      while (slot < distinct && candidates[slot] != cand) {
        ++slot;
      }
      if (slot == distinct) {
        if (distinct == 32) {
          continue;
        }
        candidates[distinct] = cand;
        tallies[distinct] = 1;
        ++distinct;
      } else {
        ++tallies[slot];
      }
      if (tallies[slot] > best_count) {
        best_count = tallies[slot];
        best = cand;
      }
    }
    if (best < 0) {
      return false;
    }
    draft[index] = best;
    prev = best;
  }
  return true;
}

/// Drafts the continuation of a near-match of the last two to six tokens
/// with exactly one substitution. Last resort after the exact spans and the
/// bigram chain: both abstain only where the tail is novel, and a single
/// novel token still leaves a near-full match worth proposing. Windows are
/// tried most-recent first, longest span at each position first; every
/// proposal is observed history the verifier still decides, and a
/// first-position miss skips the weight pass entirely (greedy d0 pre-check,
/// sampled pre-judge), so a wrong proposal costs microseconds, never a pass.
/// Returns false when no window is within one substitution.
bool gapped_draft(const std::vector<int>& history, int* draft, int need) {
  const int length = static_cast<int>(history.size());
  if (need < 1 || need > 6) {
    return false;
  }
  for (int end = length - 2; end >= 1; --end) {
    if (length - 1 - end < need) {
      continue;
    }
    for (int span = 6; span >= 2; --span) {
      if (span > end + 1 || length < span + need) {
        continue;
      }
      int mismatches = 0;
      for (int offset = 0; offset < span; ++offset) {
        if (history[static_cast<std::size_t>(length - span + offset)] !=
            history[static_cast<std::size_t>(end - span + 1 + offset)]) {
          if (++mismatches > 1) {
            break;
          }
        }
      }
      if (mismatches != 1) {
        continue;
      }
      for (int index = 0; index < need; ++index) {
        draft[index] = history[static_cast<std::size_t>(end + 1 + index)];
      }
      return true;
    }
  }
  return false;
}

/// Detects a period-1 (aaaa) or period-2 (abab) lock in recent history and
/// proposes `need` tokens along the locked cycle. Verifier-gated like all
/// drafts (the d0 pre-check still applies), so a wrong lock costs microseconds,
/// never a weight pass; returns false when not locked. Measured: locked
/// repetition accepts at the 6.00/step ceiling with byte-identical ids on
/// Qwen2.5-0.5B and Llama-3.2-3B; non-locked prompts abstain.
bool skip_locked_draft(const std::vector<int>& history, int* draft, int need) {
  const int length = static_cast<int>(history.size());
  if (need < 1 || need > 6 || length < 8) {
    return false;
  }
  const int a = history[static_cast<std::size_t>(length - 1)];
  const int b = history[static_cast<std::size_t>(length - 2)];
  bool period1 = true;
  for (int i = 1; i <= 6; ++i) {
    if (history[static_cast<std::size_t>(length - i)] != a) {
      period1 = false;
      break;
    }
  }
  bool period2 = false;
  if (!period1) {
    period2 = true;
    for (int i = 0; i < 8; ++i) {
      const int expect = (i % 2 == 0) ? a : b;
      if (history[static_cast<std::size_t>(length - 1 - i)] != expect) {
        period2 = false;
        break;
      }
    }
  }
  if (!period1 && !period2) {
    return false;
  }
  for (int index = 0; index < need; ++index) {
    if (period1) {
      draft[index] = a;
    } else {
      draft[index] = (index % 2 == 0) ? a : b;
    }
  }
  return true;
}

/// Drafts the continuation of the latest earlier match of the last one to
/// six tokens. Returns `need` tokens or nothing. Longer spans are tried
/// first: a longer context match accepts at a higher rate, and the fallback
/// to shorter spans keeps every proposal the old 3..1 range would have made.
/// When no exact span matches, a bigram chain proposes from single-token
/// statistics instead of abstaining; when the chain also abstains (novel
/// tail), one-substitution windows get the last word.
/// `need` is 6 on the greedy path (the verifier's whole batch, no padding)
/// and 5 on the sampled path (historical shape, zero behavioral delta).
/// Greedy tries the deep draft first and falls back to the historical
/// five-deep form, so every proposal the old code made is preserved.
bool prompt_lookup_draft(const std::vector<int>& history, int* draft, int need) {
  if (need < 1 || need > 6) {
    return false;
  }
  const int length = static_cast<int>(history.size());
  // Deep draft first; the historical five-deep form is the fallback, so a
  // short history that cannot supply six continuations behaves exactly as
  // before. The sampled path (need 5) runs a single identical pass.
  for (int pass = 0; pass < 2 && (pass == 0 || need > 5); ++pass) {
    const int want = (pass == 0) ? need : 5;
    for (int span = 6; span >= 1; --span) {
      if (length < span + want) {
        continue;
      }
      const auto suffix_begin = history.begin() + (length - span);
      const auto suffix_end = history.end();
      for (int end = length - 2; end >= span - 1; --end) {
        if (length - 1 - end < want) {
          continue;
        }
        if (!std::equal(suffix_begin, suffix_end,
                        history.begin() + (end - span + 1))) {
          continue;
        }
        for (int index = 0; index < want; ++index) {
          draft[index] = history[static_cast<std::size_t>(end + 1 + index)];
        }
        return true;
      }
    }
    if (bigram_chain_draft(history, draft, want)) {
      return true;
    }
    if (gapped_draft(history, draft, want)) {
      return true;
    }
  }
  return false;
}

int run_generate(int argc, char** argv) {
  std::string model_path;
  std::vector<int> prompt;
  std::string prompt_text;
  int count = 32;
  int threads = 0;
  int prefill_threads = 0;
  int pin_threads = 0;
  int context_limit = 0;
  bool print_ids = false;
  bool print_text = false;
  VocabularyMode vocabulary = VocabularyMode::Auto;
  tge::models::AttentionMode attention_mode = tge::models::AttentionMode::Auto;
  bool speculative = false;
  bool disable_fuse = false;
  bool disable_skip = false;
  bool vnni = false;
  bool kq8 = true;
  bool kq8d_head = false;
  bool kq8d_down = false;
  bool kv_half = false;
  bool kv_q8 = false;
  bool kv_auto = false;
  bool prefetch = false;
  tge::models::SamplingParams sampling;
  int seed = 0;
  tge::models::SparseConfig sparse;
  std::string gpu_mode = "auto";

  for (int index = 2; index < argc; ++index) {
    if (std::strcmp(argv[index], "--print-ids") == 0) {
      print_ids = true;
      continue;
    }
    if (std::strcmp(argv[index], "--text") == 0) {
      print_text = true;
      continue;
    }
    if (std::strcmp(argv[index], "--speculative") == 0) {
      speculative = true;
      continue;
    }
    if (std::strcmp(argv[index], "--no-fuse") == 0) {
      disable_fuse = true;
      continue;
    }
    if (std::strcmp(argv[index], "--no-skip") == 0) {
      disable_skip = true;
      continue;
    }
    if (std::strcmp(argv[index], "--sparse") == 0) {
      sparse.enabled = true;
      continue;
    }
    if (std::strcmp(argv[index], "--vnni") == 0) {
      vnni = true;
      continue;
    }
    if (std::strcmp(argv[index], "--prefetch") == 0) {
      prefetch = true;
      continue;
    }
    if (index + 1 >= argc) {
      std::fprintf(stderr, "generate: '%s' needs a value\n", argv[index]);
      return 2;
    }
    const char* flag = argv[index];
    const char* value = argv[++index];
    bool ok = true;
    if (std::strcmp(flag, "--model") == 0) {
      model_path = value;
    } else if (std::strcmp(flag, "--tokens") == 0) {
      prompt = parse_token_list(value);
    } else if (std::strcmp(flag, "--prompt") == 0) {
      prompt_text = value;
    } else if (std::strcmp(flag, "--n") == 0) {
      ok = parse_int(value, count);
    } else if (std::strcmp(flag, "--threads") == 0) {
      ok = parse_int(value, threads);
    } else if (std::strcmp(flag, "--prefill-threads") == 0) {
      ok = parse_int(value, prefill_threads);
    } else if (std::strcmp(flag, "--pin-threads") == 0) {
      ok = parse_int(value, pin_threads) &&
           (pin_threads == 0 || pin_threads == 1);
    } else if (std::strcmp(flag, "--context") == 0) {
      ok = parse_int(value, context_limit);
    } else if (std::strcmp(flag, "--vocabulary") == 0) {
      if (std::strcmp(value, "eager") == 0) {
        vocabulary = VocabularyMode::Eager;
      } else if (std::strcmp(value, "lazy") == 0) {
        vocabulary = VocabularyMode::Lazy;
      } else if (std::strcmp(value, "auto") == 0) {
        vocabulary = VocabularyMode::Auto;
      } else {
        std::fprintf(stderr, "generate: --vocabulary takes auto|eager|lazy\n");
        return 2;
      }
    } else if (std::strcmp(flag, "--seed") == 0) {
      ok = parse_int(value, seed);
    } else if (std::strcmp(flag, "--sparse-window") == 0) {
      ok = parse_int(value, sparse.window);
    } else if (std::strcmp(flag, "--sparse-stride") == 0) {
      ok = parse_int(value, sparse.stride);
    } else if (std::strcmp(flag, "--attention") == 0) {
      if (std::strcmp(value, "scalar") == 0) {
        attention_mode = tge::models::AttentionMode::Scalar;
      } else if (std::strcmp(value, "head") == 0) {
        attention_mode = tge::models::AttentionMode::Head;
      } else if (std::strcmp(value, "gqa") == 0) {
        attention_mode = tge::models::AttentionMode::Gqa;
      } else if (std::strcmp(value, "auto") == 0) {
        attention_mode = tge::models::AttentionMode::Auto;
      } else {
        std::fprintf(stderr,
                     "generate: --attention takes auto|scalar|head|gqa\n");
        return 2;
      }
    } else if (std::strcmp(flag, "--kv") == 0) {
      if (std::strcmp(value, "f16") == 0) {
        kv_half = true;
        kv_q8 = false;
      } else if (std::strcmp(value, "i8") == 0) {
        kv_q8 = true;
        kv_half = false;
      } else if (std::strcmp(value, "f32") == 0) {
        kv_half = false;
        kv_q8 = false;
        kv_auto = false;
      } else if (std::strcmp(value, "auto") == 0) {
        kv_auto = true;
        kv_half = false;
        kv_q8 = false;
      } else {
        std::fprintf(stderr, "generate: --kv takes f32|f16|i8|auto\n");
        return 2;
      }
    } else if (std::strcmp(flag, "--gpu") == 0) {
      if (std::strcmp(value, "auto") == 0 ||
          std::strcmp(value, "off") == 0 ||
          std::strcmp(value, "on") == 0) {
        gpu_mode = value;
      } else {
        std::fprintf(stderr, "generate: --gpu takes auto|off|on\n");
        return 2;
      }
    } else if (std::strcmp(flag, "--kernel") == 0) {
      if (std::strcmp(value, "kq8") == 0) {
        kq8 = true;
      } else if (std::strcmp(value, "float") == 0) {
        kq8 = false;
      } else {
        std::fprintf(stderr, "generate: --kernel takes kq8|float\n");
        return 2;
      }
    } else if (std::strcmp(flag, "--kq8d") == 0) {
      kq8d_head = false;
      kq8d_down = false;
      if (std::strcmp(value, "head") == 0) {
        kq8d_head = true;
      } else if (std::strcmp(value, "down") == 0) {
        kq8d_down = true;
      } else if (std::strcmp(value, "head,down") == 0 ||
                 std::strcmp(value, "down,head") == 0) {
        kq8d_head = true;
        kq8d_down = true;
      } else if (std::strcmp(value, "off") != 0) {
        std::fprintf(stderr,
                     "generate: --kq8d takes head|down|head,down|off\n");
        return 2;
      }
    } else if (std::strcmp(flag, "--top-k") == 0) {
      ok = parse_int(value, sampling.top_k);
    } else if (std::strcmp(flag, "--temp") == 0) {
      ok = parse_float(value, sampling.temperature);
    } else if (std::strcmp(flag, "--top-p") == 0) {
      ok = parse_float(value, sampling.top_p);
    } else if (std::strcmp(flag, "--min-p") == 0) {
      ok = parse_float(value, sampling.min_p);
    } else if (std::strcmp(flag, "--repeat-penalty") == 0) {
      ok = parse_float(value, sampling.repeat_penalty);
    } else if (std::strcmp(flag, "--penalty-window") == 0) {
      ok = parse_int(value, sampling.penalty_window);
    } else {
      std::fprintf(stderr, "generate: unknown option '%s'\n", flag);
      return 2;
    }
    if (!ok) {
      std::fprintf(stderr, "generate: invalid value for '%s'\n", flag);
      return 2;
    }
  }

  if (model_path.empty()) {
    std::fprintf(stderr, "generate: --model is required\n");
    return 2;
  }
  sampling.seed = static_cast<std::uint32_t>(seed);
  // Temp-copy compat: the invention harness used TGE_NO_FUSE=1 / TGE_NO_SKIP=1
  // in the same binary for A/B measurement. The shipped CLI keeps the flags
  // above as the documented path; the env vars remain as a fallback so old
  // probe scripts keep working. Default (both unset) is fused + SKIP on.
  if (std::getenv("TGE_NO_FUSE") != nullptr &&
      std::strcmp(std::getenv("TGE_NO_FUSE"), "1") == 0) {
    disable_fuse = true;
  }
  if (std::getenv("TGE_NO_SKIP") != nullptr &&
      std::strcmp(std::getenv("TGE_NO_SKIP"), "1") == 0) {
    disable_skip = true;
  }
  if (sparse.enabled && (sparse.window < 1 || sparse.stride < 2)) {
    std::fprintf(stderr,
                 "generate: --sparse-window must be >= 1 and --sparse-stride "
                 "must be >= 2\n");
    return 2;
  }
  if (kv_auto) {
    kv_half = tge::models::Qwen2::auto_kv_half(context_limit);
    kv_q8 = false;
  }
  return command_generate(model_path, prompt, prompt_text, count, threads,
                          print_ids, sampling, context_limit, print_text,
                          vocabulary, attention_mode, speculative, vnni, kq8,
                          kq8d_head, kq8d_down, kv_half, kv_q8, prefetch,
                          prefill_threads, pin_threads != 0, sparse, gpu_mode,
                          disable_fuse, disable_skip);
}

int run_gemm_bench(int argc, char** argv, int first) {
  int m = 1024;
  int n = 1024;
  int k = 1024;
  int repeat = 5;
  int threads = 0;
  const char* kernel = "packed";

  for (int index = first; index < argc; index += 2) {
    if (index + 1 >= argc) {
      std::fprintf(stderr, "bench gemm: '%s' needs a value\n", argv[index]);
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
    } else if (std::strcmp(flag, "--kernel") == 0) {
      kernel = value;
      ok = true;
    } else {
      std::fprintf(stderr, "bench gemm: unknown option '%s'\n", flag);
      return 2;
    }
    if (!ok) {
      std::fprintf(stderr, "bench gemm: invalid value for '%s'\n", flag);
      return 2;
    }
  }
  return command_gemm_bench(m, n, k, repeat, threads, kernel);
}

int run_gemv_bench(int argc, char** argv, int first) {
  int rows = 4096;
  int cols = 4096;
  int repeat = 5;
  int threads = 0;

  for (int index = first; index < argc; index += 2) {
    if (index + 1 >= argc) {
      std::fprintf(stderr, "bench gemv: '%s' needs a value\n", argv[index]);
      return 2;
    }
    const char* flag = argv[index];
    const char* value = argv[index + 1];
    bool ok = false;
    if (std::strcmp(flag, "--rows") == 0) {
      ok = parse_int(value, rows);
    } else if (std::strcmp(flag, "--cols") == 0) {
      ok = parse_int(value, cols);
    } else if (std::strcmp(flag, "--repeat") == 0) {
      ok = parse_int(value, repeat);
    } else if (std::strcmp(flag, "--threads") == 0) {
      ok = parse_int(value, threads);
    } else {
      std::fprintf(stderr, "bench gemv: unknown option '%s'\n", flag);
      return 2;
    }
    if (!ok) {
      std::fprintf(stderr, "bench gemv: invalid value for '%s'\n", flag);
      return 2;
    }
  }
  return command_gemv_bench(rows, cols, repeat, threads);
}

int run_mem_bench(int argc, char** argv, int first) {
  int size_mb = 256;
  int repeat = 5;
  int threads = 0;

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
    } else if (std::strcmp(flag, "--threads") == 0) {
      ok = parse_int(value, threads);
    } else {
      std::fprintf(stderr, "bench mem: unknown option '%s'\n", flag);
      return 2;
    }
    if (!ok) {
      std::fprintf(stderr, "bench mem: invalid value for '%s'\n", flag);
      return 2;
    }
  }
  return command_mem_bench(size_mb, repeat, threads);
}

int command_fidelity(const std::string& path, const std::string& corpus_path,
                     int context, int max_tokens, int threads, bool vnni,
                     bool kq8, bool kq8d_head, bool kq8d_down, bool kv_half,
                     bool kv_q8, tge::models::SparseConfig sparse,
                     const std::string& gpu_mode) {
  const tge::cpu::CpuInfo cpu = tge::cpu::detect();
  if (vnni && !cpu.features.avx_vnni) {
    std::fprintf(stderr, "fidelity: --vnni needs AVX-VNNI on this CPU\n");
    return 2;
  }
  if (kv_half && !cpu.features.f16c) {
    std::fprintf(stderr, "fidelity: --kv f16 needs F16C on this CPU\n");
    return 2;
  }
  if (kv_half && kv_q8) {
    std::fprintf(stderr, "fidelity: --kv takes one of f32|f16|i8\n");
    return 2;
  }
  if (gpu_mode == "on" &&
      tge::gpu::init() != tge::gpu::GpuStatus::Ready) {
    std::fprintf(stderr, "fidelity: --gpu on needs a ready GPU backend (%s)\n",
                 tge::gpu::info().name);
    return 2;
  }
  std::ifstream corpus_stream(corpus_path, std::ios::binary);
  if (!corpus_stream) {
    std::fprintf(stderr, "fidelity: cannot read %s\n", corpus_path.c_str());
    return 1;
  }
  std::ostringstream corpus_buffer;
  corpus_buffer << corpus_stream.rdbuf();
  const std::string corpus = corpus_buffer.str();
  if (corpus.empty()) {
    std::fprintf(stderr, "fidelity: corpus is empty\n");
    return 1;
  }

  std::string error;
  const auto load_start = Clock::now();
  // The scalar reference loads the vocabulary eagerly, because the tokenizer
  // reuses those arrays instead of parsing the file a second time.
  std::unique_ptr<tge::models::Qwen2> reference =
      tge::models::Qwen2::load(path, error, context, true, false);
  if (!reference) {
    std::fprintf(stderr, "fidelity: %s\n", error.c_str());
    return 1;
  }
  std::unique_ptr<tge::models::Qwen2> candidate =
      tge::models::Qwen2::load(path, error, context, false, kv_half, kv_q8);
  if (!candidate) {
    std::fprintf(stderr, "fidelity: %s\n", error.c_str());
    return 1;
  }
  const auto load_end = Clock::now();

  reference->set_threads(threads);
  candidate->set_threads(threads);
  reference->set_attention_mode(tge::models::AttentionMode::Scalar);
  // The reference is the full scalar oracle: scalar attention plus scalar
  // norms, RoPE and SiLU, so the KL prices the whole vectorisation program.
  // The test model runs every kernel as production would.
  reference->set_scalar_kernels(true);
  candidate->set_vnni(vnni);
  candidate->set_kq8(kq8);
  candidate->set_kq8d(kq8d_head, kq8d_down);
  // The candidate may request the GPU; the reference never does, so any KL
  // prices the GPU numerics against the exact scalar oracle.
  candidate->set_use_gpu(gpu_mode != "off");
  if (sparse.enabled) {
    if (kv_q8) {
      std::fprintf(stderr,
                   "fidelity: --sparse needs the f32|f16 KV cache (no --kv i8)\n");
      return 2;
    }
    try {
      candidate->set_sparse_config(sparse);
    } catch (const std::exception& error) {
      std::fprintf(stderr, "fidelity: %s\n", error.what());
      return 2;
    }
  }

  const std::unique_ptr<tge::models::Tokenizer> tokenizer =
      tge::models::Tokenizer::from_gguf(reference->file(), error);
  if (!tokenizer) {
    std::fprintf(stderr, "fidelity: %s\n", error.c_str());
    return 1;
  }
  const std::vector<int> tokens = tokenizer->encode(corpus);
  if (tokens.empty()) {
    std::fprintf(stderr, "fidelity: corpus has no tokens\n");
    return 1;
  }

  const int usable = std::min(reference->max_context(), candidate->max_context());
  int limit = static_cast<int>(tokens.size());
  if (max_tokens > 0) {
    limit = std::min(limit, max_tokens);
  }
  limit = std::min(limit, usable);
  if (limit < 2) {
    std::fprintf(stderr,
                 "fidelity: corpus needs two tokens within the %d-token context\n",
                 usable);
    return 2;
  }

  const int vocab_size = reference->vocab_size();
  std::vector<double> left(static_cast<std::size_t>(vocab_size));
  std::vector<double> right(static_cast<std::size_t>(vocab_size));
  double kl_sum = 0.0;
  double kl_worst = 0.0;
  double max_difference = 0.0;
  int top1_agree = 0;
  const int positions = limit - 1;

  const auto start = Clock::now();
  for (int position = 0; position < positions; ++position) {
    const float* reference_logits =
        reference->forward(tokens[static_cast<std::size_t>(position)], position);
    const float* candidate_logits =
        candidate->forward(tokens[static_cast<std::size_t>(position)], position);

    int best_reference = 0;
    int best_candidate = 0;
    for (int index = 1; index < vocab_size; ++index) {
      if (reference_logits[index] > reference_logits[best_reference]) {
        best_reference = index;
      }
      if (candidate_logits[index] > candidate_logits[best_candidate]) {
        best_candidate = index;
      }
    }
    for (int index = 0; index < vocab_size; ++index) {
      const double difference =
          std::fabs(static_cast<double>(reference_logits[index]) -
                    static_cast<double>(candidate_logits[index]));
      if (difference > max_difference) {
        max_difference = difference;
      }
    }
    if (best_reference == best_candidate) {
      ++top1_agree;
    }

    const double reference_max = reference_logits[best_reference];
    const double candidate_max = candidate_logits[best_candidate];
    double reference_sum = 0.0;
    double candidate_sum = 0.0;
    for (int index = 0; index < vocab_size; ++index) {
      left[static_cast<std::size_t>(index)] =
          std::exp(static_cast<double>(reference_logits[index]) - reference_max);
      right[static_cast<std::size_t>(index)] =
          std::exp(static_cast<double>(candidate_logits[index]) - candidate_max);
      reference_sum += left[static_cast<std::size_t>(index)];
      candidate_sum += right[static_cast<std::size_t>(index)];
    }

    double kl = 0.0;
    for (int index = 0; index < vocab_size; ++index) {
      const double p = left[static_cast<std::size_t>(index)] / reference_sum;
      if (p <= 0.0) {
        continue;
      }
      const double q = right[static_cast<std::size_t>(index)] / candidate_sum;
      if (q <= 0.0) {
        continue;
      }
      kl += p * std::log(p / q);
    }
    kl_sum += kl;
    if (kl > kl_worst) {
      kl_worst = kl;
    }
  }
  const auto finish = Clock::now();

  const char* mode_name = "auto";
  switch (candidate->attention_mode()) {
    case tge::models::AttentionMode::Auto:
      mode_name = "auto";
      break;
    case tge::models::AttentionMode::Scalar:
      mode_name = "scalar";
      break;
    case tge::models::AttentionMode::Head:
      mode_name = "head";
      break;
    case tge::models::AttentionMode::Gqa:
      mode_name = "gqa";
      break;
  }
  std::printf("fidelity : full-scalar oracle vs %s attention (%s%s%s%s)\n",
              tge::kernels().attention_name.c_str(), mode_name,
              vnni ? ", vnni" : "",
              kv_half ? ", f16 kv" : (kv_q8 ? ", i8 kv" : ""),
              sparse.enabled ? ", sparse" : "");
  std::printf("load     : %.1f ms\n",
              std::chrono::duration<double>(load_end - load_start).count() * 1e3);
  std::printf("corpus   : %d positions, context %d\n", positions, usable);
  std::printf("KL       : mean %.3e, worst %.3e nats\n", kl_sum / positions,
              kl_worst);
  std::printf("logits   : max |diff| %.3e\n", max_difference);
  std::printf("top-1    : %d/%d agree\n", top1_agree, positions);
  std::printf("time     : %.2f s\n",
              std::chrono::duration<double>(finish - start).count());
  return 0;
}

int run_fidelity(int argc, char** argv) {
  std::string model_path;
  std::string corpus_path;
  int context = 1024;
  int max_tokens = 0;
  int threads = 0;
  bool vnni = false;
  bool kq8 = true;
  bool kq8d_head = false;
  bool kq8d_down = false;
  bool kv_half = false;
  bool kv_q8 = false;
  bool kv_auto = false;
  tge::models::SparseConfig sparse;
  std::string gpu_mode = "auto";

  for (int index = 2; index < argc; ++index) {
    if (std::strcmp(argv[index], "--vnni") == 0) {
      vnni = true;
      continue;
    }
    if (std::strcmp(argv[index], "--sparse") == 0) {
      sparse.enabled = true;
      continue;
    }
    if (std::strcmp(argv[index], "--kv") == 0) {
      if (index + 1 >= argc) {
        std::fprintf(stderr, "fidelity: '--kv' needs a value\n");
        return 2;
      }
      const char* value = argv[++index];
      if (std::strcmp(value, "f16") == 0) {
        kv_half = true;
        kv_q8 = false;
      } else if (std::strcmp(value, "i8") == 0) {
        kv_q8 = true;
        kv_half = false;
      } else if (std::strcmp(value, "f32") == 0) {
        kv_half = false;
        kv_q8 = false;
      } else if (std::strcmp(value, "auto") == 0) {
        kv_auto = true;
        kv_half = false;
        kv_q8 = false;
      } else {
        std::fprintf(stderr, "fidelity: --kv takes f32|f16|i8|auto\n");
        return 2;
      }
      continue;
    }
    if (std::strcmp(argv[index], "--gpu") == 0) {
      if (index + 1 >= argc) {
        std::fprintf(stderr, "fidelity: '--gpu' needs a value\n");
        return 2;
      }
      const char* gpu_value = argv[++index];
      if (std::strcmp(gpu_value, "auto") == 0 ||
          std::strcmp(gpu_value, "off") == 0 ||
          std::strcmp(gpu_value, "on") == 0) {
        gpu_mode = gpu_value;
      } else {
        std::fprintf(stderr, "fidelity: --gpu takes auto|off|on\n");
        return 2;
      }
      continue;
    }
    if (index + 1 >= argc) {
      std::fprintf(stderr, "fidelity: '%s' needs a value\n", argv[index]);
      return 2;
    }
    const char* flag = argv[index];
    const char* value = argv[++index];
    bool ok = true;
    if (std::strcmp(flag, "--model") == 0) {
      model_path = value;
    } else if (std::strcmp(flag, "--corpus") == 0) {
      corpus_path = value;
    } else if (std::strcmp(flag, "--context") == 0) {
      ok = parse_int(value, context);
    } else if (std::strcmp(flag, "--tokens") == 0) {
      ok = parse_int(value, max_tokens);
    } else if (std::strcmp(flag, "--threads") == 0) {
      ok = parse_int(value, threads);
    } else if (std::strcmp(flag, "--kernel") == 0) {
      if (std::strcmp(value, "kq8") == 0) {
        kq8 = true;
      } else if (std::strcmp(value, "float") == 0) {
        kq8 = false;
      } else {
        std::fprintf(stderr, "fidelity: --kernel takes kq8|float\n");
        return 2;
      }
    } else if (std::strcmp(flag, "--kq8d") == 0) {
      kq8d_head = false;
      kq8d_down = false;
      if (std::strcmp(value, "head") == 0) {
        kq8d_head = true;
      } else if (std::strcmp(value, "down") == 0) {
        kq8d_down = true;
      } else if (std::strcmp(value, "head,down") == 0 ||
                 std::strcmp(value, "down,head") == 0) {
        kq8d_head = true;
        kq8d_down = true;
      } else if (std::strcmp(value, "off") != 0) {
        std::fprintf(stderr,
                     "fidelity: --kq8d takes head|down|head,down|off\n");
        return 2;
      }
    } else if (std::strcmp(flag, "--sparse-window") == 0) {
      ok = parse_int(value, sparse.window);
    } else if (std::strcmp(flag, "--sparse-stride") == 0) {
      ok = parse_int(value, sparse.stride);
    } else {
      std::fprintf(stderr, "fidelity: unknown option '%s'\n", flag);
      return 2;
    }
    if (!ok) {
      std::fprintf(stderr, "fidelity: invalid value for '%s'\n", flag);
      return 2;
    }
  }

  if (model_path.empty()) {
    std::fprintf(stderr, "fidelity: --model is required\n");
    return 2;
  }
  if (corpus_path.empty()) {
    std::fprintf(stderr, "fidelity: --corpus is required\n");
    return 2;
  }
  if (sparse.enabled && (sparse.window < 1 || sparse.stride < 2)) {
    std::fprintf(stderr,
                 "fidelity: --sparse-window must be >= 1 and --sparse-stride "
                 "must be >= 2\n");
    return 2;
  }
  if (kv_auto) {
    kv_half = tge::models::Qwen2::auto_kv_half(context);
    kv_q8 = false;
  }
  return command_fidelity(model_path, corpus_path, context, max_tokens, threads,
                          vnni, kq8, kq8d_head, kq8d_down, kv_half, kv_q8,
                          sparse, gpu_mode);
}

}  // namespace

int main(int argc, char** argv) {
#if defined(_WIN32)
  // The CRT builds narrow argv from the ANSI code page, which mangles any
  // prompt outside it (CJK arrives as '?', then mistokenizes). Rebuild the
  // argument vector from the wide command line so prompts are UTF-8 no
  // matter what the console speaks. File paths are byte-identical either
  // way on this machine, so every existing invocation is unaffected.
  int wide_argc = 0;
  wchar_t** wide_argv = CommandLineToArgvW(GetCommandLineW(), &wide_argc);
  std::vector<std::string> utf8_args;
  std::vector<char*> utf8_argv;
  if (wide_argv != nullptr && wide_argc >= 0) {
    bool converted = true;
    utf8_args.reserve(static_cast<std::size_t>(wide_argc));
    for (int index = 0; index < wide_argc && converted; ++index) {
      const int needed = WideCharToMultiByte(CP_UTF8, 0, wide_argv[index], -1,
                                             nullptr, 0, nullptr, nullptr);
      if (needed <= 1) {
        converted = false;
        break;
      }
      std::string arg(static_cast<std::size_t>(needed - 1), '\0');
      if (WideCharToMultiByte(CP_UTF8, 0, wide_argv[index], -1, arg.data(),
                              needed, nullptr, nullptr) != needed) {
        converted = false;
        break;
      }
      utf8_args.push_back(std::move(arg));
    }
    LocalFree(wide_argv);
    if (converted) {
      utf8_argv.reserve(utf8_args.size() + 1);
      for (std::string& arg : utf8_args) {
        utf8_argv.push_back(arg.data());
      }
      utf8_argv.push_back(nullptr);
      argc = wide_argc;
      argv = utf8_argv.data();
    }
  }
#endif
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

  if (command == "gguf") {
    if (argc < 3) {
      std::fprintf(stderr, "gguf: a model path is required\n");
      return 2;
    }
    return command_gguf(argv[2]);
  }

  if (command == "bench" && argc >= 3 &&
      std::strcmp(argv[2], "continuous") == 0) {
    std::string model_path;
    int requests = 8;
    int tokens = 32;
    int threads = 0;
    int sequences = 4;
    int context = 0;
    int prompt_tokens = 1;
    int prefill_threads = 0;
  bool vnni = false;
  bool kv_half = false;
    bool kv_q8 = false;
    bool kv_auto = false;
    for (int index = 3; index < argc; ++index) {
      if (std::strcmp(argv[index], "--vnni") == 0) {
        vnni = true;
        continue;
      }
      if (std::strcmp(argv[index], "--kv") == 0) {
        if (index + 1 >= argc) {
          std::fprintf(stderr, "bench continuous: '--kv' needs a value\n");
          return 2;
        }
        const char* value = argv[++index];
        if (std::strcmp(value, "f16") == 0) {
          kv_half = true;
          kv_q8 = false;
        } else if (std::strcmp(value, "i8") == 0) {
          kv_q8 = true;
          kv_half = false;
        } else if (std::strcmp(value, "f32") == 0) {
          kv_half = false;
          kv_q8 = false;
        } else if (std::strcmp(value, "auto") == 0) {
          kv_auto = true;
          kv_half = false;
          kv_q8 = false;
        } else {
          std::fprintf(stderr, "bench continuous: --kv takes f32|f16|i8|auto\n");
          return 2;
        }
        continue;
      }
      const char* flag = argv[index];
      const char* value = index + 1 < argc ? argv[index + 1] : "";
      bool ok = true;
      if (std::strcmp(flag, "--model") == 0) model_path = value;
      else if (std::strcmp(flag, "--requests") == 0) ok = parse_int(value, requests);
      else if (std::strcmp(flag, "--tokens") == 0) ok = parse_int(value, tokens);
      else if (std::strcmp(flag, "--threads") == 0) ok = parse_int(value, threads);
      else if (std::strcmp(flag, "--sequences") == 0) ok = parse_int(value, sequences);
      else if (std::strcmp(flag, "--context") == 0) ok = parse_int(value, context);
      else if (std::strcmp(flag, "--prompt-tokens") == 0) ok = parse_int(value, prompt_tokens);
      else if (std::strcmp(flag, "--prefill-threads") == 0) ok = parse_int(value, prefill_threads);
      else { std::fprintf(stderr, "bench continuous: unknown option '%s'\n", flag); return 2; }
      if (!ok) { std::fprintf(stderr, "bench continuous: invalid value\n"); return 2; }
      ++index;
    }
    if (model_path.empty()) {
      std::fprintf(stderr, "bench continuous: --model is required\n");
      return 2;
    }
    if (kv_auto) {
      kv_half = tge::models::Qwen2::auto_kv_half(context);
      kv_q8 = false;
    }
    return command_continuous_bench(model_path, requests, tokens, threads,
                                    sequences, context, prompt_tokens, vnni,
                                    kv_half, kv_q8, prefill_threads);
  }

  if (command == "bench" && argc >= 3 && std::strcmp(argv[2], "sessions") == 0) {
    std::string model_path;
    int sessions = 2;
    int tokens = 32;
    int threads_per_session = 0;
    int context = 0;
    int pin_threads = 0;
    bool kv_half = false;
    bool kv_q8 = false;
    bool kv_auto = false;
    for (int index = 3; index + 1 < argc; index += 2) {
      const char* flag = argv[index];
      const char* value = index + 1 < argc ? argv[index + 1] : "";
      bool ok = true;
      if (std::strcmp(flag, "--model") == 0) model_path = value;
      else if (std::strcmp(flag, "--sessions") == 0) ok = parse_int(value, sessions);
      else if (std::strcmp(flag, "--tokens") == 0) ok = parse_int(value, tokens);
      else if (std::strcmp(flag, "--threads") == 0) ok = parse_int(value, threads_per_session);
      else if (std::strcmp(flag, "--context") == 0) ok = parse_int(value, context);
      else if (std::strcmp(flag, "--kv") == 0) {
        if (std::strcmp(value, "f16") == 0) {
          kv_half = true;
          kv_q8 = false;
        } else if (std::strcmp(value, "i8") == 0) {
          kv_q8 = true;
          kv_half = false;
        } else if (std::strcmp(value, "f32") == 0) {
          kv_half = false;
          kv_q8 = false;
        } else if (std::strcmp(value, "auto") == 0) {
          kv_auto = true;
          kv_half = false;
          kv_q8 = false;
        } else {
          ok = false;
        }
      }
      else if (std::strcmp(flag, "--pin-threads") == 0) {
        ok = parse_int(value, pin_threads) &&
             (pin_threads == 0 || pin_threads == 1);
      }
      else { std::fprintf(stderr, "bench sessions: unknown option '%s'\n", flag); return 2; }
      if (!ok) { std::fprintf(stderr, "bench sessions: invalid value\n"); return 2; }
    }
    if (model_path.empty()) {
      std::fprintf(stderr, "bench sessions: --model is required\n");
      return 2;
    }
    if (sessions < 1) {
      std::fprintf(stderr, "bench sessions: --sessions must be positive\n");
      return 2;
    }
    if (kv_auto) {
      kv_half = tge::models::Qwen2::auto_kv_half(context);
      kv_q8 = false;
    }
    return command_sessions_bench(model_path, sessions, tokens,
                                  threads_per_session, context,
                                  pin_threads != 0, kv_half, kv_q8);
  }

  if (command == "plan") {
    std::string model_path;
    int context = 4096;
    int sessions = 1;
    float bandwidth = 28.0F;
    for (int index = 2; index < argc; index += 2) {
      if (index + 1 >= argc) {
        std::fprintf(stderr, "plan: '%s' needs a value\n", argv[index]);
        return 2;
      }
      const char* flag = argv[index];
      const char* value = argv[index + 1];
      bool ok = true;
      if (std::strcmp(flag, "--model") == 0) model_path = value;
      else if (std::strcmp(flag, "--context") == 0) ok = parse_int(value, context);
      else if (std::strcmp(flag, "--sessions") == 0) ok = parse_int(value, sessions);
      else if (std::strcmp(flag, "--bandwidth") == 0) ok = parse_float(value, bandwidth);
      else { std::fprintf(stderr, "plan: unknown option '%s'\n", flag); return 2; }
      if (!ok) { std::fprintf(stderr, "plan: invalid value\n"); return 2; }
    }
    if (model_path.empty()) {
      std::fprintf(stderr, "plan: --model is required\n");
      return 2;
    }
    return command_plan(model_path, context, sessions,
                        static_cast<double>(bandwidth));
  }

  if (command == "generate") {
    return run_generate(argc, argv);
  }

  if (command == "fidelity") {
    return run_fidelity(argc, argv);
  }

  if (command == "bench") {
    if (argc >= 3 && std::strcmp(argv[2], "batch") == 0) {
      int rows = 4864, cols = 896, tokens = 64, repeat = 5;
      bool use_vnni = false;
        bool use_kq8 = false;
        bool use_kq8d = false;
      int k_format = 0;
      for (int index = 3; index < argc; index += 2) {
        const char* flag = argv[index];
        const char* value = index + 1 < argc ? argv[index + 1] : "";
        bool ok = true;
        if (std::strcmp(flag, "--rows") == 0) ok = parse_int(value, rows);
        else if (std::strcmp(flag, "--cols") == 0) ok = parse_int(value, cols);
        else if (std::strcmp(flag, "--tokens") == 0) ok = parse_int(value, tokens);
        else if (std::strcmp(flag, "--repeat") == 0) ok = parse_int(value, repeat);
        else if (std::strcmp(flag, "--kernel") == 0) {
          use_vnni = std::strcmp(value, "vnni") == 0;
          use_kq8 = std::strcmp(value, "kq8") == 0;
          use_kq8d = std::strcmp(value, "kq8d") == 0;
        }
        else if (std::strcmp(flag, "--format") == 0) {
          if (std::strcmp(value, "q4k") == 0) {
            k_format = 4;
          } else if (std::strcmp(value, "q40") == 0) {
            k_format = 40;
          } else if (std::strcmp(value, "q6k") == 0) {
            k_format = 6;
          } else if (std::strcmp(value, "q50") == 0) {
            k_format = 50;
          } else if (std::strcmp(value, "q51") == 0) {
            k_format = 51;
          } else if (std::strcmp(value, "q5k") == 0) {
            k_format = 5;
          } else if (std::strcmp(value, "q8") != 0) {
            ok = false;
          }
        }
        else { std::fprintf(stderr, "bench batch: unknown option '%s'\n", flag); return 2; }
        if (!ok) { std::fprintf(stderr, "bench batch: invalid value\n"); return 2; }
      }
      return command_batch_bench(rows, cols, tokens, repeat, use_vnni,
                                 use_kq8, use_kq8d, k_format);
    }
    if (argc >= 3 && std::strcmp(argv[2], "mem") == 0) {
      return run_mem_bench(argc, argv, 3);
    }
    if (argc >= 3 && std::strcmp(argv[2], "gemv") == 0) {
      return run_gemv_bench(argc, argv, 3);
    }
    const int first = (argc >= 3 && std::strcmp(argv[2], "gemm") == 0) ? 3 : 2;
    return run_gemm_bench(argc, argv, first);
  }

  std::fprintf(stderr, "unknown command '%s'\n\n", command.c_str());
  print_usage();
  return 2;
}
