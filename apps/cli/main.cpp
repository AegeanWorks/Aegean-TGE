#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <memory>
#include <string>
#include <vector>

#include "tge/core/cpu.hpp"
#include "tge/core/dispatch.hpp"
#include "tge/core/thread_pool.hpp"
#include "tge/memory/allocator.hpp"
#include "tge/memory/budget.hpp"
#include "tge/memory/system_ram.hpp"
#include "tge/models/gguf.hpp"
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
      "  gguf <path>          inspect a GGUF model file\n"
      "  generate [opts]      greedy decode and token-rate measurement\n"
      "        --model PATH     GGUF model to load\n"
      "        --tokens \"a b c\"  prompt token ids\n"
      "        --prompt \"text\"   prompt as text, tokenized by the model\n"
      "        --n N            tokens to generate (default 32)\n"
      "        --threads N      worker threads (default: all logical)\n"
      "        --context N      KV cache length (default 4096, capped by model)\n"
      "        --vocabulary M   auto|eager|lazy tokenizer parsing (default auto)\n"
      "        --temp T         sampling temperature; 0 is greedy (default 0)\n"
      "        --top-k N        keep the N best tokens (default off)\n"
      "        --top-p P        nucleus probability mass (default 1.0)\n"
      "        --min-p P        drop tokens below this fraction of the best\n"
      "        --repeat-penalty R  penalty for already seen tokens (default 1)\n"
      "        --seed N         random seed for sampling\n"
      "        --print-ids      print generated token ids\n"
      "        --text           detokenize and print the generated text\n"
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

int command_generate(const std::string& path, const std::vector<int>& prompt_ids,
                     const std::string& prompt_text, int count, int threads,
                     bool print_ids,
                     const tge::models::SamplingParams& sampling,
                     int context_limit, bool print_text,
                     VocabularyMode vocabulary) {
  if (count <= 0) {
    std::fprintf(stderr, "generate: token count must be positive\n");
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
  std::unique_ptr<tge::models::Qwen2> model =
      tge::models::Qwen2::load(path, error, context_limit, eager);
  const auto load_end = Clock::now();
  if (!model) {
    std::fprintf(stderr, "generate: %s\n", error.c_str());
    return 1;
  }
  std::printf("load     : %.1f ms (%s vocabulary)\n",
              std::chrono::duration<double>(load_end - load_start).count() * 1e3,
              eager ? "eager" : "skipped");
  model->set_threads(threads);

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

  const auto prefill_start = Clock::now();
  const float* logits = model->prefill(prompt.data(),
                                       static_cast<int>(prompt.size()), 0);
  const auto prefill_end = Clock::now();
  const double prefill_seconds =
      std::chrono::duration<double>(prefill_end - prefill_start).count();

  std::printf("prefill  : %zu tokens in %.2f ms (%.1f tok/s)\n", prompt.size(),
              prefill_seconds * 1e3,
              static_cast<double>(prompt.size()) / prefill_seconds);

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

  generated.push_back(draw());

  int position = static_cast<int>(prompt.size());
  const auto decode_start = Clock::now();
  for (int index = 1; index < count; ++index) {
    logits = model->forward(generated.back(), position++);
    generated.push_back(draw());
  }
  const auto decode_end = Clock::now();
  const double decode_seconds =
      std::chrono::duration<double>(decode_end - decode_start).count();

  const int timed_tokens = static_cast<int>(generated.size()) - 1;
  if (timed_tokens > 0 && decode_seconds > 0.0) {
    std::printf("decode   : %d tokens in %.2f ms (%.2f tok/s)\n", timed_tokens,
                decode_seconds * 1e3, timed_tokens / decode_seconds);
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
                        bool use_vnni) {
  if (rows <= 0 || cols <= 0 || tokens <= 0 || repeat <= 0) {
    std::fprintf(stderr, "bench batch: all dimensions must be positive\n");
    return 2;
  }
  if (cols % 32 != 0) {
    std::fprintf(stderr, "bench batch: cols must be a multiple of 32\n");
    return 2;
  }

  const std::size_t matrix_bytes =
      static_cast<std::size_t>(rows) * static_cast<std::size_t>(cols / 32) * 34;
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
  const tge::cpu::Q8BatchFn kernel =
      use_vnni ? &tge::cpu::gemv_q8_batch_vnni : set.gemv_q8_batch;
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

  const double flops =
      2.0 * static_cast<double>(rows) * cols * static_cast<double>(tokens);
  std::printf("q8 batch %dx%d x %d tokens (%.1f MiB of weights), %s, best of %d\n",
              rows, cols, tokens,
              static_cast<double>(matrix_bytes) / (1024.0 * 1024.0),
              use_vnni ? "vnni" : "float", repeat);
  std::printf("  %.3f ms  %.2f GFLOPS  %.2f GB/s\n", best * 1e3,
              flops / best / 1e9,
              static_cast<double>(matrix_bytes) / best / 1e9);
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
  std::printf("  gemv f32      : %s\n", kernels.gemv_name.c_str());
  std::printf("  streaming     : %s\n", kernels.stream_name.c_str());
  std::printf("  threads       : %d logical, %d for decode\n",
              tge::cpu::resolve_thread_count(0),
              tge::cpu::resolve_decode_threads());
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

int run_generate(int argc, char** argv) {
  std::string model_path;
  std::vector<int> prompt;
  std::string prompt_text;
  int count = 32;
  int threads = 0;
  int context_limit = 0;
  bool print_ids = false;
  bool print_text = false;
  VocabularyMode vocabulary = VocabularyMode::Auto;
  tge::models::SamplingParams sampling;
  int seed = 0;

  for (int index = 2; index < argc; ++index) {
    if (std::strcmp(argv[index], "--print-ids") == 0) {
      print_ids = true;
      continue;
    }
    if (std::strcmp(argv[index], "--text") == 0) {
      print_text = true;
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
  return command_generate(model_path, prompt, prompt_text, count, threads,
                          print_ids, sampling, context_limit, print_text,
                          vocabulary);
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

  if (command == "gguf") {
    if (argc < 3) {
      std::fprintf(stderr, "gguf: a model path is required\n");
      return 2;
    }
    return command_gguf(argv[2]);
  }

  if (command == "generate") {
    return run_generate(argc, argv);
  }

  if (command == "bench") {
    if (argc >= 3 && std::strcmp(argv[2], "batch") == 0) {
      int rows = 4864, cols = 896, tokens = 64, repeat = 5;
      bool use_vnni = false;
      for (int index = 3; index < argc; index += 2) {
        const char* flag = argv[index];
        const char* value = index + 1 < argc ? argv[index + 1] : "";
        bool ok = true;
        if (std::strcmp(flag, "--rows") == 0) ok = parse_int(value, rows);
        else if (std::strcmp(flag, "--cols") == 0) ok = parse_int(value, cols);
        else if (std::strcmp(flag, "--tokens") == 0) ok = parse_int(value, tokens);
        else if (std::strcmp(flag, "--repeat") == 0) ok = parse_int(value, repeat);
        else if (std::strcmp(flag, "--kernel") == 0) use_vnni = std::strcmp(value, "vnni") == 0;
        else { std::fprintf(stderr, "bench batch: unknown option '%s'\n", flag); return 2; }
        if (!ok) { std::fprintf(stderr, "bench batch: invalid value\n"); return 2; }
      }
      return command_batch_bench(rows, cols, tokens, repeat, use_vnni);
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
