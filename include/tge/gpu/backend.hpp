#pragma once

// Hybrid CPU+GPU backend selector (bring-up step [1/4]).
//
// This header never includes Vulkan SDK headers, so translation units that
// depend on it compile with or without a Vulkan SDK installed. When
// TGE_HAS_VULKAN=1 the real implementation lives in
// src/gpu/vulkan_context.cpp; otherwise src/gpu/noop.cpp reports
// Unavailable and every eligibility predicate keeps the caller on the
// proven CPU kernels (exactly one fallback path, ADR-0002 pattern).

#include <cstddef>
#include <cstdint>

namespace tge::gpu {

/// Why the GPU path is or is not usable. `Disabled` is a compile-time or
/// explicit opt-out; `Unavailable` means the loader/driver/VRAM check failed
/// at runtime; `Ready` is the only state that permits offload.
enum class GpuStatus { Disabled, Unavailable, Ready };

enum class GpuVendor : std::uint8_t { Unknown, AMD, NVIDIA, Intel, Other };

struct GpuInfo {
  bool available = false;
  GpuVendor vendor = GpuVendor::Unknown;
  char name[128] = {};
  // Usable heaps, bytes. `device_local` is the offload target (VRAM);
  // `host_visible` bounds staging.
  std::uint64_t device_local_bytes = 0;
  std::uint64_t host_visible_bytes = 0;
  // Hardware INT8 dot-product acceleration (Vulkan cooperative matrix /
  // integerDotProduct8BitAccelerated). Gates any future INT8 GPU kernel.
  bool int8_dot_accelerated = false;
  std::uint32_t subgroup_min = 0;
  std::uint32_t subgroup_max = 0;
};

/// Static placement thresholds (Stage 2 revised plan §5). Constants, not
/// autotuned: dynamic per-mat timing would branch inside the 169-call/token
/// spin window and fight the fork-join handoff.
struct GpuThresholds {
  // Batch offload floor: prefill chunks (64), forward_batch and continuous
  // batching ride the GPU; the 6-token speculative verifier stays CPU so it
  // keeps the pinned per-head arithmetic of the sequential path.
  static constexpr int kMinBatchTokens = 8;
  // Grouped-query attention offload floor: mirrors uses_gqa() (span >= 256)
  // so the GPU path never diverges from the CPU schedule it replaces.
  static constexpr int kGqaMinSpan = 256;
};

/// Attempt backend bring-up. Safe to call repeatedly; the second call is a
/// no-op that returns the cached status. Never throws.
GpuStatus init();

/// Tear down. Safe to call without init.
void shutdown();

/// Cached status from the last init() (Disabled before the first call).
GpuStatus status();

/// Device facts from the last init(). All zeros when not Ready.
const GpuInfo& info();

/// True only when offload is permitted: status() == Ready.
bool use_gpu();

/// No-op compute round trip. Dispatches the embedded `noop` shader (writes
/// 1u from invocation 0 into a freshly zeroed storage buffer) and reads the
/// buffer back on the host. Returns `{false, ...}` without touching the
/// device when the backend is not Ready, so the CPU fallback needs no
/// second code path. `submit_ns` covers queue-submit through fence-wait;
/// `readback` must be exactly 1 on success.
struct NoopResult {
  bool ok = false;
  std::uint64_t submit_ns = 0;
  std::uint32_t readback = 0;
};

NoopResult run_noop();

/// Q8_0 matrix-vector product on the GPU: y = Wx, one H2D (input) + one D2H
/// (output) around a single dispatch. `weights` is a row-major Q8_0 matrix
/// (32-element blocks, int8 quants + fp16 scale, `row_stride_bytes` between
/// row starts); `rows` x `cols` with cols a multiple of 32. The weight copy
/// is uploaded to device-local memory once per (pointer, bytes) key and
/// reused across calls. Returns `{false}` without touching `output` on any
/// backend failure, so the caller always keeps its CPU result. Times are for
/// the transfer accounting beside every speedup claim.
struct GemvQ8Result {
  bool ok = false;
  std::uint64_t h2d_ns = 0;
  std::uint64_t submit_ns = 0;
  std::uint64_t d2h_ns = 0;
  // Host-side per-dispatch work: descriptor/buffer setup and command
  // recording (excludes the weight upload, counted in h2d_ns once).
  std::uint64_t alloc_ns = 0;
};

GemvQ8Result gemv_q8(const void* weights, std::size_t weight_bytes, int rows,
                     int cols, std::size_t row_stride_bytes,
                     const float* input, float* output);

// Batch form of gemv_q8 for prefill/verify/continuous batching: `tokens`
// rows of input (row-major, `cols` f32 each) against the same matrix, one
// row of output per token. Same residency, fallback and accounting rules;
// weights stream once per upload no matter how many calls share them.
using GemmQ8Result = GemvQ8Result;

GemmQ8Result gemm_q8(const void* weights, std::size_t weight_bytes, int rows,
                     int cols, std::size_t row_stride_bytes,
                     const float* input, float* output, int tokens);

/// Pure predicates, testable without a GPU ---------------------------------
/// Batch-matvec offload wish (tokens in the call). The caller conjoins the
/// type/mode guards (F32/Q8_0 only, !scalar, attention != Scalar); this
/// answers only the size question.
inline bool want_batch_gpu(int tokens) {
  return tokens >= GpuThresholds::kMinBatchTokens;
}

/// GQA-attention offload wish. Mirrors uses_gqa(): needs more than one
/// worker thread and a span at or past the grouped schedule floor.
inline bool want_gqa_gpu(int span, int threads) {
  return threads > 1 && span >= GpuThresholds::kGqaMinSpan;
}

/// Diagnostic logits transfer for one decode token (device -> host).
inline std::size_t logits_bytes(int vocab_size) {
  if (vocab_size <= 0) {
    return 0;
  }
  return static_cast<std::size_t>(vocab_size) * sizeof(float);
}

/// Hidden-state transfer for one token in either direction.
inline std::size_t hidden_bytes(int hidden) {
  if (hidden <= 0) {
    return 0;
  }
  return static_cast<std::size_t>(hidden) * sizeof(float);
}

/// Short-decode output-region budget (H2D hidden + D2H logits), bytes.
/// Sync cost (~1 vkQueueSubmit+Fence) is accounted separately in bench logs.
inline std::size_t short_decode_output_bytes(int hidden, int vocab_size) {
  return hidden_bytes(hidden) + logits_bytes(vocab_size);
}

/// Full placement predicate for one batched matvec (Stage-2 §5 conjunctions,
/// minus backend readiness, which the dispatcher adds via use_gpu()).
/// `supported_type`: the weight is Q8_0 or F32 (K-quants and VNNI-quantized
/// rows stay on the CPU kernels). `clean_flags`: no VNNI activations, no
/// scalar-oracle arm, attention schedule is not forced Scalar.
inline bool eligible_batch(bool requested, int tokens, bool supported_type,
                           bool clean_flags) {
  return requested && want_batch_gpu(tokens) && supported_type && clean_flags;
}

/// Decode-output offload wish: the single-token 151936-wide projection.
/// Same conjunctions as the batch rule minus the token floor (one H2D +
/// one D2H around a single dispatch still wins ~5 ms on this box).
inline bool eligible_decode_output(bool requested, bool supported_type,
                                   bool clean_flags) {
  return requested && supported_type && clean_flags;
}
/// GQA-attention offload predicate. `follows_schedule`: the caller's
/// uses_gqa() already holds for this span (grouped schedule, threads > 1).
/// `dense_only`: no sparse subset, no scalar-oracle arm, no forced Scalar.
inline bool eligible_gqa(bool requested, int span, int threads,
                         bool follows_schedule, bool dense_only) {
  return requested && want_gqa_gpu(span, threads) && follows_schedule &&
         dense_only;
}

}  // namespace tge::gpu
