// GPU fallback contract tests ([1/4] bring-up).
//
// Hermetic: no GPU, model or driver required. They pin the only three things
// [1/4] promises: bring-up never crashes and reports a non-Ready status
// without an SDK build; the static thresholds match the Stage-2 revised plan
// (batch >= 8, GQA span >= 256 with threads > 1); the short-decode transfer
// budget formula is exact.

#include <bit>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

#include "harness.hpp"
#include "tge/gpu/backend.hpp"

TGE_TEST(gpu_init_is_safe_and_falls_back) {
  const tge::gpu::GpuStatus first = tge::gpu::init();
  TGE_CHECK(first == tge::gpu::GpuStatus::Disabled ||
            first == tge::gpu::GpuStatus::Unavailable ||
            first == tge::gpu::GpuStatus::Ready);
  // Second call is a cached no-op with the same answer.
  TGE_CHECK(tge::gpu::init() == first);
  TGE_CHECK(tge::gpu::status() == first);
  TGE_CHECK(tge::gpu::use_gpu() == (first == tge::gpu::GpuStatus::Ready));
  tge::gpu::shutdown();
  TGE_CHECK(tge::gpu::status() == tge::gpu::GpuStatus::Disabled);
  TGE_CHECK(!tge::gpu::use_gpu());
  // Shutdown without init is safe; re-init after shutdown works.
  tge::gpu::shutdown();
  const tge::gpu::GpuStatus again = tge::gpu::init();
  TGE_CHECK(again == tge::gpu::GpuStatus::Disabled ||
            again == tge::gpu::GpuStatus::Unavailable ||
            again == tge::gpu::GpuStatus::Ready);
  tge::gpu::shutdown();
}

TGE_TEST(gpu_batch_threshold_is_static) {
  TGE_CHECK(!tge::gpu::want_batch_gpu(1));
  TGE_CHECK(!tge::gpu::want_batch_gpu(6));
  TGE_CHECK(!tge::gpu::want_batch_gpu(7));
  TGE_CHECK(tge::gpu::want_batch_gpu(8));
  TGE_CHECK(tge::gpu::want_batch_gpu(64));
  // The speculative verifier (6 tokens) stays CPU by construction.
  TGE_CHECK(!tge::gpu::want_batch_gpu(6));
}

TGE_TEST(gpu_gqa_threshold_mirrors_uses_gqa) {
  TGE_CHECK(!tge::gpu::want_gqa_gpu(255, 6));
  TGE_CHECK(tge::gpu::want_gqa_gpu(256, 6));
  TGE_CHECK(tge::gpu::want_gqa_gpu(2048, 6));
  // Single-thread runs keep the per-head path: no offload.
  TGE_CHECK(!tge::gpu::want_gqa_gpu(2048, 1));
  TGE_CHECK(!tge::gpu::want_gqa_gpu(256, 0));
}

TGE_TEST(gpu_transfer_budget_is_exact) {
  // Qwen2.5-0.5B geometry: hidden 896 f32, vocab 151936.
  constexpr int kHidden = 896;
  constexpr int kVocab = 151936;
  TGE_CHECK(tge::gpu::hidden_bytes(kHidden) ==
            static_cast<std::size_t>(3584));
  TGE_CHECK(tge::gpu::logits_bytes(kVocab) ==
            static_cast<std::size_t>(607744));
  TGE_CHECK(tge::gpu::short_decode_output_bytes(kHidden, kVocab) ==
            static_cast<std::size_t>(611328));
}

TGE_TEST(gpu_eligible_batch_conjunctions) {
  // Off by default: the request bit gates everything.
  TGE_CHECK(!tge::gpu::eligible_batch(false, 64, true, true));
  // Below the 8-token floor: the 6-token speculative verifier stays CPU.
  TGE_CHECK(!tge::gpu::eligible_batch(true, 1, true, true));
  TGE_CHECK(!tge::gpu::eligible_batch(true, 6, true, true));
  TGE_CHECK(!tge::gpu::eligible_batch(true, 7, true, true));
  // Winners: prefill chunks, forward_batch, continuous batching.
  TGE_CHECK(tge::gpu::eligible_batch(true, 8, true, true));
  TGE_CHECK(tge::gpu::eligible_batch(true, 64, true, true));
  // K-quants (and any non-Q8/F32) stay on the dequant path.
  TGE_CHECK(!tge::gpu::eligible_batch(true, 64, false, true));
  // VNNI activations, scalar-oracle arm, forced-Scalar schedule stay CPU.
  TGE_CHECK(!tge::gpu::eligible_batch(true, 64, true, false));
}

TGE_TEST(gpu_eligible_gqa_conjunctions) {
  TGE_CHECK(!tge::gpu::eligible_gqa(false, 2048, 6, true, true));
  // Below the span floor, or single-threaded: per-head CPU path.
  TGE_CHECK(!tge::gpu::eligible_gqa(true, 255, 6, true, true));
  TGE_CHECK(!tge::gpu::eligible_gqa(true, 2048, 1, true, true));
  // Schedule not grouped (short context, forced head): no offload.
  TGE_CHECK(!tge::gpu::eligible_gqa(true, 2048, 6, false, true));
  // Sparse subset or scalar arm: exact CPU paths, never offloaded.
  TGE_CHECK(!tge::gpu::eligible_gqa(true, 2048, 6, true, false));
  // Winner: long-decode grouped attention, dense.
  TGE_CHECK(tge::gpu::eligible_gqa(true, 256, 6, true, true));
  TGE_CHECK(tge::gpu::eligible_gqa(true, 2048, 6, true, true));
}

TGE_TEST(gpu_noop_roundtrip) {
  // Without a ready backend this is a silent no-op; with one (SDK build on
  // a Vulkan box) it proves the whole submit path and the exact readback.
  const tge::gpu::NoopResult result = tge::gpu::run_noop();
  if (!tge::gpu::use_gpu()) {
    TGE_CHECK(!result.ok);
    return;
  }
  TGE_REQUIRE(result.ok);
  TGE_CHECK(result.readback == 1u);
  TGE_CHECK(result.submit_ns > 0u);
}

namespace {

// Minimal Q8_0 row writer for the gemv test: 32 int8 + fp16 scale per block,
// matching the host ggml layout the shader reads.
std::uint16_t float_to_half_bits(float value) {
  const std::uint32_t bits =
      static_cast<std::uint32_t>(std::bit_cast<std::uint32_t>(value));
  const std::uint32_t sign = (bits >> 31u) & 0x1u;
  const int exponent =
      static_cast<int>((bits >> 23u) & 0xFFu) - 112;  // 127 - 15
  const std::uint32_t mantissa = bits & 0x7FFFFFu;
  if (exponent <= 0) {
    return static_cast<std::uint16_t>(sign << 15u);
  }
  if (exponent >= 31) {
    return static_cast<std::uint16_t>((sign << 15u) | 0x7BFFu);
  }
  return static_cast<std::uint16_t>((sign << 15u) |
                                    (static_cast<std::uint32_t>(exponent)
                                     << 10u) |
                                    (mantissa >> 13u));
}

}  // namespace

TGE_TEST(gpu_gemv_q8_matches_scalar_reference) {
  // 192 rows x 64 cols (2 blocks/row, 3 workgroups); deterministic PRNG.
  constexpr int kRows = 192;
  constexpr int kCols = 64;
  constexpr int kBlocks = kCols / 32;
  constexpr std::size_t kRowBytes =
      static_cast<std::size_t>(kBlocks) * 34u;
  std::vector<std::uint8_t> weights(static_cast<std::size_t>(kRows) *
                                    kRowBytes);
  std::vector<float> input(kCols);
  std::uint32_t state = 0x12345678u;
  auto next = [&]() {
    state = state * 1664525u + 1013904223u;
    return state;
  };
  for (int row = 0; row < kRows; ++row) {
    for (int block = 0; block < kBlocks; ++block) {
      // True host Q8_0 block order: fp16 scale first, then 32 int8 quants.
      std::uint8_t* base =
          weights.data() +
          (static_cast<std::size_t>(row) * kBlocks + block) * 34u;
      const float scale =
          0.005F + static_cast<float>(next() % 1000u) / 1000.0F * 0.045F;
      const std::uint16_t half = float_to_half_bits(scale);
      base[0] = static_cast<std::uint8_t>(half & 0xFFu);
      base[1] = static_cast<std::uint8_t>((half >> 8u) & 0xFFu);
      for (int lane = 0; lane < 32; ++lane) {
        base[2 + lane] = static_cast<std::uint8_t>(next() & 0xFFu);
      }
    }
  }
  for (int col = 0; col < kCols; ++col) {
    input[col] = static_cast<float>(next() % 2000u) / 1000.0F - 1.0F;
  }

  // Scalar oracle in host order (scale first, then quants; per-block dot
  // then scale, matching q8_gemv_scalar and the shader).
  std::vector<float> expected(kRows, 0.0F);
  for (int row = 0; row < kRows; ++row) {
    float sum = 0.0F;
    for (int block = 0; block < kBlocks; ++block) {
      const std::uint8_t* base =
          weights.data() +
          (static_cast<std::size_t>(row) * kBlocks + block) * 34u;
      const std::uint16_t half =
          static_cast<std::uint16_t>(base[0] | (base[1] << 8u));
      const std::uint32_t sign = (half >> 15u) & 0x1u;
      const std::uint32_t exp = (half >> 10u) & 0x1Fu;
      const std::uint32_t mant = half & 0x3FFu;
      const float scale = static_cast<float>(
          (static_cast<double>(mant) / 1024.0 + 1.0) *
          std::ldexp(1.0, static_cast<int>(exp) - 15) *
          (sign != 0u ? -1.0 : 1.0));
      float block_dot = 0.0F;
      for (int lane = 0; lane < 32; ++lane) {
        float q = static_cast<float>(base[2 + lane]);
        if (q >= 128.0F) {
          q -= 256.0F;
        }
        block_dot += q * input[block * 32 + lane];
      }
      sum += block_dot * scale;
    }
    expected[row] = sum;
  }

  std::vector<float> actual(kRows, std::numeric_limits<float>::quiet_NaN());
  const tge::gpu::GemvQ8Result result = tge::gpu::gemv_q8(
      weights.data(), weights.size(), kRows, kCols, kRowBytes, input.data(),
      actual.data());
  if (!tge::gpu::use_gpu()) {
    // No backend: silent no-op, output untouched.
    TGE_CHECK(!result.ok);
    for (float value : actual) {
      TGE_CHECK(value != value);  // still NaN
    }
    return;
  }
  TGE_REQUIRE(result.ok);
  float worst = 0.0F;
  int top_ref = 0;
  int top_got = 0;
  for (int row = 0; row < kRows; ++row) {
    worst = std::max(worst, std::fabs(actual[row] - expected[row]));
    if (expected[row] > expected[top_ref]) {
      top_ref = row;
    }
    if (actual[row] > actual[top_got]) {
      top_got = row;
    }
  }
  // fp32 per-block dots in the same order on both sides; contraction
  // differences stay orders of magnitude below any logit that matters.
  TGE_CHECK(worst <= 1e-3F);
  TGE_CHECK(top_ref == top_got);
}

TGE_TEST(gpu_gemm_q8_matches_scalar_reference) {
  // 3 tokens x 96 rows x 64 cols (2 blocks/row); same true Q8_0 layout as
  // the gemv test. Guards the batch indexing (token-major I/O).
  constexpr int kTokens = 3;
  constexpr int kRows = 96;
  constexpr int kCols = 64;
  constexpr int kBlocks = kCols / 32;
  constexpr std::size_t kRowBytes =
      static_cast<std::size_t>(kBlocks) * 34u;
  std::vector<std::uint8_t> weights(static_cast<std::size_t>(kRows) *
                                    kRowBytes);
  std::vector<float> input(static_cast<std::size_t>(kTokens) * kCols);
  std::uint32_t state = 0x76543210u;
  auto next = [&]() {
    state = state * 1664525u + 1013904223u;
    return state;
  };
  for (int row = 0; row < kRows; ++row) {
    for (int block = 0; block < kBlocks; ++block) {
      std::uint8_t* base =
          weights.data() +
          (static_cast<std::size_t>(row) * kBlocks + block) * 34u;
      const float scale =
          0.005F + static_cast<float>(next() % 1000u) / 1000.0F * 0.045F;
      const std::uint16_t half = float_to_half_bits(scale);
      base[0] = static_cast<std::uint8_t>(half & 0xFFu);
      base[1] = static_cast<std::uint8_t>((half >> 8u) & 0xFFu);
      for (int lane = 0; lane < 32; ++lane) {
        base[2 + lane] = static_cast<std::uint8_t>(next() & 0xFFu);
      }
    }
  }
  for (float& value : input) {
    value = static_cast<float>(next() % 2000u) / 1000.0F - 1.0F;
  }

  std::vector<float> expected(static_cast<std::size_t>(kTokens) * kRows,
                              0.0F);
  for (int token = 0; token < kTokens; ++token) {
    for (int row = 0; row < kRows; ++row) {
      float sum = 0.0F;
      for (int block = 0; block < kBlocks; ++block) {
        const std::uint8_t* base =
            weights.data() +
            (static_cast<std::size_t>(row) * kBlocks + block) * 34u;
        const std::uint16_t half =
            static_cast<std::uint16_t>(base[0] | (base[1] << 8u));
        const std::uint32_t sign = (half >> 15u) & 0x1u;
        const std::uint32_t exp = (half >> 10u) & 0x1Fu;
        const std::uint32_t mant = half & 0x3FFu;
        const float scale = static_cast<float>(
            (static_cast<double>(mant) / 1024.0 + 1.0) *
            std::ldexp(1.0, static_cast<int>(exp) - 15) *
            (sign != 0u ? -1.0 : 1.0));
        float block_dot = 0.0F;
        for (int lane = 0; lane < 32; ++lane) {
          float q = static_cast<float>(base[2 + lane]);
          if (q >= 128.0F) {
            q -= 256.0F;
          }
          block_dot +=
              q * input[token * kCols + block * 32 + lane];
        }
        sum += block_dot * scale;
      }
      expected[static_cast<std::size_t>(token) * kRows + row] = sum;
    }
  }

  std::vector<float> actual(expected.size(),
                            std::numeric_limits<float>::quiet_NaN());
  const tge::gpu::GemmQ8Result result = tge::gpu::gemm_q8(
      weights.data(), weights.size(), kRows, kCols, kRowBytes, input.data(),
      actual.data(), kTokens);
  if (!tge::gpu::use_gpu()) {
    TGE_CHECK(!result.ok);
    return;
  }
  TGE_REQUIRE(result.ok);
  float worst = 0.0F;
  for (std::size_t index = 0; index < expected.size(); ++index) {
    worst = std::max(worst, std::fabs(actual[index] - expected[index]));
  }
  TGE_CHECK(worst <= 1e-3F);
}

TGE_TEST(gpu_transfer_budgets_reject_negative_sizes) {
  // Regression: negative int cast to size_t wrapped to ~18 EB.
  TGE_CHECK(tge::gpu::logits_bytes(-1) == 0);
  TGE_CHECK(tge::gpu::hidden_bytes(-5) == 0);
  TGE_CHECK(tge::gpu::short_decode_output_bytes(-1, -1) == 0);
  TGE_CHECK(tge::gpu::short_decode_output_bytes(0, 0) == 0);
}
