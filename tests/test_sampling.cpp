#include <cmath>
#include <cstdint>
#include <vector>

#include "harness.hpp"
#include "tge/cpu/kernels.hpp"
#include "tge/models/sampling.hpp"

namespace {

struct Scratch {
  std::vector<float> scores;
  std::vector<int> order;
  std::uint32_t state = 12345u;
};

int draw(Scratch& scratch, const float* logits, int count,
         const tge::models::SamplingParams& params,
         const std::vector<int>& history = {}) {
  return tge::models::sample_token(logits, count, params, history.data(),
                                   static_cast<int>(history.size()),
                                   scratch.scores, scratch.order,
                                   scratch.state);
}

}  // namespace

TGE_TEST(sampling_temperature_zero_is_greedy) {
  const float logits[4] = {0.5F, 3.25F, -1.0F, 2.0F};
  tge::models::SamplingParams params;  // defaults: greedy
  Scratch scratch;
  for (int attempt = 0; attempt < 8; ++attempt) {
    TGE_CHECK(draw(scratch, logits, 4, params) == 1);
  }
}

TGE_TEST(sampling_top_k_one_is_greedy_at_any_temperature) {
  const float logits[4] = {0.5F, 3.25F, -1.0F, 2.0F};
  tge::models::SamplingParams params;
  params.temperature = 1.0F;
  params.top_k = 1;
  Scratch scratch;
  for (int attempt = 0; attempt < 8; ++attempt) {
    TGE_CHECK(draw(scratch, logits, 4, params) == 1);
  }
}

TGE_TEST(sampling_is_deterministic_for_a_seed) {
  const float logits[5] = {1.0F, 1.0F, 1.0F, 1.0F, 1.0F};
  tge::models::SamplingParams params;
  params.temperature = 1.0F;

  Scratch first;
  Scratch second;
  first.state = 987654321u;
  second.state = 987654321u;
  for (int step = 0; step < 32; ++step) {
    TGE_CHECK(draw(first, logits, 5, params) == draw(second, logits, 5, params));
  }
}

TGE_TEST(sampling_top_p_keeps_the_dominant_token) {
  // One token holds almost all of the mass, so a 0.5 nucleus must always
  // return it regardless of the random draw.
  const float logits[3] = {8.0F, 0.0F, 0.0F};
  tge::models::SamplingParams params;
  params.temperature = 1.0F;
  params.top_p = 0.5F;
  Scratch scratch;
  for (int attempt = 0; attempt < 64; ++attempt) {
    TGE_CHECK(draw(scratch, logits, 3, params) == 0);
  }
}

TGE_TEST(sampling_min_p_drops_the_tail) {
  const float logits[3] = {8.0F, 0.0F, 0.0F};
  tge::models::SamplingParams params;
  params.temperature = 1.0F;
  params.min_p = 0.5F;
  Scratch scratch;
  for (int attempt = 0; attempt < 64; ++attempt) {
    TGE_CHECK(draw(scratch, logits, 3, params) == 0);
  }
}

TGE_TEST(repetition_penalty_pushes_down_seen_tokens) {
  const float logits[3] = {5.0F, 4.0F, 0.0F};
  tge::models::SamplingParams params;
  params.repeat_penalty = 2.0F;

  Scratch scratch;
  TGE_CHECK(draw(scratch, logits, 3, params) == 0);

  // 5.0 / 2.0 = 2.5, which is now below 4.0.
  const std::vector<int> history = {0};
  TGE_CHECK(draw(scratch, logits, 3, params, history) == 1);
}

TGE_TEST(sampling_handles_a_single_token_vocabulary) {
  const float logits[1] = {-12.0F};
  tge::models::SamplingParams params;
  params.temperature = 0.7F;
  Scratch scratch;
  TGE_CHECK(draw(scratch, logits, 1, params) == 0);
}

TGE_TEST(sampling_top_p_does_not_reward_the_excluded_tail) {
  // p = (0.50, 0.30, 0.15, 0.05); top-p 0.9 keeps the first three, whose
  // renormalised masses are (0.5263, 0.3158, 0.1579, 0). The excluded tail must
  // get zero, not be handed to the weakest kept token.
  const float logits[4] = {std::log(0.50F), std::log(0.30F),
                           std::log(0.15F), std::log(0.05F)};
  tge::models::SamplingParams params;
  params.temperature = 1.0F;
  params.top_p = 0.9F;

  Scratch scratch;
  const int draws = 50000;
  int counts[4] = {0, 0, 0, 0};
  for (int attempt = 0; attempt < draws; ++attempt) {
    counts[draw(scratch, logits, 4, params)] += 1;
  }

  TGE_CHECK(counts[3] == 0);
  TGE_CHECK_NEAR(static_cast<double>(counts[0]) / draws, 0.5263, 0.02);
  TGE_CHECK_NEAR(static_cast<double>(counts[2]) / draws, 0.1579, 0.02);
}

TGE_TEST(repetition_penalty_applies_once_per_distinct_token) {
  const float logits[2] = {2.0F, 0.0F};
  tge::models::SamplingParams params;
  params.repeat_penalty = 1.1F;

  const std::vector<int> history(10, 0);
  Scratch scratch;
  // One penalty leaves 2.0 / 1.1 = 1.82, still above zero. Compounding per
  // occurrence would leave 0.77 and pick the other token.
  TGE_CHECK(draw(scratch, logits, 2, params, history) == 0);
}

TGE_TEST(repetition_penalty_uses_one_recent_occurrence) {
  const float logits[2] = {2.0F, 1.9F};
  tge::models::SamplingParams params;
  params.repeat_penalty = 1.05F;
  params.penalty_window = 1;

  const std::vector<int> history(10, 0);
  Scratch scratch;
  // Only the newest occurrence is in the window and it is penalised once:
  // 2.0 / 1.05 = 1.905 > 1.9. Compounding over the history would pick token 1.
  TGE_CHECK(draw(scratch, logits, 2, params, history) == 0);
}

TGE_TEST(sampling_top_p_lowers_the_floor_for_its_nucleus) {
  // The 1 - top_p floor holds only 80% of the mass here, so the nucleus reaches
  // below it. The candidate floor must be lowered; a full sort would also be
  // correct, but this covers the lowering path.
  const float probabilities[9] = {0.20F, 0.18F, 0.16F, 0.14F, 0.12F,
                                  0.08F, 0.06F, 0.04F, 0.02F};
  float logits[9];
  for (int index = 0; index < 9; ++index) {
    logits[index] = std::log(probabilities[index]);
  }

  tge::models::SamplingParams params;
  params.temperature = 1.0F;
  params.top_p = 0.9F;

  Scratch scratch;
  const int draws = 100000;
  int counts[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0};
  for (int attempt = 0; attempt < draws; ++attempt) {
    counts[draw(scratch, logits, 9, params)] += 1;
  }

  // The nucleus is the first seven entries, renormalised to 0.94.
  TGE_CHECK(counts[7] == 0);
  TGE_CHECK(counts[8] == 0);
  TGE_CHECK_NEAR(static_cast<double>(counts[0]) / draws, 0.20 / 0.94, 0.01);
  TGE_CHECK_NEAR(static_cast<double>(counts[6]) / draws, 0.06 / 0.94, 0.01);
}

TGE_TEST(softmax_helper_matches_hand_computation) {
  // Engine convention (shared with the no-cut path): out holds the raw
  // exp((v - max) / T) values and the return is their sum; callers
  // normalize by the sum instead of paying a second pass.
  const float logits[3] = {1.0F, 2.0F, 3.0F};
  float probs[3] = {0.0F, 0.0F, 0.0F};
  const float total = tge::models::softmax_probabilities(logits, 3, 1.0F, probs);
  const double e0 = std::exp(-2.0);
  const double e1 = std::exp(-1.0);
  const double e2 = 1.0;
  const double sum = e0 + e1 + e2;
  TGE_CHECK_NEAR(probs[0], e0, 1e-6);
  TGE_CHECK_NEAR(probs[1], e1, 1e-6);
  TGE_CHECK_NEAR(probs[2], e2, 1e-6);
  TGE_CHECK_NEAR(total, sum, 1e-5);
}

TGE_TEST(categorical_sample_matches_no_cut_scan) {
  // Same probabilities through both doors, same seed: identical draws, so
  // the extraction did not disturb the no-cut path it shares code with.
  const float logits[4] = {0.5F, 3.25F, -1.0F, 2.0F};
  tge::models::SamplingParams params;
  params.temperature = 1.0F;

  Scratch via_sampler;
  via_sampler.state = 424242u;
  float probs[4] = {0.0F, 0.0F, 0.0F, 0.0F};
  tge::models::softmax_probabilities(logits, 4, 1.0F, probs);
  std::uint32_t direct_state = 424242u;
  for (int step = 0; step < 32; ++step) {
    const int expected = draw(via_sampler, logits, 4, params);
    const int direct =
        tge::models::categorical_sample(probs, 4, direct_state);
    TGE_CHECK(direct == expected);
  }
}

TGE_TEST(speculative_accept_rate_matches_draft_probability) {
  // q[draft] = 0.35: about 35% of trials accept, within statistical noise.
  const float probs[4] = {0.35F, 0.30F, 0.25F, 0.10F};
  std::uint32_t state = 0x12345678u;
  const int trials = 20000;
  int accepted = 0;
  for (int trial = 0; trial < trials; ++trial) {
    if (tge::models::speculative_accept(probs, 4, 0, state).accepted) {
      ++accepted;
    }
  }
  TGE_CHECK_NEAR(static_cast<double>(accepted) / trials, 0.35, 0.02);
}

TGE_TEST(speculative_rejection_resamples_the_renormalised_tail) {
  // Draft token 0 holds q = 0.2; conditional on rejection the resample must
  // follow (0.3, 0.5) over tokens 1..2 renormalised to (0.375, 0.625) and
  // never return the draft token itself.
  const float probs[3] = {0.2F, 0.3F, 0.5F};
  std::uint32_t state = 0x87654321u;
  const int trials = 40000;
  int resamples[3] = {0, 0, 0};
  int rejections = 0;
  for (int trial = 0; trial < trials; ++trial) {
    const tge::models::SpeculativeVerdict verdict =
        tge::models::speculative_accept(probs, 3, 0, state);
    if (!verdict.accepted) {
      ++rejections;
      resamples[verdict.token] += 1;
    }
  }
  TGE_CHECK(rejections > 0);
  TGE_CHECK(resamples[0] == 0);
  TGE_CHECK_NEAR(static_cast<double>(resamples[1]) / rejections, 0.375, 0.02);
  TGE_CHECK_NEAR(static_cast<double>(resamples[2]) / rejections, 0.625, 0.02);
}

TGE_TEST(speculative_accept_rejects_bad_arguments_safely) {
  const float probs[2] = {0.5F, 0.5F};
  std::uint32_t state = 1u;
  TGE_CHECK(!tge::models::speculative_accept(nullptr, 2, 0, state).accepted);
  TGE_CHECK(!tge::models::speculative_accept(probs, 0, 0, state).accepted);
  TGE_CHECK(!tge::models::speculative_accept(probs, 2, -1, state).accepted);
  TGE_CHECK(!tge::models::speculative_accept(probs, 2, 2, state).accepted);
}

TGE_TEST(sampler_exp_avx2_matches_scalar) {
  // Random logits in [-3, 3]; the shared Cephes exp holds ~1e-6 relative,
  // so 1e-4 absolute on probabilities is generous but guards regressions.
  std::uint32_t state = 0x5EED5EEDu;
  const auto next = [&state]() {
    state = state * 1664525u + 1013904223u;
    return state;
  };
  for (const int count : {1, 7, 32, 1519}) {
    std::vector<float> logits(static_cast<std::size_t>(count));
    for (int index = 0; index < count; ++index) {
      logits[static_cast<std::size_t>(index)] =
          static_cast<float>(static_cast<int>((next() >> 9) & 0xFFFFu) - 32768) /
          10922.0F;
    }
    std::vector<float> scalar(static_cast<std::size_t>(count));
    std::vector<float> vectorised(static_cast<std::size_t>(count));
    const float total_scalar = tge::cpu::softmax_exp_scalar(
        logits.data(), count, 0.8F, scalar.data());
    const float total_vector = tge::cpu::softmax_exp_avx2(
        logits.data(), count, 0.8F, vectorised.data());
    TGE_CHECK_NEAR(total_vector, total_scalar,
                   1e-4 * (1.0 + (total_scalar >= 0 ? total_scalar : -total_scalar)));
    for (int index = 0; index < count; ++index) {
      TGE_CHECK_NEAR(vectorised[static_cast<std::size_t>(index)],
                     scalar[static_cast<std::size_t>(index)], 1e-4F);
    }
  }
}

TGE_TEST(sampled_no_cut_matches_expected_histogram) {
  // Chi-square-style gate for the vectorised sampler exp: 8-way distribution
  // over 100k draws must land in every bin up to the usual counting noise.
  // If the exp approximation ever biased a bin, this is where it shows.
  const float probabilities[8] = {0.30F, 0.22F, 0.16F, 0.12F,
                                  0.08F, 0.06F, 0.04F, 0.02F};
  float logits[8];
  for (int index = 0; index < 8; ++index) {
    logits[index] = std::log(probabilities[index]);
  }
  tge::models::SamplingParams params;
  params.temperature = 1.0F;

  Scratch scratch;
  scratch.state = 0xC10C1ABu;
  const int draws = 100000;
  int counts[8] = {0, 0, 0, 0, 0, 0, 0, 0};
  for (int attempt = 0; attempt < draws; ++attempt) {
    counts[draw(scratch, logits, 8, params)] += 1;
  }
  for (int index = 0; index < 8; ++index) {
    TGE_CHECK_NEAR(static_cast<double>(counts[index]) / draws,
                   probabilities[index], 0.01);
  }
}
