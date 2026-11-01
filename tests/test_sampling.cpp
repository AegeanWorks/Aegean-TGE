#include <cmath>
#include <cstdint>
#include <vector>

#include "harness.hpp"
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
