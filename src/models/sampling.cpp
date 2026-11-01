#include "tge/models/sampling.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <numeric>
#include <unordered_set>

namespace tge::models {
namespace {

float next_random(std::uint32_t& state) {
  // xorshift32: adequate for sampling, and deterministic when seeded.
  state ^= state << 13;
  state ^= state >> 17;
  state ^= state << 5;
  return static_cast<float>(state >> 8) / 16777216.0F;
}

}  // namespace

int sample_token(const float* logits, int count, const SamplingParams& params,
                 const int* history, int history_size,
                 std::vector<float>& scores, std::vector<int>& order,
                 std::uint32_t& random_state) {
  if (count <= 0) {
    return 0;
  }

  scores.assign(logits, logits + count);

  if (params.repeat_penalty > 0.0F && params.repeat_penalty != 1.0F &&
      history != nullptr && history_size > 0) {
    const int begin = params.penalty_window > 0
                          ? std::max(0, history_size - params.penalty_window)
                          : 0;
    // One penalty per distinct token, however often it appears in the window.
    std::unordered_set<int> seen;
    seen.reserve(static_cast<std::size_t>(history_size - begin));
    for (int index = history_size - 1; index >= begin; --index) {
      const int token = history[index];
      if (token < 0 || token >= count || !seen.insert(token).second) {
        continue;
      }
      float& value = scores[static_cast<std::size_t>(token)];
      value = value > 0.0F ? value / params.repeat_penalty
                           : value * params.repeat_penalty;
    }
  }

  if (params.temperature <= 0.0F) {
    int best = 0;
    for (int index = 1; index < count; ++index) {
      if (scores[static_cast<std::size_t>(index)] >
          scores[static_cast<std::size_t>(best)]) {
        best = index;
      }
    }
    return best;
  }

  // The maximum is needed for numerical stability and by min-p. It is found
  // once, without disturbing the order of the scores.
  float maximum = scores[0];
  for (int index = 1; index < count; ++index) {
    maximum = std::max(maximum, scores[static_cast<std::size_t>(index)]);
  }
  const float inverse_temperature = 1.0F / params.temperature;

  // With no cut at all the draw is a plain categorical sample over the whole
  // vocabulary: exponentiate, sum, scan. No ordering is needed, so the
  // full-vocabulary sort that used to run here disappears.
  if (params.top_k <= 0 && params.top_p >= 1.0F && params.min_p <= 0.0F) {
    float total = 0.0F;
    for (int index = 0; index < count; ++index) {
      const float probability = std::exp(
          (scores[static_cast<std::size_t>(index)] - maximum) *
          inverse_temperature);
      scores[static_cast<std::size_t>(index)] = probability;
      total += probability;
    }
    if (!(total > 0.0F)) {
      return 0;
    }
    float target = next_random(random_state) * total;
    for (int index = 0; index < count; ++index) {
      target -= scores[static_cast<std::size_t>(index)];
      if (target <= 0.0F) {
        return index;
      }
    }
    return count - 1;
  }

  const int keep = params.top_k > 0 ? std::min(params.top_k, count) : count;
  order.resize(static_cast<std::size_t>(count));
  std::iota(order.begin(), order.end(), 0);
  const auto better = [&scores](int left, int right) {
    return scores[static_cast<std::size_t>(left)] >
           scores[static_cast<std::size_t>(right)];
  };
  std::partial_sort(order.begin(), order.begin() + keep, order.end(), better);

  float total = 0.0F;
  for (int index = 0; index < keep; ++index) {
    const int token = order[static_cast<std::size_t>(index)];
    const float probability = std::exp(
        (scores[static_cast<std::size_t>(token)] - maximum) *
        inverse_temperature);
    scores[static_cast<std::size_t>(token)] = probability;
    total += probability;
  }

  // min-p removes tokens that are unlikely relative to the best one.
  if (params.min_p > 0.0F) {
    const float floor_value =
        params.min_p * scores[static_cast<std::size_t>(order[0])];
    for (int index = 0; index < keep; ++index) {
      const std::size_t token =
          static_cast<std::size_t>(order[static_cast<std::size_t>(index)]);
      if (scores[token] < floor_value) {
        total -= scores[token];
        scores[token] = 0.0F;
      }
    }
  }

  // Nucleus: keep the shortest prefix whose probability reaches top_p.
  int limit = keep;
  if (params.top_p < 1.0F) {
    float cumulative = 0.0F;
    for (int index = 0; index < keep; ++index) {
      const std::size_t token =
          static_cast<std::size_t>(order[static_cast<std::size_t>(index)]);
      cumulative += scores[token] / total;
      if (cumulative >= params.top_p) {
        limit = index + 1;
        break;
      }
    }
  }

  if (limit <= 0 || total <= 0.0F) {
    return order[0];
  }

  // Draw over the mass that survived the cuts, not over the total. Dividing by
  // the nucleus mass gives the excluded tail zero probability instead of
  // handing it to the weakest kept token.
  float kept_mass = 0.0F;
  for (int index = 0; index < limit; ++index) {
    kept_mass +=
        scores[static_cast<std::size_t>(order[static_cast<std::size_t>(index)])];
  }
  if (!(kept_mass > 0.0F)) {
    return order[0];
  }

  float target = next_random(random_state) * kept_mass;
  for (int index = 0; index < limit; ++index) {
    const int token = order[static_cast<std::size_t>(index)];
    target -= scores[static_cast<std::size_t>(token)];
    if (target <= 0.0F) {
      return token;
    }
  }
  return order[static_cast<std::size_t>(limit - 1)];
}

}  // namespace tge::models
