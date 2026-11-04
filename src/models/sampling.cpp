#include "tge/models/sampling.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <numeric>
#include <stdexcept>
#include <unordered_set>

#include "tge/core/dispatch.hpp"
#include "tge/cpu/kernels.hpp"

namespace tge::models {
namespace {

float next_random(std::uint32_t& state) {
  // xorshift32: adequate for sampling, and deterministic when seeded.
  if (state == 0) {
    // xorshift32 degenerates at 0 (every draw is exactly 0), so a
    // --seed 0 run would sample token 0 at every stochastic step.
    // Mix once to keep a reproducible but non-trivial stream.
    state = 0x9E3779B9u;
  }
  state ^= state << 13;
  state ^= state >> 17;
  state ^= state << 5;
  return static_cast<float>(state >> 8) / 16777216.0F;
}

}  // namespace

void apply_repeat_penalty(std::vector<float>& scores, int count,
                          const SamplingParams& params, const int* history,
                          int history_size) {
  if (!(params.repeat_penalty > 0.0F && params.repeat_penalty != 1.0F &&
        history != nullptr && history_size > 0)) {
    return;
  }
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

float softmax_probabilities(const float* logits, int count, float temperature,
                            float* out) {
  if (logits == nullptr || out == nullptr || count <= 0 ||
      !(temperature > 0.0F)) {
    return 0.0F;
  }
  // Fidelity note: the AVX2 path changes sampled draws at the ~1e-6 level
  // (chi-square gated, never bit-identical). Greedy decoding never reaches
  // here — temperature <= 0 returns upstream — so greedy ids cannot move.
  const auto features = cpu::detect().features;
  if (features.avx2 && features.fma) {
    return cpu::softmax_exp_avx2(logits, count, temperature, out);
  }
  return cpu::softmax_exp_scalar(logits, count, temperature, out);
}

int categorical_sample(const float* probabilities, int count,
                       std::uint32_t& random_state) {
  if (probabilities == nullptr || count <= 0) {
    return 0;
  }
  float total = 0.0F;
  for (int index = 0; index < count; ++index) {
    total += probabilities[index];
  }
  if (!(total > 0.0F)) {
    return 0;
  }
  // Same scan as the no-cut path: walk down from a uniform draw.
  float target = next_random(random_state) * total;
  for (int index = 0; index < count; ++index) {
    target -= probabilities[index];
    if (target <= 0.0F) {
      return index;
    }
  }
  return count - 1;
}

void validate_sampling_params(const SamplingParams& params,
                              const int* history, int history_size) {
  // Loud errors for meaningless inputs that used to degrade silently into
  // biased draws. The documented "off" sentinels (top_k/top_p/min_p at
  // defaults, penalty 1.0, window 0) all pass.
  if (std::isnan(params.temperature)) {
    throw std::invalid_argument("sample_token: NaN temperature");
  }
  if (params.top_k < 0) {
    throw std::invalid_argument("sample_token: negative top_k");
  }
  if (std::isnan(params.top_p) || params.top_p < 0.0F ||
      params.top_p > 1.0F) {
    throw std::invalid_argument("sample_token: top_p outside [0, 1]");
  }
  if (std::isnan(params.min_p) || params.min_p < 0.0F ||
      params.min_p > 1.0F) {
    throw std::invalid_argument("sample_token: min_p outside [0, 1]");
  }
  if (!(params.repeat_penalty > 0.0F) ||
      !std::isfinite(params.repeat_penalty)) {
    throw std::invalid_argument("sample_token: repeat_penalty must be finite > 0");
  }
  if (params.penalty_window < 0) {
    throw std::invalid_argument("sample_token: negative penalty_window");
  }
  if (history_size < 0) {
    throw std::invalid_argument("sample_token: negative history_size");
  }
  if (history_size > 0 && history == nullptr) {
    throw std::invalid_argument("sample_token: null history with size > 0");
  }
}

SpeculativeVerdict speculative_accept(const float* probabilities, int count,
                                      int draft, std::uint32_t& random_state) {
  SpeculativeVerdict verdict;
  if (probabilities == nullptr || count <= 0 || draft < 0 ||
      draft >= count) {
    return verdict;
  }
  float total = 0.0F;
  for (int index = 0; index < count; ++index) {
    total += probabilities[index];
  }
  if (!(total > 0.0F)) {
    return verdict;
  }
  const float accept = probabilities[draft] / total;
  if (next_random(random_state) < accept) {
    verdict.accepted = true;
    verdict.token = draft;
    return verdict;
  }
  // Rejection: draw from norm(max(0, q - onehot(draft))). With a one-hot
  // draft distribution every other token keeps its weight; the drafted
  // token keeps max(0, q[draft] - total) = 0 for normalized inputs, so the
  // general form below (which also tolerates unnormalized ones) never
  // returns the draft token after a genuine rejection.
  float remainder = total - probabilities[draft];
  if (remainder < 0.0F) {
    remainder = 0.0F;
  }
  if (!(remainder > 0.0F)) {
    // Degenerate: all mass was on the draft token, yet the draw rejected
    // (only possible for unnormalized inputs). Fall back to the draft.
    verdict.accepted = true;
    verdict.token = draft;
    return verdict;
  }
  float target = next_random(random_state) * remainder;
  for (int index = 0; index < count; ++index) {
    float weight = probabilities[index];
    if (index == draft && weight > total) {
      weight -= total;
    } else if (index == draft) {
      weight = 0.0F;
    }
    target -= weight;
    if (target <= 0.0F) {
      verdict.token = index;
      return verdict;
    }
  }
  verdict.token = count - 1 == draft ? 0 : count - 1;
  return verdict;
}

int sample_token(const float* logits, int count, const SamplingParams& params,
                 const int* history, int history_size,
                 std::vector<float>& scores, std::vector<int>& order,
                 std::uint32_t& random_state) {
  if (logits == nullptr) {
    throw std::invalid_argument("sample_token: null logits");
  }
  if (count <= 0) {
    return 0;
  }
  validate_sampling_params(params, history, history_size);

  scores.assign(logits, logits + count);
  apply_repeat_penalty(scores, count, params, history, history_size);

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
  // full-vocabulary sort that used to run here disappears. Shared with the
  // speculative sampler: same ops in the same order, bit-identical results.
  if (params.top_k <= 0 && params.top_p >= 1.0F && params.min_p <= 0.0F) {
    const float total = softmax_probabilities(
        scores.data(), count, params.temperature, scores.data());
    if (!(total > 0.0F)) {
      return 0;
    }
    return categorical_sample(scores.data(), count, random_state);
  }

  const auto better = [&scores](int left, int right) {
    return scores[static_cast<std::size_t>(left)] >
           scores[static_cast<std::size_t>(right)];
  };

  // Without top-k, a top-p or min-p cut still needs the largest probabilities
  // first. Sort only the tokens that could survive a cut: top-p skips tokens
  // below `1 - top_p` unless the nucleus reaches that far, in which case the
  // floor is lowered until the whole nucleus is included. min-p keeps
  // everything above its own floor.
  if (params.top_k <= 0 && (params.top_p < 1.0F || params.min_p > 0.0F)) {
    const bool min_filter = params.min_p > 0.0F;

    float total_all = 0.0F;
    float maximum_probability = -1.0F;
    int maximum_index = 0;
    for (int index = 0; index < count; ++index) {
      const float probability = std::exp(
          (scores[static_cast<std::size_t>(index)] - maximum) *
          inverse_temperature);
      scores[static_cast<std::size_t>(index)] = probability;
      total_all += probability;
      if (probability > maximum_probability) {
        maximum_probability = probability;
        maximum_index = index;
      }
    }

    const float min_floor = min_filter ? params.min_p : 2.0F;
    const float nucleus_floor =
        params.top_p < 1.0F ? 1.0F - params.top_p : 2.0F;
    float threshold = std::min(min_floor, nucleus_floor);

    float candidate_mass = 0.0F;
    for (;;) {
      order.clear();
      candidate_mass = 0.0F;
      for (int index = 0; index < count; ++index) {
        const float probability = scores[static_cast<std::size_t>(index)];
        if (probability >= threshold) {
          order.push_back(index);
          candidate_mass += probability;
        }
      }
      // min-p's candidates are the survivors by construction. For top-p, the
      // nucleus must fit above the floor; lower it until it does. At zero the
      // scan includes every token, which is the exact fallback.
      if (min_filter || params.top_p >= 1.0F ||
          candidate_mass >= params.top_p * total_all || threshold <= 0.0F) {
        break;
      }
      threshold *= 0.5F;
    }

    if (order.empty()) {
      return maximum_index;
    }

    std::sort(order.begin(), order.end(), better);

    float total = total_all;
    if (min_filter) {
      const float floor_value =
          params.min_p * scores[static_cast<std::size_t>(order[0])];
      std::size_t kept = 0;
      total = 0.0F;
      for (const int token : order) {
        const float probability = scores[static_cast<std::size_t>(token)];
        if (probability >= floor_value) {
          total += probability;
          order[kept++] = token;
        }
      }
      order.resize(kept);
    }

    int limit = static_cast<int>(order.size());
    if (limit <= 0 || total <= 0.0F) {
      return maximum_index;
    }
    if (params.top_p < 1.0F) {
      float cumulative = 0.0F;
      for (int index = 0; index < limit; ++index) {
        cumulative +=
            scores[static_cast<std::size_t>(
                order[static_cast<std::size_t>(index)])] /
            total;
        if (cumulative >= params.top_p) {
          limit = index + 1;
          break;
        }
      }
    }

    float kept_mass = 0.0F;
    for (int index = 0; index < limit; ++index) {
      kept_mass += scores[static_cast<std::size_t>(
          order[static_cast<std::size_t>(index)])];
    }
    if (!(kept_mass > 0.0F)) {
      return maximum_index;
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

  const int keep = params.top_k > 0 ? std::min(params.top_k, count) : count;
  order.resize(static_cast<std::size_t>(count));
  std::iota(order.begin(), order.end(), 0);
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

  // Draw over the mass that survived the cuts, not over the total.
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
