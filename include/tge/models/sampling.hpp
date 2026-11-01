#pragma once

#include <cstdint>
#include <vector>

namespace tge::models {

/// Sampling controls. The defaults reproduce greedy decoding exactly, so a
/// caller that sets nothing gets the behaviour the benchmarks were measured
/// with.
struct SamplingParams {
  /// 0 selects greedy. Any positive value scales the logits.
  float temperature = 0.0F;
  /// 0 disables the cut; otherwise keep the k highest scores.
  int top_k = 0;
  /// Nucleus cut, applied after top-k. 1.0 disables it.
  float top_p = 1.0F;
  /// Drop tokens below this fraction of the best probability. 0 disables it.
  float min_p = 0.0F;
  /// 1.0 disables it. Values above 1.0 push down tokens already in the history.
  float repeat_penalty = 1.0F;
  /// 0 selects a fixed default seed, so a run without --seed is reproducible.
  std::uint32_t seed = 0;
};

/// Draws one token from `logits`.
///
/// `history` feeds only the repetition penalty. The scratch vectors and the
/// random state are passed in so a decode loop does not allocate per token.
int sample_token(const float* logits, int count, const SamplingParams& params,
                 const int* history, int history_size,
                 std::vector<float>& scores, std::vector<int>& order,
                 std::uint32_t& random_state);

}  // namespace tge::models
