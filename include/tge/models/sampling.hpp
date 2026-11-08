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
  /// How many recent history tokens the penalty sees. 0 means the whole
  /// history. Each distinct token is penalised once, not once per occurrence.
  int penalty_window = 64;
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

/// Writes the raw exp((v - max) / temperature) values into `out` (which must
/// hold `count` floats) and returns their sum — the engine convention shared
/// with the no-cut path, which normalizes by the sum during the scan.
/// Dispatches to the AVX2 shared-exp kernel when available, else the scalar
/// oracle; sampled draws shift at the ~1e-6 level (chi-square gated, never
/// bit-identical across ISAs). Greedy decoding never calls it.
float softmax_probabilities(const float* logits, int count, float temperature,
                            float* out);

/// Categorical draw from `probabilities` (need not sum to exactly 1).
/// Bit-identical to the no-cut scan of `sample_token`.
int categorical_sample(const float* probabilities, int count,
                       std::uint32_t& random_state);

/// Applies the repetition penalty in place: each distinct token in the
/// recent `history` window has its score divided (positive) or multiplied
/// (negative) by `params.repeat_penalty`. Shared verbatim by single-token
/// sampling and the speculative accept loop, so both see the same target.
void apply_repeat_penalty(std::vector<float>& scores, int count,
                          const SamplingParams& params, const int* history,
                          int history_size);

/// Throws `std::invalid_argument` for a parameter set that cannot produce a
/// meaningful draw (NaN temperature, negative cuts, non-positive penalty,
/// negative window, null history with size > 0). The documented "off"
/// sentinels pass. Exposed so a caller can reject a bad flag set before a
/// model load instead of aborting inside the first decode step.
void validate_sampling_params(const SamplingParams& params, const int* history,
                              int history_size);

/// One step of the standard distribution-preserving speculation rule for a
/// one-hot draft distribution (prompt lookup proposes a single token).
/// Accepts `draft` with probability q[draft]; on rejection returns a token
/// drawn from norm(max(0, q - onehot(draft))), which is never `draft` itself
/// (unless q[draft] == 1, in which case acceptance is certain). Either way
/// the emitted token follows q exactly, so a verifier using this rule keeps
/// the sampled output distribution unchanged.
struct SpeculativeVerdict {
  bool accepted = false;
  int token = 0;
};

SpeculativeVerdict speculative_accept(const float* probabilities, int count,
                                      int draft, std::uint32_t& random_state);

}  // namespace tge::models
