#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "tge/models/gguf.hpp"
#include "tge/models/qwen2.hpp"

namespace tge::models {

/// One quantized format's share of the model.
struct FormatShare {
  gguf::TensorType type = gguf::TensorType::Unknown;
  std::uint64_t tensors = 0;
  std::uint64_t bytes = 0;
};

/// What a model costs in memory, computed from the GGUF tensor table alone.
///
/// The table states every tensor's shape and type, so the whole budget can be
/// computed without reading a byte of weight data. That is the point: a planner
/// has to answer "does this fit?" before it is safe to allocate anything, and
/// loading first to find out is exactly the failure it exists to prevent.
struct ModelPlan {
  std::int64_t parameter_bytes = 0;  ///< every tensor, as stored
  std::int64_t decode_bytes_per_token = 0;  ///< what one decode step must read
  std::int64_t kv_bytes_per_token = 0;      ///< K and V for one token in f32
  std::int64_t kv_bytes_per_session = 0;    ///< at the requested context
  int block_count = 0;
  int kv_dim = 0;
  int vocab_size = 0;
  int context = 0;
  std::vector<FormatShare> formats;
};

/// Reads the tensor table and the configuration. Throws on a malformed file.
ModelPlan analyse(const gguf::File& file, int context);

/// Turns a plan into an admission decision against real memory.
struct PlacementDecision {
  bool weights_fit = false;
  int max_context = 0;
  int max_sessions = 0;
  std::int64_t usable_bytes = 0;
  std::int64_t headroom_after_weights = 0;
  std::int64_t kv_bytes_per_session = 0;
  std::string reason;  ///< why the numbers came out this way
};

/// `sessions` is the number of concurrent contexts to reserve for; the decision
/// reports how many actually fit, which may be fewer.
PlacementDecision place(const ModelPlan& plan, std::int64_t usable_bytes,
                        int requested_context, int sessions);

/// Projected decode ceiling: bytes per token over sustained bandwidth.
double projected_tokens_per_second(const ModelPlan& plan, double bytes_per_second);

std::string describe(const ModelPlan& plan);
std::string describe(const PlacementDecision& decision);

}  // namespace tge::models
