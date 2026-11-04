#include "tge/models/plan.hpp"

#include <algorithm>
#include <sstream>
#include <stdexcept>

namespace tge::models {
namespace {

std::string format_bytes(std::int64_t bytes) {
  std::ostringstream out;
  out.setf(std::ios::fixed);
  if (bytes >= 1024LL * 1024 * 1024) {
    out.precision(2);
    out << (static_cast<double>(bytes) / (1024.0 * 1024.0 * 1024.0)) << " GiB";
  } else if (bytes >= 1024LL * 1024) {
    out.precision(1);
    out << (static_cast<double>(bytes) / (1024.0 * 1024.0)) << " MiB";
  } else {
    out.precision(1);
    out << (static_cast<double>(bytes) / 1024.0) << " KiB";
  }
  return out.str();
}

}  // namespace

ModelPlan analyse(const gguf::File& file, int context) {
  const Qwen2Config config = read_config(file);

  ModelPlan plan;
  plan.block_count = config.block_count;
  plan.kv_dim = config.kv_dim();
  plan.vocab_size = config.vocab_size;
  plan.context = std::min(config.context_length > 0 ? config.context_length
                                                    : context,
                          context);

  for (const gguf::TensorInfo& info : file.tensors()) {
    plan.parameter_bytes += static_cast<std::int64_t>(info.bytes);

    // Decoding reads every weight once per token except the embedding table,
    // where a single row is touched.
    if (info.name != "token_embd.weight") {
      plan.decode_bytes_per_token += static_cast<std::int64_t>(info.bytes);
    }

    auto share = std::find_if(plan.formats.begin(), plan.formats.end(),
                              [&info](const FormatShare& entry) {
                                return entry.type == info.type;
                              });
    if (share == plan.formats.end()) {
      plan.formats.push_back(FormatShare{info.type, 1, info.bytes});
    } else {
      share->tensors += 1;
      share->bytes += info.bytes;
    }
  }

  std::sort(plan.formats.begin(), plan.formats.end(),
            [](const FormatShare& left, const FormatShare& right) {
              return left.bytes > right.bytes;
            });

  // f32 for both K and V, per layer, per token.
  plan.kv_bytes_per_token =
      static_cast<std::int64_t>(config.block_count) * config.kv_dim() * 4 * 2;
  plan.kv_bytes_per_session = plan.kv_bytes_per_token * plan.context;
  return plan;
}

PlacementDecision place(const ModelPlan& plan, std::int64_t usable_bytes,
                        int requested_context, int sessions) {
  PlacementDecision decision;
  decision.usable_bytes = usable_bytes;
  decision.headroom_after_weights =
      usable_bytes - plan.parameter_bytes;
  decision.weights_fit = decision.headroom_after_weights >= 0;

  if (!decision.weights_fit) {
    decision.reason =
        "the weights alone exceed the usable budget; the model does not fit";
    return decision;
  }

  if (requested_context <= 0 || plan.kv_bytes_per_token <= 0) {
    decision.reason = "no usable context";
    return decision;
  }

  const std::int64_t per_session =
      plan.kv_bytes_per_token * requested_context;
  decision.kv_bytes_per_session = per_session;

  const int wanted = std::max(1, sessions);
  const std::int64_t needed =
      static_cast<std::int64_t>(per_session) * wanted;

  if (needed <= decision.headroom_after_weights) {
    decision.max_context = requested_context;
    decision.max_sessions = wanted;
    decision.reason = "requested context and sessions both fit";
    return decision;
  }

  // Shrink the thing the caller asked for first, then the session count.
  const int by_sessions = static_cast<int>(decision.headroom_after_weights /
                                           per_session);
  const std::int64_t per_session_budget =
      decision.headroom_after_weights / wanted;
  const int by_context =
      static_cast<int>(per_session_budget / plan.kv_bytes_per_token);

  decision.max_context = std::min(requested_context, std::max(0, by_context));
  decision.max_sessions =
      by_context >= 1 ? wanted : std::max(0, by_sessions);

  std::ostringstream reason;
  reason << "requested " << requested_context << " tokens x " << wanted
         << " session(s) needs " << format_bytes(needed) << " of KV but only "
         << format_bytes(decision.headroom_after_weights) << " is free";
  decision.reason = reason.str();
  return decision;
}

double projected_tokens_per_second(const ModelPlan& plan,
                                   double bytes_per_second) {
  if (plan.decode_bytes_per_token <= 0 || bytes_per_second <= 0.0) {
    return 0.0;
  }
  return bytes_per_second /
         static_cast<double>(plan.decode_bytes_per_token);
}

std::string describe(const ModelPlan& plan) {
  std::ostringstream out;
  out << "  weights          : " << format_bytes(plan.parameter_bytes)
      << " across all tensors\n";
  out << "  read per token   : " << format_bytes(plan.decode_bytes_per_token)
      << " (embedding row excluded)\n";
  out << "  kv per token     : " << format_bytes(plan.kv_bytes_per_token)
      << " over " << plan.block_count << " layers x " << plan.kv_dim
      << " values\n";
  out << "  kv at " << plan.context << " tokens: "
      << format_bytes(plan.kv_bytes_per_session) << "\n";
  out << "  vocab            : " << plan.vocab_size << "\n";
  for (const FormatShare& share : plan.formats) {
    out << "  " << gguf::type_name(share.type) << "  " << share.tensors
        << " tensors, " << format_bytes(static_cast<std::int64_t>(share.bytes))
        << "\n";
  }
  return out.str();
}

std::string describe(const PlacementDecision& decision) {
  std::ostringstream out;
  out << "  usable budget    : " << format_bytes(decision.usable_bytes) << "\n";
  out << "  after weights    : "
      << format_bytes(decision.headroom_after_weights) << "\n";
  out << "  kv per session   : "
      << format_bytes(decision.kv_bytes_per_session) << "\n";
  out << "  fits             : " << decision.max_context << " tokens x "
      << decision.max_sessions << " session(s)\n";
  out << "  verdict          : " << decision.reason << "\n";
  return out.str();
}

}  // namespace tge::models
