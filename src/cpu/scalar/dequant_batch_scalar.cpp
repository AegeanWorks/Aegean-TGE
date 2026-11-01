#include "tge/cpu/kernels.hpp"

#include <cstddef>
#include <vector>

namespace tge::cpu {

void gemv_dequant_batch_scalar(const DequantBatchArgs& args,
                               const GemvConfig& /*config*/) {
  if (args.rows <= 0 || args.cols <= 0 || args.tokens <= 0) {
    return;
  }

  const std::size_t row_bytes = quant_row_bytes(args.kind, args.cols);
  std::vector<float> weights(static_cast<std::size_t>(args.cols));

  for (int row_index = 0; row_index < args.rows; ++row_index) {
    const std::uint8_t* row =
        args.data + static_cast<std::size_t>(row_index) * row_bytes;
    dequantize_row(args.kind, row, weights.data(), args.cols);

    for (int token = 0; token < args.tokens; ++token) {
      const float* input =
          args.input + static_cast<std::size_t>(token) * args.cols;
      float total = 0.0F;
      for (int index = 0; index < args.cols; ++index) {
        total += weights[static_cast<std::size_t>(index)] * input[index];
      }
      args.out[static_cast<std::size_t>(token) * args.rows + row_index] = total;
    }
  }
}

}  // namespace tge::cpu
