#include "tge/cpu/kernels.hpp"

#include <cstddef>
#include <cstring>
#include <stdexcept>

namespace tge::cpu {

void gemv_q8_batch_scalar(const Q8BatchArgs& args, const GemvConfig& /*config*/) {
  if (args.rows <= 0 || args.cols <= 0 || args.tokens <= 0) {
    return;
  }
  if (args.cols % kQ8BlockElements != 0) {
    throw std::invalid_argument("gemv_q8_batch: cols not a whole block");
  }

  const int blocks = args.cols / kQ8BlockElements;
  const std::size_t row_bytes =
      static_cast<std::size_t>(blocks) * kQ8BlockBytes;

  for (int row = 0; row < args.rows; ++row) {
    const std::uint8_t* weight_row =
        args.data + static_cast<std::size_t>(row) * row_bytes;
    for (int token = 0; token < args.tokens; ++token) {
      const float* input =
          args.input + static_cast<std::size_t>(token) * args.cols;
      float total = 0.0F;
      for (int block = 0; block < blocks; ++block) {
        const std::uint8_t* entry = weight_row + block * kQ8BlockBytes;
        std::uint16_t scale_bits = 0;
        std::memcpy(&scale_bits, entry, sizeof(scale_bits));
        const float scale = half_to_float(scale_bits);
        const auto* values = reinterpret_cast<const std::int8_t*>(entry + 2);
        const float* vector = input + block * kQ8BlockElements;
        float accumulator = 0.0F;
        for (int index = 0; index < kQ8BlockElements; ++index) {
          accumulator += static_cast<float>(values[index]) * vector[index];
        }
        total += scale * accumulator;
      }
      args.out[static_cast<std::size_t>(token) * args.rows + row] = total;
    }
  }
}

}  // namespace tge::cpu
