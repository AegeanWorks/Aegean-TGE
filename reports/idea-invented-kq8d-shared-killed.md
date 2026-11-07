# INVENTED: Shared-activation kq8d for gate+up — KILLED (fidelity excellent, speed no gain; down-only stays)

- **Idea**: Prototype a shared Q8_K activation for the gate and up projections: quantize the FFN input once, then run the integer kq8d row kernel for both matrices with the same prequantized blocks. Cycle 21 showed separate gate/up kq8d quantized twice and was slower; sharing removes one quantizer pass per token/layer.
- **Cycle date**: 2026-11-07
- **Verdict**: **KILLED on speed.** Fidelity is excellent (mean KL **2.327e-4**, top-1 **395/399 = 99.0%**, identical to separate gate+up), but 2048 decode is not faster: shared mean **4.09 t/s**, separate **4.28**, down-only **4.77** (best 3.40 / 3.59 / 4.26). The quantizer is not the bottleneck; the gate/up shape is.
- **Tree state**: spike reverted from `include/tge/models/qwen2.hpp`, `include/tge/cpu/kernels.hpp`, `src/models/qwen2.cpp`, `src/cpu/avx2/dequant_batch_kq8_avx2.cpp`; 145/145 baseline tests rebuilt and green; no branch, no spike commit; nothing pushed.

## 1. Mechanism

New public API in `kernels.hpp`:

```text
struct Kq8Activation { int cols; vector<uint8_t> data; };  // opaque Q8_K blocks
quantize_kq8_activation(input, cols, out);
gemv_dequant_kq8_row_prequant_avx2(args, config, activation);
```

The row kernel is the same integer Q4_K/Q6_K dot; only the activation quantization is hoisted. In `forward`, when scope is `gate+up` and both weights are K-quant, quantize `h_` once and pass the scratch to both matvec calls. `TGE_KQ8D_SHARE=0` disables sharing for the A/B.

Numerically the shared path is identical to separate gate+up: same input, same deterministic quantizer, same integer dots. The 399-position fidelity confirms it.

## 2. Falsifier

- **KEEP** if shared gate+up is ≥5% faster than separate gate+up and faster than down-only, with 399-position fidelity in class and ids identical.
- **KILL** if sharing is neutral/slower or gate+up remains behind down-only.

Measured: neutral vs separate, behind down-only; kill.

## 3. Measured numbers

### Fidelity (399 positions, f32)

| path | mean KL | worst KL | top-1 |
|---|---:|---:|---:|
| shared gate+up | 2.327e-4 | 4.149e-3 | 395/399 (99.0%) |
| separate gate+up (cycle 22 baseline) | 2.327e-4 | 4.149e-3 | 395/399 |

Exactly identical, as expected.

### 2048 decode A/B (same binary, rotating order)

| round | shared gate+up | separate gate+up | down-only |
|---:|---:|---:|---:|
| 1 | 4.77 | 3.59 | 5.28 |
| 2 | 3.40 | 4.97 | 4.26 |
| **mean** | **4.09** | **4.28** | **4.77** |
| **best** | 3.40 | 3.59 | **4.26** |

Shared vs separate is inside the box noise (one round each way); neither is faster than down-only. Generated ids are **IDENTICAL** across all three modes.

## 4. Why it lost

- The activation quantizer is not the dominant cost. Sharing saves one pass over 3,072 floats per layer per token; the integer dot on the 4,864×896 gate/up shape still loses to the float dequant path.
- The down projection (896×4,864) is the only shape where the integer row kernel's savings exceed its overhead; gate/up are the opposite shape (many rows, short reduction).
- Adding a shared-activation API, a Qwen2 member and branch plumbing buys no measurable speed; complexity without a result.

## 5. Reimplementation recipe (kill-grade, for the record)

1. `kernels.hpp`: add `Kq8Activation { int cols; vector<uint8_t> data; }`, `quantize_kq8_activation`, `gemv_dequant_kq8_row_prequant_avx2`.
2. `dequant_batch_kq8_avx2.cpp`: quantize into the opaque buffer via `quantize_row_q8k`; the prequant kernel validates kind/cols/size and runs the same banded integer dot with `q8 = reinterpret_cast<const Q8KBlock*>(data)`.
3. `matvec`: add `const Kq8Activation* prequant`; if non-null and cols match, call the prequant kernel, else the internal-quantizer row kernel.
4. `forward`: when scope gate+up and both weights Q4_K/Q6_K and `dim % 256 == 0`, quantize `h_` once and pass the scratch to both.
5. Gate: 399-position fidelity (2.327e-4, 395/399), 2048 shared vs separate vs down, ids identical. Expected shared 4.09 vs separate 4.28 vs down 4.77 → do not ship.

## 6. Verdict and disposition

**KILLED.** The kq8d line is closed at **down-only**. Shared activation for gate+up is numerically excellent but does not recover the shape penalty; down-only remains the only measured decode tier. Do not revisit gate/up kq8d without a different kernel shape strategy, not another quantizer-sharing wrapper.

## 7. Exact commands

```text
cmake --build --preset release                         → 54/54
.\build\release\bin\tge_tests.exe                      → 145/145
TGE_KQ8D_SCOPE=gate+up tge fidelity ... --context 1024 --tokens 400 --kv f32
TGE_KQ8D_SCOPE=gate+up [TGE_KQ8D_SHARE=0] tge generate --tokens <2048 ids> --n 32 --threads 6 --gpu off
TGE_KQ8D_SCOPE=down tge generate ...
git checkout -- include/tge/models/qwen2.hpp include/tge/cpu/kernels.hpp \
  src/models/qwen2.cpp src/cpu/avx2/dequant_batch_kq8_avx2.cpp
python aikit.py check aegean-tge                        → 4/4
```

No push, no merge, no force-push. Spike code is not in the tree or in any commit.
