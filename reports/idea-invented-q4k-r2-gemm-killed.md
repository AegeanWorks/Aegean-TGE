# INVENTED: two-row Q4_K prefill GEMM (row-interleaved without format repack) - KILLED (median 0.99x, bit-exact)

- **Idea**: The history analysis identified row-interleaved quantized GEMM as the main remaining CPU prefill lever in llama.cpp/ik_llama.cpp. Prototype the cheapest version: keep the Q4_K_M model format, but process two output rows together so each 32-value Q8_K activation chunk is loaded once and reused for both row dots.
- **Cycle date**: 2026-11-08
- **Verdict**: **KILLED.** Four best-of-3 runs on the real prefill shape (3072x3072 Q4_K, 64 tokens, 6 threads) give ref/group2 speedups **0.999x, 0.985x, 1.087x, 0.932x** - median **~0.99x**, no consistent win. The prototype is bit-exact (maxdiff 0.000e+00), so this is not a correctness or alignment effect: sharing activation loads does not help because the Q4_K x Q8_K AVX2 dot is **madd-chain bound**, not load bound.
- **Tree state**: spike reverted from `include/tge/cpu/kernels.hpp`, `src/cpu/avx2/dequant_batch_kq8_avx2.cpp` and `tests/test_dequant.cpp`; 146/146 baseline tests rebuilt and green; `aikit.py check` 4/4; no branch, no spike commit; nothing pushed.

## 1. Mechanism

The shipped integer prefill kernel (`gemv_dequant_batch_kq8_avx2`) is row-outer/token-inner and runs `dot_q4k_q8k(row, q8_token)` per (row, token), so each dot loads the row's 144-byte weight block and the token's 292-byte Q8_K block. The prototype kept the same per-row math but reordered to block -> 32-value k-chunk -> two rows:

- one `Q8_K` chunk load feeds both rows (halves activation loads),
- per-row scale/min unpack and the `bsums` correction stay identical,
- two independent madd chains give the scheduler ILP.

The kernel was exposed as `gemv_dequant_kq8_group_r2_avx2`, differential-tested against the shipped kernel, and benchmarked with the same `DequantBatchArgs` and `GemvConfig`.

## 2. Falsifier

- **PROCEED** if the group kernel is >=1.3x on the 3072x3072 Q4_K / 64-token call with a bit-exact result.
- **KILL** if it is neutral or slower.

Measured: median 0.99x; kill.

## 3. Measured numbers

Shape: rows 3072, cols 3072, tokens 64, 6 threads, Q4_K, 64 best-of-3 runs (3 calls per round).

| run | shipped kernel | two-row group | speedup | max abs diff |
|---:|---:|---:|---:|---:|
| 1 | 8.16 ms | 8.17 ms | 0.999x | 0 |
| 2 | 13.99 ms | 14.21 ms | 0.985x | 0 |
| 3 | 12.22 ms | 11.24 ms | 1.087x | 0 |
| 4 | 7.74 ms | 8.30 ms | 0.932x | 0 |
| **median** | - | - | **~0.99x** | 0 |

The absolute times move with box load; the sign does not establish a gain. Bit-exactness confirms the arithmetic order is preserved.

## 4. Why it failed

- The Q4_K dot spends its hot-loop budget on **two madd operations per 16 values** (`maddubs` then `madd` with the per-16 scale) plus nibble extraction. On AVX2 those two madds are the port bottleneck; removing half the activation loads does not reduce them.
- Q4_K's **per-16 scales** prevent the clean AVX-VNNI `vpdpbusd` mapping: the integer dot would need a per-16 scale multiply after the dot groups, which reintroduces the madd chain. This is why the histories' big CPU prefill wins use **repacked formats with uniform per-block scales** (Q4_0_R8, Q8_0_R16, IQ*_KS_R16) plus AVX-512/VNNI, not row-interleaved Q4_K.
- The earlier 128-token chunk experiment also showed that reducing weight-pass count helps at 2048 but ties at 8k; weight streaming is not the limiting factor either. The dot itself is.

## 5. Consequence for the roadmap

- Close the "row-interleave the existing Q4_K kernel on AVX2" line. It cannot deliver the 1.3x the roadmap required.
- The remaining P1 variants require a **format change** (uniform-scale repack, i-quants) plus a quantizer and model conversion, or **AVX-512** (fused off on this CPU). Both are outside the current Q4_K_M model and this box.
- P2 (SiLU/RMSNorm + quantization fusion) and P3 (activation-quantizer tuning) are arithmetically negligible for prefill: the activation traffic is ~1.2 MB per layer per chunk against ~1.87 GB of weights, so their ceiling is well under 1%.
- The honest position: TGE's shipped Q4_K AVX2 prefill is at its kernel roofline. Closing the gap to llama.cpp on this hardware needs a different quantized format/ISA path, not more loop tiling.

## 6. Exact commands

```text
cmake --build --preset release                         → 54/54
.\build\release\bin\tge_tests.exe                      → 147/147 with the prototype bench; 146/146 after revert
# [kq8-rgroup] ref 8.16 ms, group2 8.17 ms, speedup 0.999x | maxdiff 0.000e+00
# [kq8-rgroup] ref 13.99 ms, group2 14.21 ms, speedup 0.985x
# [kq8-rgroup] ref 12.22 ms, group2 11.24 ms, speedup 1.087x
# [kq8-rgroup] ref 7.74 ms, group2 8.30 ms, speedup 0.932x
git checkout -- include/tge/cpu/kernels.hpp \
  src/cpu/avx2/dequant_batch_kq8_avx2.cpp tests/test_dequant.cpp
python aikit.py check aegean-tge                        → 4/4
```

No push, no merge, no force-push. Spike code is not in the tree or in any commit.
