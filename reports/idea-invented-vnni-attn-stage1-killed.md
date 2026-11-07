# INVENTED: AVX-VNNI Q8 QK attention dot, stage 1 — KILLED (1.19x; 8k decode saving ~4.75%, under 5%)

- **Idea**: Stage-1 kernel for the cycle-24 long-context reopen: quantize one query head to per-32 int8 blocks, bias the signed int8 key to unsigned (XOR 0x80), and run the QK dot with AVX-VNNI `vpdpbusd`, correcting by `128·sum(q)` per block and applying the query/key scales.
- **Cycle date**: 2026-11-07
- **Verdict**: **KILLED on the kernel bar.** Over an 8192-key span, the VNNI dot is **15.9 ns/key** vs the current float-query q8 dot at **18.8 ns/key** — **1.19x**, below the 1.5x gate. At the measured 8k i8 share (50% attention, QK ≈ half), 1.19x saves ~4.75% of decode, below 5%, before query-quantization error. Do not wire it.
- **Tree state**: spike reverted from `include/tge/cpu/kernels.hpp`, `src/cpu/avx2/q8_batch_vnni.cpp`, `tests/test_attention.cpp`; 145/145 baseline tests rebuilt and green; no branch, no spike commit; nothing pushed.

## 1. Mechanism

```text
per 32-value block:
  q8[b] = clamp(round(query * 127/amax), -127..127), qscale[b] = amax/127
  qsum[b] = sum(q8[b])
  key_u = key_signed XOR 0x80          // signed key -> unsigned (key+128)
  acc = vpdpbusd(0, key_u, q8)          // unsigned key × signed query
  sum = horizontal(acc) - 128*qsum[b]
  total += sum * qscale[b] * kvscale[b]
```

The float reference is the current q8 attention dot: float query × signed int8 key × per-32 KV scale, widened to float and FMA'd.

The 8192-key span is the stage-1 stand-in for the 8k context A/B: one head, 128 dims, 8192 keys, 50 iterations, best of 3.

## 2. Falsifier

- **PROCEED** only if the VNNI dot is ≥1.5x the current q8 dot including the query quantizer amortization, with query-quantization error acceptable for the i8 tier.
- **KILL** if <1.5x; a 1.19x dot cannot clear 5% at the measured share.

## 3. Measured numbers

| metric | value |
|---|---:|
| current q8 dot | **18.8 ns/key** |
| VNNI dot | **15.9 ns/key** |
| speedup | **1.19x** |
| mean abs error vs float-query dot | 3.839e-3 |
| max abs error | 2.160e-2 |
| RMS reference | 1.466 |
| mean relative (mean abs / RMS) | 0.26% |

The error is not the blocker; the speed is. At 8k, attention is 50% of decode and QK is roughly half of that; a 1.19x QK dot saves `0.19 × 0.5 × 0.5 ≈ 4.75%`, under the 5% bar, and the query quantizer adds cost on top.

## 4. Why it lost

- The current q8 dot is already efficient: four blocks of 32 int8 widened to float with FMA. The VNNI form adds XOR, `vpdpbusd`, a horizontal lane sum, a 128·qsum correction, and two scale multiplies per block; at 128 dims the fixed overhead is a large fraction of the dot.
- `vpdpbusd` gives 4 int8 products per lane instruction, but the lane-sum/correction path eats the advantage on a short 128-dim reduction.
- The earlier `maddubs` prefill result also warned that integer dot layouts can lose on this box; this is a different instruction but the same short-reduction overhead pattern.

## 5. What would reopen it

- A layout that removes the per-block horizontal sum/correction: precompute the key bias + bsums at KV-store time and accumulate VNNI results in int32 lanes across multiple blocks; or process two 32-blocks per `vpdpbusd` pair and scale after.
- A kernel that reaches ≥1.5x including query quantization; at 50% attention that is ~9.4% decode, above the 5% bar.
- An 8k phase measurement showing QK ≥60% of attention (then even 1.19x is ~6%).

Until then, the current AVX2 q8 attention stays; the 8k VNNI reopen is recorded but not actionable.

## 6. Exact commands

```text
cmake --build --preset release                         → 54/54
.\build\release\bin\tge_tests.exe                      → 146/146 with the stage-1 bench; 145/145 after revert
# [vnni-attn] ref 18.8 ns/key vnni 15.9 ns/key speedup 1.19x
# meanabs 3.839e-3 maxabs 2.160e-2 rmsref 1.466
git checkout -- include/tge/cpu/kernels.hpp src/cpu/avx2/q8_batch_vnni.cpp tests/test_attention.cpp
python aikit.py check aegean-tge                        → 4/4
```

No push, no merge, no force-push. Spike code is not in the tree or in any commit.
