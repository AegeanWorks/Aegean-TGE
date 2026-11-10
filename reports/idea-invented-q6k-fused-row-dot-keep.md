# INVENTED: fused Q6_K row dot for decode — KEEP (kernel 1.315x, bit-identical)

- **Cycle**: 20
- **Date**: 2026-11-09
- **Family**: quantized decode matvec / K-quant
- **Verdict**: **KEEP-grade** (passed the kernel falsifier; not integrated, not pushed)
- **Tree after the cycle**: spike and temporary bench reverted, branch
  `idea/q6k-fused-dot` deleted, this report is the only change

## Why this idea

Cycle 19 fused the **Q4_K** decode row dot (kernel 1.763x, bit-identical). The
same log gap covers Q6_K, and the reference Q4_K_M file contains Q6_K tensors too.
Decode for Q6_K still calls `gemv_dequant_batch_avx2`: dequantize a whole row into
a thread-local float scratch, then `dot_f32` reads it back. Single-token decode
therefore pays the scratch store+load for every 8-value group.

## Mechanism

Q6_K super-block (210 bytes, 256 values): 128 low bytes, 64 high bytes, 16 int8
scales at 192, fp16 `d` at 208. The dequant writes, per 128-value half and per
`index` in {0, 8, 16, 24}, four groups of 8 values at offsets
`index / index+32 / index+64 / index+96`, with factors `f1..f4 = d * scale[...]`
and high-bit shifts 0/2/4/6.

`dot_q6k_row_avx2` adds `fma_group_q6_k`, which:

- decodes the same 8 values with the same nibble/bit extraction;
- keeps the dequant **multiply-only** (`factor * value`, no FMA), exactly like
  `dequantize_group_q6_k`;
- FMAs the scaled 8 values with the activation.

All four groups at one `index` have offsets congruent to `index` modulo 32, so
they all target the same accumulator (`index / 8`) that `dot_f32` would have used
at those lanes; the final reduction is copied from `dot_f32`. The result is
therefore **bit-identical** to dequantize-then-dot, and the prefill path
(`tokens > 1`) keeps the row-reuse dequant unchanged.

## Predicted gain and falsifier

Predicted: remove the scratch store+load and the second row pass; Q6_K's dequant
is simpler than Q4_K's (no scale/min unpack), so expect less than Q4_K's 1.763x
but a clear win. Falsifier: keep if the kernel A/B is >=1.05x, the output is
exactly equal, and fidelity/ids do not move.

## Measured — kernel A/B

In-process, alternating old/new in the same binary, 6 threads, Q6_K
8192 x 8192 (55 MB), single token, best of 40 alternating rounds:

| base `gemv_dequant_batch_avx2` | fused `gemv_q6k_fused_avx2` | ratio |
|---:|---:|---:|
| 3920.7 us | 2981.5 us | **1.315x** |

## Measured — end to end

3B Q4_K_M, greedy, story prompt, n=64, `--kv f32 --threads 6 --gpu off`; only the
Q6_K tensors take the fused path (Q4_K is still the old kernel in this cycle),
4 alternating pairs:

| pair | plain tok/s | fused tok/s | ratio |
|---:|---:|---:|---:|
| 1 | 6.90 | 7.40 | 1.083x |
| 2 | 7.20 | 7.40 | 1.020x |
| 3 | 6.80 | 7.37 | 1.088x |
| 4 | 6.90 | 7.39 | 1.071x |

Best 6.90 -> 7.40 (1.072x); every pair is positive, ids identical. This is the
Q6_K-only contribution on a mixed-quant file; it stacks with the cycle-19 Q4_K
keep, which covers the bulk of the weights. The combined end-to-end number is an
integration-time measurement, not claimed here.

## Fidelity

`tge fidelity --model 3B-Q4_K_M --corpus docs/architecture.md --context 512
--tokens 200 --threads 1 --gpu off` (199 positions), fused binary:

| mean KL | worst KL | top-1 |
|---:|---:|---:|
| 1.157e-11 | 2.330e-10 | 199/199 |

Identical to the master measurement from cycle 19 (1.157e-11, 199/199), as the
bit-identity argument requires.

## Tests

169/169 with the spike: the existing suite plus
`dequant_q6k_fused_row_matches_batch` (two rows, 512 cols, exact `==` against
`gemv_dequant_batch_avx2`) and the env-gated `dequant_q6k_fused_bench_env_gated`.
After the revert: 167/167, `aikit.py check aegean-tge` 4/4.

## Anchors (spike, reverted)

- `src/cpu/avx2/dequant_avx2.cpp` — `fma_group_q6_k`, `dot_q6k_row_avx2`.
- `src/cpu/avx2/dequant_batch_avx2.cpp` — `gemv_q6k_fused_avx2` (decode-only,
  falls back for `tokens > 1`).
- `src/models/qwen2.cpp:990` — Q6_K single-token dispatch after the kq8d branch.
- `include/tge/cpu/kernels.hpp:188` — declarations.
- `tests/test_dequant.cpp` — differential test and bench.

## Full reimplementation recipe

1. Add `fma_group_q6_k(low, high, low_shift, high_shift, factor, input, acc)`
   beside `dequantize_group_q6_k`: same `loadl_epi64` + shift/mask/or/sub-32
   decode, `values = cvt(combined)`, `scaled = factor * values`, then
   `acc = fmadd(scaled, input, acc)`.
2. Add `dot_q6k_row_avx2`: four accumulators; per block read low/high/scales/d;
   per 128-half and `index` in {0,8,16,24}, compute `f1..f4` exactly as the
   dequant, select `accs[index/8]`, and call the group helper for the four
   offsets `index`, `+32`, `+64`, `+96` with shifts (0,0), (0,2), (4,4), (4,6);
   advance low/high/scales by 64/32/8 per half; reduce exactly as `dot_f32`.
3. Add `gemv_q6k_fused_avx2` with the batch kernel's band skeleton; require
   `kind == Q6_K && tokens == 1`, else fall back.
4. Dispatch Q6_K to it in `Qwen2::matvec` when `!scalar_kernels_ && avx2`.
5. Tests: exact equality against the batch kernel; env-gated bench.

Pitfalls: all four groups at one index share the same accumulator (do not offset
it by 0/1/2/3); keep the dequant multiply-only and only the dot fused; keep the
unaligned loads; Q6_K rows are whole 256-value blocks.

## Disposition / next

KEEP. The lossless K-quant decode line now has two verified pieces: cycle 19
Q4_K (1.763x kernel) and this Q6_K (1.315x). The next approved integration batch
should land both together and measure the combined 3B decode; nothing else in
this family remains unfused for the reference file. Nothing was pushed; 27
commits remain ahead of `origin/master`.
