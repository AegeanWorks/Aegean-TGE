# INVENTED: fused Q4_K row dot for decode — KEEP (kernel 1.763x, bit-identical)

- **Cycle**: 19
- **Date**: 2026-11-09
- **Family**: quantized decode matvec / K-quant
- **Verdict**: **KEEP-grade** (passed the kernel falsifier; not integrated, not pushed)
- **Tree after the cycle**: spike and temporary bench reverted, branch
  `idea/q4k-fused-dot` deleted, this report is the only change

## Why this idea

The log's oldest open kernel gap: *"Fused K-quant dot: still true gap (6.71 vs
21.85 GB/s), still not fused."* Decode for Q4_K weights calls
`gemv_dequant_batch_avx2`, which dequantizes a whole row into a thread-local
float scratch (`cols` floats) and then runs `dot_f32` over that row. For a
single token (decode) the scratch is written once and read back immediately, so
the dequant work and the dot cannot overlap and every weight makes a
store+load round trip through L1.

## Mechanism

`dot_q4k_row_avx2` dequantizes each 32-value group **into registers** and FMAs
it with the activation immediately:

- the nibble extraction and lane mapping are byte-for-byte the same as
  `write32` (low nibbles = first 32 values of the chunk, high nibbles = next
  32);
- the value is `q * scale - offset` with **multiply then subtract, never
  fused**, exactly like the stored path;
- each 8-lane vector is FMA'd into the same accumulator that `dot_f32` would
  have used at that lane offset (0/8/16/24 within every 32-value stride), and
  the final reduction is copied from `dot_f32`.

Therefore the reduced dot is **bit-identical** to dequantize-then-dot, not an
approximation. Prefill (`tokens > 1`) still dequantizes once and reuses the row
across tokens; the fused path only replaces the single-token decode call.

## Predicted gain and falsifier

Predicted: removing the scratch store+load and the second row pass should lift
Q4_K decode from the measured 6.7-17 GB/s toward the Q8_0 path. Falsifier: keep
if the in-process kernel A/B is >=1.05x, the fused row dot equals the batch
dot **exactly**, and fidelity/ids do not move.

## Measured — kernel A/B

In-process, alternating old/new in the same binary, 6 threads, Q4_K
8192 x 8192 (37.7 MB, enough to stream past cache), single token, best of 40
alternating rounds:

| base `gemv_dequant_batch_avx2` | fused `gemv_q4k_fused_avx2` | ratio |
|---:|---:|---:|
| 5061.8 us | 2870.5 us | **1.763x** |

Both kernels read the same 37.7 MB of quantized weights; base runs at ~7.4 GB/s
and fused at ~13.1 GB/s, so the gap was instruction/scratch-bound, not only
DRAM-bound. The synthesized rows are valid Q4_K blocks (fixed fp16 d/dmin,
random payload), so this is the real decode shape.

## Measured — end to end

3B Q4_K_M, greedy, story prompt, n=64, `--kv f32 --threads 6 --gpu off`,
4 alternating pairs:

| pair | plain tok/s | fused tok/s | ratio |
|---:|---:|---:|---:|
| 1 | 3.27 | 4.25 | 1.300x |
| 2 | 3.80 | 4.49 | 1.182x |
| 3 | 4.25 | 4.49 | 1.056x |
| 4 | 4.20 | 4.49 | 1.069x |

Best 4.25 -> 4.49 (1.056x); every pair is positive. The box was externally
loaded during this run (0.5B Q8_0 control 41 t/s vs 69 t/s in a quiet window;
the 0.5B pairs swing 1.052/0.971 with no code on that path, i.e. ~5% noise).
So the end-to-end number is a floor, not the kernel's ceiling; the 3B still has
Q6_K tensors on the old path, which caps the aggregate.

## Fidelity and ids

`tge fidelity --model 3B-Q4_K_M --corpus docs/architecture.md --context 512
--tokens 200 --threads 1 --gpu off` (199 positions), base and fused:

| binary | mean KL | worst KL | top-1 |
|---|---:|---:|---:|
| master | 1.157e-11 | 2.330e-10 | 199/199 |
| fused | 1.157e-11 | 2.330e-10 | 199/199 |

Identical to the last digit, as the bit-identity argument predicts. Greedy ids
on the 3B story run are identical base-vs-fused.

## Tests

169/169 with the spike: the existing suite plus
`dequant_q4k_fused_row_matches_batch` (two rows, 512 cols, exact `==` against
`gemv_dequant_batch_avx2`) and an env-gated `dequant_q4k_fused_bench_env_gated`
(bench placeholder inert without `TGE_Q4K_BENCH`). After the revert: 167/167,
`aikit.py check aegean-tge` 4/4.

## Anchors (spike, reverted)

- `src/cpu/avx2/dequant_avx2.cpp` — `fma32` (anonymous namespace) and
  `dot_q4k_row_avx2`.
- `src/cpu/avx2/dequant_batch_avx2.cpp` — `gemv_q4k_fused_avx2` (band skeleton,
  decode-only, falls back for `tokens > 1`).
- `src/models/qwen2.cpp:990` — dispatch for Q4_K single-token matvec, after the
  kq8d opt-in branch and before the generic batch path.
- `include/tge/cpu/kernels.hpp:186` — declarations.
- `tests/test_dequant.cpp` — differential test and bench.

## Full reimplementation recipe

1. In `dequant_avx2.cpp`, add `fma32(quants, high_nibble, scale, offset, input,
   acc0..acc3)`: copy the nibble extraction and lane mapping from `write32`;
   compute `_mm256_sub_ps(_mm256_mul_ps(values, factor), base)` (no FMA in the
   dequant); then `_mm256_fmadd_ps` each 8-lane value into acc0..acc3.
2. Add `dot_q4k_row_avx2(source, input, cols)`: for each 256-value block read
   d/dmin, scales and quants exactly as `dequantize_q4_k_avx2`; for each 64-value
   chunk unpack the two scales/mins and call `fma32` low then high; reduce with
   the same `(acc0+acc1)+(acc2+acc3)` and 128-bit horizontal as `dot_f32`.
3. Add `gemv_q4k_fused_avx2` reusing the batch kernel's band/thread skeleton,
   calling the row dot per row; require `kind == Q4_K && tokens == 1`, else fall
   back to `gemv_dequant_batch_avx2`.
4. In `Qwen2::matvec`, after the `allow_kq8d` branch, dispatch Q4_K to the fused
   kernel when `!scalar_kernels_ && avx2`.
5. Tests: exact equality against the batch kernel; bench env-gated.

Pitfalls: keep multiply-then-subtract in the dequant (FMA changes the rounding);
keep the four-accumulator lane order; do not fuse the multi-token prefill path
(it deliberately dequantizes once and reuses the row).

## Disposition / next

KEEP. Integration is recommended in the next approved code batch; it is
lossless, decode-only, and the kernel A/B is 1.763x. The same fused pattern
applies to **Q6_K** (present in Q4_K_M), which is the next cycle candidate for
this family. Nothing was pushed; 26 commits remain ahead of `origin/master`.
