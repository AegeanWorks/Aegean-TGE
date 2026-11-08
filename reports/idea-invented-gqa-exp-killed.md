# INVENTED: Fast scalar exp for the GQA online softmax — KILLED (libm exp already 4.2–4.8 ns; polynomial twin 4.7–5.7 ns)

- **Idea**: The GQA decode kernel evaluates one exponential per key per head in its online softmax (`std::exp(score - maximum)`). At 4–8k spans that is millions of scalar libm calls per decoded token, so a scalar twin of the vectorised `exp_avx2` polynomial looked like a large decode lever.
- **Cycle date**: 2026-11-08
- **Verdict**: **KILLED by microbenchmark.** MinGW's `std::exp` is already **4.20–4.82 ns/call**; the polynomial twin is **4.65–5.66 ns/call** — 0.79–1.00x, i.e. no faster. The group kernel's best-of-3 time at span 4096 is identical: **0.731 ms (libm) vs 0.740 ms (fast)**. The exponential is not a bottleneck, and even a free exp would save only ~12 ms of a ~80 ms/token attention budget at 4k.
- **Tree state**: spike reverted from `src/cpu/avx2/attention_avx2.cpp` and `tests/test_attention.cpp`; 145/145 baseline tests rebuilt and green; `aikit.py check` 4/4; no branch, no spike commit; nothing pushed.

## 1. Mechanism

The spike added `exp_fast_scalar`: clamp to `[-87, 88]`, `n = nearbyint(x·log2e)`, `r = x − n·ln2`, degree-6 Taylor polynomial, and `2^n` reconstructed by bit assembly — the scalar analogue of `exp_avx2`. It replaced the two `std::exp` calls in `attention_group_avx2` (the new-max rescale and the ordinary beta), behind `TGE_EXP_FAST=0` for a same-binary A/B.

A degree-5 version initially failed the existing `attention_group_avx2_matches_scalar` sums check (the ~1e-6 per-term error accumulates over 257 keys past the 1e-4 tolerance); degree-6 (~1e-7) passes all 145 tests without weakening anything.

## 2. Falsifier

- **KEEP** if the exp microbenchmark is materially faster, the group kernel best-of-3 improves, and decode gains ≥5% with identical ids.
- **KILL** if the scalar exp is no faster than libm or the kernel is unchanged.

Measured: no speedup; kill.

## 3. Measured numbers

### Scalar exp microbenchmark (2,000,000 calls, 3 rounds)

| round | std::exp | polynomial twin | ratio |
|---:|---:|---:|---:|
| 1 | 4.40 ns | 5.37 ns | 0.82x |
| 2 | 4.20 ns | 5.30 ns | 0.79x |
| 3 | 4.82 ns | 5.66 ns | 0.85x |
| 4 | 4.65 ns | 4.66 ns | 1.00x |

### Group kernel, span 4096, i8 KV, best of 3 rounds of 10 calls

| binary | ms/call |
|---|---:|
| libm exp | 0.731 |
| polynomial exp | 0.740 |

Identical within noise.

### Model A/B (4096 prompt, i8 KV, 8 decoded tokens, 2 pairs)

| pair | libm | fast | ratio | matvec libm/fast | attention libm/fast |
|---:|---:|---:|---:|---:|---:|
| 1 | 3.09 t/s | 2.00 t/s | 0.647 | 1,658 / 2,598 ms | 570 / 859 ms |
| 2 | 3.76 t/s | 1.73 t/s | 0.460 | 1,343 / 3,020 ms | 480 / 973 ms |

The matvec phase — which the exp change cannot touch — was 1.57x and 2.25x slower in the fast arms, so these ratios measure box drift, not the kernel. The best-of-3 kernel comparison above shows no effect; ids were identical in both pairs.

## 4. Why it failed

- **MinGW's `expf` is already fast.** At ~4.5 ns it is a handful of instructions plus a table/series, comparable to the polynomial's dependency chain (round, subtract, six FMAs, bit assembly).
- **The polynomial trades a call for a serial chain.** In the online-softmax loop the exp result feeds the accumulator immediately, so latency matters; the scalar twin does not shorten that chain.
- **Exp is a small share.** 2.75M calls per decoded token at 4k × 4.5 ns ≈ 12 ms, versus ~80 ms/token of attention in the same window. A free exp would save <7% decode — below the 5–15% bar the loop requires.

## 5. Disposition

- Keep `std::exp` in `attention_group_avx2`.
- Close the "cheaper scalar exp" direction: libm is not the cost, and the vector exp already covers the head path's long softmax.
- If decode attention is revisited, target the value accumulation / memory traffic (as cycle 28 showed for GQA reuse) or the QK dot layout, not the exponential.

## 6. Exact commands

```text
cmake --build --preset release                         → 54/54
.\build\release\bin\tge_tests.exe                      → 145/145
TGE_EXP_FAST=0|on .\build\release\bin\tge_tests.exe    → [group-exp] 0.731 / 0.740 ms
TGE_EXP_FAST=0|on tge generate --tokens <4096 ids> --n 8 --threads 6 --gpu off --kv i8 --context 4352
git checkout -- src/cpu/avx2/attention_avx2.cpp tests/test_attention.cpp
python aikit.py check aegean-tge                        → 4/4
```

No push, no merge, no force-push. Spike code is not in the tree or in any commit.
