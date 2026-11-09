# INVENTED: exponent-histogram threshold for the top-p sampler — KILLED (slower in both distributions; halving's early exit wins)

- **Idea**: The top-p/min-p branch finds the nucleus threshold by repeatedly scanning the whole vocabulary and halving the floor (`threshold *= 0.5`) until the candidate mass reaches `top_p`. Replace those repeated scans with one exponent histogram (mass per fp32 exponent) to jump directly to the crossing bucket, then run the existing candidate scan once at that bucket's lower bound.
- **Cycle date**: 2026-11-08
- **Verdict**: **KILLED.** The histogram version is slower in both measured shapes: **779.8 -> 952.7 µs** per sample on realistic peaked logits (0.82x) and **14,330 -> 15,041 µs** on the pathological large-nucleus shape (0.95x). Sampled sequences are identical (same 200-call sink), so the cut set matches, but the extra full bit-extraction pass plus the 255-bucket walk costs more than the halving's early-exit scans. Keep the shipped halving.
- **Tree state**: spike reverted from `src/models/sampling.cpp` and `tests/test_sampling.cpp`; branch `idea/top-p-histogram-threshold` deleted; `git status` clean; master rebuilt 162/162; nothing committed except this report.

## 1. Mechanism

Baseline:

```text
threshold = min(min_p, 1 - top_p)
for (;;) {
  scan all count: collect p >= threshold, sum mass
  if (mass >= top_p * total || threshold <= 0) break
  threshold *= 0.5
}
sort candidates; cumulative top-p cut; draw
```

Spike: after the AVX2 softmax writes probabilities, one pass buckets their mass by fp32 exponent (`(bits >> 23) & 0xFF`, skipping 0 and 255), then walks exponents 254..1 accumulating until `mass >= top_p * total` and sets `threshold = ldexp(1, exponent - 127)`. The existing candidate scan, sort and cumulative cut are unchanged; the halving loop remains as a fallback but normally breaks on the first scan.

## 2. Falsifier

- **KEEP-GRADE** if the sampler is faster in the peaked shape (the representative LLM case) and never slower in the large-nucleus shape.
- **KILL** if either shape regresses.

Measured: both regress. Kill.

## 3. Measured numbers

128,256 logits, temp 0.8, top_p 0.9, 200 calls, best of three; two test binaries differing only in the sampler:

| logits shape | baseline (halving) | histogram | ratio (base/spike) | sampled sink |
|---|---:|---:|---:|---:|
| Gumbel-scaled (peaked nucleus) | **779.8 µs** | 952.7 µs | 0.82x | identical 167533 |
| Uniform (large nucleus) | **14,330.1 µs** | 15,040.9 µs | 0.95x | identical 164155 |

- On peaked logits the halving needs only a few scans; the histogram always pays a full 128k pass with a `memcpy` bit extract and a bucket RMW, plus the 255-step walk.
- On uniform logits the repeated scans are many, but the **candidate sort dominates** (~13 ms of the ~14 ms); replacing ~1-2 ms of scans with a more expensive single pass does not help.
- The sampled sequences are identical in both shapes, so the cut-set semantics are correct; this is a pure performance kill.

Suite 163/163 with the temporary bench. No fidelity/sampled-id diff owed beyond the matching sinks (the shipped exp path is untouched by this idea).

## 4. Why it failed

- The halving loop has an early exit: on real (peaked) logits the first or second threshold already covers `top_p`, so it costs 1-3 scans, not the worst-case 14-20.
- The histogram's pass is more expensive per element than a compare-and-add: float bit extraction (`memcpy`), exponent mask, bucket index, indexed RMW. Even one extra full pass can lose to one simple scan.
- For large nuclei the O(k log k) candidate sort is the real cost; the threshold search is not the bottleneck.

## 5. Full reimplementation recipe (for the record)

1. `src/models/sampling.cpp`, top-p/min-p branch: after `threshold = min(min_p, 1 - top_p)`, if `!min_filter && top_p < 1`, build `float bucket_mass[255] = {0}`; for each `probability > 0`, `memcpy` its bits, `exponent = (bits >> 23) & 0xFF`, skip 0/255, `bucket_mass[exponent] += probability`.
2. Walk `exponent = 254..1`, accumulate, and on `mass >= top_p * total_all` set `threshold = std::ldexp(1.0F, exponent - 127); break;`.
3. Leave the existing candidate scan/halving fallback, sort, min-p floor and cumulative cut unchanged.
4. Bench with `TGE_SAMPLER_TOPP_BENCH=1` (200 calls) and `TGE_SAMPLER_TOPP_MODE=uniform|gumbel`; build baseline/spike test binaries and compare the sink.
5. Result: slower; do not integrate.

## 6. Remaining lever (named, not built)

The large-nucleus cost is the candidate `std::sort` (~13 ms on 128k uniform). A future cycle should target the **selection/cut**, not the threshold: e.g., an exact probability-threshold selection via a radix/histogram refinement plus a partial selection to the cumulative boundary, or a replacement of the sort with an O(n) bucketed cumulative walk. Any such change must preserve the current sampled sequences on the two recorded shapes (or demonstrate a distribution-level chi-square match), and the greedy path is unaffected.
