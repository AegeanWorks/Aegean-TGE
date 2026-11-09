# INVENTED: LSD radix sort for the top-p candidate cut — KEEP-GRADE (2.56x sampler on large nuclei; flat decode +11.6% 0.5B; sampled ids identical)

- **Idea**: The top-p/min-p branch sorts every candidate token by probability with `std::sort` (O(k log k)). For a large nucleus (high temperature / top_p), k approaches the whole 128k vocabulary and the sort dominates the sampler (~13 ms). Replace it with a stable 4-pass LSD radix sort on the fp32 probability bit patterns (descending), which is O(k) and keeps the exact ordering.
- **Cycle date**: 2026-11-08
- **Verdict**: **KEEP-GRADE.** The sampler falls **9,771 -> 3,820 µs per sample on the large-nucleus shape (2.56x)** and **705.5 -> 650.8 µs on peaked logits (1.08x)**, with identical 200-call sampled sequences in both shapes. End-to-end flat top-p decode (temp 1.5, top_p 0.95, n=64, seed 7): 0.5B **35.60 -> 39.73 t/s best-of (+11.6%)**, 3B 4.06 -> 4.39 t/s (+8.1%, noisy), **sampled ids identical**; the greedy thinking set is unchanged.
- **Tree state**: spike reverted from `src/models/sampling.cpp` and `tests/test_sampling.cpp`; branch `idea/radix-sampler-sort` deleted; `git status` clean; master rebuilt 162/162; nothing committed except this report.

## 1. Mechanism

The branch collects candidate indices with `probability >= threshold`, then previously:

```text
std::sort(order.begin(), order.end(), [&](int a, int b){ return scores[a] > scores[b]; });
```

The spike replaces that with `sort_candidates_by_score(order, scores)`:

- For each candidate, take its probability's fp32 bits and flip them (`key = ~bits`). Positive finite floats order by their bit pattern, so ascending `key` is descending probability.
- Four LSD counting passes over 8-bit digits; two index buffers and two key buffers are swapped each pass. 256-entry offset array per pass, prefix-summed and scattered.
- Stable: equal probabilities keep the candidate-scan (index) order. `std::sort` leaves ties unspecified; the sorted prefix cut and draw are otherwise unchanged.

The candidate scan, threshold halving, min-p floor, top-p cumulative cut and draw are untouched. Greedy never enters the branch; the top-k path still uses `partial_sort`.

## 2. Falsifier

- **KEEP-GRADE** if the sampler is >=1.5x faster on the large-nucleus shape and not slower on peaked logits, the sampled sequences match (or are distribution-equivalent under tie reordering), and greedy ids are unchanged.
- **KILL** if peaked regresses or sampled sequences move beyond tie effects.

Measured: 2.56x / 1.08x, sinks identical, greedy unchanged. Keep.

## 3. Measured numbers

### Isolated sampler (128,256 logits, temp 0.8, top_p 0.9, 200 calls, best of three)

Two test binaries differing only in the candidate sort:

| logits shape | std::sort | radix | speedup | sampled sink |
|---|---:|---:|---:|---:|
| Gumbel-scaled (small nucleus) | 705.5 µs | **650.8 µs** | **1.08x** | identical 167533 |
| Uniform (large nucleus) | 9,771.1 µs | **3,819.9 µs** | **2.56x** | identical 164155 |

Suite 163/163 with the temporary bench.

### End-to-end flat top-p decode

`--temp 1.5 --top-p 0.95 --n 64 --threads 6 --gpu off --kv f32 --context 2048 --seed 7 --print-ids`, alternating order:

| model | baseline best | radix best | best ratio | sampled ids |
|---|---:|---:|---:|---|
| Qwen2.5-0.5B Q8_0 | 35.60 t/s | **39.73 t/s** | **0.896** (radix +11.6%) | identical |
| Llama-3.2-3B Q4_K_M | 4.06 t/s | 4.39 t/s | 0.925 (radix +8.1%, noisy) | identical |

Both models reproduce the same 64-token sampled sequence at seed 7, so the stable tie order did not change any recorded draw. Greedy prompts A/B match the fixed ids exactly (greedy bypasses the branch).

## 4. Full reimplementation recipe

1. `src/models/sampling.cpp`: add `<array>`, `<cstdint>`, `<cstring>`.
2. In a file-local anonymous namespace before `sample_token`, add `sort_candidates_by_score(std::vector<int>& order, const std::vector<float>& scores)`:
   - `keys[i] = ~bit_cast<uint32_t>(scores[order[i]])`; `order_tmp`/`keys_tmp` buffers.
   - For `pass = 0..3`, `shift = pass*8`: count digits, prefix-sum into offsets, scatter `(order, keys)` into the temporaries, swap both.
   - Return early for `count < 2`.
3. In the top-p/min-p branch replace `std::sort(order.begin(), order.end(), better);` with `sort_candidates_by_score(order, scores);`. Leave the `better` lambda for the top-k `partial_sort`.
4. Bench: `TGE_SAMPLER_TOPP_BENCH=1` with `TGE_SAMPLER_TOPP_MODE=uniform|gumbel`, 200 calls, and compare the sink between baseline/radix binaries.
5. End-to-end: flat sampling (`--temp 1.5 --top-p 0.95 --seed 7 --print-ids`) on 0.5B/3B; compare best-of decode and the sampled ids.

## 5. Disposition

**KEEP-GRADE: integrate as a normal sampler change.** The sort is exact (same order for distinct probabilities, stable for ties), the sampled sequences recorded here are identical, and the win is large exactly where the old sort dominated: high-temperature/high-top_p sampling. Typical peaked sampling is neutral-to-slightly-positive.

Tie caveat: `std::sort` ordering of equal probabilities is unspecified while the radix sort is stable; a future sampled sequence could differ only through the arbitrary choice among equal-probability tokens, which is distribution-equivalent. The greedy path and the top-k path are untouched.
