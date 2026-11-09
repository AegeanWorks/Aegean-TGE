# INVENTED: radix sort for the top-k sampler branch — KILLED (10.4x slower and wrong for negative logits)

- **Idea**: The top-k sampler branch fills an index array, sorts the whole 128k vocabulary with `std::partial_sort`, and keeps the first `k`. Replace that with the radix candidate sort already shipped for the top-p cut (O(n) bit-pattern passes).
- **Cycle date**: 2026-11-09
- **Verdict**: **KILLED on performance and correctness.** The radix full sort is **212.2 -> 2217.5 µs per sample (10.4x slower)** for `top_k=40`, and it returns the wrong token when the inputs contain negative values: `sampling_top_k_one_is_greedy_at_any_temperature` failed 8 assertions. `partial_sort` is the right tool for a small top-k over a large vocabulary; the radix helper only orders non-negative floats and must not sort raw logits.
- **Tree state**: spike reverted from `src/models/sampling.cpp` and `tests/test_sampling.cpp`; branch `idea/topk-radix-partial` deleted; `git status` clean; master rebuilt 165/165; nothing committed except this report.

## 1. Mechanism

Top-k branch before:

```text
order.resize(count); iota(order, 0);                 // 128k indices
std::partial_sort(order.begin(), order.begin()+keep, order.end(),
                  [&](a,b){ return scores[a] > scores[b]; });
```

Spike:

```text
order.resize(count); iota(order, 0);
sort_candidates_by_score(order, scores);             // full 4-pass LSD radix, descending
order.resize(keep);
```

`sort_candidates_by_score` (added with the top-p radix keep) maps each value to `key = ~bits` and radix-sorts ascending by key, which is descending by **positive** float value. In the top-p branch it is called after the softmax, so every value is a non-negative probability. The top-k branch calls it on the penalized **logits**, which can be negative.

## 2. Falsifier

- **KEEP** if the top-k sampler is >=1.3x faster with identical sampled outputs.
- **KILL** if slower or any sampled output moves.

Measured: 10.4x slower and a deterministic top-k=1 test fails. Kill.

## 3. Measured numbers

Isolated sampler (128,256 logits in [0,1), temp 0.8, top_k 40, 200 calls, best of three):

| binary | per call | sink |
|---|---:|---:|
| `std::partial_sort` | **212.2 µs** | 97725 |
| full radix sort | 2217.5 µs (0.096x) | 97725 |

The sink matches because this bench's logits are non-negative, so the radix order is valid there. The deterministic unit test uses logits with negative values and exposes the domain error: top_k=1 returned a non-argmax token in all 8 assertions, failing `sampling_top_k_one_is_greedy_at_any_temperature`.

Full suite on the spike: 166 cases, 1 failed (8 assertions). After revert: **165/165**.

## 4. Why it failed

- **Algorithm fit**: `partial_sort` is O(n log k). For n=128k and k=40 that is a few hundred thousand cheap comparisons; a full radix sort always pays four scatter passes over 128k indices plus key computation, which is far more work. The radix sort won for the top-p cut only because that branch had already collected a candidate list and the comparison sort there was `std::sort` over all candidates (often the whole vocabulary) with an indirect comparator; top-k's partial sort avoids touching the tail order.
- **Domain**: `key = ~bits` is an order-preserving map only for non-negative floats. Raw logits include negatives, where the IEEE sign-magnitude layout inverts the intended order (`-1.0 < -2.0` by bits as unsigned, but as values `-2.0 < -1.0`). The helper's comment says "positive fp32 bit patterns"; the top-k call violated it.
- **Test value**: the synthetic bench's non-negative logits hid the bug; the property test `top_k=1 is greedy at any temperature` caught it. Kernel benches need adversarial input distributions, not just the natural one.

## 5. Full reimplementation recipe (for the record; do not ship)

1. In `src/models/sampling.cpp`, in the `params.top_k > 0` branch, replace
   `std::partial_sort(order.begin(), order.begin() + keep, order.end(), better);`
   with `sort_candidates_by_score(order, scores); order.resize(keep);`
   and delete the now-unused `better` lambda.
2. Add a temporary `TGE_SAMPLER_TOPK_BENCH=1` test: 128k logits, temp 0.8, top_k 40, 200 calls, print µs/call and the sink.
3. Measure baseline vs spike binaries; run the suite (the top-k test fails on negative logits).
4. Result: 10.4x slower and incorrect. Revert.

## 6. Reopen trigger

- A **partial** selection radix (find the k-th key, scatter only the top k) using a sign-safe order key (flip the sign bit for negatives) could beat `partial_sort` only when k approaches the vocabulary size; for typical top-k it cannot.
- Re-test only if top-k grows to hundreds or if the branch is shown to dominate sampling on a real workload.
