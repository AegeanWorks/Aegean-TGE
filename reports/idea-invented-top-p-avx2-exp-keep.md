# INVENTED: AVX2 exp in the top-p/min-p sampler branch — KEEP-GRADE (sampler 1.68x on peaked logits; sampled ids identical)

- **Idea**: The top-p/min-p branch computed 128k probabilities with scalar `std::exp` plus a per-value maximum comparison, while the no-cut path already uses the AVX2 softmax kernel. Swap the branch to `softmax_probabilities` (AVX2 exp, internal max and sum) and carry the argmax index out of the existing max scan for the degenerate-cut fallback.
- **Cycle date**: 2026-11-08
- **Verdict**: **KEEP-GRADE.** On realistic peaked logits the top-p sampler falls **816.2 µs -> 486.9 µs per 128k sample (1.68x)**, saving 329 µs per sampled token, and the 200-call sampled sequence is identical. End-to-end top-p decode (temp 0.8, top_p 0.9, n=64, seed 7): 0.5B best **52.57 -> 54.64 t/s (+3.9%)**, 3B best 5.43 -> 6.19 t/s (+14%, load-noisy); **sampled ids identical on both models**, and the greedy thinking set is unchanged.
- **Tree state**: spike reverted from `src/models/sampling.cpp` and `tests/test_sampling.cpp`; branch `idea/top-p-avx2-exp` deleted; `git status` clean; master rebuilt 162/162; nothing committed except this report.

## 1. Mechanism

Top-p/min-p branch before:

```text
float total_all = 0, maximum_probability = -1; int maximum_index = 0;
for (index = 0; index < count; ++index) {
  p = std::exp((scores[index] - maximum) * inverse_temperature);
  scores[index] = p; total_all += p;
  if (p > maximum_probability) { ... }
}
```

After:

```text
// maximum_index now comes from the existing max scan at the top of the
// non-greedy section (strict greater, first index).
const float total_all = softmax_probabilities(
    scores.data(), count, params.temperature, scores.data());
```

`softmax_probabilities` dispatches to `softmax_exp_avx2` when AVX2+FMA is present (the no-cut path has used it since the sampler work): it computes its own max, writes `exp((x-max)/T)` eight lanes at a time, and returns the index-order sum. The maximum probability is 1.0 at the argmax by construction, so the removed `maximum_probability` tracking is not needed; the empty/degenerate fallbacks already use `maximum_index`.

The threshold-halving search, `std::sort` of candidates, min-p floor and final draw are unchanged. Greedy sampling never enters this branch.

## 2. Falsifier

- **KEEP-GRADE** if the top-p sampler is >=1.3x faster on realistic peaked logits, the greedy thinking ids are unchanged, and the recorded sampled sequences match at a fixed seed.
- **KILL** if the sampler is slower or any sampled sequence diverges on the recorded prompts.

Measured: 1.68x, greedy ids unchanged, sampled ids identical. Keep.

## 3. Measured numbers

### Isolated sampler (128,256 logits, temp 0.8, top_p 0.9, 200 calls, best of 3)

Two test binaries differing only in the branch; logits from a Gumbel-scaled distribution so the nucleus is small like real LLM logits.

| binary | per call | sampled-sequence sink |
|---|---:|---:|
| scalar exp | **816.2 µs** | 167533 |
| AVX2 exp | **486.9 µs** | 167533 |

Speedup **1.68x**, saving **329 µs/sample**; the identical sink means all 200 sampled tokens matched.

Distribution dependence: with pathological near-uniform logits (all values in [0,1)) top-p keeps ~90% of the vocabulary, and the threshold/sort loops dominate: 9.7 ms vs 11.5 ms per sample, i.e. the exp swap is neutral-to-negative there. Real model logits are peaked; the Gumbel bench is the representative one.

### End-to-end sampled decode

Prompt: repeated fox sentence; `--temp 0.8 --top-p 0.9 --n 64 --threads 6 --gpu off --kv f32 --context 2048 --seed 7 --print-ids`; alternating arm order.

| model | baseline best | AVX2 best | best ratio | sampled ids |
|---|---:|---:|---:|---|
| Qwen2.5-0.5B Q8_0 | 52.57 t/s | **54.64 t/s** | **0.962** (spike +3.9%) | identical |
| Llama-3.2-3B Q4_K_M | 5.43 t/s | 6.19 t/s | 0.877 (spike +14%, noisy) | identical |

The 3B runs were load-noisy (baseline 5.43/2.61, spike 6.19/5.50); only the best-of is usable. The 0.5B runs are tight (baseline 46.8-52.6, spike 49.5-54.6). The sampled token sequences match exactly at seed 7 on both models, so the ~1e-6 exp difference did not change any draw in these 128 sampled tokens.

### Greedy thinking set

Prompts A and B produce the fixed ids exactly on the spike binary (`... 16053` and `... 9290 25`); greedy never enters this branch. Suite 163/163. No fidelity run is owed: logits, greedy decoding and teacher-forced paths are untouched; only sampled probabilities change at the kernel's documented ~1e-6 level (the same kernel the no-cut sampler already uses).

## 4. Full reimplementation recipe

1. `src/models/sampling.cpp`, non-greedy section: after the existing scalar max scan, track `int maximum_index` on strict `>` and keep `maximum` (the top-k path still uses it).
2. In the `top_k <= 0 && (top_p < 1 || min_p > 0)` branch, delete the scalar exp/total/max-index loop and replace it with
   `const float total_all = softmax_probabilities(scores.data(), count, params.temperature, scores.data());`.
   Remove `maximum_probability`; `maximum_index` now comes from step 1.
3. Leave the threshold-halving loop, `order` construction, `std::sort`, min-p floor, top-p cumulative cut and draw unchanged.
4. Bench: a `TGE_SAMPLER_TOPP_BENCH=1` test with Gumbel-scaled logits (`value = -log(-log(u))*2`), temp 0.8, top_p 0.9, 200 calls; build baseline/spike test binaries; require the sink to match.
5. End-to-end: `generate --temp 0.8 --top-p 0.9 --n 64 --seed 7 --print-ids` on 0.5B and 3B, alternating; compare sampled ids (they matched here) and best-of decode.

## 5. Disposition and next lever

**KEEP-GRADE: integrate as a normal sampler change.** It is local to the cut branch, reuses an already-shipped kernel, and is exact enough that the recorded sampled sequences match.

The remaining top-p hotspot is the **threshold-halving plus candidate sort** when the nucleus is large. A single-pass exponent-histogram threshold (one mass pass, then one candidate pass) followed by a selection/radix cut would remove the repeated full scans and the O(k log k) sort; the uniform-logit bench (9.7-11.5 ms) is the shape where it would pay. That is a separate cycle with a distribution-level correctness gate.
