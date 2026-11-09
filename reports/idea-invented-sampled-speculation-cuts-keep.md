# Idea 16 — Sampled speculation under top-k/top-p/min-p cuts

- **Cycle**: 16
- **Date**: 2026-11-09
- **Family**: token generation / speculative decoding
- **Verdict**: **KEEP-grade** (passed its falsifier; not integrated, not pushed — the
  push gate applies because this is a code change, not a report)
- **Tree after the cycle**: spike reverted, spike branch deleted, this report is the
  only change

## Why this idea

The CLI has shipped distribution-preserving sampled speculation for one
temperature-only configuration. `speculate_sampled` was deliberately fenced to

```
temperature > 0 && top_k <= 0 && top_p >= 1 && min_p <= 0
```

so that `--speculative --temp 0.8 --top-p 0.9` (the common chat setting) silently ran
plain sampling. The fence existed because the verifier's target distribution was the
full softmax; applying the accept rule to the untruncated distribution would not match
what `sample_token` draws. That is a pure availability loss: the most common sampling
configuration gets zero speculation.

## Mechanism

`speculative_accept` already works on an **unnormalized** probability row: it computes
`total = sum(q)`, accepts the draft with `q[draft] / total`, and draws the replacement
from `q - onehot(draft)` normalized by `total - q[draft]`. Therefore, if the verifier
row is the **post-cut** probability row (non-kept entries zeroed, kept entries left
unnormalized), the rule emits exactly the truncated target distribution. No
renormalization and no change to the accept rule are required.

The only missing piece was a reusable cut: `apply_sampling_cuts(probabilities, count,
params, scratch)` that mirrors `sample_token`'s selection:

- `top_k > 0`: `partial_sort` the top k, keep exactly those;
- otherwise: threshold halve (`threshold = min(min_p or 2, (1 - top_p) or 2)`) while the
  candidate mass is below `top_p * total_all`, so the subsequent sort only sees nucleus
  candidates;
- `min_p > 0`: drop candidates below `min_p * max` (relative floor, as `sample_token`
  does after the absolute prefilter);
- `top_p < 1`: radix-sort descending and keep the shortest prefix whose cumulative mass
  over `total` reaches `top_p`;
- zero every non-kept entry.

Wiring: `speculate_sampled = speculative && temperature > 0`; inside the existing
`penalized_softmax` lambda (which already copies the row, applies the repeat penalty and
softmaxes), call `apply_sampling_cuts` when any cut is active. Both the accept rule and
the bonus-token `categorical_sample` then see the same truncated row.

## Predicted gain and math

Per verify step the draft proposes up to five tokens; the verifier is one 6-row batch
forward. Emitted tokens per step are `1 + A`, where `A` is the accepted-draft count.
The speedup is therefore

```
speedup = (1 + A) * t_plain / t_verify
```

where `t_plain` is one single-row forward plus one `sample_token` cut, and `t_verify` is
one 6-row forward plus one cut per verified row. The 6-row batch reuses each weight read
across six tokens, so `t_verify` is far below `6 * t_plain`; with `A >= 1` the idea wins.
The 0.5B is the risky case because the cut share of a token is larger. Predicted:
`>=1.2x` on the 3B reference under top-p sampling, distribution-preserving; the 0.5B
tolerated only if the cut stays O(nucleus).

Falsifier: keep only if (a) sampled speculation with cuts is `>=1.2x` plain sampling on
the 3B reference at temp 0.8 / top-p 0.9 **and** top-k, (b) the emitted distribution over
the truncated support matches the normalized cut (mask reference plus an accept-rule
histogram), and (c) the full suite and the project gate pass.

## Measurement — first attempt (naive cut) and the correction

First implementation applied the cuts but **sorted the whole 128k vocabulary** on every
verified row. It preserved correctness but lost time on the small model:

| Config (best-of-N) | plain tok/s | spike tok/s | ratio plain/spike |
|---|---:|---:|---:|
| 0.5B Q8_0, temp 0.8, top-p 0.9 (4 pairs) | 40.91 | 38.65 | 1.058 (spike 5.5% slower) |
| 0.5B Q8_0, temp 0.8, top-k 40 (3 pairs) | 42.51 | 39.63 | 1.073 (spike 6.7% slower) |
| 3B Q4_K_M, temp 0.8, top-p 0.9 (2 pairs) | 5.11 | 8.09 | 0.632 (**1.58x faster**) |

Diagnosis: the helper sorted 128k candidates per row; `sample_token` first prefilters by
threshold halving and sorts only the nucleus. The fix was to mirror the prefilter (and
the min-p absolute-then-relative-floor order) exactly. That single change moved the
small-model result from a regression to a large win.

## Measurement — final helper

Same prompt and settings (`The quick brown fox ...` repeated, `--n 64`, seed 7,
`--gpu off`, `--threads 6`), best-of-pairs, alternating A/B:

| Config | plain tok/s | spike tok/s | ratio plain/spike | spec steps / accepted per step / emitted |
|---|---:|---:|---:|---|
| 0.5B Q8_0, temp 0.8, top-p 0.9 (4 pairs) | 40.84 | **174.88** | 0.234 (**4.28x**) | 24 / 1.62 / 62 |
| 0.5B Q8_0, temp 0.8, top-k 40 (3 pairs) | 41.39 | **121.97** | 0.339 (**2.95x**) | 23 / 1.30 / 52 |
| 3B Q4_K_M, temp 0.8, top-p 0.9 (2 pairs) | 3.35 | **4.92** | 0.681 (**1.47x**) | 17 / 2.41 / 57 |

The 3B baseline moved between runs (5.11 then 3.35 tok/s) on this box; the two measured
3B ratios are 1.58x and 1.47x, both above the 1.2x bar. The 0.5B numbers are stable
within a few percent and show the batch-forward reuse clearly.

`--speculative` remains opt-in, so this changes no default behavior; it only removes a
silent disable.

## Correctness evidence

- **167/167 tests**, including two new tests:
  - `sampling_cuts_match_the_reference_set`: six parameter sets (top-k, top-p, min-p,
    top-k+top-p, all three, no cuts) checked entry-by-entry against an independent
    reference that implements `sample_token`'s exact selection, including the absolute
    threshold prefilter and the relative min-p floor. Also checks that a cut-away draft
    is always rejected and the replacement is inside the support.
  - `speculative_accept_preserves_the_truncated_distribution`: 40,000 accept/remainder
    draws on a synthetic 8-token row cut with `top_p = 0.75`. Every emitted token is in
    the cut support, and every token frequency is within 0.01 of the normalized cut
    probability (standard error <= 0.0025, deterministic RNG).
- The resolved line now reports `speculation=sampled` for any temperature > 0; the
  `disabled (sampling cuts)` state is gone.
- Sampled speculation does not reproduce plain sampling's exact IDs: the accept rule
  consumes RNG differently. The promise is distribution preservation, not stream
  equivalence. The model-level cross-seed histogram was not usable as a gate here: after
  `The` the 0.5B first-token distribution is effectively one-hot even at temp 1.5, while
  at temp 2.0 it is so flat that 20/20 seeds are unique. The synthetic accept histogram
  is the stronger, deterministic gate.

## Anchors (spike, reverted)

- `include/tge/models/sampling.hpp:83` — `apply_sampling_cuts` declaration.
- `src/models/sampling.cpp:233` — helper implementation (threshold prefilter, min-p
  floor, radix cut, zeroing).
- `apps/cli/main.cpp:750` — `speculate_sampled` fence removed.
- `apps/cli/main.cpp:1092` — cut applied inside `penalized_softmax` (used by the first
  verdict, every verify row, and the bonus draw).
- `apps/cli/main.cpp:625` — `speculation=sampled` reporting.
- `tests/test_sampling.cpp:494` and `:668` — the two tests.
- Patch artifact: `build/idea/sampled-spec-cuts.patch` (also copied to the session
  scratchpad as `sampled-spec-cuts.patch`).
- Binaries: `build/idea/tge-speccuts-base.exe` (master) and
  `build/idea/tge-speccuts-spike2.exe` (final helper).

## Reimplementation recipe

1. In `include/tge/models/sampling.hpp`, declare:

   ```cpp
   void apply_sampling_cuts(std::vector<float>& probabilities, int count,
                            const SamplingParams& params,
                            std::vector<int>& scratch);
   ```

2. In `src/models/sampling.cpp`, implement immediately before `sample_token`. Compute
   `total_all` first. For `top_k > 0` use `partial_sort` over all indices and keep
   exactly k; set `total = sum(top-k)`. Otherwise run the threshold loop, keeping
   indices with `p >= threshold`, recomputing the mass; break on
   `min_p > 0 || top_p >= 1 || mass >= top_p * total_all || threshold <= 0`, halving
   otherwise. Fall back to the argmax if the candidate list is empty. Apply the min-p
   relative floor (`min_p * max(candidates)`) and recompute `total` over survivors. For
   `top_p < 1`, `sort_candidates_by_score` (positive values only — the `~bits` radix key
   is invalid for negative logits), then keep the shortest prefix whose cumulative
   `p / total` reaches `top_p`. Finally zero every index not in the kept set.
3. In `apps/cli/main.cpp`, set
   `speculate_sampled = speculative && sampling.temperature > 0.0F`; in
   `penalized_softmax`, after `softmax_probabilities`, call `apply_sampling_cuts` when
   `top_k > 0 || top_p < 1 || min_p > 0`. Update the resolved label to `sampled`.
4. Add the two tests above. The reference in the mask test must use the **absolute**
   threshold prefilter, not just the relative min-p floor, or it will disagree with
   `sample_token` for `min_p < 1`.
5. Verify: `cmake --build --preset release`; `tge_tests` (expect 167 green);
   `python aikit.py check aegean-tge` with the map regenerated.

## Pitfalls found

- Sorting all 128k candidates per verified row turns the 0.5B win into a 5–7% loss. The
  threshold prefilter is not an optimization detail; it is the difference between the
  idea failing and a 4.28x win.
- `speculative_accept` wants unnormalized rows; do not renormalize the cut. Dividing by
  the kept mass is already inside the rule.
- The bonus token must be drawn from the same cut row as the accept rule. Applying the
  cut only around `speculative_accept` would let the bonus ignore the nucleus.
- The min-p order matters: `sample_token` prefilters at the absolute `min_p`, then
  applies the relative floor. A reference built only from the relative floor keeps a
  slightly different set when `max < 1`.

## Tree and gate state

- Spike reverted (`git status` clean on `master`), branch `idea/sampled-spec-cuts`
  deleted, this report is the only change.
- With the spike present and the generated map refreshed, `aikit.py check aegean-tge`
  reported **4/4 gates passed**. After reverting, the map returns to the committed
  version. Note: the root workspace has an in-progress `scripts/update_knowledge_base.py`
  change (new "Map Coverage" header) that makes the regenerated map differ from the
  committed one independent of this idea; that drift is pre-existing and was left
  untouched.
- 21 commits remain unpushed ahead of `origin/master`; nothing from this cycle was
  pushed.
