# INVENTED: greedy sampler fast path (skip the vocab copy) — KILLED on share, but it exposed a 13x scalar argmax

- **Idea**: `sample_token` always copies the full logits row into `scores` and runs the penalty pass before the greedy argmax. A no-penalty greedy call can argmax the logits directly. The copy is pure overhead and the returned token is bit-identical.
- **Cycle date**: 2026-11-08
- **Verdict**: **KILLED on share.** The fair two-binary A/B shows the library call **138.6 µs -> 127.6 µs (1.086x, 11 µs saved per token)**. At the reference model's decode rate (~250 ms/token) that is **0.004%**; at the 0.5B's ~22 ms/token it is **0.05%**, far below any keep bar. The more important finding is inside the same measurement: the library's argmax scan is **127 µs**, while an identical loop in the test translation unit is **9-10 µs** — the reduction is not vectorized in `sampling.cpp`. That is a separate, larger lever.
- **Tree state**: spike reverted from `src/models/sampling.cpp` and `tests/test_sampling.cpp`; branch `idea/greedy-sampler-fastpath` deleted; `git status` clean; master rebuilt 162/162; nothing committed except this report.

## 1. Mechanism

`samples_token` (sampling.cpp):

```text
validate...
scores.assign(logits, logits + count);        // 128,256 floats
apply_repeat_penalty(scores, ...);            // no-op when penalty == 1
if (temperature <= 0) { scalar argmax over scores; }
```

For greedy with no active penalty, the copy and the penalty call do nothing useful. The spike inserts, immediately after validation:

```text
const bool penalty_active = repeat_penalty > 0 && repeat_penalty != 1 &&
                            history != nullptr && history_size > 0;
if (temperature <= 0 && !penalty_active) {
  scalar argmax over logits; return best;
}
```

The argmax sees exactly the same values (the penalty pass is a no-op in this case), so the token is bit-identical.

## 2. Falsifier

- **KEEP** if the absolute saving is at least 0.5% of decode time on the reference model (or if the sampler call improves enough to matter at the measured token rate).
- **KILL** if the absolute saving is below that bar.

Measured: 11 µs/token, <=0.05% of decode at the fastest observed rate. Kill.

## 3. Measured numbers

Two test binaries with the same bench test, differing only in `sample_token`; 2000 greedy calls on a 128,256-float row, best of three rounds:

| binary | per call | note |
|---|---:|---|
| baseline (`copy + penalty + argmax`) | 138.6 µs | 277.246 ms / 2000 |
| fast path (`argmax` on logits) | 127.6 µs | 255.161 ms / 2000 |
| hand-written copy+argmax inside the test TU | 9.0-10.0 µs | same operations, vectorized by the compiler |

- Fast path speedup: **1.086x**, absolute saving **11 µs/token**.
- Decode share: 11 µs / 22 ms (0.5B at 45 t/s) = **0.05%**; 11 µs / 250 ms (3B at 4 t/s) = **0.004%**.
- The copy itself is ~11 µs; the remaining 127 µs is the scalar argmax. The test-TU loop runs 13-14x faster, so the reduction in `sampling.cpp` is not vectorized while the test's copy+scan is.
- Ids match the fixed thinking set exactly (`320 2028 ... 16053` and `4815 34 ... 9290 25`); suite 163/163.

## 4. Why it was killed

The copy is real but tiny: 0.5 MB per token at ~45 GB/s effective is 11 µs, and decode is dominated by weight streaming (22-250 ms/token). A 0.004-0.05% gain adds a branch to the sampler for no measurable engine effect. Keep the uniform copy+penalty+argmax path.

## 5. Full reimplementation recipe

1. `src/models/sampling.cpp`, `sample_token`, after `validate_sampling_params` and before `scores.assign`:
   `const bool penalty_active = params.repeat_penalty > 0.0F && params.repeat_penalty != 1.0F && history != nullptr && history_size > 0;`
2. If `params.temperature <= 0.0F && !penalty_active`, scan `logits` directly for the strict-greater argmax and return it.
3. Nothing else changes; the slow paths keep the copy.
4. Bench recipe: a `TGE_SAMPLER_BENCH=1` test that times 2000 greedy calls on a 128,256-float row, in two binaries (baseline vs spike) to avoid compiler differences between translation units.
5. Expected: same ids, 1.08x on the sampler call, <0.1% end-to-end.

## 6. The real follow-up: vectorize the argmax

The same measurement says `sample_token`'s greedy scan is **127 µs** for 128k floats (~0.4 floats/cycle), while a compiler-vectorized copy+scan of the same data is 9-10 µs (~13 G floats/s). The reduction is a serial max/index loop in `sampling.cpp`, which is compiled without AVX2; even SSE2 could cut it several-fold with a max-value pass plus a first-index pass.

Where it matters:

- Greedy single-token decode: 127 µs/token -> ~0.05% at 0.5B, ~0.005% at 3B (still small).
- Speculative verification: `Qwen2::argmax` (also scalar) runs over up to 6 x 128k logits per step; at 127 µs/row that is ~0.76 ms/step, ~3% of a 0.5B speculative step and ~0.3% of a 3B step.
- Continuous-batch benches use the same argmax per decode row.

Next cycle candidate: a vectorized argmax helper (AVX2/SSE2, first-max index) shared by `sample_token` and `Qwen2::argmax`, with a microbench at 128k and an end-to-end speculative decode A/B. Falsifier: >=3% faster speculative decode on the 0.5B model with identical ids, or >=1% at the reference 3B; otherwise report the kernel win and keep the default path.
