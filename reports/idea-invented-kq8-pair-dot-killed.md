# INVENTED: two-row Q4_K x Q8_K dot sharing the activation block — KILLED (+7% isolated at chunk sizes, 0.91-0.99x end-to-end)

- **Idea**: Follow-up prescribed by `idea-invented-kq8-band1-killed.md`: the kq8 prefill kernel is compute-bound in the maddubs chain, and each weight row re-loads the whole Q8_K activation block. A two-row Q4_K dot that loads each 64-value q8 chunk once and computes two weight-row dots from it would halve the activation-side loads and raise maddubs issue efficiency. Target: >= 1.15x at 64 tokens in `bench batch` before any model wiring; then end-to-end prefill.
- **Cycle date**: 2026-11-11
- **Verdict**: **KILLED.** Isolated kernel gains are real but small and shape-dependent: **1.071x at 64 tokens** (372.4 vs 347.3 GFLOPS), **1.071x at 128** (374.9 vs 350.1), **1.067x at 256** (375.9 vs 352.3), **0.986x at 32**, **0.968x at 1**, **0.845x at 8**. End-to-end prefill is **0.911x / 0.967x** at 510 tokens and **0.994x** at 2,050 — no win. The pre-stated 1.15x at 64 tokens was not met, so the kernel never qualified for model wiring; the end-to-end A/B was run anyway for the record and confirms the kill.
- **Tree state at close**: spike reverted from `src/cpu/avx2/dequant_batch_kq8_avx2.cpp`; `git status` clean; stash holds only the pre-existing 2026-11-04 peer entry; rebuild clean, `tge_tests` **171/171**, `python aikit.py check aegean-tge` **4/4**. No branch, no spike commit, nothing pushed.

## 1. Mechanism

`gemv_dequant_batch_kq8_avx2` quantises every token row to Q8_K once, then bands weight rows; inside a band each (row, token) dot is `dot_q4k_q8k`, which per 64 values loads two 32-byte q8 chunks and one 32-byte q4 chunk. The spike adds `dot_q4k_q8k_pair(xa, xb, y, blocks, out[2])`: the same block loop, but `utmp`/`mins_and_scales`/`bsums`-based min terms are computed for both rows, the 64-value loop loads `q8l`/`q8h` once and runs both rows' nibble masks, `maddubs`+`madd` chains. All arithmetic per row is identical to `dot_q4k_q8k` (same order, same masks, same scale shuffle), so results are bit-identical. The Q4_K branch of the batch kernel steps two rows at a time (single-row tail), Q6_K stays single.

## 2. Predicted gain (with the math)

Per 64 values a single-row dot moves 64 B of q8 + 32 B of q4 + 2 B scales, and issues 6 `maddubs`/`madd` ops; the pair moves the same q8 (64 B) plus 2x q4 (64 B) and issues 12 ops for two rows — i.e. q8 bytes per row halve (64 -> 32) and the q8 load/uop count halves, while the madd chain doubles usefully. If issue slots or load ports were the limiter, 1.15-1.3x. If the kernel is latency/port-bound on maddubs only, close to 1.0x. Measured 1.07x at sustained sizes — the madd chain, not the q8 loads, is the limiter.

## 3. Falsifier (stated before the spike)

**PROCEED** if `bench batch --rows 3072 --cols 3072 --tokens 64 --format q4k --kernel kq8` gains >= 1.15x over the single-row kernel, with the row differential still exact. **KILL** otherwise (and no model wiring).

Measured 1.071x -> kill. The batch row-differential test (`dequant_batch_kq8_avx2_matches_generic`) stayed green, confirming exactness.

## 4. File anchors (current HEAD, read-before-write verified)

- `src/cpu/avx2/dequant_batch_kq8_avx2.cpp:143-216` — `dot_q4k_q8k` (the row dot the pair mirrors).
- `:222-313` — `dot_q6k_q8k` (unchanged).
- `:324-387` — `gemv_dequant_batch_kq8_avx2` (banding, token loop the pair loop replaces).
- `:359-363` — band computation (the previous cycle's knob).
- `apps/cli/main.cpp` `command_batch_bench` — the isolation harness (`--tokens N --format q4k --kernel kq8`).
- `reports/idea-invented-kq8-band1-killed.md` — why the traffic hypothesis failed and this follow-up.
- `reports/llama-cpp-comparison-2026-11-11.md` — the prefill gap this line is chasing.

## 5. What was done (temporary spike, fully reverted)

Added `dot_q4k_q8k_pair` (two-row shared-activation dot) and a Q4_K pair loop in the batch kernel (single-row tail, Q6_K unchanged); built; 171/171; benchmarked isolated at tokens 1/8/32/64/128/256 against a baseline binary built from the stashed tree; ran end-to-end prefill A/B at 510 (2 pairs) and 2,050 (1 pair); reverted; rebuilt; 171/171; gates 4/4; clean tree.

## 6. Measured numbers

### Isolated (`bench batch --rows 3072 --cols 3072 --repeat 3 --format q4k --kernel kq8`)

| tokens | single GFLOPS | pair GFLOPS | ratio |
|---:|---:|---:|---:|
| 1 | 51.44 | 49.79 | 0.968 |
| 8 | 256.62 | 216.82 | 0.845 |
| 32 | 287.57 | 283.45 | 0.986 |
| 64 | 347.25 | 372.37 | **1.071** |
| 128 | 350.12 | 374.94 | **1.071** |
| 256 | 352.32 | 375.89 | **1.067** |

### End-to-end prefill (`generate --n 4 --threads 10 --gpu off`)

| tokens | pair | single t/s | pair t/s | ratio |
|---:|---:|---:|---:|---:|
| 510 | 1 | 68.3 | 62.2 | 0.911 |
| 510 | 2 | 57.2 | 55.3 | 0.967 |
| 2050 | 1 | 51.0 | 50.7 | 0.994 |

Ids unchanged (both arms exact).

## 7. Why it failed

- The isolated +7% is real at chunk sizes >= 64, but prefill spends a large share outside the kq8 matvec (attention, norms, and the non-KQ8 LM head), and the model's row sets are banded so unevenly that the pair loop's two-row stride can perturb the worker balance (the 510-token pair 1 shows -9% while the isolated kernel is +7%).
- The 8-token regression (-15%) is the give-away that the pair's extra live registers compete with the q8 chunk registers; the 16-YMM budget again (same family as the dot8acc kill). At sustained sizes the compiler can hide it; below that it cannot.
- The remaining prefill gap to llama is therefore *not* reachable by sharing activation loads at two rows. It needs a genuinely different maddubs schedule: 4-value `vpmaddubsw` with wider accumulators is already what we do; the next candidate is a 2x2 tile (two rows x two 64-value chunks) with q4 double-buffering, or accepting VNNI's priced fidelity.

## 8. Reimplementation recipe (rebuild only to re-verify the kill)

1. Add `inline void dot_q4k_q8k_pair(const Q4KBlock* xa, const Q4KBlock* xb, const Q8KBlock* y, int blocks, float* out)` beside `dot_q4k_q8k`: per block unpack both rows' 12 scale bytes into `utmp_a`/`utmp_b` with the same kmask sequence, build both `mins_and_scales`, load `y[i].bsums` once and compute both min-term products, then a 64-value loop that loads `q8l`/`q8h` once, loads both q4 chunks, and runs the two maddubs/madd chains into `sumi_a`/`sumi_b`; finish with the same two-level float reduction per row.
2. In `gemv_dequant_batch_kq8_avx2`, for Q4_K step two rows (`index + 1 < end`) calling the pair, with a single-row tail; leave Q6_K single.
3. Build + `tge_tests` 171/171 (the differential is exact). `bench batch --tokens 64 --format q4k --kernel kq8` -> expect ~1.07x, below gate; end-to-end prefill 0.91-0.99x. Kill.

## 9. Exact commands run and results

```text
cmake --build projects\aegean-tge\build\release --config Release   -> clean (56/56)
projects\aegean-tge\build\release\bin\tge_tests.exe                -> 171 cases, 0 failed
tge bench batch --rows 3072 --cols 3072 --tokens {1|8|32|64|128|256} --repeat 3 \
  --format q4k --kernel kq8                                      -> §6 isolated table
tge generate ... --prompt <fox x51|x205> --n 4 --threads 10 --gpu off -> §6 e2e table
git checkout -- src/cpu/avx2/dequant_batch_kq8_avx2.cpp
cmake --build ... -> clean; tge_tests -> 171/171
python aikit.py check aegean-tge                                 -> 4/4 gates passed
```

No push, no merge, no force-push. Baseline binary lives at `build/idea/tge-pair-base.exe`; the committed tree is unchanged except this report.
