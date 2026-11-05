# IDEA-1327: AVX2 Exact-Int8 Dual-Accumulator Q8 GEMV Dot — MEASURED AND KILLED

- **Idea ID**: IDEA-1327 (dossier `scout-2026-11-tge-kernels.md`: "AVX2 Exact-Int8 Dual-Accumulator GEMV Dot Product")
- **Cycle date**: 2026-11-06
- **Verdict**: KILLED — no repeatable >5% gain (falsifier triggered). Single-thread regresses 3.7%; ten-thread gains only +2.2%. Numerics clean, thinking set identical.
- **Tree state at close**: `git status` clean, no diff, spike branch `spike-1327-dual-accum` created and deleted, no spike commit. Stash untouched (`stash@{0}` leftover peer work). `master` still ahead of `origin/master` by the same 8 pre-existing commits — nothing pushed (human gate).

## 1. Mechanism

Split the single `__m256` accumulator chain in the Q8_0 decode dot product into two
independent accumulators so the two FMA/cvt dependency chains overlap and hide
each other's latency:

- Inner level: the four 8-element chunks of each 32-element Q8 block are dealt
  alternately — even chunks (elements 0–7, 16–23) into `accA`, odd chunks
  (8–15, 24–31) into `accB` — then `block_sum = accA + accB`.
- Outer level: even blocks accumulate into `total0`, odd blocks into `total1`;
  one `_mm256_add_ps(total0, total1)` before the existing horizontal sum.
- Everything else unchanged: same `madd`-free widening (`cvtepi8→cvtepi32→cvtps` +
  `fmadd`), same per-block fp16 scale broadcast, same unaligned loads, same
  row banding/thread pool, same `kQ8BlockBytes = 34` / `kQ8BlockElements = 32` layout.

## 2. Predicted gain (with the math)

Dossier premise: single-threaded decode is FMA-latency-bound, so breaking one
4-deep dependent FMA chain (32 elements = 4 sequential `fmadd`s per block, plus one
`fmadd` per block into the running `total` across ~128 blocks for a 4096-wide row)
into two independent chains should approach up to ~2x on the latency-bound portion.
Falsifier threshold taken from the dossier: **>5% GB/s (here: decode t/s) gain,
repeatable best-of-3**, else kill. Decode on the Q8_0 model is ~96% matvec at 1
thread, so a dot-loop win converts almost 1:1 into decode t/s.

## 3. Falsifier (from the dossier, adapted to the available harness)

- Kill if no repeatable >5% gain best-of-3 (`bench gemv` covers only the f32 path,
  so end-to-end `generate` decode t/s on the Q8_0 model is the measuring instrument,
  same binary A/B, warm page cache, interleaved).
- Kill if `tge fidelity` mean KL > 1e-9 vs the scalar oracle, or any top-1 flip.
- Kill if the fixed + creative thinking-set ids diverge between binaries.

## 4. File anchors (current `master`)

- `src/cpu/avx2/q8_gemv_avx2.cpp` — `dot_row_q8()` (the spiked function) and
  `gemv_q8_0_avx2()` (unchanged banding/dispatch).
- `src/cpu/scalar/q8_gemv_scalar.cpp` — `gemv_q8_0_scalar()`, the oracle (untouched).
- `include/tge/cpu/kernels.hpp` — `kQ8BlockBytes`, `kQ8BlockElements`, `Q8GemvArgs`
  (untouched); header comment already notes decode is bandwidth/sync-bound.
- `src/models/qwen2.cpp` (~line 862) — decode matvec call site routing Q8_0
  weights through `kernels().gemv_q8` (untouched).
- `src/core/dispatch.cpp` (line 21) — dispatches `gemv_q8` to AVX2 vs scalar (untouched).
- `tests/test_q8.cpp` — `q8_gemv_matches_scalar_reference` tolerance
  `1e-4 * cols + 1e-3` (spike passed).
- Note: `tge bench gemv` exercises only the **f32** `gemv_avx2` path; `tge bench batch
  --format q8` exercises `gemv_q8_batch_avx2` (multi-token prefill), **not**
  `gemv_q8_0_avx2`. There is no isolated bench for the spiked kernel, hence
  end-to-end `generate` on the Q8_0 model.

## 5. What was done (temporary spike, fully reverted)

1. `git checkout -b spike-1327-dual-accum` (no commit ever made on it).
2. Edited only `dot_row_q8()` as described in §7; no other file touched.
3. `cmake --build --preset release` — clean (54/54).
4. `ctest --preset release` — pass; `tge_tests.exe` — **141/141, 0 failed**.
5. Spike `generate` (Qwen2.5-0.5B-Q8_0, `--gpu off` to force the CPU kernel,
   warm page cache): 1-thread best-of-3 + ids; saved spike `tge.exe` to temp.
6. `git checkout -- src/cpu/avx2/q8_gemv_avx2.cpp`, rebuilt baseline, re-ran
   `ctest` (pass, 141/141), baseline `generate` best-of-3 + ids + `fidelity`.
7. Cross-checked 10-thread best-of-3 on both binaries; second creative prompt on
   both binaries for the thinking-set gate.
8. `python ../../aikit.py check aegean-tge` — 3/4 (hygiene INCONCLUSIVE: no
   scannable Python/asset files in this C++-only project; guidelines + secrets pass;
   pre-existing, unrelated to the spike which touched no Python).
9. Deleted the branch (`git branch -D spike-1327-dual-accum` — pointer was still at
   `master` HEAD, nothing committed), verified `git status` clean and
   `git stash list` unchanged. No push.

## 6. Measured numbers

Model: `qwen2.5-0.5b-instruct-q8_0.gguf` (Q8_0 decode path). Prompt A:
"The quick brown fox jumps over the lazy dog. Explain why." (`--n 32`, 31 decoded
tokens, `--gpu off`). Prompt B (thinking set): "Write a haiku about circuits
dreaming." (`--n 24`, 23 tokens). Box note: loaded machine; ratios within one
session are reliable, absolutes carry the documented ±20%.

### 6a. Single-thread decode (the hypothesis's best case), Prompt A

| run | baseline `--threads 1` | spike `--threads 1` |
|---|---|---|
| 1 | 21.50 t/s (1442.15 ms) | 20.70 t/s (1497.39 ms) |
| 2 | 21.42 t/s (1447.17 ms) | 20.55 t/s (1508.79 ms) |
| 3 | 21.35 t/s (1452.06 ms) | 20.66 t/s (1500.75 ms) |
| **best** | **21.50** | **20.70 (−3.7%)** |

All 3 spike runs slower than all 3 baseline runs: a consistent regression, not noise.

### 6b. Ten-thread decode, Prompt A

| run | baseline `--threads 10` | spike `--threads 10` |
|---|---|---|
| 1 | 83.71 t/s | 88.21 t/s |
| 2 | 84.58 t/s | 87.24 t/s |
| 3 | 86.28 t/s | 87.84 t/s |
| **best** | **86.28** | **88.21 (+2.2%)** |

Direction consistent (3/3 spike above baseline) but **+2.2% < 5% falsifier bar**.

### 6c. Fidelity (teacher-forced scalar oracle vs AVX2 candidate)

Corpus: 86 positions from a fixed English text file, `--context 512 --tokens 200
--threads 1 --gpu off`.

| | baseline | spike |
|---|---|---|
| mean KL | 2.121e-11 nats | 1.890e-11 nats |
| worst KL | 5.858e-10 | 2.789e-10 |
| max \|logit diff\| | 1.621e-04 | 1.117e-04 |
| top-1 agreement | 86/86 | 86/86 |

Both far under the 1e-9 gate. Reordered summation prices at ~1e-11: numerics are free.

### 6d. Thinking set (fixed + creative)

- Prompt A ids byte-identical baseline vs spike
  (`576 3974 13876 38835 ... 1345 1854 13`, 31 ids; promptids identical).
- Prompt B ids byte-identical baseline vs spike
  (`758 264 1879 ... 25722 23607 11`, 23 ids; promptids identical).
- Verdict: no behavioral delta; the kernel change is fidelity-neutral and
  text-neutral.

## 7. Full reimplementation recipe (to rebuild this exact spike from `master`)

Touch exactly one function in exactly one file:
`dot_row_q8()` in `src/cpu/avx2/q8_gemv_avx2.cpp` (lines ~27–59 on `master`).

1. Replace `__m256 total = _mm256_setzero_ps();` with two running totals:
   `__m256 total0 = _mm256_setzero_ps(); __m256 total1 = _mm256_setzero_ps();`
2. Inside the per-block loop (after loading `scale`, `values`, `vector_block`),
   replace the single `__m256 accumulator` + `for (index = 0; index < 32; index += 8)`
   loop with two accumulators `accA`, `accB` (both zeroed) and four explicit,
   unrolled chunks (no loop):
   - chunk 0: `p0 = _mm_loadl_epi64(values+0)`, `f0 = cvt(p0)`,
     `accA = fmadd(f0, loadu(vector_block+0), accA)`;
   - chunk 1: same at offset 8 into `accB`;
   - chunk 2: offset 16 into `accA`;
   - chunk 3: offset 24 into `accB`.
   Each cvt is `_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(p))`, each multiply is
   `_mm256_fmadd_ps(f, _mm256_loadu_ps(...), accX)` — identical intrinsics to the
   baseline, only the destination accumulator alternates.
3. After the four chunks: `block_sum = _mm256_add_ps(accA, accB)`; then
   `if ((block & 1) == 0) total0 = _mm256_fmadd_ps(scale, block_sum, total0);`
   `else total1 = _mm256_fmadd_ps(scale, block_sum, total1);`
4. After the block loop: `total = _mm256_add_ps(total0, total1);` then keep the
   existing horizontal-sum epilogue verbatim (extract128, add, movehl, shuffle,
   `cvtss`).
5. Constants/layout: unchanged — `kQ8BlockBytes = 34` (2-byte fp16 scale +
   32 int8), `kQ8BlockElements = 32`, row stride `blocks * 34`, unaligned loads
   only, no new includes, no dispatch or header changes.
6. Rebuild with `cmake --build --preset release`, expect `tge_tests` 141/141
   (the `q8_gemv_matches_scalar_reference` tolerance absorbs the reorder).

Kill-grade note: the extra `_mm256_add_ps` per block plus doubled running state
costs more than the saved stall on this memory-bound loop — matches the standing
header comment ("decode here is limited by memory and per-call synchronisation,
not by the widening chain").

## 8. Verdict and why

**KILLED by its own falsifier.** The dossier's required repeatable >5% gain never
appears: −3.7% single-threaded (the mechanism's claimed best case), +2.2%
ten-threaded (consistent but under bar). Fidelity (KL ~2e-11, 86/86) and
thinking-set ids (both prompts identical) pass, so the idea is *safe but useless*:
it buys nothing on a bandwidth-bound loop and adds one extra add per block. Do not
revisit without (a) a single-thread latency-bound profile contradicting today's
96%-matvec/20–21 t/s measurement, or (b) new silicon where the cvt+FMA chain, not
DRAM, is the ceiling.

## 9. Exact commands run and their results

- `git checkout -b spike-1327-dual-accum` → new branch at `d1bca9a`-era `master`.
- Edit `src/cpu/avx2/q8_gemv_avx2.cpp` (dual-accumulator `dot_row_q8`).
- `cmake --build --preset release` → 54/54, link OK.
- `ctest --preset release` → 1/1 pass; `tge_tests.exe` → 141 cases, 0 failed.
- `tge.exe fidelity --model <q8_0> --corpus <temp 86-pos> --context 512 --tokens 200 --threads 1 --gpu off` (spike) → mean KL 1.890e-11, 86/86.
- `tge.exe generate --model <q8_0> --prompt <fox> --n 32 --threads 1 --gpu off` ×3 (spike) → 20.70 / 20.55 / 20.66 t/s; ids `576 ... 1854 13`.
- Saved spike binary to temp (`tge_spike_1327.exe`).
- `git checkout -- src/cpu/avx2/q8_gemv_avx2.cpp` → file reverted; `git diff --stat` empty.
- `cmake --build --preset release` → 54/54; `ctest` → pass, 141/141.
- Same `generate` ×3 (baseline, 1T) → 21.50 / 21.42 / 21.35 t/s; ids identical.
- Same `fidelity` (baseline) → mean KL 2.121e-11, 86/86.
- `generate --threads 10` ×3 both binaries → baseline best 86.28, spike best 88.21 (+2.2%).
- `generate <haiku> --n 24 --threads 10 --gpu off --print-ids` both binaries → ids identical (`758 ... 23607 11`).
- `python ../../aikit.py check aegean-tge` → 3/4 (hygiene INCONCLUSIVE, pre-existing; guidelines + secrets pass).
- `git checkout master; git branch -D spike-1327-dual-accum` → branch deleted, nothing committed; `git status` clean; stash list unchanged; ahead count unchanged (8, unpushed per human gate).
