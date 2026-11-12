# INVENTED: split integer accumulators in the kq8 dots — KILLED (neutral; kernel is not add-chain-bound)

- **Idea**: `dot_q4k_q8k` accumulates every 32-value half into one `__m256i sumi` (`sumi += p16l + p16h`, four serial adds per block). `dot_q6k_q8k` does the same with two adds per block. Integer adds are associative, so splitting into two accumulators (`sumi0 += p16l`, `sumi1 += p16h`, final add) is bit-identical and halves the dependent chain — the classic fix if the loop is add-latency-bound. Target: prefill chunk sizes (32-256 tokens), where the kq8 kernel is compute-bound.
- **Cycle date**: 2026-11-11
- **Verdict**: **KILLED.** Interleaved best-of-3 at 3072x3072 Q4_K: 32 tokens 397.5 vs 415.7 GFLOPS (0.96x), 64 tokens 399.6 vs 415.0 (0.96x), 128 tokens 400.5 vs 396.4 (1.01x). No gain; the serial add is not the limiter. Bit-identical, as expected from integer associativity (differential test green).
- **Tree state at close**: spike reverted from `src/cpu/avx2/dequant_batch_kq8_avx2.cpp`; `git status` clean; stash holds only the pre-existing 2026-11-04 peer entry; rebuild clean, `tge_tests` **171/171**, `python aikit.py check aegean-tge` **4/4**. No branch, no spike commit, nothing pushed.

## 1. Mechanism

Both dots (`src/cpu/avx2/dequant_batch_kq8_avx2.cpp:143-216`, `:222-313`) end each 64-value iteration with an add into a single `sumi`. Spike: two accumulators, `sumi0` for the low-nibble `p16l` and `sumi1` for the high-nibble `p16h` in Q4_K; `sumi0` for `p16_0+p16_1` and `sumi1` for `p16_2+p16_3` in Q6_K; a single `add` before the per-block `fmadd(cvt(sumi), d)`. Integer addition is exact under regrouping, so outputs are bit-identical.

## 2. Predicted gain (with the math)

Per block the Q4_K loop has 4 dependent adds (each `sumi = add(sumi, add(p16l,p16h))` is two adds with one on the critical path); the madd chain feeding them is longer: 4 x (2 maddubs + 2 madd) = 16 ops. With 2 accumulators the add path halves to 2 serial adds while madd work is unchanged, so the prediction was 1.03-1.08x only if adds were on the critical path. Measured <= 1.01x, i.e. the adds were already hidden under the madd throughput.

## 3. Falsifier (stated before the spike)

**PROCEED** if best-of-3 at 64 or 128 tokens gains >= 1.05x over the single-accumulator binary. **KILL** otherwise.

Measured 0.96/0.96/1.01 -> kill.

## 4. File anchors (current HEAD, read-before-write verified)

- `src/cpu/avx2/dequant_batch_kq8_avx2.cpp:143-216` — `dot_q4k_q8k` (single `sumi`, four adds/block).
- `:222-313` — `dot_q6k_q8k` (single `sumi`, two adds/block).
- `:324-387` — `gemv_dequant_batch_kq8_avx2` (the prefill caller).
- `apps/cli/main.cpp` `command_batch_bench` — isolation harness.
- `reports/idea-invented-kq8-band1-killed.md`, `reports/idea-invented-kq8-pair-dot-killed.md` — the two preceding prefill attempts; all three now bound the same conclusion: the madd chain is the ceiling, not loads, not bands, not adds.

## 5. What was done (temporary spike, fully reverted)

Split `sumi` in both dots, built, 171/171 (differential exact), built a single-accumulator baseline binary, ran interleaved best-of-3 at 32/64/128 tokens, reverted, rebuilt, 171/171, gates 4/4, clean tree. Box variance in this window was large (base 285-415 GFLOPS across runs of the same binary), which is why the table uses interleaved best-of and why the 32/64 negative readings are read as "no gain" rather than a loss.

## 6. Measured numbers

`bench batch --rows 3072 --cols 3072 --repeat 3 --format q4k --kernel kq8`, three runs per arm, interleaved base/split.

| tokens | base GFLOPS (3 runs) | best | split GFLOPS (3 runs) | best | best ratio |
|---:|---|---:|---|---:|---:|
| 32 | 285.1, 279.4, 415.7 | 415.7 | 374.8, 263.8, 397.5 | 397.5 | 0.956 |
| 64 | 311.3, 415.0, 388.7 | 415.0 | 349.5, 392.1, 399.6 | 399.6 | 0.963 |
| 128 | 396.4, 291.2, 396.3 | 396.4 | 390.6, 400.5, 389.8 | 400.5 | 1.010 |

Ids/differential: `dequant_batch_kq8_avx2_matches_generic` and the row differentials all green (bit-identical regrouping).

## 7. Why it failed and what it leaves

- The critical path is the `maddubs`+`madd` chain feeding the add, not the add; the compiler already interleaves the four iterations' madd work, hiding the add latency.
- Combined with the band and two-row results, the kq8 prefill kernel is now bounded on three sides: activation re-sweep (band change, neutral), activation loads across rows (pair, +7% isolated only), and add latency (this, neutral). The remaining gap to llama's prefill is in madd throughput/issue efficiency; a larger structural change (different q4 layout with per-64 madd input, or AVX-512 on another machine) would be needed, not a loop tweak. VNNI remains the priced alternative (`--vnni`, KL 1.5e-3 class).
- The measurement lesson is recorded: this box drifts up to 45% between runs of the same binary; all kernel comparisons must interleave arms and quote best-of-3.

## 8. Reimplementation recipe (rebuild only to re-verify the kill)

1. In `dot_q4k_q8k`: declare `sumi0`/`sumi1`; replace the per-j add with `sumi0 += p16l; sumi1 += p16h;`; after the loop `const __m256i sumi = add(sumi0, sumi1);` before the `fmadd`.
2. In `dot_q6k_q8k`: same split with `sumi0 += p16_0+p16_1; sumi1 += p16_2+p16_3;`, final add, then the existing `q8sclsub` subtract.
3. Build + `tge_tests` 171/171; build a baseline binary from the reverted tree; run interleaved `bench batch --tokens 32|64|128` with `--repeat 3` and compare best-of-3 -> expect 0.95-1.01x. Revert.

## 9. Exact commands run and results

```text
cmake --build projects\aegean-tge\build\release --config Release   -> clean (56/56)
projects\aegean-tge\build\release\bin\tge_tests.exe                -> 171 cases, 0 failed
tge bench batch --rows 3072 --cols 3072 --tokens {32|64|128} --repeat 3 --format q4k --kernel kq8
  base best 415.7 / 415.0 / 396.4 ; split best 397.5 / 399.6 / 400.5
git checkout -- src/cpu/avx2/dequant_batch_kq8_avx2.cpp
cmake --build ... -> clean; tge_tests -> 171/171
python aikit.py check aegean-tge                                   -> 4/4 gates passed
```

No push, no merge, no force-push. Baseline binary lives at `build/idea/tge-sumi1-base.exe`; the committed tree is unchanged except this report.
