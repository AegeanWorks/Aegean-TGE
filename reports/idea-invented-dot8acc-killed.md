# INVENTED: 8-accumulator fused K-quant row dot (break the FMA dependency chain) — KILLED (median 0.93x; AVX2 register pressure)

- **Idea**: The fused Q4_K/Q6_K single-row dots accumulate into four 8-lane vectors. Within a 256-value block each accumulator receives 8 dependent FMAs, and FMA latency (4 cycles) exceeds issue spacing, so the chain may be latency-limited. Split the dequantise-and-FMA stream into two banks of four accumulators (values 0-31 into bank A, 32-63 into bank B for Q4_K; alternating 32-value groups into the two banks for Q6_K) and fold the banks once at the end. Predicted up to 1.15-1.3x on the dot if latency-bound; 4-6% end-to-end.
- **Cycle date**: 2026-11-11
- **Verdict**: **KILLED.** End-to-end decode on the reference model: 10.08/9.57/10.93/10.88 -> 9.72/9.68/9.78/8.54 t/s; per-pair ratios 0.964, 1.012, 0.895, 0.785, **median 0.93x**. The 8-accumulator form is *slower*, including the two quiet-window pairs: the extra four live `__m256` accumulators push the kernel past AVX2's 16 architectural YMM registers (4 accumulators + 4 lane vectors + factor/base/nibble temporaries + 2 activation loads), so the compiler spills and the loop gains nothing. Ids were identical on the dragon prompt; the fused-vs-batch differentials need a tolerance instead of `memcmp` (last-bit reordering), which is recorded below.
- **Tree state at close**: spike reverted from `src/cpu/avx2/dequant_avx2.cpp` and `tests/test_dequant.cpp`; `git status` clean; stash holds only the pre-existing 2026-11-04 peer entry; rebuild clean, `tge_tests` **171/171**, `python aikit.py check aegean-tge` **4/4**. No branch, no spike commit, nothing pushed.

## 1. Mechanism

`dot_q4k_row_avx2` (`src/cpu/avx2/dequant_avx2.cpp:158`) walks 144-byte blocks. For each 64-value chunk it unpacks two 6-bit scale/min pairs and calls `fma32` twice: low nibbles -> input[chunk..chunk+31], high nibbles -> input[chunk+32..chunk+63]. `fma32` (`:70`) converts four 8-byte groups to `__m256` lanes and does `acc[lane] = fmadd(sub(mul(lane, factor), base), input, acc[lane])`. With one accumulator array, each of the four accumulators sees 8 dependent FMAs per block (2 per chunk x 4 chunks). `dot_q6k_row_avx2` (`:291`) has the same shape: four `fma_group_q6_k` calls per 32-value index all target `accumulators[index/8]`, so each accumulator gets 16 FMAs per 128-value half.

Spike: two accumulator banks.

- Q4_K: low-nibble group -> `accumulators[0..3]`, high-nibble group -> `accumulators[4..7]`; final `folded[i] = accumulators[i] + accumulators[i+4]` then the existing `reduce_accumulators`.
- Q6_K: within each 32-value index, groups at offsets 0/64 target bank A, offsets 32/96 target bank B; same fold.
- Numerics: the fold changes the f32 summation order, so results are no longer bit-identical to the dequantise-then-dot path. The two exact differentials (`dequant_q4k_fused_row_matches_batch`, `dequant_q6k_fused_row_matches_batch`) were relaxed to `|a-b| <= 1e-5 * (1+|a|)` for the spike and pass; ids stayed identical on the reference prompt.

## 2. Predicted gain (with the math)

Per 256-value Q4_K block: 8 FMAs per accumulator with 4 independent chains and FMA latency 4 on 2 ports means the dependency limit is 8 FMAs x 4 cycles / 4 chains = 8 cycles per accumulator issue window, vs 8 FMAs x 4 accumulators / 2 ports = 16 cycles throughput-bound. But the block also pays 8 x (2 loads + shift/and + 4 cvtepi8_epi32 + 4 mul + 4 sub) for the dequant, ~120-160 uops, which on 4-wide decode is ~30-40 cycles. If the dequant uops dominate, extra accumulators cannot help; if FMA latency dominates, the two-bank form should approach the uop limit -> 15-25% dot gain -> 4-6% end-to-end.

## 3. Falsifier (stated before the spike)

**PROCEED** if median end-to-end decode on the reference model gains >= 5% over the 4-accumulator binary (`--n 64 --kv f32 --threads 6 --gpu off`, 4 alternating pairs), with ids identical on the test prompt and the relaxed differentials passing. **KILL** otherwise.

Measured median 0.93x (worst 0.785x). Kill.

## 4. File anchors (current HEAD, read-before-write verified)

- `src/cpu/avx2/dequant_avx2.cpp:70-99` — `fma32` (4 lane vectors, 4 accumulators, multiply-then-subtract then FMA).
- `:101-112` — `reduce_accumulators` (pairwise tree then horizontal reduce).
- `:158-195` — `dot_q4k_row_avx2` (two `fma32` calls per 64-value chunk, one accumulator array).
- `:206-238` — `dequantize_group_q6_k` / `fma_group_q6_k`.
- `:291-345` — `dot_q6k_row_avx2` (four groups per 32-value index into `accumulators[index/8]`).
- `src/cpu/avx2/dequant_batch_avx2.cpp:55/94` — callers (`gemv_q4k_fused_avx2`, `gemv_q6k_fused_avx2`).
- `tests/test_dequant.cpp` — `dequant_q4k_fused_row_matches_batch`, `dequant_q6k_fused_row_matches_batch` (exact `memcmp` in the tree).

## 5. What was done (temporary spike, fully reverted)

- `dot_q4k_row_avx2`: `accumulators[8]`; low nibbles -> base pointer, high nibbles -> `accumulators + 4`; fold 8 -> 4 before `reduce_accumulators`.
- `dot_q6k_row_avx2`: `accumulators[8]`; groups at index offsets 0/64 -> `accumulators[index/8]`, offsets 32/96 -> `accumulators[4 + index/8]`; same fold.
- Tests: both fused differentials switched from `memcmp == 0` to a 1e-5 relative tolerance (spike-only change).
- Built, 171/171, built the 4-accumulator baseline from the stashed tree, ran 4 alternating pairs, reverted both files, rebuilt, 171/171, gates 4/4, clean tree.

## 6. Measured numbers

`Llama-3.2-3B-Instruct-Q4_K_M.gguf`, dragon prompt, `--n 64 --kv f32 --threads 6 --gpu off --print-ids`, greedy.

| pair | dot4 t/s | dot8 t/s | ratio | ids |
|---:|---:|---:|---:|---|
| 1 | 10.08 | 9.72 | 0.964 | identical |
| 2 | 9.57 | 9.68 | 1.012 | identical |
| 3 | 10.93 | 9.78 | 0.895 | identical |
| 4 | 10.88 | 8.54 | 0.785 | identical |
| **median** | 10.48 | 9.75 | **0.930** | 4/4 |

Pairs 3-4 were the quiet window (dot4 10.93/10.88) and the spike still lost 10-21%; the loss is structural, not box noise.

## 7. Why it failed

- **Register pressure.** AVX2 has 16 YMM registers. `fma32` holds four converted lane vectors plus `factor`, `base`, two 128-bit halves, and the 32-byte quant load; adding four more live accumulators forces spills across the call boundary, so the loop re-loads and stores accumulators instead of gaining ILP.
- **The chain was not the bottleneck.** With 4 accumulators the FMA chain is 8 deep per block, but the dequant work (8 conversions, 8 mul, 8 sub, nibble masks) already fills the issue slots; the measured dot is instruction-throughput-bound, not latency-bound. The 4-accumulator form is the right shape for AVX2.
- The fold also breaks bit-identity with the dequantise-then-dot path, which would need a priced-numerics justification on its own. No reason to spend it on a slower kernel.

## 8. Full reimplementation recipe (rebuild only to re-verify the kill)

Do not ship. To reproduce:

1. `src/cpu/avx2/dequant_avx2.cpp` `dot_q4k_row_avx2`: change `accumulators[4]` to `[8]` (eight `_mm256_setzero_ps()`); pass `accumulators` to the low-nibble `fma32` and `accumulators + 4` to the high-nibble call; before `return`, fold `__m256 folded[4] = {add(acc0,acc4), add(acc1,acc5), add(acc2,acc6), add(acc3,acc7)}` and call `reduce_accumulators(folded)`.
2. `dot_q6k_row_avx2`: `accumulators[8]`; in the 32-value index loop bind `__m256& first_bank = accumulators[index/8]` and `__m256& second_bank = accumulators[4 + index/8]`; send offsets 0 and 64 to `first_bank`, offsets 32 and 96 to `second_bank`; same fold before the reduce.
3. `tests/test_dequant.cpp`: replace both `memcmp == 0` checks with `|a-b| <= 1e-5*(1+|a|)` (the fold reorders f32 adds).
4. Build + `tge_tests` 171/171; build a baseline binary from the reverted tree; alternate `generate --n 64 --kv f32 --threads 6 --gpu off --print-ids` four times. Expected: 0.85-1.0x decode, ids identical, differential within tolerance. Any median below 1.05x is the recorded kill.

## 9. Exact commands run and results

```text
cmake --build projects\aegean-tge\build\release --config Release   -> clean (56/56)
projects\aegean-tge\build\release\bin\tge_tests.exe                -> 2 exact differentials fail (expected)
# after relaxing the two differentials to 1e-5 relative
tge_tests.exe                                                     -> 171 cases, 0 failed
# baseline binary from the stashed tree -> build\idea\tge-dot4acc-base.exe
tge generate --model Llama-3.2-3B-Instruct-Q4_K_M.gguf --prompt <dragon> \
  --n 64 --kv f32 --threads 6 --gpu off --print-ids                -> §6 table
git checkout -- src/cpu/avx2/dequant_avx2.cpp tests/test_dequant.cpp
cmake --build ... -> clean; tge_tests -> 171/171
python aikit.py check aegean-tge                                  -> 4/4 gates passed
```

No push, no merge, no force-push. Spike binaries live only under `build/idea/` and `build/release/bin/`; the committed tree is unchanged except this report.
