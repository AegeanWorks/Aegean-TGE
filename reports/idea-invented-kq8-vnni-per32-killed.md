# INVENTED: per-32 activation scales feeding a VNNI Q4_K dot (no weight multiply) — KILLED (0.62-0.99x; vpdpbusd + cvt per chunk loses to maddubs+madd)

- **Idea**: The previous VNNI attempt lost because the 6-bit weight scale needed a `vpmulld` per 32-value chunk after `vpdpbusd`. Move the scale to the float side: quantise activations per 32 values (`Q8VBlock`, one scale per chunk), fold the weight scale into the existing per-chunk float factor, and the VNNI accumulation becomes a plain `vpdpbusd` + `cvt` + `fmadd`. Predicted: removes the `vpmulld` tax and beats the stock maddubs+madd chain at prefill chunk sizes, at the cost of per-32 activation quantisation (a priced numerics change, kq8d-class ~1.1% row error).
- **Cycle date**: 2026-11-11
- **Verdict**: **KILLED on the speed gate before any fidelity pricing.** 3072x3072 Q4_K, best-of-3: 32 tokens 208.1 vs 210.8 GFLOPS (0.99x), 64 tokens 220.0 vs 333.9 (0.66x), 128 tokens 271.9 vs 392.1 (0.69x), 256 tokens 245.3 vs 392.9 (0.62x). The `vpdpbusd` + `cvt` + `fmadd` triple per 32 values is still slower than the stock `maddubs` + `madd` + block-level fma, and the per-32 quantiser adds cost on top. No fidelity run was owed (speed gate first, per the project's kill order), so the numerics trade was never reached.
- **Tree state at close**: spike reverted from `src/cpu/avx2/dequant_batch_kq8_avx2.cpp`; `git status` clean; stash holds only the pre-existing 2026-11-04 peer entry; rebuild clean, `tge_tests` **171/171**, `python aikit.py check aegean-tge` **4/4**. No branch, no spike commit, nothing pushed.

## 1. Mechanism (as spiked)

- `Q8VBlock { float d[8]; int8_t qs[256]; int16_t sums32[8]; }` and `quantize_row_q8v` — the per-32 quantiser from the decode per-32 cycle (signed max, `iscale = -127/max`, `d = 1/iscale`).
- `dot_q4k_q8v_vnni(x, y, blocks)`: per 32-value chunk, `vpdpbusd(0, nibbles_g, q8_g)` -> `cvt` -> `fmadd` with `act_d[g] * dw * sc[g]`; min term from `sums32[g]`. Marked `__attribute__((target("avxvnni")))` for MinGW.
- Selected by `getenv("TGE_KQ8V32")` in the Q4_K branch of `gemv_dequant_batch_kq8_avx2` (spike only; a parallel `Q8VBlock` buffer is built when enabled).

## 2. Predicted gain (with the math)

Stock per 64 values: 4 `maddubs` + 4 `madd` + amortised block fma. VNNI-per32 per 64 values: 2 `vpdpbusd` + 2 `cvt` + 2 `fmadd` and no `madd`. Op count drops from ~8 to ~6 per 64 values, and the expensive `vpmulld` is gone. If instruction count were the limiter, 1.1-1.2x; if `vpdpbusd`'s 2-uop cost plus the extra `cvt` dominates, ~1.0x. Measured 0.62-0.69x at sustained sizes — worse than even the previous VNNI form, because `cvt` is also 2 uops and now runs per 32 values instead of per block.

## 3. Falsifier (stated before the spike)

**PROCEED** if `bench batch --tokens 64|128 --format q4k --kernel kq8` with `TGE_KQ8V32=1` gains >= 1.15x over stock; **then** price fidelity (f32, 399 positions, KL <= 1.6e-3, top-1 >= 390/399) before any model wiring. **KILL at the first failing gate.**

Speed gate failed at 64/128/256 -> kill. No fidelity run owed.

## 4. File anchors (current HEAD, read-before-write verified)

- `src/cpu/avx2/dequant_batch_kq8_avx2.cpp:143-216` — `dot_q4k_q8k` (the stock comparison).
- `:324-387` — `gemv_dequant_batch_kq8_avx2` (hook site).
- `src/cpu/avx2/q8_batch_vnni.cpp`, `CMakeLists.txt:101-109` — shipped VNNI tier (Q8_0 only).
- `reports/idea-invented-kq8-vnni-killed.md` — the first VNNI form and the `--vnni` Q8_0-only correction.
- `reports/idea-invented-kq8d-per32-activation-keep.md` — the per-32 quantiser reused here (valid for decode fidelity, not for prefill speed).

## 5. What was done (temporary spike, fully reverted)

Added `Q8VBlock`/`quantize_row_q8v`/`unpack_q4k_scale_min`/`dot_q4k_q8v_vnni` and the env-gated parallel buffer + dot selection; first build failed on a missing helper order, fixed by placing `unpack_q4k_scale_min` above the VNNI block; built; 171/171; benched 32/64/128/256 tokens with and without `TGE_KQ8V32`; reverted; rebuilt; 171/171; gates 4/4; clean tree.

## 6. Measured numbers

`bench batch --rows 3072 --cols 3072 --repeat 3 --format q4k --kernel kq8`, best of 3.

| tokens | stock GFLOPS | per-32 VNNI GFLOPS | ratio |
|---:|---:|---:|---:|
| 32 | 210.8 | 208.1 | 0.987 |
| 64 | 333.9 | 220.0 | 0.659 |
| 128 | 392.1 | 271.9 | 0.693 |
| 256 | 392.9 | 245.3 | 0.624 |

## 7. Why it failed and what it leaves

- On this core, `vpdpbusd` is 2 uops (ports 0/5) and `vcvtdq2ps` is 2 uops; the stock `maddubs`+`madd` pair is 2 cheap uops for the same 4-value groups *including* the scale multiply. Adding a per-chunk convert and a per-chunk fma cannot win, with or without the weight multiply.
- The VNNI line is now closed in both forms: weight-scaled (`vpmulld` tax) and activation-scaled (`cvt`+`fmadd` tax). VNNI ships only for Q8_0 weights, which is the one layout where its scale handling is free.
- Prefill now has five measured bounds (bands, two-row sharing, split adds, VNNI weight-scaled, VNNI activation-scaled). The remaining llama gap is structural: llama's repacked formats and different q4 unpacking, not a dot-level instruction swap available to this codebase.

## 8. Reimplementation recipe (rebuild only to re-verify the kill)

1. Add `unpack_q4k_scale_min` (6-bit unpack) above a new `Q8VBlock`/`quantize_row_q8v` pair (per-32 signed-max quantiser, `sums32[8]`).
2. Add `__attribute__((target("avxvnni"))) float dot_q4k_q8v_vnni(const Q4KBlock*, const Q8VBlock*, int)`: 8 groups per block, `dpbusd` -> `cvt` -> `fmadd(cvt, set1(act*dw*sc[g]))`, min term from `sums32`.
3. In `gemv_dequant_batch_kq8_avx2`, when `getenv("TGE_KQ8V32")` and kind is Q4_K, build a parallel `std::vector<Q8VBlock>` and use the VNNI dot; otherwise stock.
4. Build + 171/171; `bench batch --tokens 64|128` -> expect 0.62-0.69x. Revert. Do not run fidelity: the speed gate fails first.

## 9. Exact commands run and results

```text
cmake --build projects\aegean-tge\build\release --config Release   -> clean (56/56)
projects\aegean-tge\build\release\bin\tge_tests.exe                -> 171 cases, 0 failed
set TGE_KQ8V32=1 ; tge bench batch --rows 3072 --cols 3072 --tokens {32|64|128|256} \
  --repeat 3 --format q4k --kernel kq8                           -> §6 table
git checkout -- src/cpu/avx2/dequant_batch_kq8_avx2.cpp
cmake --build ... -> clean; tge_tests -> 171/171
python aikit.py check aegean-tge                                   -> 4/4 gates passed
```

No push, no merge, no force-push. The spike exists only in the rebuilt binary; the committed tree is unchanged except this report.
