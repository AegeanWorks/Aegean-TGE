# INVENTED: AVX-VNNI Q4_K x Q8_K dot for prefill — KILLED (-24% at sustained sizes; the 16-bit scale madd is cheaper than per-chunk 32-bit multiply)

- **Idea**: Prefill is the only axis where TGE trails llama.cpp (1.3-1.5x on matched f16 KV). The kq8 prefill dot is compute-bound in the maddubs+madd chain. AVX-VNNI's `vpdpbusd` folds four u8 x i8 products into one int32 lane add, halving that chain. If the per-32 weight scale is applied with a 32-bit multiply after each `vpdpbusd`, the row result should be bit-identical to `dot_q4k_q8k` (the existing 16-bit madd multiplies equal scale pairs, i.e. the same product). Test whether VNNI closes the prefill gap for Q4_K weights.
- **Cycle date**: 2026-11-11
- **Verdict**: **KILLED.** Isolated 3072x3072 Q4_K: **310.7 vs 409.8 GFLOPS at 128 tokens**, **315.7 vs 411.6 at 64**, **344.0 vs 415.2 at 256** — VNNI is **24% slower** at the chunk sizes prefill uses. It only looks better at 32 tokens (284.8 vs 256.7), inside this box's large noise and below the 64+ regime that matters. The cause is structural: the existing kernel applies one 16-bit scale pair per `madd` over 4 values, while the VNNI form needs a 32-bit `vpmulld` per 32-value chunk (8 per superblock); `vpmulld` costs more than the `madd` it removes.
  **Side finding**: `--vnni` in the tree routes only **Q8_0** weights (`gemv_q8_batch_vnni`); on the Q4_K_M reference model it is a no-op, so the old "VNNI costs 1.6e-3 fidelity" price applies to Q8_0 models, not to this one. The prefill gap cannot be closed by the shipped `--vnni` flag.
- **Tree state at close**: spike reverted from `src/cpu/avx2/dequant_batch_kq8_avx2.cpp`; `git status` clean; stash holds only the pre-existing 2026-11-04 peer entry; rebuild clean, `tge_tests` **171/171**, `python aikit.py check aegean-tge` **4/4**. No branch, no spike commit, nothing pushed.

## 1. Mechanism (as spiked)

- `dot_q4k_q8k_vnni(const Q4KBlock*, const Q8KBlock*, int)` beside `dot_q4k_q8k`: same 12-byte scale unpack (`utmp`, kmasks), same `bsums`-based min correction, same final `d * cvt(sumi)` fma; the 64-value loop is replaced by eight 32-value groups where `sumi += vpmulld(vpdpbusd(0, nibbles, q8), set1(sc[g]))`.
- Selected by `getenv("TGE_KQ8_VNNI")` inside `gemv_dequant_batch_kq8_avx2` (spike-only switch, no CLI change); function carries `__attribute__((target("avxvnni")))` because this build is MinGW-GCC and the TU is compiled with `-mavx2 -mfma -mf16c` (the existing `q8_batch_vnni.cpp` gets `-mavxvnni` per-file; the attribute avoids touching CMake for a spike).
- Integer arithmetic is exact, so a row result equals `dot_q4k_q8k` (no float reassociation).

## 2. Predicted gain (with the math)

Per 64 values the stock loop issues 4 `maddubs` + 4 `madd` (8 ops) for two 32-value halves; the VNNI form issues 2 `vpdpbusd` + 2 `vpmulld` (4 ops) — half the chain, so 1.2-1.4x if the chain were the limiter and `vpmulld` were free. `vpmulld` is a 2-uop, port-0/1 instruction on this core, ~2-3x the cost of `madd`, which is why the prediction fails: 4 ops where one pair is expensive beats 8 cheap ops only if the expensive pair is cheaper than the three cheap ops it replaces — it is not.

## 3. Falsifier (stated before the spike)

**PROCEED** if `bench batch --tokens 64|128 --format q4k --kernel kq8` with the VNNI dot gains >= 1.15x over the stock dot, and ids/differentials stay exact. **KILL** otherwise.

Measured -24% at both -> kill. (Correctness itself was not the issue; the spike never reached the model path.)

## 4. File anchors (current HEAD, read-before-write verified)

- `src/cpu/avx2/dequant_batch_kq8_avx2.cpp:143-216` — `dot_q4k_q8k` (the 16-bit scale madd the VNNI form replaced).
- `:112-136` — `k4_shuffle_mask` / `k_scale_shuffle`.
- `:324-387` — `gemv_dequant_batch_kq8_avx2` (spike switch site).
- `src/cpu/avx2/q8_batch_vnni.cpp`, `CMakeLists.txt:101-109` — the only shipped VNNI tier (Q8_0 weights, per-file `-mavxvnni`).
- `src/models/qwen2.cpp:978-993` / `:1090-1105` — VNNI dispatch requires `TensorType::Q8_0`.
- `reports/llama-cpp-comparison-2026-11-11.md` — the prefill gap this targeted.
- `reports/idea-invented-kq8-band1-killed.md`, `...-pair-dot-killed.md`, `...-split-sumi-killed.md` — the other three prefill bounds; this cycle adds the fourth (instruction choice).

## 5. What was done (temporary spike, fully reverted)

Added the VNNI dot + env switch; first build failed under MinGW (`-mavxvnni` missing), fixed with a target attribute; built, 171/171; benched 32/64/128/256 tokens with and without `TGE_KQ8_VNNI`; reverted; rebuilt; 171/171; gates 4/4; clean tree.

## 6. Measured numbers

`bench batch --rows 3072 --cols 3072 --repeat 3 --format q4k --kernel kq8`, best of 3 per point.

| tokens | stock GFLOPS | VNNI GFLOPS | ratio |
|---:|---:|---:|---:|
| 32 | 256.7 | 284.8 | 1.110 |
| 64 | 411.6 | 315.7 | 0.767 |
| 128 | 409.8 | 310.7 | 0.758 |
| 256 | 415.2 | 344.0 | 0.829 |

The 32-token point is inside this box's run-to-run spread (same-binary runs have spanned 257-416 GFLOPS in earlier cycles); the 64-256 points are consistent and decisive.

## 7. Why it failed and what it leaves

- `vpmulld` is the tax. The stock kernel's 16-bit `madd` applies the scale for free inside the multiply-add; VNNI has no scaled dot, so the scale must be a separate 32-bit multiply. That is strictly worse on AVX2/VNNI for this layout.
- A different formulation could avoid the per-chunk multiply: scale the *activations* per 32 and use VNNI without weight scaling — but that is per-32 activation quantisation (already kept for decode fidelity) and would change prefill numerics; it would also need the q8 quantiser to become per-32 in prefill, which is the same error class as kq8d (1.1% row error) — not a default-safe change.
- Four prefill attempts are now bounded: band count (neutral), two-row activation sharing (+7% isolated only), split integer accumulators (neutral), VNNI instruction choice (-24%). The remaining gap to llama is issue-width/format-level (llama's repacked Q4 layouts amortise unpacking differently), not reachable by a drop-in dot. VNNI remains valid only for Q8_0 weights, which is what ships.

## 8. Reimplementation recipe (rebuild only to re-verify the kill)

1. Add `dot_q4k_q8k_vnni` beside `dot_q4k_q8k`: copy the stock body, keep the `utmp` unpack, `mins_and_scales`, `bsums` min term, and final `d * cvt(sumi)` fma; replace the 4-iteration 64-value loop with 8 groups of 32 where `sumi = add(sumi, mullo(dpbusd(0, nibbles_g, q8_g), set1(scale_byte[g])))`, with `scale_byte` read from the transformed `utmp[0..1]` (same bytes as `unpack_q4k_scale_min`). Mark the function `__attribute__((target("avxvnni")))` on GCC; add `-mavxvnni` per-file on MSVC if porting.
2. Gate it behind `getenv("TGE_KQ8_VNNI")` in the Q4_K branch of `gemv_dequant_batch_kq8_avx2`.
3. Build + 171/171; `bench batch --tokens 64|128` with and without the env -> expect 0.76-0.83x. Revert.

## 9. Exact commands run and results

```text
cmake --build projects\aegean-tge\build\release --config Release   -> clean (56/56)
projects\aegean-tge\build\release\bin\tge_tests.exe                -> 171 cases, 0 failed
$env:TGE_KQ8_VNNIset TGE_KQ8_VNNI=1; tge bench batch --rows 3072 --cols 3072 --tokens {32|64|128|256} \
  --repeat 3 --format q4k --kernel kq8                           -> §6 table
git checkout -- src/cpu/avx2/dequant_batch_kq8_avx2.cpp
cmake --build ... -> clean; tge_tests -> 171/171
python aikit.py check aegean-tge                                   -> 4/4 gates passed
```

No push, no merge, no force-push. The spike exists only in the rebuilt binary; the committed tree is unchanged except this report.
