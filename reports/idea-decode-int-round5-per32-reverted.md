# IDEA-DECODE-INT ROUND 5: Per-32 Activation Dots — BUILT, FALSIFIED, REVERTED

- **Idea**: Per-32-block activation quantization with matching Q4_K integer dots, to fix round 3's verdict (per-256-block amax drowns small values on heavy-tailed activations: rel p99 10%, 10 flips). Prescribed by round 4 (per-32 quantizer p99 1.1% vs 2.9%; kernel exonerated at 6e-5).
- **Cycle date**: 2026-11-07
- **Verdict**: REVERTED. The build is correct (see below) and strictly better than old kq8 (KL 3.86e-5 vs 7.35e-4, 4 flips vs 10, +3.9% decode on 0.5B), but the stated falsifier required top-1 held (1023/1023) and the result is deterministic 1019/1023. Per the loop rule (worse than baseline → revert), no per-32 code survives in-tree. Everything needed to rebuild is in §7.
- **Tree state at close**: clean, no diff, stash untouched (pre-existing entry only). Spike branch created and deleted with zero commits on it. `master` ahead of `origin/master` by 171 after this report. Nothing pushed (human gate).

## 1. What was built (all reverted; recipe in §7)

Q8P32Block (32 int8 + per-32 float scale + per-32 bsum), per-32 nearest-away quantizer (same mantissa trick as upstream), `dot_q4k_q8p32` (verbatim lane structure of dot_q4k_q8k with per-group float accumulation + per-group dmin bias), `gemv_dequant_kq8p32_row_avx2` row entry (Q4_K-only, falls back to the existing row kernel), temp matvec wiring + capture + differential drivers (all reverted).

## 2. Gate results

- Gate 1 (unit differential vs float on synthetic uniform/spike/broadtail/constant): new matches or beats old everywhere (spike 2x better: 0.005 vs 0.013). PASS. Also verified: bit-exact horizontal sum vs scalar, maxdiff 6e-6 on deterministic input.
- Gate 2 (live-data capture vs float, 0.5B-Q4_K_M decode): new rel p99 1.53%, max 1.62% (old: p99 10%, max 28.6%). PASS (bar was <2%).
- Gate 3 (fidelity, 0.5B-Q4_K_M, arch corpus, 1023 pos): KL 3.864e-05 (bar ≤1.6e-3: PASS by 40x), worst 1.4e-3, max|diff| 0.17, **top-1 1019/1023 (bar: held — FAIL, deterministic across runs)**.
- Gate 4 (bench pulse + decode A/B): kq8d-bench-redirect end-to-end decode +3.9% on 0.5B short (needs re-measurement on 3B before any future keep claim).

## 3. Debugging log (so nobody repeats it)

Three independent implementation bugs were found and fixed in-cycle; the final build is verified correct (gates 1–2 pass). (a) unpacked-scale uaux writeback dropped (`utmp[2] = uaux` missing — caught by review, would have scrambled half the scales); (b) minmax read at offset +8 instead of +0 (out-of-bounds read of uninitialized stack — caught by inspection after the differential stayed red); (c) a quantizer double-advance (`x[g*32+i]` plus `x += 32` — caught by code read). Lesson restated: verify each AVX2 transplant against the original with a bit-exact differential BEFORE wiring it into any model path.

## 4. What remains (not attempted; would need its own falsifier)

Closing the last 4 flips needs finer-than-32 or outlier-aware activation (keep the 4-bit weight path intact; quantize activations per-16 or spill outlier channels to fp32 like LLM.int8), or a proof that the 4 flips are all sub-threshold argmax ties (max|diff| says otherwise: 0.17 can flip). Do NOT retry per-256 anything (rounds 2–3 killed it twice with real-data evidence), and do NOT re-litigate the bias math (verified bit-exact against the scalar path in round 5's selftest: reldiff 6e-8).

## 5. What was done and exact commands

- Spike branch (zero commits, deleted at close); ~300 lines temp AVX2 + driver reverted via `git checkout --` (empty diff verified).
- Builds 54/54 and `tge_tests` 145/145 green throughout (spike + reverted states).
- `spike-kq8p32-diff` (synthetic, all four distributions, best-of-3 A/B vs old kq8d + float); live capture (`TGE_KQ8P32_CAP=1` generate, 168 rows); fidelity with `TGE_KQ8P32=1` ×2 (deterministic 1019 both runs); end-to-end decode A/B.
- Final state: clean tree, `git status` clean, stash untouched; report committed; no push.

(End of file.)
