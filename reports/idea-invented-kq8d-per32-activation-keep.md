# INVENTED: per-32 activation scales in the kq8d row kernel — KEEP (whole-FFN stack now passes: 6.015e-4 / 392/399, ~1.25x; `ffn` KL drops to 9.05e-5)

- **Idea**: The per-256 Q8_K activation layout is what put the whole-FFN kq8d stack (`--kq8d ffn,down`, gate+up+down) one top-1 flip under the class bar (389/399 = 97.49% vs 97.6%). Round 4 already showed per-32 symmetric activation scales cut live-row roundtrip error from 2.9% to 1.1% p99. Give the single-row kq8d kernel a per-32 activation layout for Q4_K (the FFN matrices), still quantising once per matvec, and re-price the FFN tiers. Predicted: the stack clears the bar at roughly the same speed (8 scale multiplies per 256 values instead of 1).
- **Cycle date**: 2026-11-11
- **Verdict**: **KEEP (report-only; integration is a human-gated decision).** With the per-32 integer kernel, `--kq8d ffn,down` clears the class bar at **392/399 (98.25%)**, mean KL **6.015e-4** (was 389/399 and 9.100e-4), and decodes at **~1.25x** over the fused-float default (four rotated pairs: 1.138, 1.323, 1.274, 1.233). `--kq8d ffn` improves its KL 2.8x to **9.054e-5 / 196/199** at ~1.12x (per-256 was 2.495e-4 / 198/199 at ~1.20x — the per-32 kernel costs ~5-7% speed for the finer quant). Down at 199: 6.402e-4 / 196/199. The ideal probe (per-32 quant, exact f32 weights) prices at 2.353e-4 / 393/399, so the integer dot's rounding costs ~2.5e-4 KL but still passes.
- **Tree state at close**: spike reverted from `src/cpu/avx2/dequant_batch_kq8_avx2.cpp`, `include/tge/models/qwen2.hpp`, `src/models/qwen2.cpp`, `apps/cli/main.cpp`, `tests/test_dequant.cpp`; `git status` clean; stash holds only the pre-existing 2026-11-04 peer entry; rebuild clean, `tge_tests` **171/171**, `python aikit.py check aegean-tge` **4/4**. No branch, no spike commit, nothing pushed.

## 1. Mechanism

The kq8d single-row kernel (`gemv_dequant_kq8_row_avx2`) quantises the activation row once per matvec into Q8_K blocks: one symmetric `d` per 256 values, integer `qs[256]`, and 16-value `bsums` for the weight-min correction. The weight side (Q4_K 6-bit per-32 scales and mins) was already finer than the activation side.

Spike adds a per-32 activation layout **for Q4_K only**:

- `struct Q8K32Block { float d[8]; int8_t qs[256]; int16_t sums32[8]; };` — one symmetric scale and one integer sum per 32-value chunk, 304 bytes per superblock.
- `quantize_row_q8k32(const float* x, Q8K32Block* y, int count)` — same signed-max scheme as `quantize_row_q8k` (`iscale = -127/max`, `d = 1/iscale`, round-to-nearest, clamp at +127), applied per 32-value chunk; stores `sums32[g]` for the min term.
- `unpack_q4k_scale_min(scales, index, sc, m)` — the same 6-bit unpack as the dequant path and the shipped dot, per sub-block 0..7.
- `dot_q4k_q8k32(x, y, blocks)` — per 256-value block: unpack 8 (sc, m) pairs; for each 32-value chunk `g`, load the q4 byte group `q4 + (g/2)*32`, take low nibbles for even `g` and high nibbles for odd `g`, `maddubs(nibbles, q8)`, `madd(ones, p16)`, then `acc = fmadd(cvt(p32), act_d[g]*dw*sc[g], acc)`; accumulate `min_term += act_d[g]*dminw*mn[g]*sums32[g]`; return `hsum(acc) - min_term`.
- The row kernel branches on kind: Q4_K builds a `Q8K32Block` vector and runs `dot_q4k_q8k32`; Q6_K keeps the per-256 `Q8KBlock` + `dot_q6k_q8k` (its consumer is the LM head, whose shipped tier already passes and whose dot works on 16-value scales).
- Spike also re-applied the `ffn`/`ffn,down` scope wiring from the previous two cycles so the stack could be priced.

**Load-bearing debugging lesson**: the first probe used `thread_local std::vector<float> act32` and captured it by reference in the pool lambda. Worker threads resolve a `thread_local` reference to *their own* TLS slot, which was never resized, so `data()` was null and rows=2048 crashed with 0xC0000005 (small row counts happened to run on the caller). Use a plain local vector for anything captured into a pool lambda.

## 2. Predicted gain (with the math)

Per 256-value block the per-32 dot does 8× (maddubs + madd + cvtepi32 + fmadd) instead of 4× (maddubs low + maddubs high + 2 madd) plus two fmadds — roughly +8 int-to-float converts, +6 fmadds, and 8 scalar scale/min unpacks per block. The dot is not the bandwidth bottleneck (weights dominate), so the model predicted a small (~5%) matvec cost for a large activation-error cut (2.9% → 1.1% p99 live-row). Fidelity: the stack's previous 9.1e-4 mean KL was dominated by per-256 activation error on the wide down/gate/up sums; per-32 should cut it to the 2-6e-4 class, which is the band the passing tiers live in.

Measured: ffn,down 6.015e-4 (was 9.100e-4); ffn 9.054e-5 (was 2.495e-4); speed ffn ~1.12x (was ~1.20x), stack ~1.25x (was ~1.28x). Speed cost ~5-7% on the tier, as predicted.

## 3. Falsifier (stated before the kernel)

**KEEP** if (a) `--kq8d ffn,down` at 399 positions has mean KL <= 1.6e-3 and top-1 >= 390/399, (b) at 199 positions top-1 >= 195/199, (c) the stack still decodes >= 1.15x over the default in rotated pairs, and (d) the kernel's row error against the float oracle stays in the known per-32 class (~1% relative, not the 2.9% per-256 class). **KILL** at the first gate that fails.

All passed: 392/399 and 195/199; 1.23-1.32x; row error 1.1% (matches round 4's 1.1% p99).

## 4. File anchors (current HEAD, read-before-write verified)

- `src/cpu/avx2/dequant_batch_kq8_avx2.cpp:21-27` — `Q8KBlock`; `:63-98` — `quantize_row_q8k`; `:143-216` — `dot_q4k_q8k`; `:222-313` — `dot_q6k_q8k`; `:397-456` — `gemv_dequant_kq8_row_avx2` (single-row entry, thread-local Q8_K scratch in the tree; the spike replaces it with per-32 Q4_K / per-256 Q6_K branches).
- `src/cpu/avx2/dequant_avx2.cpp:158` / `:291` — `dot_q4k_row_avx2` / `dot_q6k_row_avx2` (the probe's exact-float dots, used to isolate activation error).
- `src/models/qwen2.cpp:860-861` (head), `:2228-2229` (gate/up with the ffn flag), `:2233-2234` (down) — kq8d call sites.
- `include/tge/models/qwen2.hpp:241-259` — `set_kq8d` scopes; `apps/cli/main.cpp:2286-2306`/`:2805-2825` — `--kq8d` parsers.
- `reports/idea-invented-kq8d-ffn-keep.md` (per-256 ffn 2.495e-4 / 198/199), `reports/idea-invented-kq8d-ffn-down-killed.md` (the 389/399 kill this cycle reverses), `reports/idea-decode-int-round4-granularity-probe.md` (per-32 error 2.9% → 1.1% p99 on live rows).

## 5. What was done (temporary spike, fully reverted)

1. Probe first: per-32 activation quant + dequant + the exact fused float dot inside the row kernel, to measure the fidelity ceiling with no integer-dot rounding. Ran `--kq8d ffn,down` at 199 and 399 positions.
2. Crash found at rows=2048: `thread_local` vector captured by reference into the pool lambda (workers see their own empty TLS). Fixed with a plain local vector; suite green.
3. Fast kernel: added `Q8K32Block`, `quantize_row_q8k32`, `unpack_q4k_scale_min`, `dot_q4k_q8k32`; row kernel branches Q4_K→per-32, Q6_K→per-256. Re-applied the `ffn`/`ffn,down` wiring.
4. Validation: tightened the kq8d row differential to 0.5% relative; it fails at 1.1%, which is the *expected* per-32 activation error class (round 4's 1.1% p99), so the kernel is behaving as designed; the shipped 5% tolerance is unchanged.
5. Fidelity: ffn/down/ffn,down at 199; ffn,down at 399. Speed: rotated off/ffn/ffn,down pairs at n=48, threads 10.
6. Reverted all five files; rebuilt; 171/171; gates 4/4; clean tree.

## 6. Measured numbers

### Fidelity (f32 KV, scalar oracle, docs/architecture.md)

| scope | layout | positions | mean KL | worst KL | top-1 | bar |
|---|---|---|---:|---:|---:|---|
| ffn | per-256 (prior) | 199 | 2.495e-4 | 3.883e-3 | 198/199 (99.50%) | pass |
| ffn | **per-32** | 199 | **9.054e-5** | 8.077e-4 | 196/199 (98.49%) | pass |
| down | per-256 (prior, 399) | 399 | 7.954e-4 | — | 390/399 (97.74%) | pass |
| down | **per-32** | 199 | **6.402e-4** | 8.327e-3 | 196/199 (98.49%) | pass |
| ffn,down | per-256 (killed) | 199 | 9.609e-4 | 7.940e-3 | 194/199 (97.49%) | **fail by 1** |
| ffn,down | per-32 probe (ideal f32 weights) | 199 | 2.345e-4 | 1.594e-3 | 196/199 (98.49%) | pass |
| ffn,down | **per-32 integer kernel** | 199 | **7.187e-4** | 9.310e-3 | **195/199 (97.99%)** | pass |
| ffn,down | per-256 (killed) | 399 | 9.100e-4 | 7.940e-3 | 389/399 (97.49%) | **fail by 1** |
| ffn,down | per-32 probe (ideal) | 399 | 2.353e-4 | 4.080e-3 | 393/399 (98.50%) | pass |
| ffn,down | **per-32 integer kernel** | 399 | **6.015e-4** | 9.310e-3 | **392/399 (98.25%)** | pass |

The integer dot's rounding costs ~2.5e-4 KL and one top-1 vs the ideal probe; the stack still clears both sizes of the bar.

### Kernel differential (random activations, rows 3, cols 1024/8192)

Relative row error 0.94-1.11% — the per-32 activation class (round 4: 1.1% p99), versus the 2.9% class of per-256. The shipped 5% gate still passes; a 0.5% gate does not, and should not.

### Decode (dragon prompt, n=48, f32, threads 10, rotated order)

| pair | off | ffn | ffn,down | stack/off | ffn/off |
|---:|---:|---:|---:|---:|---:|
| 1 | 15.28 | 17.07 | 17.39 | 1.138 | 1.117 |
| 2 | 12.03 | 13.59 | 15.92 | 1.323 | 1.130 |
| 3 | 15.35 | - | 19.57 | 1.274 | - |
| 4 | 15.75 | - | 19.42 | 1.233 | - |
| 5 | 15.74 | 15.93 | - | - | 1.012 |
| 6 | 13.93 | 15.96 | - | - | 1.146 |

Stack/off: 1.138, 1.323, 1.274, 1.233 -> median **~1.25x**. ffn/off: 1.117, 1.130, 1.012, 1.146 -> median **~1.12x**. `-` marks arms not run in that pair; comparisons are only within adjacent off pairs.

## 7. Integration consequences (for the human-gated decision)

- If integrated, the per-32 Q4_K layout changes every shipped Q4_K kq8d scope: `down` and `head,down` get their fidelity improved and roughly -5% speed; their priced KL/top-1 must be re-stated (this cycle's 199-position numbers start that, but 399-position rows are owed for `down` and `head,down`).
- The `ffn,down` tier becomes shippable (1.25x, 98.25% top-1). `ffn` remains the cheaper-fidelity choice (9.05e-5) at 1.12x.
- Q6_K stays per-256: the LM head's `head` tier is unaffected. A per-32 Q6_K dot is a possible follow-up but was not required for the gate.
- The ids flip on the dragon prompt for the stack (`A1-62-BF` vs `6B-10-8A`); expected for a lossy tier and consistent with `head`'s behaviour. Speculation must keep the verifier on the same arithmetic as before.

## 8. Full reimplementation recipe

1. `src/cpu/avx2/dequant_batch_kq8_avx2.cpp`, anonymous namespace: add `Q8K32Block { float d[8]; int8_t qs[256]; int16_t sums32[8]; }`; add `quantize_row_q8k32(const float*, Q8K32Block*, int)` (per 32-value chunk: signed max, `iscale=-127/max`, `d=1/iscale`, nearest-int clamp at +127, store `sums32`); add `unpack_q4k_scale_min(const uint8_t* scales, int index, uint8_t& sc, uint8_t& m)` (index<4: `sc=scales[index]&63, m=scales[index+4]&63`; else `sc=(scales[index+4]&15)|(((scales[index-4]>>6)&3)<<4)`, `m=(scales[index+4]>>4)|(((scales[index]>>6)&3)<<4)`); add `dot_q4k_q8k32` exactly as in §1 (8 chunks per 256-block, `q4 + (g/2)*32`, low nibble for even g, high for odd, `maddubs`+`madd(ones)`, `fmadd(cvt(p32), act*dw*sc, acc)`, scalar min term, return `hsum(acc)-min_term`).
2. In `gemv_dequant_kq8_row_avx2`, after the thread count/band setup: if `args.kind == Q4_K`, build `std::vector<Q8K32Block>` and run a banded `parallel_for` with `dot_q4k_q8k32`; otherwise keep the existing `Q8KBlock` + `dot_q6k_q8k` path. **Do not use a `thread_local` buffer captured by a pool lambda** — use a plain local vector.
3. To price the FFN stack, apply the `ffn`/`ffn,down` wiring from `reports/idea-invented-kq8d-ffn-keep.md` §8 and `reports/idea-invented-kq8d-ffn-down-killed.md` §8.
4. Build + `tge_tests` 171/171. Fidelity: `tge fidelity --kq8d ffn,down --context 512 --tokens 200 --threads 1` and `--context 1024 --tokens 400`. Gate: mean KL <= 1.6e-3, top-1 >= 195/199 and >= 390/399.
5. Speed: rotated pairs of `generate --n 48 --kv f32 --threads 10 --gpu off [--kq8d ffn|ffn,down]`; expect stack ~1.25x, ffn ~1.12x.
6. Revert (report-only), or ship behind the opt-in flag after human approval (push gate), re-pricing `down` and `head,down` at 399 positions first.

## 9. Exact commands run and results

```text
cmake --build projects\aegean-tge\build\release --config Release   -> clean (56/56)
projects\aegean-tge\build\release\bin\tge_tests.exe                -> 171 cases (probe build: crash at rows=2048, fixed to a local buffer)
tge fidelity ... --kq8d ffn,down --context 512  --tokens 200 --threads 1 -> 7.187e-4, 195/199
tge fidelity ... --kq8d ffn,down --context 1024 --tokens 400 --threads 1 -> 6.015e-4, 392/399
tge fidelity ... --kq8d ffn  --context 512 --tokens 200 --threads 1 -> 9.054e-5, 196/199
 tge fidelity ... --kq8d down --context 512 --tokens 200 --threads 1 -> 6.402e-4, 196/199
tge generate ... --n 48 --kv f32 --threads 10 --gpu off [--kq8d ffn|ffn,down] -> §6 speed table
git checkout -- src/cpu/avx2/dequant_batch_kq8_avx2.cpp include/tge/models/qwen2.hpp \
  src/models/qwen2.cpp apps/cli/main.cpp tests/test_dequant.cpp
cmake --build ... -> clean; tge_tests -> 171/171
python aikit.py check aegean-tge                                   -> 4/4 gates passed
```

No push, no merge, no force-push. The spike exists only in the rebuilt binary; the committed tree is unchanged except this report.
