# IDEA-1558: Scatter-Accumulate Exact-Int Dot for Q4_0 — SPIKED, MEASURED, KILLED (2.9–7.6x slower)

- **Idea ID**: IDEA-1558 (dossier `scout-2026-11-goal-lgpapers.md`: "Scatter-accumulate exact-int dot for low-bit matvec", arXiv 2610.31913 — length-n k-bit dot in n+2^k adds + ≤2^k decode-multiplies vs n MACs; exact integer; needs 2^k≪n)
- **Cycle date**: 2026-11-06
- **Verdict**: KILLED by its own falsifier — the AVX2 scatter-add + 16 fused decode-multiplies are 2.9x slower than the widen path at 1024² and 7.6x slower at 3072². Spike fully reverted; only this report committed.
- **Tree state at close**: `git status` clean (no diff), spike branch created and deleted with nothing committed, stash unchanged (pre-existing peer work only), `master` ahead of `origin/master` by 13 after this report. Nothing pushed (human gate).

## 1. Mechanism

For k≤4-bit weights (Q4_0: 16 values, validated by the register history as the
only feasible target — Q8 needs 256 bins over the 16-YMM budget), replace the
per-element widen+cvt+fmadd plus temp-row write/read with scatter-accumulate:
per 32-weight block, 16 float bins collect activations by nibble value
(`bins[lo] += x[i]`, `bins[hi] += x[i+16]`), then one fused decode-multiply per
bin (`Σ bins[v]·(v−8)`, vectorized as two AVX2 FMAs) times the block scale.
Weight entry read once per block and reused across the token group (same
amortization as `gemv_q8_batch_avx2`'s kGroup blocking). Exact in the integer
domain; float bins reorder summation, so the gate is the house tolerance, not
bit identity. Bench-only prototype, never wired to the model.

## 2. Predicted gain (with the math)

Per 32-weight block, baseline pays: 32 nibble-unpacks + 32 cvt/scale mults + 32
temp-float writes + 32 re-reads + 32 FMAs (two passes over 128 B of temp).
Scatter pays: 32 unpacks + 32 scalar adds + 16 mult-adds + 1 scale mult, single
pass, no temp traffic (12 KB saved per 3072-wide row per token group). If the
bin-add chain hid behind the saved memory passes, expected +5–15% on the
bandwidth-bound prefill path. Falsifier (dossier): scalar + AVX2 prototype for
k=4, n=1024–4096 — reject if not faster than the widen path, if FP degrades past
the house tolerance, or if the bin bank spills YMM.

## 3. Falsifier (from the dossier)

AVX2 scatter-add + 16 fused decode-multiplies must beat the widen path at
n=1024–4096 with FP error within the conventional bound; bin bank must fit
registers; Q4-only evaluation required (256 accumulators fail — respected: Q4_0
only, 16 bins = 2 YMM).

## 4. File anchors (current `master`, read before writing)

- `src/cpu/scalar/dequant.cpp:24-38` — Q4_0 layout (fp16 scale + 16 quant bytes;
  lo nibbles → weights 0–15, hi → 16–31; value `d·(nib−8)`).
- `src/cpu/avx2/dequant_batch_avx2.cpp:54-102` — baseline: `dequantize_row_avx2`
  into thread-local scratch + `dot_f32` 4-accumulator dot (the widen path).
- `src/cpu/scalar/dequant_batch_scalar.cpp` — scalar baseline.
- `include/tge/cpu/kernels.hpp` — `DequantBatchFn` signature reused for the
  prototype (same `DequantBatchArgs` layout needs).
- `apps/cli/main.cpp` — `command_batch_bench` q40 dispatch + `--kernel` parse
  (spike added `scatter`, q40-only, fail-loud otherwise).
- `tests/test_dequant.cpp` — house differential style (`max_abs_diff` vs
  `1e-4·cols+1e-3`).

## 5. What was done (temporary spike, fully reverted)

1. `git checkout -b spike-1558-scatter-q40` (no commit ever made on it).
2. `kernels.hpp`: declared `gemv_q4_0_scatter_batch_avx2/scalar` (bench-only).
3. Scalar prototype (`dequant_batch_scalar.cpp`): token-outer/grouped bins as
   in §7; fail-loud on nulls / non-Q4_0 kind / cols%32.
4. AVX2 prototype (`dequant_batch_avx2.cpp`): identical loop shape with banded
   thread pool + thread-local totals; scalar bin adds (unavoidable RAW chain),
   vectorized 16-bin reduction (2 FMAs vs constant (v−8) coeffs).
5. `main.cpp`: `--kernel scatter` parse + dispatch (q40-only, error otherwise)
   + usage line + kernel label.
6. `tests/test_dequant.cpp`: `q40_scatter_matches_dequant_reference`
   (differential vs `gemv_dequant_batch_scalar` at 3 shapes) +
   `q40_scatter_rejects_bad_args` (nulls/kind/block guards).
7. `cmake --build --preset release` clean; `ctest` pass; `tge_tests.exe`
   **143/143, 0 failed** (141 + 2 new).
8. `bench batch --format q40` best-of-5, default vs scatter, 1024²×6 and
   3072²×6 (same binary A/B — stronger than separate builds, no build variance).
9. `generate` on the reference Llama-3.2-3B-Q4_K_M with the spike binary:
   ids byte-identical to the recorded baseline (model path untouched by
   construction). No fidelity run owed: bench-only, model path never called the
   prototype (same precedent as bench-only kq8/kq8d).
10. Reverted all 5 files (`git checkout --`), verified `git diff` empty,
    `git checkout master`, deleted the branch. Final gates on reverted master:
    build clean, `ctest` pass, `aikit.py check` 2/4 (test + secrets pass;
    hygiene + guidelines fail pre-existing).

## 6. Measured numbers

`bench batch --format q40`, best-of-5, warm page cache, same binary:

| shape | default (widen) best | scatter best | verdict |
|---|---|---|---|
| 1024×1024×6 tok (0.6 MiB) | 0.289 ms (43.55 GFLOPS) | 0.852 ms (14.76) | **2.9x slower** |
| 3072×3072×6 tok (5.1 MiB) | 0.638 ms (177.50 GFLOPS) | 4.880 ms (23.21) | **7.6x slower** |

The gap widens with n — the opposite of the dossier's scaling hope. Cause: the
bin-accumulation RAW chain (each activation targets exactly one of 16 bins, so
no SIMD lane can own the adds; ~32 serial scalar adds per block) dominates, while
the saved temp-row traffic is L1-resident (128 B/block never leaves cache) and
worth ~nothing. The 16 fused decode-multiplies save 16 FMAs/block against ~64
extra scalar ops. YMM budget held (2 regs), FP held (differentials within
`1e-4·cols+1e-3` at all 3 test shapes), but speed fails by 3–8x. No NLL pricing
needed: rejected on speed alone per the falsifier's first arm.

Thinking set: Llama-3B ids identical spike vs recorded baseline
(`320 2028 374 ... 220 16 25`, 15 ids; promptids identical) — prototype never
touched the model path.

Tests: 143/143 on spike (2 new pass); 141/141 on reverted master.

## 7. Full reimplementation recipe (kill-grade — rebuild only to re-verify the kill)

1. `kernels.hpp`: declare both entries taking `(const DequantBatchArgs&, const
   GemvConfig&)` (kind must be `Q4_0`; fail loud otherwise).
2. Scalar (`dequant_batch_scalar.cpp`): per row, zero per-token totals; per
   32-block: `memcpy` fp16 scale → `half_to_float`; per token: `float bins[16] =
   {}`, 16× (`bins[byte&0xF] += x[i]`, `bins[byte>>4] += x[i+16]`),
   `decoded = Σ bins[v]·(v−8)`, `totals[token] += scale·decoded`; write out
   `[token·rows+row]`.
3. AVX2 (`dequant_batch_avx2.cpp`): same shape under the banded pool with
   thread-local totals; replace the scalar 16-reduction with two FMAs against
   constant coeff vectors (`[-8..-1]`, `[0..7]`) + horizontal sum + scale
   broadcast; bin adds stay scalar (fundamental).
4. `main.cpp`: `use_scatter` flag on `--kernel scatter`, trailing param on
   `command_batch_bench`, dispatch to the AVX2 entry only when `k_format == 40`
   (error `bench batch: --kernel scatter needs --format q40` otherwise), usage
   + label lines.
5. Tests: `make_q4_0_matrix` helper (0x2C00 scale, random nibbles); differential
   vs `gemv_dequant_batch_scalar` within `1e-4·cols+1e-3` at (1,32,1),
   (3,256,2), (7,1024,6); null/kind/block guard tests.
6. Gate: `bench batch --format q40 [--kernel scatter] --repeat 5` at 1024²×6
   and 3072²×6 — expect the 3–8x loss to reproduce.

## 8. Verdict and why

**KILLED by measurement, exactly as the dossier's falsifier prescribes.**
Scatter-accumulate is the wrong trade on this engine: it replaces vectorized
widen+FMAs and L1-resident temp traffic with a serial scalar RAW chain on 16
bins, losing 2.9x at small n and 7.6x at large n. The idea's own precondition
analysis holds (Q4-only, bins fit YMM, FP exact-ish), but the cost model
G(n,k) omitted the scatter-serialization term that dominates on AVX2 without
scatter-add hardware. Do not revisit without SIMD scatter-add (AVX-512F
conflict-detection or similar) — reordering the same 16 bins cannot fix a
serial chain. Bench-only prototypes remain the right vehicle for kernel ideas:
this one never risked the model path.

## 9. Exact commands run and their results

- `git checkout -b spike-1558-scatter-q40` → new branch.
- 5-file spike (§5.2–5.6; one stray test edit made and immediately reverted,
  verified via `git diff`).
- `cmake --build --preset release` → 54/54 after fixing one ternary type
  error (`*config.pool`); `ctest` → pass; `tge_tests.exe` → 143/143.
- `bench batch --format q40 --repeat 5` default vs `--kernel scatter` at
  1024²×6 (0.289 vs 0.852 ms) and 3072²×6 (0.638 vs 4.880 ms).
- `generate` (Llama-3B-Q4_K_M, spike binary) → ids identical to baseline.
- `git checkout -- <5 files>` → diff empty; `git checkout master`;
  `git branch -D spike-1558-scatter-q40`.
- Final: build clean, `ctest` pass, `aikit.py check` 2/4 (pre-existing fails).
- No push.
