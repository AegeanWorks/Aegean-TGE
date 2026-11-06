# INVENTED (1636 follow-up): Batched Scale Conversion in dot_row_q8 — KILLED (1T +12%, 10T −9%: the thread split repeats, shipped default loses)

- **Idea**: The 1636 follow-up ("batch 8 scales per cvtph"): in `dot_row_q8` (`src/cpu/avx2/q8_gemv_avx2.cpp`), replace the per-block scalar `half_to_float` + broadcast with group-of-8 batching — gather 8 strided fp16 scales, convert with ONE `_mm256_cvtph_ps`, broadcast per block from the converted vector. 1/8 the conversions, zero lane-crossing extracts (the extract sat on the per-block fmadd critical path and regressed 10 threads in 1636).
- **Cycle date**: 2026-11-06
- **Verdict**: KILLED on the stated falsifier (10T best-of-3 ≥ +5% to proceed). Measured: 1T +11.9%, 10T −8.9% best-of-3, interleaved 10T pairs −10%/−14% — the 1636 thread split reproduces almost exactly (+8%/−4..−13% there). Conversion is off the 10-thread critical path (memory/sync-bound per the file comment); batching it better only helps the latency-bound 1-thread config users don't run. Bit-safety proven (fidelity digit-identical, ids identical). Nothing to keep; kill-grade recipe below is rebuild-only.
- **Tree state at close**: `git status` clean, no diff, stash unchanged (only the pre-existing `leftover peer work shelved 2026-11-04`). Spike branch `spike-batchcvtph` created (zero commits) and deleted; the one spike file reverted with empty diff verified. Saved spike binary + scratch outputs deleted from temp. `master` ahead of `origin/master` by 94 after this report. Nothing pushed (human gate).

## 1. Mechanism (as spiked, reverted)

File-local `dot_row_q8_batched` beside `dot_row_q8`: blocks in groups of 8 (bulk = blocks & ~7) gather 8 scales via scalar 2B loads + `_mm_set_epi16`, single `_mm256_cvtph_ps`, `_mm256_store_ps` to an `alignas(32)` temp, then the unchanged dot loop per block with `_mm256_set1_ps(conv_scales[k])`; scalar tail (manual `half_to_float`) for blocks % 8. Same floats on the same bytes as `half_to_float` (1636 exhaustive proof covers the lane conversion), same block order in `total`. Dispatch in `gemv_q8_0_avx2`: F16C resolved once per call (`tge::cpu::detect().features.f16c`, 1636 pattern; TU already builds with `-mf16c`), batched vs legacy per row.

## 2. Predicted gain — split as feared

Conversion uops ÷8: 1T latency-bound → expected +2–8% (conversion-heavy front end); 10T memory/sync-bound → expected +0–3%, sign uncertain (extract removal helps, gather hurts). Measured §6.

## 3. Falsifier outcome — 10T arm fails, kill

PROCEED required 10T best-of-3 ≥ +5% AND 1T no-regress AND fidelity digit-identical AND ids identical. Fidelity + ids pass; 1T passes (+11.9%); 10T fails (−8.9% best-of-3, −10/−14% interleaved). Verdict: KILL. The 1T win is real but not shippable — special-casing the non-default path was already not-recommended in 1636, and this round re-proves why: the win inverts exactly where users run.

## 4. File anchors (master = pre-spike; spike reverted)

- `src/cpu/avx2/q8_gemv_avx2.cpp` (`dot_row_q8` :27–59, dispatch :156–165; spike added the `tge/core/cpu.hpp` include, `dot_row_q8_batched`, and the per-call F16C branch — all reverted).
- `include/tge/cpu/kernels.hpp` (:273–307: `kQ8BlockBytes = 34` strided layout, branchy scalar `half_to_float` — the cost being attacked).
- `CMakeLists.txt` :40 (`-mavx2;-mfma;-mf16c` on AVX2 TUs — no flag work owed).
- Grounds: `reports/idea-1636-scale-pack-split.md` (§6 split, §7 follow-up, exhaustive cvtph proof).

## 5. What was done (temporary spike, fully reverted)

1. `git checkout -b spike-batchcvtph` (zero commits, deleted at close). Baseline: build 54/54, `tge_tests` 141/141.
2. ANALYZE: scale layout (strided 34B), TU flags (F16C present), `half_to_float` body (branchy ~20 uops), runtime-gate pattern — gathered honestly, including the gather cost that weakens the story (§2).
3. Applied the spike (§1). Rebuilt 54/54, `tge_tests` 141/141.
4. Box health: `bench mem` read 58.72 GB/s (reference band 54–61).
5. Spike-side: story 1T×3 + 10T×3 (best-of-3), fidelity (arch corpus, ctx 512, 86→85 pos), thinking set (story + creative, saved); spike binary saved to temp.
6. Reverted the file, rebuilt baseline 54/54, `tge_tests` 141/141; baseline-side: same 1T/10T best-of-3 + same fidelity; interleaved pairs (saved spike vs baseline: 10T S-B-S-B, 1T S-B); master thinking set; ids + fidelity compared.
7. `aikit.py check aegean-tge` → 2/4 (test + secrets PASS; hygiene + guidelines fail pre-existing, same pair as every prior cycle).
8. Deleted spike binary + scratch outputs; `git status` clean, stash untouched.

## 6. Measured numbers (Qwen2.5-0.5B-Q8_0, story 30 tok, `--gpu off`, warm page cache)

| config | spike (tok/s) | baseline (tok/s) | delta |
|---|---|---|---|
| 1T best-of-3 | 22.94 / 23.71 / **23.94** | 21.39 / 20.96 / **21.21** | **+11.9%** (best-of-best 23.94/21.39) |
| 10T best-of-3 | **76.50** / 70.55 / 75.44 | 79.58 / **83.95** / 75.53 | **−8.9%** (76.50/83.95) |
| 10T interleave S/B/S/B | 71.55 / 76.19 | 83.15 / 84.82 | −14.0% / −10.2% per pair |
| 1T interleave S/B | 19.35 | 14.07 | box-shifted late pair, collapsed regime — unreliable; 1T claim rests on the best-of-3 blocks |

Fidelity (both binaries, arch corpus, ctx 512): KL mean **1.135e-11**, worst **1.081e-10**, max |logit diff| **8.583e-05**, top-1 **85/85** — digit-identical, so the conversion change is bit-safe (NaN-scale caveat inherits from 1636, untriggered here). Thinking set: story + creative ids byte-identical spike-vs-baseline.

Reading: the 10T regression is over-determined (best-of-3 −8.9% + two interleaved pairs −10%/−14%, no overlap of run ranges except one baseline 75.53). The 1T win (+11.9%) matches the 1636 class (+8%) and the cross-session baseline (21.39 vs their 21.26) confirms the setup. Conclusion: conversion work is not on the 10-thread decode critical path; removing it better cannot move the configuration that ships.

## 7. Full reimplementation recipe (kill-grade — rebuild only to re-verify)

1. Re-apply §1 (one include, one function, one dispatch branch in `q8_gemv_avx2.cpp`).
2. Gate: build 54/54, `tge_tests` 141/141; story 1T/10T best-of-3 + interleaved pairs per §5–6; fidelity digit-identity; ids identity. Expect the §6 split to reproduce (1T win, 10T loss).
3. Revival trigger ONLY (named, not built): a decode shape where per-block conversion is measured ON the threaded critical path (e.g., a future kernel whose dots are already free) — until such a shape exists, no scale-conversion variant re-proves the thread split.

## 8. Verdict and why

**KILLED: batching the conversion wins the wrong configuration.** The mechanism works exactly as designed (1T +12%, bit-exact), but the prize is in 10-thread decode where it loses −9..−14%: conversion is off that critical path, and the batched gather + vector traffic interacts worse with it than the scalar original. This closes the 1636 follow-up with a second independent measurement of the same split — the scale-conversion line is now exhausted (manual → cvtph-extract → batched: +8/−13, +12/−9). Next-cycle candidates: S2 pre-packed-B stays caller-less (1325 — skip unless a production GEMM caller appears); the live fronts are the 1517 epic's stage-2 skip harness and the human-gated ports (mprefix-table, sparse no-gather, batchsnap kind tags, timers fix).

## 9. Exact commands run and their results

- `git checkout -b spike-batchcvtph` → branch (zero commits, deleted at close).
- Baseline: `cmake --build --preset release` → 54/54; `tge_tests.exe` → 141/141.
- Spike (§1); rebuild → 54/54; `tge_tests` → 141/141.
- `bench mem` → read 58.72 GB/s (healthy).
- Spike: `generate --model <0.5B-Q8_0> --threads {1,10} --gpu off --prompt <fox> --n 31` ×3 each → §6 spike rows; `fidelity --corpus docs/architecture.md --context 512 --tokens 86 --threads 6` → §6 fidelity row; thinking set (saved); binary saved to temp.
- Revert file → clean; `git checkout master`; branch deleted; rebuild → 54/54; `tge_tests` → 141/141.
- Baseline: same 1T/10T ×3 + same fidelity → §6 rows (digit-identical fidelity); interleaved pairs (saved spike vs baseline) → §6; master thinking set → ids identical; `aikit.py check` → 2/4 (test + secrets pass; pre-existing fails).
- Spike binary + scratch deleted; `git status` clean; report committed; no push.

(End of file.)
