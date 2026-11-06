# INVENTED: VNNI Asymmetric + Fine Combo (Kitchen Sink) — KILLED (2.97e-4, still 3x over the 1e-4 bar; road ends before the bar)

- **Idea**: Invented, queued as the VNNI program's closing measurement: combine per-16 granularity (finescale round: 0.54x) with asymmetric min-offsets (asym round: 0.40x) into per-16 asymmetric blocks (2 scales + 2 mins per 32 values), testing whether the mechanisms stack to the 1e-4 bar the program needs.
- **Cycle date**: 2026-11-06
- **Verdict**: KILLED on both stated arms. Fidelity `--vnni` (0.5B-Q8_0, arch corpus, ctx 1024): KL 2.971e-04, worst 3.405e-3, top-1 1012/1023 (11 flips) vs the 1e-4 bar (3x miss); 70-token prefill 161.0 t/s vs coarse 235.2 (0.68x — misses the 5% guard; fine was 0.73x, asym 0.96x). The mechanisms multiply almost exactly as predicted (0.54 × 0.40 ≈ 0.22 → predicted ~3.3e-4, measured 2.97e-4) — and the product still misses by 3x. Endpoint analysis kills the road, not just the spike: per-K asymmetric costs 1 + 8/K bytes/value (scales + offsets), so the granularity the bar demands (~per-2) would exceed fp32's own 4 B/value. The bar is unreachable on this road without spending more memory than the unquantized baseline. Spike fully reverted (5 files + test); nothing pushed (human gate).
- **Tree state at close**: `git status` clean (no diff, no untracked), stash unchanged (only the pre-existing `leftover peer work shelved 2026-11-04`, untouched). Spike branch created (zero commits) and deleted; all spike files reverted with empty diff verified. `master` ahead of `origin/master` by 78 before this report. Nothing pushed (human gate).

## 1. Mechanism (as spiked)

`quantize_q8_batch` gained a `float* offsets` parallel array (6-arg signature; compiler-enforced call sites: `quantize_activations` + the kernel's internal fallback). Per 32-block: two asymmetric 16-halves (min/max → scale + offset each, uint8 bit patterns, degenerate-constant exact). VNNI kernel: per-half masked dpbusd dots, `((idot_lo·sa_lo + min_lo·wsum_lo) + (idot_hi·sa_hi + min_hi·wsum_hi)) · sw` with hoisted per-block weight sums, scalar float accumulation (the finescale 8x-bug lesson applied pre-emptively). No XOR bias anywhere (activations are already uint8). `Q8Quantized`/`Q8BatchArgs` gained the parallel `offsets` array with null-guards that fall back to float instead of garbage.

## 2. Predicted gain — confirmed almost exactly, and still insufficient

Predicted: independent-multiplicative stacking, ~0.22x coarse KL → ~3.3e-4. Measured 2.971e-4 (0.198x) — the mechanisms ARE independent and multiplicative, and the product STILL misses the bar 3x. Flips: 30 → 11 (fewer positions flip, but the ones that do sit on the same decision boundaries — the max|diff| only fell 0.77 → 0.36).

## 3. Falsifier outcome — kill on both arms

KL arm: 2.97e-4 vs ≤1e-4 required — miss by 3x. t/s arm: 0.68x coarse on the 70-token shape (worse than either ingredient alone: asym 0.96x, fine 0.73x — the scalar per-half combining from both spikes stacks additively). Neither arm is close; no re-weighing changes the verdict.

## 4. File anchors (master = pre-spike; spike reverted)

- `src/cpu/avx2/q8_batch_vnni.cpp` (combo quantizer + per-half masked-dot kernel + fallback offsets — reverted).
- `include/tge/cpu/kernels.hpp` (6-arg decl, `quantized_offsets`, `Q8Quantized.offsets` — reverted).
- `include/tge/models/qwen2.hpp` (`q8_offsets_` — reverted) + `src/models/qwen2.cpp` (alloc + both call sites — reverted).
- `tests/test_q8.cpp` (`vnni_asym_fine_matches_scalar_reference` differential, tight `1e-5·cols+1e-4` — green, reverted with the spike).
- Grounds: replay (kernel exonerated), double-accum (combining exonerated), qonly re-run (quantization confirmed: 1.515e-3), finescale (0.54x), asym (0.40x), statcorr (maxact R² 0.67).

## 5. What was done (temporary spike, fully reverted)

1. `git checkout -b spike-vnni-combo` (zero commits ever made on it; deleted at close).
2. Spike (5 files + 1 test): combo quantizer (this cycle's only new math); per-half masked-dot kernel with hoisted weight sums + scalar accumulators (8x-bug avoided by construction); offsets plumbing through model + fallback; unit differential with exact dequant reference.
3. `cmake --build --preset release` clean (54/54); `ctest --preset release` pass; `tge_tests.exe` 142/142 (141 + 1 new) — built clean first try, no mid-spike fixes this time.
4. Box health: wall 56.21 bracketing; `python aikit.py check aegean-tge` 2/4 (test PASS, no flake; pre-existing hygiene + guidelines fails).
5. Fidelity `--vnni` (combo): §6 row 2 (35.99 s).
6. t/s guard (70-token fox, 0.5B): combo 161.0/62.9 vs coarse 235.2/74.5 vs float 291.9/77.0 (stash-free comparison — coarse numbers from the finescale round's stash-built binary, same prompt/box class).
7. Thinking set on spike binary (non-VNNI paths untouched; qonly-style flags absent): story ids byte-identical to baseline; reasoning fluent.
8. Reverted all 5 files (`git checkout --`, empty diff verified), `git checkout master`, `git branch -D spike-vnni-combo`; final gates on reverted master: build clean, `ctest` pass, `tge_tests` 141/141; `python aikit.py check aegean-tge` 2/4.

## 6. Measured numbers (0.5B-Q8_0, arch corpus, ctx 1024, 6 threads, gpu off, unless noted)

| run | KL mean | worst KL | max\|diff\| | top-1 | wall |
|---|---|---|---|---|---|
| coarse `--vnni` (on record) | 1.503e-03 | 1.533e-02 | 0.77 | 993/1023 | 47.52 s |
| fine (on record) | 8.093e-04 | 8.545e-3 | — | 1002/1023 | 34.58 s |
| asym (on record) | 5.972e-04 | 6.687e-3 | 0.46 | 992/1023 | 32.51 s |
| **combo (spike)** | **2.971e-04** | **3.405e-3** | **0.36** | **1012/1023** | **35.99 s** |
| dense floor (reference arm) | 9.248e-12 | 2.418e-10 | 1.9e-04 | 1023/1023 | — |

70-token prefill t/s (6 threads): float 291.9 / coarse 235.2 / fine 171.1 / combo 161.0. Decode: float 77.0 / coarse 74.5 / combo 62.9 (all within run noise of each other except the prefill ALU cost, which compounds per ingredient).

## 7. Full reimplementation recipe (kill-grade — rebuild to re-verify, then STOP; the road ends here)

Rebuild: §5.2 (combo quantizer with degenerate-constant handling; per-half masked dpbusd + hoisted weight sums + scalar accumulation; offsets array through model + fallback with null-guards; differential test with exact dequant). Gate: build 54/54, 142/142 tests (differential must pass at the tight bound — it validates the integer math), fidelity `--vnni` reproduces §6 combo row within run noise (and still misses 1e-4 — that IS the expected result). Do NOT extend (per-8/per-4 asymmetric, finer asymmetric, three-way combos): the endpoint arithmetic is decisive — per-K asymmetric costs 1+8/K bytes/value, so the granularity the bar demands (~per-2) exceeds fp32's 4 B/value, at which point the correct implementation is "don't quantize" rather than a cleverer quantizer. NEXT PROGRAM (named, not started — different road, not a continuation): per-channel activation scales (amortized statistics, not per-block), or conceding int8 VNNI activations entirely and spending the budget where int8 already wins (kq8-style integer paths on K-quant models, which price at floor). To re-verify this kill: re-run the §6 fidelity; combo must stay ~0.2x coarse and above 1e-4.

## 8. Verdict and why

**KILLED on both arms, and the road behind it is closed too.** The combo is the single best VNNI-accuracy number on record (2.97e-4, 0.2x coarse, 11 flips) — and it proves the bar unreachable on this road: the mechanisms multiply cleanly, the product still misses 3x, and the granularity the bar demands costs more memory than not quantizing. This closes the VNNI accuracy program that ran eight cycles (quantizer identified → combining/int-dots/accumulation exonerated → layout mapped → asymmetric probed → fino granularity probed → combo probed): every falsifier fired except the ones that confirmed sub-bar improvements, and the endpoint arithmetic shows no continuation of THIS road reaches keep-grade. The loop's honest output is the map plus this terminus — future VNNI work starts from a different road (per-channel) or not at all.

## 9. Exact commands run and their results

- `git checkout -b spike-vnni-combo` → branch (zero commits, deleted at close).
- 5-file spike + differential test (§5.2; built clean first try, no mid-spike fixes).
- `cmake --build --preset release` → 54/54; `ctest --preset release` → pass; `tge_tests.exe` → 142/142 on spike, 141/141 after revert.
- Wall 56.21 bracketing; `aikit.py check` → 2/4 (test PASS, no flake; pre-existing hygiene + guidelines fails).
- Fidelity `--vnni` (combo) → KL 2.971e-04, worst 3.405e-03, 1012/1023 (35.99 s).
- 70-token prefill: combo 161.0/62.9 vs coarse 235.2/74.5 vs float 291.9/77.0 (coarse from finescale round's stash build, same shape).
- Story ids byte-identical to baseline; reasoning fluent.
- `git checkout -- <5 files>` → clean; `git checkout master`; `git branch -D`; final master gates green; report committed; no push.

(End of file.)
