# INVENTED: VNNI Per-16 Finer Activation Scales — KILLED (KL halves to 8.1e-4 but misses the 1e-4 bar; 27% slower than coarse VNNI)

- **Idea**: Invented, queued by the qonly re-run (activation quant confirmed at 1.515e-3): halve the VNNI activation-quant block from per-32 to per-16 symmetric scales (two scales per 32-value block, same int8 values layout, contiguous-traffic-preserving), so one outlier crushes 15 neighbours instead of 31. Falsifier (stated): kill if fidelity KL stays above 1e-4 (no meaningful move toward the 1e-9 bar) OR prefill t/s regresses >5% vs coarse VNNI on the same shape.
- **Cycle date**: 2026-11-06
- **Verdict**: KILLED on BOTH arms. Fidelity `--vnni` with fine scales: KL 8.093e-04, worst 8.545e-3, top-1 1002/1023 (21 flips) vs coarse 1.503e-03 / 993/1023 (30 flips) — direction confirmed (KL ≈ halved, flips −30%), magnitude insufficient (8x above the 1e-4 bar). Timing on a 70-token prefill (0.5B-Q8_0): fine 171.1 t/s vs coarse 235.2 (−27%) vs float 291.9 — fails the 5% guard (and coarse itself is 0.81x float on this small-shape window, so the guard is shape-sensitive; the extra scalar lane-store/adds in this spike's combining are implementation, not intrinsic). Extrapolation kills the family at this granularity ladder: per-32 → 1.50e-3, per-16 → 0.81e-3 (≈0.54x per halving); reaching 1e-6 would need ~10 more halvings — the residual is structural (symmetric, uncentered), not granular. Named next: asymmetric (zero-point) activation quantization or per-channel scales, gated on the same 1e-4 bar — a new idea, not a tuning of this one. Spike fully reverted (4 files + test); nothing pushed (human gate).
- **Tree state at close**: `git status` clean (no diff, no untracked), stash unchanged (only the pre-existing `leftover peer work shelved 2026-11-04`, untouched). Spike branch created (zero commits) and deleted; all spike files reverted with empty diff verified. `master` ahead of `origin/master` by 72 before this report. Nothing pushed (human gate).

## 1. Mechanism (as spiked)

`quantize_q8_batch` emits two scales per 32-value block (per-16-half max/round, interleaved [lo0,hi0,…]); the VNNI group + tail paths split the bias-corrected int32 lanes into halves (lanes 0–3 bytes 0–15, lanes 4–7 bytes 16–31 — the correction is already folded per-lane, no second subtraction) and combine `half = lo*sa_lo + hi*sa_hi` with one float fmadd × weight scale. Same memory traffic pattern (values unchanged; scales buffer doubled: tokens×blocks → tokens×2·blocks). One honest implementation note: halves are combined with scalar lane stores/adds (clarity over cleverness for a diagnostic); a shuffle-based variant would cut the ALU overhead the t/s guard prices below.

## 2. Predicted gain — direction right, magnitude short

+9 flips recovered (30 → 21), KL −46%, worst-KL −44% (1.53e-2 → 8.55e-3). The predicted scaling held almost exactly (0.54x per granularity halving), which is also what kills it: the ladder extrapolates to ~10 more halvings for 1e-6 — absurd, so the remaining error is not granularity.

## 3. Falsifier outcome — both arms fire

KL arm: 8.093e-04 >> 1e-4 bar — kill. t/s arm: fine/coarse = 171.1/235.2 = 0.73 (−27%) — kill (bar: within 5%). Caveats recorded honestly: (a) coarse itself is 0.81x float here (small model + 70-token prefill is overhead/ALU-bound, not the load-bound regime where VNNI's 1.9x was priced — the guard is shape-sensitive and this window disfavors all int paths); (b) part of fine's deficit is the spike's scalar lane-store/add combining, not the scales themselves (a shuffle-based variant is the fair rematch — but with KL missing the bar 8x, no rematch is owed).

## 4. File anchors (master = pre-spike; spike reverted)

- `src/cpu/avx2/q8_batch_vnni.cpp` (`quantize_block_fine` + per-16 `quantize_q8_batch` + half-split group/tail combining — reverted).
- `include/tge/cpu/kernels.hpp` (scales contract `[tokens][2·cols/32]` — reverted).
- `src/models/qwen2.cpp` (`quantize_activations` scales sizing ×2 — reverted).
- `tests/test_q8.cpp` (`vnni_finescale_matches_scalar_reference` differential, tight tolerance `1e-5·cols+1e-4`, VNNI-guarded — green, reverted).
- Grounds: qonly re-run (1.515e-3 = the error to beat), replay (int-dot exoneration), statcorr (maxact R² 0.67).

## 5. What was done (temporary spike, fully reverted)

1. `git checkout -b spike-vnni-finescale` (zero commits ever made on it; deleted at close).
2. Spike (4 files + 1 test): fine quantizer; half-split group + tail paths; doubled scale buffers (local + model); contract comments; differential test.
3. Two honest mid-spike bugs, both caught by the new test before any benchmark: (a) double-subtracted bias correction (sums already corrected per-lane) — fixed; (b) 8x blowup from broadcasting a scalar half to all 8 lanes before `horizontal_sum` — diagnosed from the exact 8.000x ratio of ref vs opt (ref0 11.9044 vs opt0 95.2355), fixed with scalar float accumulators; differential then passed with wide headroom (ref==opt to 5+ digits).
4. `cmake --build --preset release` clean (54/54); `ctest --preset release` pass; `tge_tests.exe` 142/142 (141 + 1 new).
5. Box health: wall 64.46 + aikit 2/4 (test PASS, no flake; hygiene + guidelines fail pre-existing).
6. Fidelity `--vnni` (0.5B-Q8_0, arch corpus, ctx 1024, 6 threads, gpu off): §6 row 1.
7. t/s guard (70-token fox prompt, 0.5B): fine-VNNI vs coarse-VNNI (stash-rebuild dance) vs float, §6 table; story ids + reasoning (thinking set).
8. Reverted all 4 files (`git checkout --`, empty diff verified), `git checkout master`, `git branch -D spike-vnni-finescale`; final gates on reverted master: build clean, `ctest` pass, `tge_tests` 141/141; `python aikit.py check aegean-tge` 2/4.

## 6. Measured numbers

| run (0.5B-Q8_0, arch ctx1024, 6 threads) | KL mean | worst KL | top-1 | wall |
|---|---|---|---|---|
| coarse `--vnni` (on record) | 1.503e-03 | 1.533e-02 | 993/1023 | 47.52 s |
| fine `--vnni` (spike) | 8.093e-04 | 8.545e-03 | 1002/1023 | 34.58 s |
| dense floor (reference arm) | 9.248e-12 | 2.418e-10 | 1023/1023 | — |

| 70-token prefill + 8 decode, 0.5B-Q8_0, 6 threads | prefill t/s | decode t/s |
|---|---|---|
| float (no flag) | 291.9 | 77.0 |
| coarse `--vnni` (stash-built baseline) | 235.2 (0.81x) | 74.5 |
| fine `--vnni` (spike) | 171.1 (0.59x float, 0.73x coarse) | 61.7 |

Fine-VNNI story ids (short prompt): byte-identical to the dense f32 story (`11 1052 572 … 14126 13`) — consistent with flips concentrating on the natural-text corpus, not toy prompts. Reasoning output fluent.

## 7. Full reimplementation recipe (kill-grade — rebuild to re-verify, then STOP; asymmetric is next)

Rebuild: §5.2 (fine quantizer; half-split group + tail with scalar accumulators — NOT set1+horizontal_sum; doubled scale buffers + contract; differential test with tight tolerance). Gate: build 54/54, 142/142 tests (differential must pass with headroom — it validates the combining, and would have caught both mid-spike bugs); fidelity `--vnni` reproduces §6 row 2 within run noise (and still fails the 1e-4 bar — that IS the expected result). Do NOT extend to per-8/per-4 (the 0.54x-per-halving trend says the bar needs ~10 more halvings). NEXT SPIKE (named, not built): asymmetric activation quantization (per-block zero-point; absorbs outliers without crushing the other 31 values) or per-channel scales; falsifier: fidelity `--vnni` KL ≤ 1e-4 AND prefill t/s within 5% of coarse VNNI on the same 70-token shape (shuffle-based combining required — the scalar lane-store version is disqualified by this round's t/s arm). To re-verify this kill: re-run the §6 pair; fine must stay ~half the coarse KL and ~0.7x its prefill.

## 8. Verdict and why

**KILLED on both stated arms.** Per-16 symmetric scales halve the VNNI error (mechanism confirmed, direction right) but the ladder extrapolates to absurdity — the residual is the symmetric/uncentered construction facing outliers, not granularity. The loop's honest output: a priced negative with a named successor (asymmetric/per-channel), plus two debugging exhibits for the library (double-subtraction and the 8x broadcast bug, both caught by the new differential before any benchmark ran).

## 9. Exact commands run and their results

- `git checkout -b spike-vnni-finescale` → branch (zero commits, deleted at close).
- 4-file spike + 1 test (§5.2; two mid-spike bugs caught by the differential: double-subtracted correction; set1+horizontal_sum 8x — diagnosed from the exact 8.000x ratio, fixed with scalar accumulators).
- `cmake --build --preset release` → 54/54; `ctest --preset release` → pass; `tge_tests.exe` → 142/142 on spike, 141/141 after revert.
- Wall 64.46; `aikit.py check` → 2/4 (test PASS, no flake; pre-existing hygiene + guidelines fails).
- Fidelity `--vnni` (fine) → KL 8.093e-04, worst 8.545e-03, 1002/1023 (34.58 s).
- 70-token story ×3 arms (§6 t/s table: float 291.9, coarse 235.2 via stash dance, fine 171.1); fine story ids == dense; reasoning fluent.
- `git checkout -- <4 files>` → clean; `git checkout master`; `git branch -D`; final master gates green; report committed; no push.

(End of file.)
