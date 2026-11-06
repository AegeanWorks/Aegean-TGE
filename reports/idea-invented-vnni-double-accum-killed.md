# INVENTED: VNNI Double-Accumulation Audit — KILLED (1.504e-03 vs 1.503e-03; combining precision exonerated; localization harness named next)

- **Idea**: Invented follow-up, queued by the VNNI error-split report (§8): replace the per-block float fmadd accumulation in `gemv_q8_batch_vnni` (group-of-6 + scalar tail) with double accumulation (exact int32→double conversion, 53-bit mantissa), on the hypothesis that 96 float combinings per row explain the 1.5e-3 KL. Falsifier (stated): kill if KL stays above 1e-4 (no meaningful move toward the 1e-9 bar) or prefill t/s regresses >5% vs VNNI.
- **Cycle date**: 2026-11-06
- **Verdict**: KILLED on the KL arm. Double accumulation prices KL 1.504e-03, worst 1.294e-02, 993/1023 — the same class as float accumulation (1.503e-03, worst 1.533e-02, 993/1023). Combining precision contributes nothing measurable; the int-dot error is neither activation quant (qonly at floor, previous cycle) nor accumulation. Remaining suspects, narrowed: bias-correction path (integer-exact on paper — needs an empirical no-bias variant to close), fp16 weight-scale application order, or an outright indexing bug in the group/tail paths that only real weight/activation distributions trigger (unit differentials pass with tight tolerance on random data). Next spike named in §8: per-layer max-diff localization harness (dump per-matmul VNNI vs float outputs on a fixed prompt, find the first diverging layer/matmul) — the efficient step, not more blind kernel variants. Spike fully reverted (3 files + test); nothing pushed (human gate).
- **Tree state at close**: `git status` clean (no diff, no untracked), stash unchanged (only the pre-existing `leftover peer work shelved 2026-11-04`, untouched). Spike branch created (zero commits) and deleted; all spike files reverted with empty diff verified. `master` ahead of `origin/master` by 66 before this report. Nothing pushed (human gate).

## 1. Mechanism (as spiked)

In `gemv_q8_batch_vnni` (group path): per-block int32 partials converted with `_mm256_cvtepi32_pd` (exact — all partial sums fit int32) and combined with `_mm256_fmadd_pd` into `__m256d` accumulators with the scale in double; row end via double horizontal sum, single float conversion. Tail path: scalar double accumulation over the 8 lanes. Everything else identical (same bias correction, same scales, same blocking, same threading). A 1-token portability fix rode along inside the spiked file (reverted with it): `_mm_extractf128_ps` (MSVC-only) → `_mm256_extractf128_ps` — see §9.

## 2. Predicted gain — not reached, by measurement

If combining rounding were the error, double accumulation should have moved KL from 1.5e-3 toward the 1.4e-11 kq8 floor (≥2 orders to clear the kill bar at 1e-4). It moved nowhere (1.504e-03 vs 1.503e-03, same 993/1023, worst-KL same class). The t/s arm is moot (kill on KL first) — no bench A/B run, honestly recorded as not-run rather than implied.

## 3. Falsifier outcome — KL arm fires, accumulation exonerated

Kill bar was KL staying above 1e-4: measured 1.504e-03 — fires. Combined with the previous cycle: activation quant exonerated (qonly at floor 9.248e-12 digit-identical) AND accumulation exonerated (this cycle) — two of the four dot-path suspects dead with measurements. The int math (bias correction, dpbusd, lane sums) is exact on paper; that a real-weight error of 0.77 max|logit| persists through exact math + exact combining points at scale application order or an input-indexing divergence visible only on real distributions (random-data differentials pass tight) — hence the localization harness, not another arithmetic variant.

## 4. File anchors (master = pre-spike; spike reverted)

- `src/cpu/avx2/q8_batch_vnni.cpp:24-46` (`quantize_block` + `quantize_q8_batch` — untouched this cycle), `:68-209` (VNNI dots: bias correction, per-block fmadd, horizontal sum — the group path; tail `:194-224` — audited region).
- `include/tge/cpu/kernels.hpp:109-142` (Q8BatchArgs/quantized-buffer contract, `quantize_q8_batch` decl — untouched).
- `tests/test_q8.cpp` (float-batch differentials the spike differential mirrors; the spike test `vnni_double_accum_matches_scalar_reference` green, reverted).
- Grounds: VNNI error-split report (quant exonerated, engagement confined to Q8_0, kq8 true floor 1.006e-11 on Q4_K_M full corpus); 1394 report (1.5e-3 price + gate).

## 5. What was done (temporary spike, fully reverted)

1. `git checkout -b spike-vnni-double-accum` (zero commits ever made on it; deleted at close).
2. Spike (3 files): double accumulators + `horizontal_sum_pd` + scalar double tail + portable-extract drive-by in `q8_batch_vnni.cpp`; `dequantize_q8_batch` decl+impl re-added for the test (reverted with it); `vnni_double_accum_matches_scalar_reference` differential (VNNI-quantized input vs scalar on dequantized-rounded input — isolates combining arithmetic; tight tolerance `1e-5*cols+1e-4`; VNNI ISA-guarded skip).
3. Mid-spike test failure, diagnosed honestly: first version compared quantized-input VNNI vs float-input scalar (activation rounding, worst 0.04–0.98) — wrong comparison, fixed to rounded-inputs-both-sides, green. Recorded so the tolerance choice is auditable.
4. `cmake --build --preset release` clean; `ctest --preset release` pass; `tge_tests.exe` 142/142 (141 + 1 new).
5. Box health gate: wall 59.70 read (healthy). `python aikit.py check aegean-tge` 2/4 (test + secrets pass incl. a 17.8 s PASS with no flake this cycle; hygiene + guidelines fail pre-existing).
6. Fidelity `--vnni` (0.5B-Q8_0, arch corpus, ctx 1024, 6 threads, gpu off): §6.
7. Thinking set on spike binary (VNNI file changed but `--vnni` off → identical dispatched paths): story ids byte-identical to baseline, reasoning prompt fluent.
8. Reverted all 3 files (`git checkout --`, empty diff verified), `git checkout master`, `git branch -D spike-vnni-double-accum`; final gates on reverted master: build clean, `ctest` pass, `tge_tests` 141/141.

## 6. Measured numbers (wall 59.70 healthy bracketing)

| run | KL mean | worst KL | top-1 | wall |
|---|---|---|---|---|
| float accum `--vnni` (1394 cycle) | 1.503e-03 | 1.533e-02 | 993/1023 | 47.52 s |
| **double accum `--vnni` (spike)** | **1.504e-03** | **1.294e-02** | **993/1023** | 46.83 s |

Same class to the digit-group: mean within 0.07%, same 30 flips, worst-KL same order. The differential unit test passes with the tight tolerance on random data (both combining arithmetics exact-ish there) — the fidelity gap opens only on real weights/activations, consistent with a data-dependent divergence (scale application or indexing), not precision.

## 7. Full reimplementation recipe (kill-grade — rebuild only to re-verify, then STOP; the next spike is different)

Rebuild: §5.2 + double tail + differential test + portability one-liner; gate: build 54/54, 142/142 tests, fidelity `--vnni` reproduces §6 row 2 within run noise (and still fails the 1e-4 bar — that IS the expected result). Do NOT extend this spike (compensated sums, wider accumulators — the double result already bounds the whole family at zero). INSTEAD, next spike (named, not built): per-layer max-diff localization harness — fixed prompt, dump per-matmul outputs for VNNI vs float dots, report the first diverging (layer, matrix) with magnitudes; falsifier for that round: harness must name ONE matmul whose isolated substitution (float dots for that matmul only) recovers ≥90% of the 1.5e-3 gap, else the error is distributed and the program moves to scale-application order. To re-verify this kill: re-run the §6 fidelity and confirm still-1.5e-3-class.

## 8. Verdict and why — plus two findings for the owner

**KILLED on the KL arm: accumulation exonerated.** The VNNI-accuracy program is now a localization problem, not an arithmetic one (quant: floor; combining: no move). Findings for the owner, no action taken: (1) VNNI has no differential unit test on record — the 1.5e-3 lived in fidelity-only territory until the spike test (reverted with the spike; consider keeping a version of it — it passes on master too, since the float path it pins is unchanged); (2) `q8_batch_vnni.cpp:18` uses the MSVC-only `_mm_extractf128_ps` alias and does NOT compile under the pinned MinGW preset (exposed when the spike forced a recompile — the shipped .obj predates the toolchain); any future touch of that file must carry the one-token portable fix or the build breaks.

## 9. Exact commands run and their results

- `git checkout -b spike-vnni-double-accum` → branch (zero commits, deleted at close).
- 3-file spike (§5.2) + 1 test (with one honest mid-spike test-design fix).
- `cmake --build --preset release` → 54/54 (after the portability one-liner); `ctest --preset release` → pass; `tge_tests.exe` → 142/142 on spike, 141/141 after revert.
- Wall 59.70; `aikit.py check` → 2/4 (test PASS, no flake; pre-existing hygiene + guidelines fails).
- Fidelity `--vnni` (double accum) → KL 1.504e-03, worst 1.294e-02, 993/1023 (46.83 s).
- Story ids identical; reasoning fluent (default paths).
- `git checkout -- <3 files>` → clean; `git checkout master`; `git branch -D`; final master gates green; report committed; no push.

(End of file.)
