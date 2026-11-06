# INVENTED: VNNI qonly Re-Run WITH Engagement Proof — MEASURED (activation quant CONFIRMED at 1.515e-3; prior floor result retracted as silent non-engagement)

- **Idea**: Mandated arbiter from the replay round: re-run the quantize-then-dequantize-then-float-dots diagnostic (`--vnni-qonly`) WITH engagement proof — stderr `vnni-qonly engaged #N rows/cols/tokens` for the first 3 executions plus a `, qonly` fidelity candidate label — to decide whether the prior cycle's digit-identical floor was real (exoneration stands) or silent non-engagement (exoneration falls).
- **Cycle date**: 2026-11-06
- **Verdict**: MEASURED — ACTIVATION QUANTIZATION CONFIRMED AS THE ERROR. Engaged qonly: KL 1.515e-03, worst 1.762e-02, max|diff| 0.951, top-1 992/1023 (36.18 s) vs VNNI-int on record: 1.503e-03, worst 1.533e-02, 993/1023 — the same error class (identical mechanism: the int dots add nothing measurable on top of the rounding; the 12e-6 gap and 1-flip delta are run noise). The falsifier's "exonerated" arm is dead; the "confirmed" arm fires: finer activation scales (Q8_K-style sub-block or per-16) is the named next spike, gated on KL ≤ 1e-6 with t/s within 5% of VNNI. The prior cycle's floor is RETRACTED as silent non-engagement — root cause found: all 15 `quantize_activations` call sites gate on `vnni_`, and the old spike set only its own flag, so the quantized buffer was never built and every matmul silently took the float path (digit-identical floor explained exactly). The replay round's core findings STAND (int-dot arithmetic exonerated at 1.7e-5; inputs carry the divergence) and are now consistent: activation rounding IS the input-side error. Spike fully reverted (6 files + test); nothing pushed (human gate).
- **Tree state at close**: `git status` clean (no diff, no untracked), stash unchanged (only the pre-existing `leftover peer work shelved 2026-11-04`, untouched). Spike branch created (zero commits) and deleted; all spike files reverted with empty diff verified. `master` ahead of `origin/master` by 71 before this report. Nothing pushed (human gate).

## 1. Mechanism (as spiked — qonly-v2 with engagement proof)

Same dequantize-then-float-dots diagnostic as the error-split round, plus three engagement instruments the original lacked: (1) a process-wide execution counter printing `vnni-qonly engaged #N rows/cols/tokens` for the first 3 qonly-path executions; (2) a `, qonly` fidelity candidate label beside `, vnni`; (3) a `maybe_quantize`-free minimal diff — the ONLY functional changes vs the dead cycle are the counter/label plus the engagement fix below. Root-cause fix for the non-engagement (spike-only): `--vnni-qonly` now also sets `vnni` (the 15 quantize call sites gate on `vnni_`; the qonly branches take precedence in both matvec paths, so VNNI int dots never run in qonly mode) — documented at both parse sites as deliberate spike coupling. Falsifier (stated): engaged (counter + label present) AND KL ≈ 1.5e-3 class → activation quant confirmed, recipe for finer scales; engaged AND KL ≈ floor → exoneration reinstated, paradox reopened; no counter/label → spike broken, kill it.

## 2. Predicted gain — diagnostic, delivered

Value was a binary verdict with a named next step in each branch. Delivered: confirmed-quantization, with finer-scales as the specified follow-up. Bonus finding: the non-engagement mechanism (caller-side `vnni_` gates), which audits every future activation-path spike — any new quantize consumer must be engagement-proven, never assumed.

## 3. Falsifier outcome — engaged, and the confirmed arm fires

Engagement triple-proven: stderr shows `engaged #1 rows=896 cols=896 tokens=4` (then #2/#3 on k/v) on a plain story run; the fidelity candidate label reads `(auto, vnni, qonly)`; the number moved two orders (floor → 1.5e-3), which no silent path can produce. KL 1.515e-03 vs VNNI-int 1.503e-03 (Δ 0.8%, inside box noise), worst-KL same order (1.76e-2 vs 1.53e-2), top-1 992 vs 993 of 1023 (Δ 1 flip). Int dots add nothing on top of rounding — the error budget is ~100% activation quantization (per-32 symmetric, outlier-crush).

## 4. File anchors (master = pre-spike; spike reverted)

- `src/models/qwen2.cpp` (quantize call sites — the `vnni_` caller-side gates that caused the silent run; qonly branches in both dispatch sites + execution counter + `write_vnni_replay`-adjacent helpers — reverted).
- `include/tge/cpu/kernels.hpp` + `src/cpu/avx2/q8_batch_vnni.cpp` (`dequantize_q8_batch` decl/impl — reverted).
- `include/tge/models/qwen2.hpp` (`vnni_qonly_` + setter — reverted).
- `apps/cli/main.cpp` (`--vnni-qonly` ×2 + trailing params + setters + `, qonly` label + vnni-coupling with comment — reverted).
- `tests/test_q8.cpp` (roundtrip test, green — reverted with the spike).
- Grounds: replay report (1.7e-5 int-dot exoneration + input-side divergence), statcorr (R² 0.67), bisection, layergate.

## 5. What was done (temporary spike, fully reverted)

1. `git checkout -b spike-vnni-qonly2` (zero commits ever made on it; deleted at close).
2. Spike (6 files + 1 test): qonly v2 per §7 of the error-split report + engagement counter/label + coupling fix.
3. Mid-spike self-catch #1: same-run story showed NO engagement lines — diagnosed within minutes as the caller-side `vnni_` gate (the prior cycle's exact failure mode, reproduced deliberately then fixed); coupled the flags with an explicit comment.
4. `cmake --build --preset release` clean (54/54); `ctest --preset release` pass; `tge_tests.exe` 142/142 (141 + roundtrip).
5. Box health: wall 56.35 + aikit 2/4 (test PASS, no flake).
6. Engagement check (story): 3 announcement lines + normal run; fidelity `--vnni-qonly` (label `, qonly` confirmed): §6; thinking set (default paths): story ran nominal, reasoning fluent.
7. Reverted all 6 files (`git checkout --`, empty diff verified), `git checkout master`, `git branch -D spike-vnni-qonly2`; final gates on reverted master: build clean, `ctest` pass, `tge_tests` 141/141; `python aikit.py check aegean-tge` 2/4 (test PASS; hygiene + guidelines fail pre-existing).

## 6. Measured numbers (0.5B-Q8_0, arch corpus, ctx 1024, 6 threads, gpu off)

| run | KL mean | worst KL | max\|diff\| | top-1 | wall | reads as |
|---|---|---|---|---|---|---|
| VNNI int (on record) | 1.503e-03 | 1.533e-02 | 0.77 | 993/1023 | 47.52 s | the priced error |
| qonly engaged (this run) | 1.515e-03 | 1.762e-02 | 0.95 | 992/1023 | 36.18 s | same class: quant is the error |
| qonly silent (prior cycle) | 9.248e-12 | 2.418e-10 | 1.9e-04 | 1023/1023 | 33.30 s | RETRACTED (float path, untouched) |

The engaged run differs from VNNI-int within run noise on every column while differing from the silent run by two orders — the spike measures what it claims, and the old number measured the float path.

## 7. Full reimplementation recipe (arbiter — rebuild to re-verify, then STOP; finer scales next)

Rebuild: §5.2 (helper decl/impl; model flag + setter; 2 qonly branches WITH the engagement counter first; CLI ×2 with the vnni-coupling + label; roundtrip test). Gate: build 54/54, 142/142 tests; story shows the 3 announcement lines; fidelity label prints `, qonly`; fidelity `--vnni-qonly` reproduces §6 row 2 within run noise. NEXT SPIKE (named, not built): finer activation scales for the VNNI path — Q8_K-style sub-block scales (per-16 or per-8 group scales inside the 32-block) or per-16 blocks with a repacked activation layout; falsifier: fidelity `--vnni` KL ≤ 1e-6 (two orders down) with prefill t/s within 5% of coarse VNNI; kill on either miss (then the error is the symmetric-clipping shape itself, not granularity — next: asymmetric zero-point or outlier-aware routing). To re-verify this arbitration: confirm the digit-identity argument is now moot (engaged numbers on record) and never cite the retracted floor again.

## 8. Verdict and why — plus an explicit retraction

**MEASURED: activation quantization CONFIRMED at 1.515e-03; prior floor RETRACTED.** The VNNI-accuracy program is now a quantization program: per-32 symmetric activation rounding explains ~100% of the priced error; int dots, combining, threading, outliers-as-such, bias correction, and scale application order are all exonerated or bounded out. RETRACTION, stated plainly for the library record: the error-split report's "quantization exonerated (floor 9.248e-12)" verdict rested on an unengaged run (caller-side `vnni_` gates, mechanism identified and reproduced); that verdict is withdrawn and replaced by this one. The replay round's kernel exoneration (1.7e-5) and input-side localization (0.324) STAND — they are now consistent with, and explained by, this result.

## 9. Exact commands run and their results

- `git checkout -b spike-vnni-qonly2` → branch (zero commits, deleted at close).
- 6-file spike + 1 test (§5.2 + coupling fix after the mid-spike no-engagement catch).
- `cmake --build --preset release` → 54/54; `ctest --preset release` → pass; `tge_tests.exe` → 142/142 on spike, 141/141 after revert.
- Wall 56.35; `aikit.py check` → 2/4 (test PASS, no flake; pre-existing hygiene + guidelines fails).
- Story engagement: 3 announcement lines; fidelity `--vnni-qonly` (label `, qonly`) → KL 1.515e-03, worst 1.762e-02, 992/1023 (36.18 s); thinking set fluent.
- `git checkout -- <6 files>` → clean; `git checkout master`; `git branch -D`; final master gates green; report committed; no push.

(End of file.)
