# INVENTED: VNNI Per-Matmul Dump Localization — MEASURED (depth-growing divergence, FFN-led; threading/outlier/group-tail dead; layer-gated VNNI named next)

- **Idea**: Invented follow-up, queued by the bisection round ("per-matmul dump harness"). Temporary `--vnni-dump N` (generate only): instruments the first N Q8_0 VNNI matmuls with an inline float-path reference (`gemv_q8`/`gemv_q8_batch` on the same float input) and prints `rows/cols/tokens/maxdiff` to stderr in deterministic call order. Falsifier (stated): one shape class diverging ≥10x the rest names the locus (next: audit its banding/indexing); all-small (≈1e-6, rounding-class) means the 1.5e-3 is composition/amplification across layers, not per-matmul error (next: compounding analysis); all-large-and-flat means uniform systematic error (next: scale-application order + bias-correction empirical audit).
- **Cycle date**: 2026-11-06
- **Verdict**: MEASURED — outcome (d): depth-growing divergence, FFN-led, no cliff by shape. Layer-0 per-matmul maxdiff ~0.01 (q 0.014, k 0.0138, gate 0.0148, up 0.0129, down 0.0087; o/v ~0.0007); layer-3 peaks (up 0.324, gate 0.162, q 0.079, down 0.051); all seven shapes participate every layer. Threading dead (single-threaded rerun digit-identical: 0.01408, 0.0137905, ...). Outlier trigger dead (synthetic one-huge-outlier-per-block: worst 3.05e-05 vs scalar on rounded inputs). Group/tail dead (covered identically in the differential suite + both paths active in every dumped run). Remaining, narrowed: data-dependent divergence that grows with activation statistics through depth — composited error, amplification, or an indexing/scale interaction invisible to uniform/outlier synthetic data. Next spike named in §8: layer-gated VNNI (zero-signature variant — gate the `quantize_activations` call sites on layer index) to separate compounding from intrinsic depth effects. Spike fully reverted (4 files + tests); nothing pushed (human gate).
- **Tree state at close**: `git status` clean (no diff, no untracked), stash unchanged (only the pre-existing `leftover peer work shelved 2026-11-04`, untouched). Spike branch created (zero commits) and deleted; all spike files reverted with empty diff verified. `master` ahead of `origin/master` by 68 before this report. Nothing pushed (human gate).

## 1. Mechanism (as spiked)

In both Q8_0 dispatch sites (`matvec` single-token, `batch_matvec` batch): after the VNNI call, when the countdown member is positive, run the production float kernel on the same float input into a thread-local buffer, max-abs-diff vs the VNNI output, `fprintf` one line (`rows/cols/tokens/maxdiff`), decrement. `--vnni-dump N` on generate only (diagnostic scope), trailing defaulted param, setter beside `set_vnni`. No kernel changes, no math changes — pure observation.

## 2. Predicted gain — diagnostic, not product

Value: replace blind kernel variants with a ranked suspect list. Delivered a ranking plus three more deaths (threading, outliers, group/tail) and one crisp next step.

## 3. Falsifier outcome — none of the three scripted branches; the data chose a fourth

No shape class diverges ≥10x the rest (FFN leads by ~4-5x at peak, all classes grow together) — not (a). Not all-small — per-matmul diffs reach 0.32, three orders above rounding — not (b). Not all-flat — layer-0 ~0.01 vs layer-3 ~0.3, monotonic growth — not (c). Recorded outcome (d): depth-growing, FFN-led, deterministic, thread-independent, outlier-independent divergence. The 1.5e-3 is therefore neither a localized bug nor uniform rounding — it is a data-dependent effect that strengthens through depth (candidates, in order: activation-statistic interaction with per-32 symmetric scales on drifting distributions; scale-application order under real scale spreads; an indexing interaction visible only on real shapes — the dump cannot separate these, the layer-gate can start to).

## 4. File anchors (master = pre-spike; spike reverted)

- `src/models/qwen2.cpp` Q8_0 dispatch in `matvec` + `batch_matvec` (the two instrumented sites, reverted).
- `include/tge/models/qwen2.hpp` (`vnni_dump_remaining_` mutable member + setter + `kVnniMinRowsDefault`-style... no — plain default 0; reverted).
- `apps/cli/main.cpp` (`--vnni-dump N`, generate only, trailing param; reverted).
- `tests/test_q8.cpp` (`vnni_outlier_blocks_divergence_characterization`, worst 3.05e-05; reverted with the spike).
- Grounds: bisection ladder (distributed + k/v amplification), double-accum kill, error-split (quant exonerated).

## 5. What was done (temporary spike, fully reverted)

1. `git checkout -b spike-vnni-dump` (zero commits ever made on it; deleted at close).
2. Spike (4 files + 1 test): dump branches (both dispatch sites, thread-local ref buffers, stderr lines, countdown member + setter + CLI flag + trailing param); outlier characterization test (dequant helper re-added for it); unit-test portability note: none needed this round (no MSVC-alias touched).
3. Mid-spike red herring caught and killed: a stray `kv_auto` local from the prior cycle's recipe draft appeared in one edit's context — removed before build; tree-diff verified clean of cross-cycle leakage.
4. `cmake --build --preset release` clean (54/54); `ctest --preset release` pass; `tge_tests.exe` 142/142 (141 + 1 characterization).
5. Box health: not re-measured separately this round (same window class as the bisection round; story decode nominal).
6. Dump run (0.5B-Q8_0, fox×7 = 70-token prompt, `--n 6`, 6 threads, `--vnni --vnni-dump 48`): §6 table (first 48 matmuls ≈ 7 prefill layers).
7. Falsifier follow-ups, same binary: single-threaded rerun (`--threads 1 --vnni-dump 16`) → digit-identical lines (threading/aliasing dead); outlier test → 3.05e-05 (trigger dead).
8. Thinking set on spike binary (`--vnni` off by default → production float path, identical behavior): story ids byte-identical to baseline; reasoning fluent.
9. Reverted all spike files (`git checkout --`, empty diff verified), `git checkout master`, `git branch -D spike-vnni-dump`; final gates on reverted master: build clean, `ctest` pass, `tge_tests` 141/141; `python aikit.py check aegean-tge` 2/4 (test + secrets pass; hygiene + guidelines fail pre-existing).

## 6. Measured numbers (70-token prompt, 64-token prefill chunk, 6 threads)

Per-matmul VNNI-vs-float max|diff|, layer by layer (q/k/v/o/gate/up/down order):

| layer | q(896) | k(128) | v(128) | o(896) | gate(4864) | up(4864) | down(896×4864) |
|---|---|---|---|---|---|---|---|
| 0 | 0.0141 | 0.0138 | 0.00067 | 0.00073 | 0.0148 | 0.0129 | 0.0087 |
| 1 | 0.0168 | 0.0182 | 0.0048 | 0.0085 | 0.0225 | 0.0166 | 0.0062 |
| 2 | 0.0232 | 0.0229 | 0.0059 | 0.0074 | 0.0291 | 0.0692 | 0.0740 |
| 3 | 0.0790 | 0.0690 | 0.0235 | 0.0074 | 0.1616 | 0.3241 | 0.0512 |
| 4 | 0.0426 | 0.0710 | 0.0272 | 0.0047 | 0.0426 | 0.0398 | 0.0129 |
| 5 | 0.0348 | 0.0277 | 0.0154 | 0.0070 | 0.0398 | 0.0759 | 0.0118 |
| 6 (partial) | 0.0283 | 0.0243 | … | … | … | … | … |

Single-threaded rerun, first 16 matmuls: digit-identical (0.01408, 0.0137905, 0.000674348, …). Synthetic outlier test (one 50.0 among 0.01s per block, 32×256×8): worst 3.05e-05. Decode ids for the dump run: `576 3974 13876 38835 34208 916` (6 tokens, VNNI active, fluent).

## 7. Full reimplementation recipe (diagnostic — rebuild to re-verify, then STOP; the layer gate is next)

Rebuild: §5.2 (countdown member + setter; two dump branches with thread-local ref buffers + stderr lines; `--vnni-dump N` on generate + trailing param + setter call; outlier characterization test + dequant helper). Gate: build 54/54, 142/142 tests, story dump reproduces the §6 layer-0 row within run noise and the depth growth qualitatively (magnitudes depend on prompt/box, pattern should not). Then STOP — do not extend the dump (per-block dumps, file output). NEXT SPIKE (named, not built): layer-gated VNNI — gate the `quantize_activations` call sites on layer index (`index >= vnni_from_layer`), zero signature changes (callers know the layer; `quantized==nullptr` falls through to float); falsifier: VNNI-from-late-layers-only fidelity vs full-VNNI — if late-only ≈ full 1.5e-3, the error is intrinsic to deep activations (not compounded history) and the program moves to scale-application/bias empirical variants on deep-layer captures; if late-only ≈ floor, the error compounds through the residual stream and the program moves to perturbation-injection (prove amplification) or early-layer clamping.

## 8. Verdict and why

**MEASURED: depth-growing, FFN-led, deterministic divergence — three more hypotheses dead (threading, outliers, group/tail), compounding-vs-intrinsic named as the arbiter.** The VNNI-accuracy program now fits on one line: everything arithmetic is exonerated (quant, combining, correction-by-proof, threading, outliers); what remains is WHERE in the network the error is born vs amplified, and the layer gate answers exactly that. No future cycle should touch the VNNI kernel again until the gate speaks.

## 9. Exact commands run and their results

- `git checkout -b spike-vnni-dump` → branch (zero commits, deleted at close).
- 4-file spike (§5.2) + 1 test (+1 stray-line removal pre-build).
- `cmake --build --preset release` → 54/54; `ctest --preset release` → pass; `tge_tests.exe` → 142/142 on spike, 141/141 after revert.
- Dump run `--vnni --vnni-dump 48` (§6 table); single-threaded rerun → digit-identical; outlier test → 3.05e-05.
- `python aikit.py check aegean-tge` → 2/4 (test PASS; pre-existing hygiene + guidelines fails).
- Story ids byte-identical; reasoning fluent.
- `git checkout -- <4 files + test>` → clean; `git checkout master`; `git branch -D`; final master gates green; report committed; no push.

(End of file.)
