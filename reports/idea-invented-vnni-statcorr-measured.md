# INVENTED: VNNI Activation-Statistic Correlation — MEASURED (maxdiff ~ maxact^0.6, R2 0.67; weight scales clean at 0.06; replay audit named as arbiter)

- **Idea**: Invented follow-up, queued by the layergate round (statistic hunt named as next). Temporary `--vnni-dump N` (generate only; same name/shape as the dump round): per instrumented Q8_0 VNNI matmul, log rows/cols/tokens/maxdiff-vs-float-reference PLUS max|activation input|, max activation block-scale, and max weight fp16-scale; offline log-log correlation of maxdiff against each candidate driver over a full-prefill dump (199 matmuls). Falsifier (stated): best R2 < 0.3 kills the distribution-shift theory (back to scale-application/bias empirical variants); R2 ≥ 0.5 with a clear driver names the next fix.
- **Cycle date**: 2026-11-06
- **Verdict**: MEASURED — distribution-shift CONFIRMED as the major driver (clears the 0.5 bar): log(maxdiff) vs log(maxact) r=0.818, R2=0.670, slope ≈ 0.60, n=199; max weight-scale R2=0.064 (clean); matmul order R2=0.038 (depth per se is not the driver — activation magnitude is, depth only correlates). The sublinear slope (0.6, not 1.0) plus the qonly floor plus double-accum no-move leave the mechanism open: the error scales with activation magnitude but is neither pure rounding (qonly) nor combining (double). Named next: replay audit — dump full inputs of one high-diff matmul, recompute offline float-vs-int; replay reproduces ⇒ kernel-arithmetic audit on exact bytes; replay agrees ⇒ runtime dataflow/aliasing (bias-correction path first). Spike fully reverted (3 files); nothing pushed (human gate).
- **Tree state at close**: `git status` clean (no diff, no untracked), stash unchanged (only the pre-existing `leftover peer work shelved 2026-11-04`, untouched). Spike branch created (zero commits) and deleted; all spike files reverted with empty diff verified. `master` ahead of `origin/master` by 70 before this report. Nothing pushed (human gate).

## 1. Mechanism (as spiked)

Same 4-site shape as the dump round (member + setter, Q8_0 dispatch branches in `matvec`/`batch_matvec`, `--vnni-dump N` on generate, trailing param), extended line: `maxact` (max |float input| over tokens×cols), `maxasc` (max activation block-scale), `maxwsc` (max fp16 weight scale over all row-blocks, via `half_to_float`). Reference = production float `gemv_q8`/`gemv_q8_batch` on the same float input. One indexing slip caught pre-build (output looped over cols instead of rows — fixed before measuring; the prior round's recipe explicitly guards this). Stderr to file (UTF-16 PowerShell redirect noted for the offline parser).

## 2. Predicted gain — diagnostic, not product

Value: replace the mechanism hunt with a named statistic. Delivered: maxact R2 0.67, everything else clean.

## 3. Falsifier outcome — distribution-shift confirmed (R2 0.67 ≥ 0.5 bar), weight scales and order clean

| pair (log-log unless noted) | r | R2 | n | reads as |
|---|---|---|---|---|
| maxdiff vs maxact | 0.818 | 0.670 | 199 | major driver; slope ≈ 0.60 |
| maxdiff vs max act scale | 0.818 | 0.670 | 199 | same variable as maxact (/127) |
| maxdiff vs max weight scale | 0.252 | 0.064 | 199 | weight side clean |
| maxdiff vs matmul order | 0.195 | 0.038 | 199 | depth per se is not the driver |
| maxdiff vs maxact (raw) | 0.755 | 0.569 | 199 | — |

By shape class (mean maxdiff): gate/up 4864-wide 0.050 (max 0.324) > k/v 128-wide 0.032 ≈ q/o 896-wide 0.028 ≈ down 0.028. The sublinear slope (0.6) rules out pure activation-rounding scaling (which would track maxact ~linearly) — consistent with the qonly floor — and the weight-scale cleanliness rules out weight-side distribution effects.

## 4. File anchors (master = pre-spike; spike reverted)

- `src/models/qwen2.cpp` Q8_0 dispatch branches (the two instrumented sites, reverted) + `quantize_block` per-32 symmetric quant (the scale law behind maxasc≡maxact/127).
- `include/tge/models/qwen2.hpp` (`vnni_dump_remaining_` + setter — reverted).
- `apps/cli/main.cpp` (`--vnni-dump N`, generate only — reverted).
- Offline: `vnni_corr.py` recipe (UTF-16 read, whitespace-joined regex, log-log R² + slope) — §7.
- Grounds: dump ranking (depth growth, FFN-led), bisection (distributed), double-accum kill, qonly exoneration, layergate (intrinsic growth).

## 5. What was done (temporary spike, fully reverted)

1. `git checkout -b spike-vnni-statdump` (zero commits ever made on it; deleted at close).
2. Spike (3 files, no new unit test — the dump lines are the instrument; default path verified by story ids): member + setter; two branch instrumentations (reference + 3 stats + stderr line); CLI flag + trailing param + setter call.
3. `cmake --build --preset release` clean (54/54); `ctest --preset release` pass; `tge_tests.exe` 141/141 (no new tests this round — the gates pin the default path, the dump is the measurement).
4. Box health: not separately re-gated (same window class as the bisection round; story decode nominal).
5. `python aikit.py check aegean-tge` 2/4 (test PASS; hygiene + guidelines fail pre-existing).
6. Dump run (0.5B-Q8_0, fox×7 = 70-token prompt, `--n 6`, 6 threads, gpu off, `--vnni --vnni-dump 200`, stderr to temp file): 199 usable matmul lines (168 prefill + 31 decode singles); prompt ids recorded (19-token periodic fox prompt).
7. Offline correlation (§3 table + slope 0.60).
8. Thinking set on spike binary (default paths — dump gated off): story ids byte-identical to baseline; reasoning fluent.
9. Reverted all 3 files (`git checkout --`, empty diff verified), `git checkout master`, `git branch -D spike-vnni-statdump`; final gates on reverted master: build clean, `ctest` pass, `tge_tests` 141/141.

## 6. Measured numbers (dump window: 199 matmuls, 70-token prompt + 6 decode tokens)

Correlation table in §3 (R²: maxact 0.670, weight-scale 0.064, order 0.038; slope 0.60; shape means: gate/up 0.050, k/v 0.032, q/o/down ~0.028). Dump-run story ids: `576 3974 13876 38835 34208 916 …` (repetitive-fox prompt, 19 prompt tokens as tokenized). Decode singles included in the 199 (matvec path instrumented too) — no shape class shows zero diff, consistent with the distributed verdict.

## 7. Full reimplementation recipe (kill-grade instrument — rebuild to re-verify, then STOP; replay audit is next)

Rebuild: §5.2 (member + setter; two branch instrumentations with reference + maxact/maxasc/maxwsc + stderr line — mind the rows-vs-cols output indexing; CLI flag + param + setter); gate: build 54/54, 141/141 tests, story ids identical (dump must not move numerics), dump run reproduces the §3 ranking (R² maxact ≥ 0.5, weight-scale < 0.2). NEXT SPIKE (named, not built): replay audit — extend the dump to emit full inputs (int8 activations + scales + weight blocks + float reference out) for the single worst matmul to a file; offline recompute float-vs-int on the exact bytes; falsifier: replay reproduces (≥50% of the dumped maxdiff) ⇒ kernel-arithmetic audit on exact bytes (bias-correction variants first: the one untested path); replay agrees (≤1e-5) ⇒ runtime dataflow/aliasing audit (buffer lifetime under the thread pool, despite the blocking contract) — then scale-application order. To re-verify this correlation: re-run dump + offline script; maxact R² ≥ 0.5 must reproduce or the distribution-shift theory reopens.

## 8. Verdict and why

**MEASURED: distribution-shift confirmed as the major driver (R² 0.67, slope 0.6), weight scales and order clean.** The VNNI-accuracy program now reads: error ~ maxact^0.6 per matmul, FFN-led, no cliff, no compounding, no threading, no outliers-as-such, no combining, no activation-rounding — with the replay audit named as the mechanism arbiter. The loop's honest output is the correlation plus the recipe; no future cycle re-runs blind kernel variants without replay bytes in hand.

## 9. Exact commands run and their results

- `git checkout -b spike-vnni-statdump` → branch (zero commits, deleted at close).
- 3-file spike (§5.2; one indexing slip fixed pre-build).
- `cmake --build --preset release` → 54/54; `ctest --preset release` → pass; `tge_tests.exe` → 141/141 (no new tests; gates pin defaults).
- `aikit.py check` → 2/4 (test PASS; pre-existing hygiene + guidelines fails).
- Dump run `--vnni --vnni-dump 200` → 199 usable lines (stderr to temp file); offline correlation (§3 + slope 0.60).
- Story ids byte-identical; reasoning fluent.
- `git checkout -- <3 files>` → clean; `git checkout master`; `git branch -D`; final master gates green; report committed; no push.

(End of file.)
