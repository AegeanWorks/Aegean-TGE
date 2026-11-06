# INVENTED: VNNI Replay Audit — MEASURED (kernel arithmetic exonerated to 1.7e-5; divergence sits in the inputs; prior qonly floor flagged SUSPECT, re-run mandated)

- **Idea**: Invented arbiter, queued by the statcorr round: dump the single worst-diverging Q8_0 VNNI matmul's exact bytes (quantized activations + scales, weight blocks, both outputs) via temporary `--vnni-replay PATH` (generate only; keeps the running max, overwrites on new worst), then recompute offline three ways — float dots on dequantized bytes, exact integer emulation (dpbusd + bias-correction semantics in int64, double combining), and cross-checks against both binary outputs. Falsifier (stated): replay reproduces the dump gap (≥50%) → kernel-arithmetic audit with bias-correction first; replay agrees (≤1e-5) → runtime dataflow/aliasing audit.
- **Cycle date**: 2026-11-06
- **Verdict**: MEASURED — and the arbiter splits the case three ways. (1) Kernel arithmetic EXONERATED: offline exact-int emulation vs binary VNNI = 1.66e-05 (no indexing bug, no bias-correction bug, no saturation — the engine implements exactly the specified integer arithmetic). (2) Divergence sits in the INPUTS: offline float on dequantized bytes vs binary float on original floats = 0.324142, reproducing the dump's binary-vs-binary 0.324137 to 5 digits (residual = float64-vs-float32 accumulation, as expected) — i.e., per-32 symmetric activation rounding shifts this matmul's outputs by up to 0.32. (3) Offline int vs offline float on identical bytes: 4.3e-14 — int and float dots agree to machine epsilon, so no dot arithmetic of any kind explains the gap. CONSEQUENCE: the prior cycle's qonly-at-floor result (dequantized-inputs float dots == dense floor digit-for-digit) is now SUSPECT — it contradicts (2) directly (same rounded inputs must show the same ~0.3 shifts), and digit-identity to 4 sig figs across 1023 positions is essentially impossible for two genuinely different computations. The likeliest account is silent non-engagement of the qonly flag in that run (no engagement marker was printed). MANDATED next cycle: qonly re-run WITH engagement proof (stderr counter/label), falsifier — engaged-qonly ≈ 1.5e-3 class confirms activation quant as the error (spike finer scales next); engaged-qonly ≈ floor reinstates the exoneration and reopens the paradox honestly. Spike fully reverted (3 files); nothing pushed (human gate).
- **Tree state at close**: `git status` clean (no diff, no untracked), stash unchanged (only the pre-existing `leftover peer work shelved 2026-11-04`, untouched). Spike branch created (zero commits) and deleted; all spike files reverted with empty diff verified. `master` ahead of `origin/master` by 70 before this report. Nothing pushed (human gate).

## 1. Mechanism (as spiked)

`--vnni-replay PATH` (generate only): in both Q8_0 dispatch branches, after the VNNI call + inline float reference, track max|diff|; on a new running max, write the VNIR v1 record (magic `0x564E4952`, dims, int8 acts, fp32 act-scales, raw Q8_0 weight bytes, VNNI out, float ref) and print one stderr line. Worst matmul of the run wins the file. Offline (`vnni_replay.py`, numpy): parse, fp16→double conversion audited against the 0x2C00=0.0625 known value, three recomputes + cross-comparisons (§6).

## 2. Predicted gain — diagnostic, and it delivered the arbiter's verdict

Value was a decision procedure, not a speedup: it fired on the second branch (inputs, not arithmetic) with a bonus finding (the qonly contradiction). Two offline-script bugs were caught and fixed mid-cycle by their own absurd outputs (fp16 exponent dropped → 6267-scale garbage; flat biased array → reshape error) — the script distrusts itself; every number below survived a fix-and-rerun.

## 3. Falsifier outcome — neither scripted branch; the evidence forces a third reading

Replay reproduces the 0.324 gap (0.324142 vs 0.324137) — but the reproduction sits in the INPUT comparison (rounded vs original), not in any dot arithmetic (all dot-vs-dot comparisons ≤ 1.7e-5). So neither "audit the bias correction" nor "audit the dataflow" is the right next step as stated: the bias correction is verified exact by construction-plus-emulation, and the dataflow is verified by determinism (single-threaded dump runs digit-identical, §5 of the dump round). The right next step is the qonly re-run with engagement proof — the one result the whole exoneration chain rests on and the one result whose digit-identity now reads as non-engagement, not cleanliness.

## 4. File anchors (master = pre-spike; spike reverted)

- `src/models/qwen2.cpp` (both Q8_0 dispatch branches — the instrumented sites, reverted) + file-local `write_vnni_replay` + `kVniReplayMagic/Version` (reverted).
- `include/tge/models/qwen2.hpp` (`vnni_replay_path_` + `vnni_replay_best_` + setter — reverted).
- `apps/cli/main.cpp` (`--vnni-replay PATH`, generate only, trailing defaulted param + setter — reverted).
- Offline: `vnni_replay.py` (parse → fp16 audit → (a)(b)(c)(d) cross-comparisons → top-rows; recipe in §7).
- Grounds: all six prior VNNI rounds (quant/dots/bisect/dump/layergate/statcorr); the 1394 price being priced.

## 5. What was done (temporary spike, fully reverted)

1. `git checkout -b spike-vnni-replay` (zero commits ever made on it; deleted at close).
2. Spike (3 files): member + setter; file-local writer (one indexing slip caught pre-build: scales count is tokens×blocks, not tokens×(blocks/2)); two branch instrumentations (VNNI + float ref + maxdiff + conditional write + stderr line); CLI flag + param + setter.
3. `cmake --build --preset release` clean (54/54); `ctest --preset release` pass; `tge_tests.exe` 141/141 (no new unit tests — the offline cross-checks are the verification; gates pin the default paths).
4. Box health: not separately re-gated (same window class as the statcorr round; story decode nominal).
5. Replay run (0.5B-Q8_0, fox×7 = 70-token prompt, `--n 6`, 6 threads, gpu off, `--vnni --vnni-replay <temp>`): worst matmul gate/up-class 4864×896 (maxdiff 0.324137, final of 12 improving records), 7.2 MB record.
6. Offline analysis (§6; two self-caught script bugs fixed by rerun).
7. Thinking set on spike binary (default paths — replay gated off): story ids byte-identical to baseline; reasoning fluent.
8. Reverted all 3 files (`git checkout --`, empty diff verified), `git checkout master`, `git branch -D spike-vnni-replay`; final gates on reverted master: build clean, `ctest` pass, `tge_tests` 141/141; `python aikit.py check aegean-tge` 2/4 (test PASS; hygiene + guidelines fail pre-existing).

## 6. Measured numbers (replay file: 4864 rows × 896 cols × 64 tokens, 7,185,428 bytes, zero trailing)

| comparison | max\|diff\| | reads as |
|---|---|---|
| (a) offline-float (dequantized bytes) vs file float-ref (original floats) | 0.324142 | activation-rounding effect, reproduced |
| (b) offline-int (exact dpbusd+bias, int64) vs binary VNNI | 1.66e-05 | kernel implements the spec — no bug to find here |
| (c) offline-int vs offline-float (identical bytes) | 4.3e-14 | int vs float dots agree to machine epsilon |
| (d) binary VNNI vs file float-ref (dump's own number) | 0.324137 | (a) reproduces it to 5 digits — same gap |
| top divergent rows (int-vs-float) | ≤ 4.3e-14 everywhere | no row stands out — uniform, not localized |

Worst single output: row 3573, token 0: VNNI −17.29465 vs ref −16.97052 (2% relative on a large logit; output magnitude max 59.96, mean 1.58 — the gap is NOT float rounding, which would sit at ~1e-5 here).

## 7. Full reimplementation recipe (arbiter — rebuild to re-verify, then STOP; qonly re-run is next)

Rebuild: §5.2 (writer + member + setter + 2 branch sites + CLI); gate: build 54/54, 141/141 tests, replay run reproduces the §6 table within run noise (worst matmul FFN-class, maxdiff same order; cross-comparisons (b) ≤ 1e-4 and (c) ≤ 1e-12 must hold exactly — they are arithmetic identities). Then STOP extending the dump (per-matmul files, scale audits — all moot until qonly is resolved). NEXT SPIKE (mandated, not optional): `--vnni-qonly` re-run WITH engagement proof (stderr counter of qonly-path executions + candidate label suffix), same corpus/model/ctx; falsifier: engaged-qonly ≈ 1.5e-3 class (KL and flips) CONFIRMS activation quant as the error — next: finer activation scales (Q8_K-style sub-block or per-16) with the 1e-9 bar; engaged-qonly ≈ floor REINSTATES the exoneration and reopens the paradox honestly (then: bias-correction-off variant as the only untested arithmetic).

## 8. Verdict and why — including a self-correction

**MEASURED: kernel arithmetic exonerated (1.7e-5), inputs convicted (0.324), prior qonly result flagged SUSPECT.** This cycle both advances the program (the int-dot question is closed — stop auditing arithmetic) and corrects it (the quant exoneration rested on a run whose digit-identity now reads as non-engagement; that claim is SUSPENDED, not cited, until the engagement-proven re-run lands). The loop's honest output includes the correction: a library that cannot retract a load-bearing measurement is a ledger, not science. No future cycle may cite "qonly at floor" without the re-run.

## 9. Exact commands run and their results

- `git checkout -b spike-vnni-replay` → branch (zero commits, deleted at close).
- 3-file spike (§5.2; one indexing slip fixed pre-build).
- `cmake --build --preset release` → 54/54; `ctest --preset release` → pass; `tge_tests.exe` → 141/141 (no new tests; offline cross-checks verify).
- `aikit.py check` → 2/4 (test PASS; pre-existing hygiene + guidelines fails).
- Replay run `--vnni --vnni-replay` (§6 record, 7.2 MB); offline analysis (§6 table; two self-caught script bugs fixed by rerun).
- Story ids byte-identical; reasoning fluent.
- `git checkout -- <3 files>` → clean; `git checkout master`; `git branch -D`; final master gates green; report committed; no push.

(End of file.)
