# INVENTED: VNNI Shape-Bisection Localization — MEASURED (distributed, not localized: smooth ladder, no cliff; k/v amplified, FFN diluted)

- **Idea**: Invented follow-up, queued by the double-accum kill (localization named next). Temporary `--vnni-min-rows N` gate (VNNI int dots only for weight matrices with ≥N rows, else the production float path) bisects the 1.5e-3 error by matmul shape class on 0.5B-Q8_0 (all-Q8_0, so VNNI engages everywhere by default): T=0 (all), T=200 (all but 128-row k/v), T=1000 (4864-row gate/up + 151936-row output only), T=100000 (output only). Falsifier (stated): a shape class that restores floor names the next audit target (banding math for its row count); a smooth ladder with no cliff means distributed error and sends the program to per-matmul dump localization.
- **Cycle date**: 2026-11-06
- **Verdict**: MEASURED — smooth ladder, NO CLIFF. T=0: 1.503e-03 (993/1023); T=200: 1.200e-03 (996/1023); T=1000: 9.097e-04 (998/1023); T=100000: 7.515e-04 (995/1023). No single matmul class carries the error (output-only still holds half the KL; every excluded class removes only its share) — a localized indexing/banding bug is ruled out; the error is systematic per-FLOP with downstream sensitivity: k/v rows run ~10x the average KL/row (attention softmax amplification), gate/up ~7x below (residual-stream dilution), output ≈ average (direct to logits). Spike fully reverted (4 files + test); nothing pushed (human gate).
- **Tree state at close**: `git status` clean (no diff, no untracked), stash unchanged (only the pre-existing `leftover peer work shelved 2026-11-04`, untouched). Spike branch created (zero commits) and deleted; all spike files reverted with empty diff verified. `master` ahead of `origin/master` by 67 before this report. Nothing pushed (human gate).

## 1. Mechanism (as spiked)

`Qwen2::vnni_min_rows_` (default 0 = all VNNI, identical behavior) + setter, checked in both Q8_0 dispatch sites (`matvec` single-token, `batch_matvec` batch) as `weight.rows >= vnni_min_rows_`; `--vnni-min-rows N` on generate + fidelity via trailing defaulted params (no signature breaks). Quantization still runs ungated (wasted work when gated off — noted, harmless, same as VNNI-on-Q4_K_M today). Row classes on 0.5B-Q8_0: k/v 128, q/o/down 896, gate/up 4864, output 151936.

## 2. Predicted gain — diagnostic, not product

No t/s claimed (the gate ADDS a comparison; VNNI already ships). Value: discriminate localized-bug (cliff at some threshold → audit that shape's banding/indexing) from distributed-precision-deficit (smooth ladder → dump harness). Delivered the latter.

## 3. Falsifier outcome — smooth ladder, no cliff; distributed error with downstream sensitivity

Marginal KL shares (indicative — KL is not strictly additive, stated as approximation): k/v ≈ 0.30e-3 (20%), q/o/down ≈ 0.29e-3 (19%), gate/up ≈ 0.16e-3 (11%), output ≈ 0.75e-3 (50%). Per-row rates: k/v 4.9e-8/row (~10x average — softmax amplification of K/V perturbations), q/o/down/output 4.5–4.9e-9/row (average — direct or post-attention paths), gate/up 6.8e-10/row (~7x below — residual dilution). A banding/indexing bug would show a cliff (one class restoring floor); instead every excluded class removes only its share — the bug theory for shape-dependent paths is dead alongside the precision theories (quant exonerated: qonly floor; combining exonerated: double no-move).

## 4. File anchors (master = pre-spike; spike reverted)

- `src/models/qwen2.cpp` Q8_0 dispatch sites (`matvec` + `batch_matvec` — the two gated branches, reverted) + `quantize_activations` ungated callers (the noted wasted-work path).
- `include/tge/models/qwen2.hpp` (`vnni_` member area — the spike's `vnni_min_rows_` + `kVnniMinRowsDefault` + setter/getter lived here, reverted).
- `apps/cli/main.cpp` (`--vnni-min-rows` on generate + fidelity, trailing defaulted params — reverted).
- `tests/test_attention.cpp` (`vnni_min_rows_defaults_to_zero` — green on spike, reverted with it).
- Grounds: VNNI error-split report (quant exonerated, engagement confined) + double-accum kill (combining exonerated).

## 5. What was done (temporary spike, fully reverted)

1. `git checkout -b spike-vnni-bisect` (zero commits ever made on it; deleted at close).
2. Spike (4 files): member + constexpr default + setter/getter; 2 rows-gate conditions; 2 CLI flags + 2 trailing params + 2 setter calls; 1 default-value test.
3. `cmake --build --preset release` clean (54/54); `ctest --preset release` pass; `tge_tests.exe` 142/142 (141 + 1 new).
4. Box health gate: wall 59.33 + aikit 2/4 (test PASS, no flake; hygiene + guidelines fail pre-existing).
5. Fidelity ×4 (0.5B-Q8_0, arch corpus, ctx 1024, 6 threads, gpu off, `--vnni --vnni-min-rows {0,200,1000,100000}`): §6.
6. Shared-scratch race hypothesis checked and killed on the record: pool contract is blocking `parallel_for` (header doc), so the next quantize cannot clobber in-flight VNNI reads — no measurement needed, contract suffices.
7. Thinking set on spike binary (gate default 0 → VNNI everywhere as usual; non-VNNI paths byte-identical by construction): story ids identical to baseline, reasoning fluent.
8. Reverted all 4 files (`git checkout --`, empty diff verified), `git checkout master`, `git branch -D spike-vnni-bisect`; final gates on reverted master: build clean, `ctest` pass, `tge_tests` 141/141.

## 6. Measured numbers (wall 59.33 healthy bracketing)

| VNNI scope (min-rows) | KL mean | worst KL | top-1 | wall | reads as |
|---|---|---|---|---|---|
| 0 (all) | 1.503e-03 | 1.533e-02 | 993/1023 | 35.36 s | reproduces the priced error exactly |
| 200 (all but k/v) | 1.200e-03 | 1.864e-02 | 996/1023 | 35.43 s | k/v carry ~20% |
| 1000 (gate/up/output) | 9.097e-04 | 3.682e-03 | 998/1023 | 35.32 s | q/o/down carry ~19% more |
| 100000 (output only) | 7.515e-04 | 5.825e-03 | 995/1023 | 35.77 s | output alone holds ~half; rest is upstream |

Worst-KL does not monotonic-track the mean (T=200 worst 1.86e-02 exceeds all-VNNI 1.53e-02 — tail behavior on few flips, noted not chased). Top-1 climbs 993 → 996 → 998 then 995 (output-only flips 28 vs gate/up/output's 25 — same class, noise-scale on 1023).

## 7. Full reimplementation recipe (kill-grade bisection — rebuild to re-verify, then STOP; the dump harness is next)

Rebuild: §5.2 (member + constexpr + setter/getter; 2 gate conditions; CLI flags + params + setter calls; default test); gate: build 54/54, 142/142 tests, fidelity 4-arm reproduces the §6 ladder within run noise (means ±10%, same order). Do NOT extend the bisection (finer thresholds subdivide the same smooth curve — diminishing returns proven by monotonicity). NEXT SPIKE (named, not built): per-matmul dump harness — dump inputs+outputs of each VNNI matmul on a fixed prompt (first 64 positions suffice), recompute in float offline, report the first diverging (layer, matrix) with magnitudes; falsifier: isolated float substitution for the named matmul must recover ≥90% of the 1.5e-3 gap, else the harness itself is wrong and the program reverts to scale-application-order audit.

## 8. Verdict and why

**MEASURED: distributed, not localized.** Three exonerations now stand (activation quant: floor; float combining: no-move; localized bug: no cliff), plus one confinement (Q8_0 weights only) and one true floor (kq8 on K-quants, full corpus). The remaining space is narrow and named: bias-correction empirical variant, fp16 scale-application order, or a data-dependent divergence the dump harness will name. The loop's honest output is the map plus the ladder — no future cycle re-bisects shapes.

## 9. Exact commands run and their results

- `git checkout -b spike-vnni-bisect` → branch (zero commits, deleted at close).
- 4-file spike (§5.2) + 1 test.
- `cmake --build --preset release` → 54/54; `ctest --preset release` → pass; `tge_tests.exe` → 142/142 on spike, 141/141 after revert.
- Wall 59.33; `aikit.py check` → 2/4 (test PASS, no flake; pre-existing hygiene + guidelines fails).
- Fidelity 4-arm (§6 table; ~35 s each).
- Story ids identical to baseline; reasoning fluent (default paths).
- `git checkout -- <4 files>` → clean; `git checkout master`; `git branch -D`; final master gates green; report committed; no push.

(End of file.)
