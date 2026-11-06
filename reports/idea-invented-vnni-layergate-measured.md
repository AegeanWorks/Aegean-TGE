# INVENTED: VNNI Layer-Gated Compounding Arbiter — MEASURED (intrinsic growth with depth, not compounding; late layers dominate on clean inputs)

- **Idea**: Invented follow-up, queued by the dump round (layer gate named as the arbiter). Temporary `--vnni-from-layer N` (generate + fidelity): VNNI int dots apply only at layers ≥ N (and the output projection, counted as block_count); earlier layers run the production float path on identical inputs. Falsifier (stated): late-only VNNI ≈ full 1.5e-3 → error intrinsic to deep activations (move to scale-application/bias empirical variants on deep-layer captures); late-only ≈ floor → error compounds through the residual stream (move to perturbation-injection or early-layer clamping).
- **Cycle date**: 2026-11-06
- **Verdict**: MEASURED — intrinsic growth, NOT compounding. N=25 (all float): 9.248e-12 digit-identical floor (gate verified); N=18 (layers 18–23 + output): 1.027e-03 (68% of full); N=12 (layers 12–23 + output): 1.200e-03 (80%); full VNNI: 1.503e-03; output-only (bisection round): 7.515e-04. Per-layer marginals on clean inputs grow monotonically with depth (0–11: ~0.025/layer; 12–17: ~0.029; 18–23: ~0.046; output: 0.75 alone) — late layers dominate on CLEAN inputs, so early-error amplification cannot be the main story; the error is intrinsic to deep-layer activations and roughly additive across layers (parts sum ≈ whole: 0.30+0.17+0.28+0.75 ≈ 1.50). Spike fully reverted (4 files + test); nothing pushed (human gate).
- **Tree state at close**: `git status` clean (no diff, no untracked), stash unchanged (only the pre-existing `leftover peer work shelved 2026-11-04`, untouched). Spike branch created (zero commits) and deleted; all spike files reverted with empty diff verified. `master` ahead of `origin/master` by 69 before this report. Nothing pushed (human gate).

## 1. Mechanism (as spiked)

`maybe_quantize_activations(input, tokens, cols, layer)` replaces all 15 bare `vnni_ ? quantize...` call sites (forward_batch/forward/prefill_run/output, each with its own loop variable; output projection passes `block_count`); returns empty below the threshold so the existing null-buffer fallthrough selects float dots — no dispatch changes, no kernel changes. `--vnni-from-layer N` on generate + fidelity via trailing defaulted params. Default -1 (every layer) = current behavior, pinned by a constexpr unit test.

## 2. Predicted gain — diagnostic, not product

Value: separate compounding (early errors amplified → early layers dominate; fix = clamp/perturb early) from intrinsic growth (late layers dominate on clean inputs; fix = characterize deep-activation statistics). Delivered the latter, decisively.

## 3. Falsifier outcome — intrinsic growth wins; compounding loses

Late-only (N=18) reproduces 68% of the full error with only the last quarter VNNI on clean early layers; extending to N=12 adds 0.173e-3; the first 12 layers add ~0.30e-3. Had compounding dominated, early layers (whose errors traverse the most downstream layers) would dominate — they contribute the least per layer. The parts (≈0.30 + 0.17 + 0.28 + 0.75) sum to the whole (≈1.50) within KL-nonadditivity slop, i.e. errors add roughly linearly without multiplicative amplification — consistent with the dump's smooth depth growth AND the bisection's smooth shape ladder. The remaining question is no longer WHERE (answered: everywhere, growing with depth) but WHAT about deep activations defeats per-32 symmetric int dots while sparing float dots on identical rounded inputs.

## 4. File anchors (master = pre-spike; spike reverted)

- `src/models/qwen2.cpp` (15 quantize call sites across `forward_batch`/`forward`/`prefill_run`/output-projection + verifier path — the gated set; `quantize_activations` itself untouched).
- `include/tge/models/qwen2.hpp` (`vnni_from_layer_` + `kVnniFromLayerDefault` + setter + `maybe_quantize_activations` decl — reverted).
- `apps/cli/main.cpp` (`--vnni-from-layer` on generate + fidelity, trailing params — reverted).
- `tests/test_attention.cpp` (`vnni_from_layer_defaults_to_all` — green, reverted).
- Grounds: bisection ladder (distributed + k/v amplification), dump ranking (depth growth, FFN-led), double-accum kill, error-split (quant exonerated, confinement).

## 5. What was done (temporary spike, fully reverted)

1. `git checkout -b spike-vnni-layergate` (zero commits ever made on it; deleted at close).
2. Spike (4 files): member + constexpr + setter + helper decl/impl; 15 call-site gates (each with its own loop variable verified by scope audit — one misattribution caught at build time: the `forward_batch` output site sits outside the layer loop and takes `block_count`, fixed before measuring); CLI flag ×2 + params + setters; default test.
3. Scope audit: every gated site verified against its enclosing layer loop (walk-back script + targeted reads); the two output sites (logits + verifier path) correctly take `block_count`.
4. `cmake --build --preset release` clean (54/54, after the one scope fix); `ctest --preset release` pass; `tge_tests.exe` 142/142 (141 + 1 new).
5. Box health: wall 54.73 bracketing the ladder start; aikit 2/4 (test + secrets pass; hygiene + guidelines fail pre-existing — no flake this round; the 7.5s-flake signature did not recur).
6. Fidelity ladder (0.5B-Q8_0, arch corpus, ctx 1024, 6 threads, gpu off): N=25 → N=18 → N=12 (§6; on-record full-VNNI 1.503e-03 as the top rung).
7. Thinking set on spike binary (default -1 → identical VNNI-everywhere behavior; non-VNNI paths untouched): story ids byte-identical to baseline; reasoning fluent.
8. Reverted all 4 files (`git checkout --`, empty diff verified), `git checkout master`, `git branch -D spike-vnni-layergate`; final gates on reverted master: build clean, `ctest` pass, `tge_tests` 141/141.

## 6. Measured numbers (wall 54.73 healthy bracketing)

| VNNI scope | KL mean | worst KL | top-1 | wall | reads as |
|---|---|---|---|---|---|
| N=25 (all float control) | 9.248e-12 | 2.418e-10 | 1023/1023 | 73.57 s | digit-identical to dense floor — gate verified |
| N=12 (layers 12–23 + out) | 1.200e-03 | 8.866e-03 | 993/1023 | 35.75 s | latter half holds 80% |
| N=18 (layers 18–23 + out) | 1.027e-03 | 8.376e-03 | 998/1023 | 38.06 s | last quarter holds 68% |
| full VNNI, on record | 1.503e-03 | 1.533e-02 | 993/1023 | 47.52 s | — |
| output-only, on record | 7.515e-04 | 5.825e-03 | 995/1023 | 35.77 s | — |

Marginals (indicative — KL is not strictly additive): output 0.75; layers 18–23 ≈ 0.28; layers 12–17 ≈ 0.17; layers 0–11 ≈ 0.30 (by remainder). Per-layer on clean inputs: 0.025 → 0.029 → 0.046 → output 0.75 — monotonic increase with depth, late-dominated. Worst-KL falls monotonically with less VNNI (1.53e-02 → 8.87e-03 → 8.38e-03 → 2.42e-10), consistent with fewer int-dot layers, not with a single bad layer (no cliff at any gate).

## 7. Full reimplementation recipe (kill-grade arbiter — rebuild to re-verify, then STOP; the statistic hunt is next)

Rebuild: §5.2 (member + constexpr + setter; helper; 15 gated call sites with per-site loop variables audited; CLI ×2; default test); gate: build 54/54, 142/142 tests, N=25 reproduces the floor digit-for-digit (gate-correctness proof), N=18/N=12 reproduce the §6 ladder within run noise (and beat the 1e-4 bar the other way — the ladder holds). NEXT SPIKE (named, not built): activation-statistic correlation — extend the dump run to log per-matmul max|activation| + max block-scale alongside maxdiff, offline-correlate divergence with outlier magnitude/scale variance; falsifier: no correlation (R² < 0.3) kills the distribution-shift theory and sends the program back to scale-application/bias empirical variants on deep-layer captures. To re-verify this arbitration: re-run the §6 ladder — late-dominance on clean inputs must reproduce, else the compounding question reopens.

## 8. Verdict and why

**MEASURED: intrinsic growth with depth, not compounding.** Late layers dominate on clean inputs; parts sum to the whole; no cliff anywhere. The VNNI-accuracy program is now a statistics problem (WHAT about deep activations defeats symmetric per-32 int dots while sparing float dots on identical rounded inputs), not an arithmetic or dataflow problem — every one of those has been exonerated on the record (quant, combining, threading, outliers, group/tail, localization, compounding). The loop's honest output is the arbitration plus the named statistic hunt.

## 9. Exact commands run and their results

- `git checkout -b spike-vnni-layergate` → branch (zero commits, deleted at close).
- 4-file spike (§5.2 + 1 test; one scope fix pre-build).
- `cmake --build --preset release` → 54/54 (after fix); `ctest --preset release` → pass; `tge_tests.exe` → 142/142 on spike, 141/141 after revert.
- Wall 54.73; `aikit.py check` → 2/4 (test PASS, no flake; pre-existing hygiene + guidelines fails).
- Fidelity N=25 → floor digit-for-digit; N=18 → 1.027e-03; N=12 → 1.200e-03 (§6 table).
- Story ids byte-identical; reasoning fluent.
- `git checkout -- <4 files>` → clean; `git checkout master`; `git branch -D`; final master gates green; report committed; no push.

(End of file.)
