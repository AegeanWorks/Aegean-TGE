# IDEA-1451: Mid-Depth Unified KV Scoring and Eviction — SPIKED AS SCORED-MIDDLE PROXY, KILLED ON THE SPEED ARM (2x attention cost for +3 flips)

- **Idea ID**: IDEA-1451 (dossier `arxiv-sweep-aegean-tge.md`: arXiv:2611.02230 §4–5 — single calibration-selected mid-depth scorer (0.4–0.75 depth) with one unified token set for all caches; mid-layer + single-set beats own-layer and per-head SnapKV +18–43pp at 5–15% budgets, no extra forward pass)
- **Cycle date**: 2026-11-06
- **Verdict**: KILLED on the stated speed falsifier, with a measured quality tiebreak that is too small to matter. The spike tests the dossier's load-bearing sub-claim — scored middle beats static stride at a matched kept budget — with a per-head key-row-L2-norm proxy (stronger per-position information than a unified mid-layer set, weaker in cross-layer amortization). Result: +3 flips quality edge (978 vs 975/1023) at ~2x prefill attention cost (+64–114% paired, 3/3 same sign across load states). A scoring pass that costs more than the attention it selects cannot ship. Spike fully reverted (4 files checked out, branch deleted); only this report committed. Nothing pushed (human gate).
- **Tree state at close**: `git status` clean (no diff, no untracked), stash unchanged (only the pre-existing `leftover peer work shelved 2026-11-04`, untouched), `master` ahead of `origin/master` by 50 before this report. Nothing pushed (human gate).

## 1. Mechanism (as dossiered, and as spiked)

Dossier: score every KV cache from a calibration-selected mid-depth layer with one unified token set, mmap-first, no retraining/CUDA. What must be true: the eviction path adopts calibration-prompt layer selection plus a unified set.

Spike (faithful to the load-bearing sub-claim, scoped to one cycle): keep sink + window intact and replace the content-blind strided middle (`build_sparse_index`: every 4th position) with the top-scoring middle positions by key-row L2 norm, keeping EXACTLY the count the stride would keep (matched budget: at span 2000, sink 4 + middle 371 + window 512 = 887 kept both arms). Same gather into thread-local scratch, same kernel over compacted rows — only the index list changes. Per-head scoring (each head scores its own key rows) is a strictly stronger quality baseline than a unified set and a strictly weaker cost baseline (no cross-layer amortization); the report prices both directions honestly in §8.

## 2. Predicted gain (with the math)

Quality arm (load-proof): fewer top-1 flips than static's 48/1023 at ctx-1024 natural text (measurement-log §50 baseline: KL 2.597e-02, worst 7.867, 975/1023) at the same kept count — same kernel traffic, better-chosen rows.

Speed arm: same kept count implies same dot/value traffic, so prefill attention phase should pair at parity (±noise); the extra cost is one norm pass over the full span per head per layer (~span x head_dim mult-adds ≈ the attention it selects — flagged before measuring as the thing that could kill it). Stated falsifier: kill if scored flips >= static flips on the §50 corpus, or scored prefill attn-phase slower than static by >5% paired, or short-span ids diverge (dense fallback must stay bit-identical below sink + window).

## 3. Falsifier (stated for this spike) — speed arm fires, 3/3 same sign

- Scored flips (45) < static flips (48): quality bar passes, marginally.
- Scored prefill attn-phase vs static, interleaved best-of-3, 2000-token prompt: +114% / +97% / +64% (R1/R2/R3) — kill bar was >5%. Consistent sign across a healthy window AND a collapsed-box round (R3 static itself fell to 118 t/s) → structural, not load.
- Short-span ids: byte-identical all three arms (dense fallback holds).

## 4. File anchors (current `master`, read before writing; spike has been reverted)

- `include/tge/models/qwen2.hpp:24-41` (`SparseConfig`: static sink + strided middle + window, content-blind by its own comment; dense exact path below sink + window; f32-only, never Scalar).
- `src/models/qwen2.cpp:1177-1200` (gather site: static `build_sparse_index`, gather kept rows into thread-local scratch, same kernel), `:143-171` (static builder), `:1216-1220` (`uses_sparse`).
- `apps/cli/main.cpp` (`--sparse/--sparse-window/--sparse-stride` on generate + fidelity; `--sparse-scored` existed only on the spike).
- `tests/test_attention.cpp` (sparse index/gather differential style the spike test followed).
- Baseline: measurement-log §50 (static sparse kept opt-in: KL 2.597e-02, 975/1023, prefill attn wins vs dense on long prompts).

## 5. What was done (temporary spike, fully reverted)

1. `git checkout -b spike-1451-scored-sparse` (no commit ever made on it — the branch pointed at the report commit throughout and was deleted).
2. Spike (4 files): `SparseConfig.scored` + `build_scored_sparse_index` decl; impl (squared key-row L2 norms, `(score desc, position asc)` rank via `nth_element` + sorted top-K where K = static middle count, sink + selected + window merged in order, null-keys/strided-degenerate inputs fail safe to the static set); gather-site ternary (scored ? scored-builder : static); `--sparse-scored` on generate + fidelity (+ print + fidelity label); new `sparse_scored_matches_budget_and_fallbacks` test (same count as static at 1024/2048/8192, sorted/unique, sink + window intact, determinism, dense fallback, null-keys safety).
3. `cmake --build --preset release` clean (54/54); `ctest` pass; `tge_tests.exe` 142/142 (141 + 1 new).
4. Box health gate: `bench mem` read 57.92 GB/s (healthy window) before timed runs.
5. Fidelity A/B (Qwen2.5-0.5B-Q8_0, `docs/architecture.md`, ctx 1024, 6 threads, `--gpu off`): static `--sparse`, then scored `--sparse-scored`.
6. Fallback identity: story short prompt ×3 arms (dense / static / scored), ids diffed.
7. Long-prefill A/B: 2000-token prompt (fox sentence ×200, 8999 chars → 2000 tokens), interleaved static/scored ×3, prefill t/s + prefill attn ms.
8. Thinking set (spike binary, dense default): reasoning prompt + haiku, fluent with ids recorded.
9. `python aikit.py check aegean-tge` → 2/4 (test + secrets pass; hygiene + guidelines fail pre-existing).
10. Reverted all 4 files (`git checkout --`), verified empty diff, `git checkout master`, `git branch -D spike-1451-scored-sparse`; final gates on reverted master: build clean, `ctest` pass, `tge_tests` 141/141.

## 6. Measured numbers

Fidelity (1023 positions, ctx 1024, Q8_0 0.5B):

| arm | KL mean | worst KL | max \|logit\| | top-1 | wall |
|---|---|---|---|---|---|
| static `--sparse` | 2.597e-02 | 7.867 | 11.67 | 975/1023 | 42.77 s |
| scored `--sparse-scored` | 2.401e-02 | 4.608 | 8.52 | 978/1023 | 63.61 s |

Static reproduces §50 exactly (2.597e-02 / 7.867 / 975/1023). Scored trims the tail (worst 7.867 → 4.608) and flips 3 fewer (−7.5% KL, +0.3pp top-1) — a real but marginal quality edge at the same kept budget.

Fallback identity (story, 4-tok prompt, 15 ids): dense vs static vs scored byte-identical (`11 1052 572 ... 13`). t/s across the three unclaimed (tiny-sample noise).

Long prefill (2000 tokens, interleaved, same window order S/C ×3):

| round | static prefill | static attn | scored prefill | scored attn | attn delta |
|---|---|---|---|---|---|
| R1 | 274.7 t/s | 1505.7 ms | 206.7 t/s | 3216.7 ms | **+114%** |
| R2 | 234.2 t/s | 1711.0 ms | 184.9 t/s | 3375.5 ms | **+97%** |
| R3 (box collapsed) | 118.1 t/s | 3304.1 ms | 102.8 t/s | 5433.6 ms | **+64%** |

Reading: matvec drifted with load (R3 static itself halved), but the attn delta keeps sign and scale in all three pairs (attn +64–114% while matvec moved +3–12% within pairs) — structural. Cause, confirmed by construction: scoring re-reads all 2000 key rows for norms then gathers the 887 kept, ~3x key-row traffic vs static's subset-only gather; the norm pass costs more than the subset attention saves. The fidelity wall times agree (63.61 vs 42.77 s, +49% on the same corpus).

Thinking set (spike binary, dense default — the shipped path): reasoning answer fluent (ids `576 12884 ... 315` recorded), haiku fluent ("Nature's gentle embrace."). Dense paths are untouched by construction (`scored` defaults false; static arm calls the identical function with identical args apart from one branch).

## 7. Full reimplementation recipe (kill-grade — rebuild only to re-verify or to test a sub-linear scorer)

1. `qwen2.hpp`: add `bool scored = false;` to `SparseConfig` + decl `build_scored_sparse_index(span, config, keys, kv_stride, head_dim)`.
2. `qwen2.cpp`: impl per §5.2 (squared norms, nth_element top-K with K = static middle count, positional tie-break, sorted merge sink + selected + window; dense fallback byte-identical to static; null/short/degenerate fail-safe to `build_sparse_index`); gather-site ternary.
3. `main.cpp`: `--sparse-scored` (sets enabled + scored) in both `run_generate` and `run_fidelity` parse loops, print suffix + fidelity label.
4. `test_attention.cpp`: `sparse_scored_matches_budget_and_fallbacks` per §5.2.
5. Gate: build 54/54, 142/142 tests, fidelity pair on the §50 corpus (expect §6 row 2 within run noise: deterministic per binary), 2000-token interleaved prefill pair (expect attn +60–115% scored), short-prompt ids identical all arms. To overturn the kill, a scorer must price its pass at sub-linear key traffic (chunk-mean amortization as in CISA/IDEA-1452, or reuse of already-computed quantities) AND clear both falsifier arms at once — the per-row norm construction measured here cannot, by traffic arithmetic.

## 8. Verdict and why

**KILLED on the stated speed falsifier.** Scored middle is a marginal quality win (+3 flips, −41% worst-KL tail) that costs ~2x attention: the scoring pass re-reads every key row the subset was meant to skip. Per-head key-norm is the strongest cheap signal available without cross-layer plumbing, so this also sets the dossier's burden of proof: the full mid-depth unified machinery (calibration + capture + same-order scoring traffic, minus cross-layer amortization) must demonstrate a quality win far larger than +3/1023 to justify ~2x attention — and the §50 static path keeps its place as the cheap opt-in until such a construction exists. Named triggers, not attempted: sub-linear scoring (chunk means, reused quantities), or a quality deficit on record where static's 48 flips actually hurt an end-to-end task.

## 9. Exact commands run and their results

- `git checkout -b spike-1451-scored-sparse` → new branch (no commits ever made on it).
- 4-file spike (§5.2) + 1 test.
- `cmake --build --preset release` → 54/54; `ctest --preset release` → pass; `tge_tests.exe` → 142/142 (new `sparse_scored_matches_budget_and_fallbacks` OK).
- `bench mem` → read 57.92 GB/s (healthy).
- Fidelity static → KL 2.597e-02, worst 7.867, 975/1023 (42.77 s, reproduces §50 exactly).
- Fidelity scored → KL 2.401e-02, worst 4.608, 978/1023 (63.61 s).
- Story short ×3 arms → ids byte-identical; reasoning + haiku fluent on spike dense default.
- 2000-token prefill S/C ×3 → attn +114/+97/+64% scored (table §6).
- `aikit.py check` → 2/4 (pre-existing hygiene + guidelines fails).
- `git checkout -- <4 files>` → empty diff; `git checkout master`; `git branch -D spike-1451-scored-sparse`.
- Final on reverted master: build clean, `ctest` pass, `tge_tests` 141/141, `git status` clean, stash unchanged (pre-existing peer entry only).
- No push.

(End of file.)
