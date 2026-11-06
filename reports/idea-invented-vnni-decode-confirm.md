# INVENTED (1394 hole): VNNI Decode Trajectory Stability — CONFIRM (0 flips on 56 ids, fluent, no crash; decode under the flag pays ~5-13%)

- **Idea**: Close the hole the 1394 correction flags explicitly ("decode ids under --vnni unmeasured"): single-token `forward()` quantizes activations and runs the VNNI int dots whenever `--vnni` is set, the per-position price is measured (KL 1.5e-3, 993/1023), but no free-running decode trajectory under the flag was ever checked. Run the fixed thinking set under `--vnni` on the priced config (0.5B-Q8_0) and compare trajectories vs the default path.
- **Cycle date**: 2026-11-06
- **Verdict**: CONFIRM — the priced gate extends to decode trajectories. Story 32/32 + creative 24/24 ids byte-identical under `--vnni` (0 flips, inside the ~3% prefill price class with margin), both texts fluent, both runs clean exits. No spike (flag-gated measurement, precedented); no branch, no edits, nothing to revert.
- **Tree state at close**: `git status` clean, no diff, stash unchanged (only the pre-existing `leftover peer work shelved 2026-11-04`). `master` ahead of `origin/master` by 99 after this report. Nothing pushed (human gate).

## 1. Mechanism (as run — no code)

`generate --vnni` on Qwen2.5-0.5B-Q8_0, threads 6, gpu off: story (fox, n=32) + creative (haiku, n=24), `--print-ids`; same prompts without the flag; diff ids, count flips, eyeball fluency.

## 2. Predicted gain — stability, landed better than the price class

Expected flips within the prefill price (~3% → 0-2 on 56 ids). Measured: **zero** on both prompts. Greedy argmax absorbs the 1.5e-3-class perturbation without flipping on these shapes.

## 3. Falsifier outcome — CONFIRM on all arms

1. **Flips**: 0/32 story, 0/24 creative (bar: ≤ prefill-class rate; no crash, no hang).
2. **Fluency**: both VNNI texts read normally (fox explanation; haiku-form circuits text).
3. **Side number (new)**: VNNI decode is SLOWER than float decode here — story 68.29 vs 78.12 t/s (−12.6%), creative 75.33 vs 79.03 (−4.7%). Same lesson as the kq8d 0.8x: at tokens==1 the activation quantizer costs more than the integer dots save. This is why `--vnni` stays a prefill/batch flag in practice; decode under the flag is priced-stable but not faster.

## 4. File anchors (current `master`, read but untouched)

- `src/models/qwen2.cpp:1683-1742` (single-token `forward()`: `quantize_activations` + VNNI matvecs whenever `vnni_` — the engaged path).
- `apps/cli/main.cpp` (`--vnni` flag; timers/sampler paths unchanged by it).
- Grounds: `idea-1394-quantize-once-integrated.md` (the correction flagging this hole; per-position price 1.5e-3, 993/1023) + `idea-1582-rust-intdot-split.md` (VNNI 1.9x prefill, 1.6e-3 priced).

## 5. What was done (measurement only — no spike per the flag-gated precedent)

1. ANALYZE: confirmed the single-token VNNI engagement on today's tree (re-read, not cited from memory).
2. Gates on the unmodified tree: `cmake --build --preset release` 54/54; `tge_tests.exe` 141/141; `aikit.py check` 2/4 (test + secrets pass; hygiene + guidelines fail pre-existing, same pair as every prior cycle).
3. Story + creative under `--vnni` (saved) and without (saved): ids diffed, flips counted, fluency read; timings recorded.
4. No fidelity run owed (no numerics changed — flag-gated invocation of priced kernels); no bench A/B owed (no second implementation exists); thinking set IS the measurement (story + creative diffed).
5. Scratch outputs deleted; no branch, no edits; `git status` clean, stash untouched.

## 6. Measured numbers (0.5B-Q8_0, 6 threads, gpu off, warm page cache)

| prompt | --vnni ids | plain ids | flips | --vnni t/s | plain t/s |
|---|---|---|---|---|---|
| story, n=32 | 576 3974 … 1854 13 | identical | 0/32 | 68.29 | 78.12 (−12.6%) |
| creative, n=24 | 758 264 … 23607 11 | identical | 0/24 | 75.33 | 79.03 (−4.7%) |

Prefill price (cited, not re-run): KL 1.5e-3, 993/1023 top-1.

## 7. Full reimplementation recipe (re-run to re-verify — nothing to build)

To reproduce: the §1 commands on the priced config; expect 0-2 flips per 56 ids, fluent text, clean exits, and VNNI decode ~5-13% slower than float. Revisit ONLY if a future VNNI kernel change moves the per-position price (re-price fidelity first) or a decode trajectory diverges (then narrow the flag's gate to prefill/batch-only). Revival trigger: none today — the hole is closed.

## 8. Verdict and why

**CONFIRM: the VNNI price covers decode trajectories on the priced config.** Zero flips on 56 ids with fluent text means the 1.5e-3-class perturbation does not compound in free-running greedy decode here — the flag needs no narrowing, and the 1394 hole is filled with a re-runnable two-command check. The side finding (VNNI decode slower, quantizer-dominated at tokens==1) matches the kq8d precedent and explains the flag's prefill-shaped use. Next-cycle candidates: the 1517 epic's stage-2 skip harness remains the only live structural front; the human-gated port shelf stands (mprefix-table, sparse no-gather, batchsnap kind tags, timers fix, detect memoize).

## 9. Exact commands run and their results

- `cmake --build --preset release` → 54/54; `tge_tests.exe` → 141/141; `aikit.py check aegean-tge` → 2/4 (test + secrets pass; pre-existing hygiene + guidelines fails).
- `generate --model <0.5B-Q8_0> --threads 6 --gpu off --vnni --prompt <fox|haiku> --n {32,24} --print-ids` → §6 VNNI rows (clean exits, fluent).
- Same without `--vnni` → §6 plain rows; ids byte-identical both prompts (0 flips).
- Scratch deleted; no branch, no edits; report committed; no push.

(End of file.)
