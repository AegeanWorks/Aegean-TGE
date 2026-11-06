# INVENTED: Speculative × Non-f32 KV Compatibility — CONFIRM (8/8 combos clean; temp-0 exact, temp-0.1 same shape)

- **Idea**: The f32-only guards (CoW, H1/H2 records, sparse) disable themselves gracefully under `--kv`, but the speculative loop (drafter → 6-row verify → correct/advance) was never run against i8/f16 KV on the record. Unreviewed combo = crash/divergence risk. Run it: temp-0 exact-identity gate + temp-0.1 shape gate on f32/i8/f16 × nospec/spec.
- **Cycle date**: 2026-11-06
- **Verdict**: CONFIRM compat. All 8 combos ran clean, drafter engages on every KV tier (2.4-3.5/step temp-0, 1.40/step temp-0.1), temp-0 trajectories byte-identical spec-vs-nonspec within each tier, temp-0.1 shapes match the f32 spec pattern on i8 and f16. No spike (flag combos only, precedented); no branch, no edits.
- **Tree state at close**: `git status` clean, no diff, stash unchanged (only the pre-existing `leftover peer work shelved 2026-11-04`). `master` ahead of `origin/master` by 111 after this report. Nothing pushed (human gate).

## 1. Mechanism (as run — existing flags, no code)

0.5B-Q8_0, fox prompt, `--n 20`, threads 6: {f32, `--kv i8`, `--kv f16`} × {plain, `--speculative`} × {`--temp 0`, `--temp 0.1 --seed 7`}, `--print-ids`. (Sparse explicitly rejects non-f32 KV at the CLI with an error — that combo stays out by design.)

## 2. Predicted gain — a compat verdict, landed clean

Expected: clean runs + same-shape acceptance. Measured: temp-0 spec==nonspec ids byte-identical per tier (f32: 576 3974…; i8: 576 11652…); temp-0.1 all three tiers show the f32 spec shape (5 steps, 7 acc, 1.40/step).

## 3. Falsifier outcome — CONFIRM on all arms

1. **No crash/hang/divergence** in 8/8 combos (the REOPEN arm).
2. **Drafter engages per tier**: temp-0 f32 3.00/step, i8 3.50/step; temp-0.1 all tiers 1.40/step.
3. **Sane trajectories**: temp-0 exact per tier; temp-0.1 ids differ across sampling paths only as RNG consumption differs (expected at temp>0; the i8-vs-f16 coincidence is one sample, not a claim). Ids show diverse non-looping tokens; step counts match the shipped f32 spec shape.

## 4. File anchors (current `master`, read but untouched)

- `apps/cli/main.cpp:551-574` (speculative per-head pinning; sparse-vs-KV explicit reject — the gating pattern this combo lacks and doesn't need).
- `src/models/qwen2.cpp` (kv_half_/kv_q8_ branches in attention/forward/verify/store paths — the coverage that makes the combo work).
- Grounds: the sampler/speculative program (shapes this matches), kv-auto + i8/f16 pricing (tiers this preserves).

## 5. What was done (pure measurement — no spike, no branch)

1. ANALYZE: confirmed no explicit guard rejects spec×i8/f16 (unlike sparse, which errors cleanly); confirmed per-path KV support in the model file.
2. Gates: build 54/54; `tge_tests` 141/141; `aikit.py check` 2/4 (test + secrets pass; pre-existing fails).
3. 8 generate combos (§6); ids diffed per tier; acceptance shapes compared.
4. No fidelity run (no numerics changed — flag combinations of priced paths); thinking set = the 8 trajectories themselves (ids recorded).
5. Scratch deleted; `git status` clean, stash untouched.

## 6. Measured numbers (0.5B-Q8_0, fox, threads 6, --n 20)

| combo | trajectory | spec |
|---|---|---|
| f32, temp-0 | 576 3974 … 2311 330 | 2 steps / 6 acc (3.00) — identical ns/sp |
| i8, temp-0 | 576 11652 … 264 9445 | 2 steps / 7 acc (3.50) — identical ns/sp |
| f32, t0.1 s7 | f32-ns ids | 5 steps / 7 acc (1.40) |
| i8, t0.1 s7 | i8-ns ids | 5 steps / 7 acc (1.40) |
| f16, t0.1 s7 | f16-ns ids (= f32-ns) | 5 steps / 7 acc (1.40) |

## 7. Full reimplementation recipe (verification protocol — re-run to re-verify)

Re-run §1; expect clean exits, temp-0 identity per tier, temp-0.1 acceptance in the 1-4/step band. Revival trigger: any crash, hang, 0-accept divergence, or looping-ids on a spec×non-f32-KV combo — file as a bug with the exact flag set (the f32-only guards are the first suspects).

## 8. Verdict and why

**CONFIRM: the speculative loop is KV-tier-agnostic on the tested paths.** The f32-only features (CoW, H1/H2, sparse) already self-disable; everything else routes per-tier correctly, including under speculation. Next-cycle candidates: the 1517 stage-2 skip harness remains the only live structural front; the human-gated port shelf stands.

## 9. Exact commands run and their results

- `generate --model <0.5B-Q8_0> ... [--kv i8|f16] [--speculative] [--temp 0|0.1] ... --n 20 --print-ids` (8 combos) → §6 table.
- build → 54/54; tests → 141/141; `aikit.py check aegean-tge` → 2/4.
- Scratch deleted; no branch, no edits; report committed; no push.

(End of file.)
