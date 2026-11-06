# INVENTED (trigger S2): Pre-Packed-B GEMM Retry — KILLED analytic (no caller + code gone: a quiet-box rerun could only move bench GEMM)

- **Idea**: The parked S2 trigger: re-run the `gemm_packed_preb` (persistent-B cache) spike on a quiet box with the same interleaved pairs, to settle the sign-flipping ±30% (R1 +0.9%/+15%, R2 +30%/−54%) that killed it as box state rather than kernel.
- **Cycle date**: 2026-11-06
- **Verdict**: KILLED analytic — the retry is pointless regardless of box state. Re-grepped on today's tree: (1) production prefill still never calls any packed GEMM — `batch_matvec` routes Q8_0/K-quant/F32-per-token through GEMV kernels, `gemm_f32` callers remain bench + tests only (1325's map holds verbatim); (2) the preb code itself is fully gone — zero `preb` matches outside the measurement-log entry, the temporary `--kernel preb` flag reverted with the spike. A quiet-box rerun would rebuild a bench-only kernel from scratch to re-measure a bench-only number: best possible outcome 0% production. No spike per the dead-idea rule; no branch, no edits, nothing to revert.
- **Tree state at close**: `git status` clean, no diff, stash unchanged (only the pre-existing `leftover peer work shelved 2026-11-04`). `master` ahead of `origin/master` by 96 after this report. Nothing pushed (human gate).

## 1. Mechanism (as spiked in S2, now absent)

Persistent-B cache for `gemm_packed` (same panels/order, pack once, reuse across calls) behind temporary `--kernel preb`. Reorder-only → KL ~0 by construction; the only question was wall time on 1024³.

## 2. Predicted gain — 0% production by carrier absence

Even granting a clean +5% bench win (never observed — R1 +0.9%, R2 sign-flipped), production prefill tokens never reach the kernel: Q8_0 → `gemv_q8_batch`, K-quants → `gemv_dequant_batch`/kq8, F32 → per-token `gemv_f32` loop. The retry's ceiling is 0% before any box enters the picture.

## 3. Falsifier outcome — kill on both arms

Stated falsifier: kill the retry if (a) no production GEMM caller exists on today's tree, or (b) the preb code is absent (retry = rebuild, not rerun). Both hold: (a) `gemm_f32` callers = `apps/cli/main.cpp:195` (bench) + `tests/test_cpu.cpp:49` + `tests/test_tensor.cpp:46` only; `pack_a|pack_b` transient-only in `gemm_avx2.cpp`; `batch_matvec` gemm-free; (b) `preb` zero matches in code. Revival trigger ONLY (named, not built): F32 GEMM becomes the production prefill path AND a packed-resident budget is approved against mmap-first — the same trigger 1325 sets, still unset.

## 4. File anchors (current `master`, read but untouched)

- `src/cpu/avx2/gemm_avx2.cpp:103` (`pack_b`, transient), `:120-206` (`gemm_packed`), `:206/:213` (packed + unpacked bench controls).
- `src/core/dispatch.cpp:17` (`set.gemm_f32`); `apps/cli/main.cpp:195` (bench caller); `src/models/qwen2.cpp:894,1000` (production F32 GEMV, not GEMM).
- Grounds: verify-then-build S2 entry (`docs/measurement-log.md:3770-3778`), `reports/idea-1325-pack-on-load-split.md` (no-caller kill), `reports/idea-1569-inside-gemm-killed-analytic.md` (bench-only autotune).

## 5. What was done (ANALYZE only — no spike per the dead-idea rule)

1. Re-ran the carrier greps on today's tree (§3 — both kill arms verified live, not cited from memory).
2. Confirmed the preb spike's full revert (zero code matches; flag gone).
3. Gates on the unmodified tree: `cmake --build --preset release` 54/54; `tge_tests.exe` 141/141; `aikit.py check` 2/4 (test + secrets pass; hygiene + guidelines fail pre-existing, same pair as every prior cycle).
4. No fidelity run owed (no arithmetic exists to price); no bench A/B owed (rebuilding a bench-only kernel to re-measure it is the thing being killed); no thinking-set diff owed (same binary, zero behavior change).

## 6. Measured numbers (inventory — no new benchmark belongs to an analytic kill)

No new numbers: the applicable ones already exist (S2 R1/R2 sign-flip on record; 97% roof in §28ak; carrier map in 1325 §4/§5). This cycle adds the two absences (§3) and retires the trigger.

## 7. Full reimplementation recipe (deliberately NOT recommended — rebuild only to re-verify the kill)

To reproduce the negative result (do not ship): re-apply the S2 spike from the measurement-log description (persistent-B cache + `--kernel preb`), run 1024³ best-of-5 interleaved on a quiet box, then grep the callers and confirm bench-only. Expected: whatever the bench delta, production prefill unchanged (no caller). Revisit ONLY if the §3 trigger fires.

## 8. Verdict and why

**KILLED analytic: the S2 retry is retired, not postponed.** Box state was never the binding constraint — caller absence is. With preb fully reverted and production GEMM-free, the retry trigger I listed two cycles ago is removed from the candidate list. Remaining live fronts: the 1517 epic's stage-2 skip harness and the human-gated ports (mprefix-table, sparse no-gather, batchsnap kind tags, timers fix).

## 9. Exact commands run and their results

- Carrier greps (`preb`, `pack_b`, `gemm_f32`, `gemv_f32`, `gemm_packed`) — §3 map, both kill arms hold.
- `cmake --build --preset release` → 54/54; `tge_tests.exe` → 141/141; `aikit.py check aegean-tge` → 2/4 (test + secrets pass; pre-existing hygiene + guidelines fails).
- No branch created, no edits made, no benchmarks beyond the gate suite, report committed, no push.

(End of file.)
