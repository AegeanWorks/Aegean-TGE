# INVENTED (f16-path follow-up): Llama f16 Grouped-vs-Per-Head — CONFIRM uniformity (+4-6% decode, −33% attn; f16 follows f32 on the reference model too)

- **Idea**: The f16-path cycle closed the 0.5B map (grouped wins big for f16) but explicitly deferred Llama/Phi transfer. Test the reference model at 2k (fox×150 natural text): does grouped f16 also win, or is the 0.5B result shape-specific?
- **Cycle date**: 2026-11-06
- **Verdict**: CONFIRM uniformity — grouped wins both swapped rounds (R1 +5.7%, R2 +4.1% decode; attn −33% both), ids byte-identical all 4. Same sign as 0.5B, smaller magnitude (attn is 11-16% of Llama decode here vs ~50% at 0.5B/8k — the win scales with attn share, as it should). No model-specific surprise; the "cheap dequant → grouped wins" rule transfers.
- **Tree state at close**: `git status` clean, no diff, stash unchanged (only the pre-existing `leftover peer work shelved 2026-11-04`). No branch, no edits. `master` ahead of `origin/master` by 116 after this report. Nothing pushed (human gate).

## 1. Mechanism (as run — no code)

Llama-3.2-3B-Q4_K_M, fox×150 natural text (~1950 tok), ctx 4096, threads 10, --kv f16, --attention head|gqa, --n 31, swapped pair, attn phases + ids.

## 2. Predicted gain — uniformity confirmed within the expected scaling

Falsifier: grouped ≥ per-head on Llama (same direction as 0.5B, any magnitude ≥0%) with identical ids → CONFIRM; a grouped LOSS would break uniformity and reopen model-dependence. Measured +4-6% grouped, identical ids.

## 3. Falsifier outcome — CONFIRM

1. **Direction holds on the reference model**: grouped wins both orders (excludes drift as the driver).
2. **Magnitude scales with attn share**: +4-6% decode here (attn 11-16% of wall) vs +35-44% on 0.5B@8k (attn ~50%) — exactly the expected scaling, not a discrepancy.
3. Ids byte-identical all 4 (natural text).

## 4. File anchors (current `master`, read but untouched)

- `apps/cli/main.cpp:2055-2068` (the two flags).
- Grounds: f16-path confirm (0.5B map), nattext-llama (prose on this model), GQA-8k (grouped f32 baseline).

## 5. What was done (pure measurement — no spike, no branch)

1. CREATE with the stated uniformity falsifier.
2. Gates: build 54/54; tests 141/141; `aikit.py check` 2/4 (test + secrets pass; pre-existing fails).
3. Llama fox×150 f16 head/gqa swapped pair (§6) + ids.
4. Scratch deleted; `git status` clean, stash untouched.

## 6. Measured numbers (Llama-3.B-Q4_K_M, fox×150, ctx 4096, threads 10, --kv f16, 30 tok)

| round | per-head (attn) | grouped (attn) | delta |
|---|---|---|---|
| 1 (h,g) | 8.51 t/s (552 ms, 16%) | 8.99 (368, 11%) | +5.7% |
| 2 (g,h) | 8.59 (546) | 8.94 (367) | +4.1% |

## 7. Full reimplementation recipe (verification — re-run to re-verify)

Re-run §1; expect grouped ≥ per-head with identical ids on any model where attn ≥10% of decode; below that share the comparison is noise-dominated (report honestly instead of claiming). Phi transfer remains explicitly open but low-priority (same uniform-dequant argument covers it; run only if a Phi-specific doubt ever materializes).

## 8. Verdict and why

**CONFIRM: the f16-grouped rule is model-independent within the tested family.** Combined with the 0.5B map, the path×KV×arch story is now: grouped wins for f32/f16 everywhere measured (magnitude follows attn share); per-head wins for i8 everywhere measured. The remaining open threads are the human-gated auto-routing port (which can now be specified unconditionally: grouped unless (KV==i8 AND (long ctx OR always))) and the 1517 stage-2 harness. Next-cycle candidates: those two, in that order of actionability — both blocked on human approval, so the loop continues with whatever new falsifiable direction appears.

## 9. Exact commands run and their results

- Llama fox×150 f16 head/gqa swapped pair → §6 table + ids.
- build → 54/54; tests → 141/141; `aikit.py check` → 2/4.
- Scratch deleted; no branch, no edits; report committed; no push.

(End of file)
