# INVENTED: f16 Per-Head vs Grouped at 8k — CONFIRM (f16 follows f32: grouped wins +44%/+36%; path×KV matrix complete)

- **Idea**: The path×KV matrix had one empty cell: f16 was only ever run grouped (auto). Does f16 behave like f32 (grouped wins big via online+parallelism) or like i8 (grouped neutral/hostile — dequant-saturated)? 0.5B, 7.8k repetitive prompt, ctx 8192, `--kv f16`, `--attention head|gqa` swapped pair, ids recorded.
- **Cycle date**: 2026-11-06
- **Verdict**: CONFIRM — grouped wins big: R1 32.49→47.09 (+44.9%), R2 34.24→46.73 (+36.5%) swapped; attn 531→268 / 508→270 (~−48%). f16 follows f32, not i8: lightweight/no dequant → grouping's online+parallel formulation dominates; heavyweight dequant (i8) → the win evaporates. The matrix is now complete on 0.5B: f32 grouped≫head, f16 grouped≫head, i8 grouped≈head.
- **Tree state at close**: `git status` clean, no diff, stash unchanged (only the pre-existing `leftover peer work shelved 2026-11-04`). No branch, no edits. `master` ahead of `origin/master` by 116 after this report. Nothing pushed (human gate).

## 1. Mechanism (as run — existing flags, no code)

0.5B-Q8_0, 7799 repetitive tokens, ctx 8192, threads 10, --kv f16, --attention head|gqa, --n 31, interleaved h,g / g,h; per-head and grouped attn phases + ids.

## 2. Predicted gain — falsifier satisfied with margin

Stated: ≥10% either way decides f16's allegiance. Measured +44.9% / +36.5% grouped — f16 is f32-like. (If i8-like, grouped would tie-or-lose as in the interaction cycle.)

## 3. Falsifier outcome — CONFIRM

1. **Direction**: grouped wins both rounds, swapped order (rules out drift as the driver).
2. **Magnitude**: attn halved (531→268, 508→270) — same class as f32's grouped win, larger because f16 halves the traffic the grouped path streams.
3. **Ids**: byte-identical all 4 (degenerate prompt, noted as weak identity evidence; the f16 fidelity price stands separately).

## 4. File anchors (current `master`, read but untouched)

- `apps/cli/main.cpp:2055-2068` (the `--attention` / `--kv` levers).
- `src/cpu/avx2/attention_avx2.cpp:121-288` vs `:290+` (the two paths whose interaction with KV format is now fully mapped on 0.5B).
- Grounds: GQA-8k (f32 grouped win), grouped-i8 + thermal check (i8 neutral/hostile), f16-8k (f16 wins over f32), q8micro (dequant costs are smooth — the path interaction is structural, not arithmetic).

## 5. What was done (pure measurement — no spike, no branch)

1. CREATE with the stated allegiance falsifier.
2. Gates: build 54/54; tests 141/141; `aikit.py check` 2/4 (test + secrets pass; pre-existing fails).
3. f16 head/gqa swapped pair (§6) + ids.
4. Scratch deleted; `git status` clean, stash untouched.

## 6. Measured numbers (0.5B-Q8_0, 7799 tok, ctx 8192, threads 10, --kv f16, 30 tok; attn ms)

| run | per-head | grouped | delta |
|---|---|---|---|
| R1 (h,g) | 32.49 t/s (531.5) | 47.09 t/s (267.6) | **+44.9%** |
| R2 (g,h) | 34.24 (508.5) | 46.73 (270.3) | **+36.5%** |

Complete 0.5B path×KV picture: f32 grouped +45% / f16 grouped +44% / i8 grouped ~0% (thermal-check best-of-3).

## 7. Full reimplementation recipe (verification — re-run to re-verify)

Re-run §1; expect grouped ≥ per-head +10% on f16 at long ctx with identical ids. The remaining open cells (Llama/Phi f16 paths) are explicitly deprioritized: 0.5B decides the allegiance question, and per-model transfer for f16 has no mechanism to differ (cvt dequant is uniform and cheap — unlike i8's widen, which motivated the per-model work).

## 8. Verdict and why

**CONFIRM: the path×KV interaction map is now complete and coherent — lightweight KV (f32, f16) loves the grouped path; heavyweight-KV (i8, widen-per-element) is neutral-to-hostile to it.** No single default serves both, which is exactly the KV-aware-auto port the loop keeps naming. Nothing further to measure here without touching code. Next-cycle candidates: the 1517 stage-2 harness remains the only live structural front; the human-gated port shelf stands (KV-aware auto now fully specified by three confirming cycles).

## 9. Exact commands run and their results

- 0.5B f16 head/gqa swapped pair → §6 table + ids.
- build → 54/54; tests → 141/141; `aikit.py check` → 2/4.
- Scratch deleted; no branch, no edits; report committed; no push.

(End of file.)
