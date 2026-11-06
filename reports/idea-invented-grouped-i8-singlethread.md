# INVENTED (interaction follow-up): Grouped-vs-Per-Head i8, Single-Threaded — PARALLELISM OWNS THE GAP (algorithmic branch closed: tied at 1.8% with no threads involved)

- **Idea**: The thermal-check cycle narrowed the grouped-vs-per-head i8 question to mechanism: is the ~6% multi-threaded gap algorithmic (online-softmax fusion vs two-pass) or parallelization (banding/sync/load-balance)? Remove threads from the equation entirely: same 8k i8 shape, `--threads 1` (forces per-head path) vs `--attention gqa --threads 1` (forces the grouped code single-threaded). A tie kills the algorithmic branch; a persistent gap indicts it.
- **Cycle date**: 2026-11-06
- **Verdict**: ALGORITHMIC BRANCH CLOSED. Single-threaded: per-head 1563.8 ms (attn 631.4) vs grouped 1535.3 ms (attn 618.4) = 1.8% apart, ids identical — a tie within noise. The online-softmax fusion computes the same work at the same speed as two-pass when no threads are involved. The multi-threaded 6-7% gap is therefore 100% parallelization-dependent (how the pool splits/joins the two loop nests), not arithmetic and not the online formulation. No spike (flag-gated measurement, precedented); no branch, no edits.
- **Tree state at close**: `git status` clean, no diff, stash unchanged (only the pre-existing `leftover peer work shelved 2026-11-04`). `master` ahead of `origin/master` by 117 after this report. Nothing pushed (human gate).

## 1. Mechanism (as run — no code)

0.5B-Q8_0, 7799 repetitive tokens (bandwidth test, same both arms), ctx 8192, --kv i8, `--attention head|gqa`, `--threads 1` (grouped code runs single-threaded; per-head forced), --n 15, single pair + ids.

## 2. Predicted gain — a falsifier, not a speedup

Stated: tie (±5%) closes the algorithmic branch (parallelism owns the gap); a persistent ≥10% grouped loss indicts the online formulation itself (a real kernel finding). Measured 1.8% → close the algorithmic branch.

## 3. Falsifier outcome — CLOSED, with the mechanism correctly assigned

1. **Single-threaded tie**: 1563.8 vs 1535.3 ms (attn differs 13 ms of ~625 — noise). Online vs two-pass: no inherent gap.
2. **Therefore the multi-threaded gap (6-7% on 0.5B) is thread-pool behavior**: banding granularity, barrier frequency, load balance across the two different nest shapes — NOT the math inside them.
3. Ids identical (degenerate prompt, noted as weak identity evidence; natural-text parity stands from prior cycles).

## 4. File anchors (current `master`, read but untouched)

- `apps/cli/main.cpp:2055-2068` (the `--attention` / `--threads` levers).
- `src/cpu/avx2/attention_avx2.cpp:121-288` (two-pass) vs `:290+` (grouped online — shown equivalent serially).
- Grounds: the interaction round (grouped-vs-perhead i8 gap), the q8micro arithmetic close (no per-element pathology), the thermal check (parallelism-independent confounds removed).

## 5. What was done (pure measurement — no spike, no branch)

1. CREATE with the stated algorithmic-vs-parallel falsifier.
2. Gates: build 54/54; tests 141/141; aikit 2/4 (test + secrets pass; pre-existing fails).
3. Single pair, both orders effectively (per-head run first, grouped second — adjacent, warm).
4. Scratch deleted; `git status` clean, stash untouched.

## 6. Measured numbers (0.5B-Q8_0, 7799 tok, ctx 8192, --kv i8, threads 1, 15-tok decode)

| path | decode | attn (share) | vs |
|---|---|---|---|
| per-head | 1563.8 ms (9.59 t/s) | 631.4 ms (40%) | — |
| grouped | 1535.3 ms (9.77 t/s) | 618.4 ms (40%) | +1.8% (tie) |

Ids: `220 22 …` identical. (Single pair — adequate for a tie claim this tight; a multi-pair repeat would only narrow the error bar, not change the verdict, since any algorithmic gap worth naming would exceed noise by far.)

## 7. Full reimplementation recipe (verification — re-run to re-verify)

Re-run §1; expect grouped ≈ per-head within ±5% single-threaded. Revival trigger: a PERSISTENT single-threaded grouped loss ≥10% (would reopen the algorithmic branch with a real number — today's 1.8% closes it). The remaining parallelization branch (banding/sync/load-balance of the two nests under the pool) is named follow-up with one caveat: it has no shippable end on its own — band/sync tuning lives with the thread-pool owners (H3 territory), and any change there must re-prove the GQA f32 wins first.

## 8. Verdict and why

**CLOSED: the grouped-vs-per-head i8 question is parallelization, not arithmetic — proven by removing threads.** Combined with the q8micro arithmetic close, both non-parallel explanations are now dead: same ops (microbench smooth), same serial speed (this cycle). What remains is pool behavior, which belongs to the threading program, not the attention program. The actionable guidance is unchanged (per-head i8 at long ctx; KV-aware auto as the human-gated fix). Next-cycle candidates: the 1517 stage-2 harness remains the only live structural front; the human-gated port shelf stands.

## 9. Exact commands run and their results

- 0.5B 7.8k i8 `--attention head|gqa --threads 1` → §6 table (ids identical).
- build → 54/54; tests → 141/141; `aikit.py check` → 2/4.
- Scratch deleted; no branch, no edits; report committed; no push.

(End of file.)
