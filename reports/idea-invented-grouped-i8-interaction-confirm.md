# INVENTED (i8-geometry follow-up): Grouping × i8 Interaction — CONFIRMED on MHA (+24% per-head over grouped-i8); BONUS: per-head i8 beats per-head f32 everywhere (+13 to +24%)

- **Idea**: The geometry puzzle (0.5B −18% / Llama +19% / Phi −6% i8-vs-f32, all auto=grouped) named the grouped q8 path as the suspect. Test head-vs-grouped × f32-vs-i8 on all three architectures (Q4_K everywhere, 3.9k repetitive prompt, ctx 4096, threads 10, --n 30): if the path gap tracks the puzzle, the interaction is confirmed with a mechanism; if uniform, grouping is ruled out.
- **Cycle date**: 2026-11-06
- **Verdict**: CONFIRM with a bonus. (1) Interaction CONFIRMED where it matters: Phi grouped-i8 is 24% slower than per-head-i8 (2822 vs 1470 ms attn on the same run shape — groups-of-1 pay the full online-grouped machinery for zero splitting benefit); on GQA the path gap is ~noise (0.5B 0.6%, Llama 3-10% inside box+matvec wobble). (2) BONUS, bigger than the question asked: per-head i8 beats per-head f32 on ALL THREE — 0.5B +13.5%/+24.3% (best-of-2, swapped order), Llama +16.0%, Phi +19.4% (single pairs, noted). The "i8 loses" verdicts were all measured through the grouped path; on per-head the sign flips everywhere.
- **Tree state at close**: `git status` clean, no diff, stash unchanged (only the pre-existing `leftover peer work shelved 2026-11-04`). No branch created (flag-gated measurement only). `master` ahead of `origin/master` by 114 after this report. Nothing pushed (human gate).

## 1. Mechanism (as run — existing flags, no code)

`--attention head|gqa` × default-f32/i8 on 0.5B-Q4_K_M, Llama-3.2-3B-Q4_K_M, Phi-3.5-mini-Q4_K_M; repetitive 3900-token prompt (fits cmdline; same both arms); threads 10; --n 30; ids recorded.

## 2. Predicted gain — an interaction test, landed bigger

Falsifier: path gap ≥10% tracking the puzzle → mechanism lead. Phi delivers +24% (CONFIRM); GQA arms show ~0-3% (grouping ruled OUT there — the 0.5B/Llama puzzle signs must come from elsewhere: KV width vs dequant balance per §5 of the geometry report stands). Bonus: per-head i8-vs-f32 wins everywhere, which reframes the whole i8 decode question.

## 3. Falsifier outcome — CONFIRM (MHA) + bonus (universal per-head i8 win)

1. **0.5B**: f32 head 817/gqa 685 (−16%: grouped helps f32, matches GQA-8k); i8 head 719-681/gqa 724 (+0-6%: grouped adds nothing to i8); cross: per-head i8 beats f32 +13.5%/+24.3%.
2. **Llama**: f32 head≈gqa (4644/4573, noise); i8 head 4006/gqa 4125 (noise); cross: i8 beats f32 +16%/+11% on both paths.
3. **Phi**: f32 head 6027/gqa 6306 (head +4.6%); i8 head 5078/gqa 6304 (head **+24.3%**); cross: i8 beats f32 +19.4% on per-head, ties on grouped.
4. Ids identical everywhere (degenerate prompt, noted).

## 4. File anchors (current `master`, read but untouched)

- `apps/cli/main.cpp:2055-2068` (the `--attention` flag — the whole test surface) and `:551-557` (speculative per-head pinning — the existing precedent for path selection by workload).
- `src/models/qwen2.cpp:1177-1214` (`uses_gqa` — the auto rule this cycle informs).
- `src/cpu/avx2/attention_avx2.cpp:121+` (per-head, two-pass) vs `:290+` (grouped, online — the machinery whose per-group cost bites groups-of-1).
- Grounds: geometry round-robin (the puzzle), mha-i8 + i8-8k (the verdicts this refines), int8qk (dequant family).

## 5. What was done (pure measurement — no spike, no branch)

1. Full 2×2×3 grid (path × KV × arch) + 0.5B per-head i8-vs-f32 confirmation pair (swapped order) + ids.
2. Gates: build 54/54; tests 141/141; `aikit.py check` 2/4 (test + secrets pass; pre-existing fails).
3. Scratch deleted; `git status` clean, stash untouched.

## 6. Measured numbers (repetitive 3.9k, ctx 4096, threads 10, 30-tok decode; attn ms)

| model | f32 head | f32 gqa | i8 head | i8 gqa |
|---|---|---|---|---|
| 0.5B | 817 (347) | 684 (173) | 719/681 (239/219) | 724 (252) |
| Llama | 4644 (1599) | 4573 (1593) | 4006 (1015) | 4125 (921) |
| Phi | 6027 (2628) | 6306 (2891) | 5078 (1470) | 6304 (2822) |

i8-vs-f32 per-head: +13.5/+24.3%, +16.0%, +19.4%. i8-vs-f32 grouped: −5.5%, +10.8%, 0%. (Single pairs except 0.5B per-head best-of-2; box ±10% noted where relevant — only sub-10% deltas are flagged as such.)

## 7. Full reimplementation recipe (two keep-grade items, both human-gated)

1. **Usage rule (no code)**: at long ctx, prefer per-head attention with i8 KV (`--attention head --kv i8`); the auto path stays correct but concedes up to ~24% on MHA and ~5% on GQA-small-KV. Validate on natural text before publishing (repetitive prompt used here for cmdline length).
2. **KV-aware auto (one-condition port)**: in `uses_gqa`/dispatch, skip grouping for i8-KV when groups would be small (MHA: always; small-KV GQA: measure) — the grouped i8 path never beats per-head i8 anywhere measured. Gate the port: rebuild, full suite, this grid reproduced (per-head-i8 ≥ grouped-i8 everywhere, ids identical), aikit clean.
3. Revival trigger for the OLD guidance (grouped always at long ctx): a shape where grouped-i8 beats per-head-i8 by ≥10% with identical ids — none observed.

## 8. Verdict and why

**CONFIRM: the grouped path is i8-hostile (neutral at best, −24% at worst), and per-head i8 beats f32 everywhere measured (+13 to +24%).** This upgrades two verdicts at once: the geometry puzzle gets its mechanism lead (group-of-1 overhead + no i8 benefit from grouping on small KV), and the i8 decode story flips from "loses/neutral" to "wins on the right path". The auto router is now the bottleneck, not the format. Next-cycle candidates: natural-text validation of the per-head-i8 rule; the 1517 stage-2 harness (whose batch admission should note the path finding); the human-gated port shelf (now led by KV-aware auto + no-gather-decode).

## 9. Exact commands run and their results

- `--attention {head,gqa} [--kv i8]` × 3 models, interleaved within-model → §6 table (ids identical throughout).
- 0.5B per-head i8-vs-f32 swapped-order confirmation → +24.3% (backs R1's +13.5%).
- build → 54/54; tests → 141/141; `aikit.py check` → 2/4.
- Scratch deleted; no branch, no edits; report committed; no push.

(End of file.)
