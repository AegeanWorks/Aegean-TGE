# INVENTED: GQA-Grouped vs Per-Head Attention at 8k — CONFIRM (+14 to +51% grouped, ids identical; the auto ≥256 switch is validated at long ctx)

- **Idea**: The only attention-path A/B on record (§34) stopped at 300 tokens (+23% prefill / +16% decode for grouped, ids identical). The auto threshold (grouped at position ≥256) was never validated where it matters most — long-context decode, where position-splitting should beat per-head-only parallelization. Force `--attention head` vs `--attention gqa` at 8192 on 0.5B, best-of-3 interleaved.
- **Cycle date**: 2026-11-06
- **Verdict**: CONFIRM — grouped beats per-head by +51.1%/+13.7%/+34.7% at 8k decode with byte-identical ids on both natural-text prompts (story + creative). The auto switch is validated with teeth at long ctx; the warmup auto run matches the grouped arm, confirming auto actually takes the grouped path there. No spike (existing CLI flags, precedented measurement); no branch, no edits.
- **Tree state at close**: `git status` clean, no diff, stash unchanged (only the pre-existing `leftover peer work shelved 2026-11-04`). `master` ahead of `origin/master` by 110 after this report. Nothing pushed (human gate).

## 1. Mechanism (as run — existing flags, no code)

0.5B-Q8_0, 7799-token repetitive prompt (same-shape bandwidth test, precedented), ctx 8192, threads 10, `--n 31`, `--attention head` vs `--attention gqa`, best-of-3 interleaved + auto warmup. Natural-text parity: story + creative, both flags, ids diffed.

## 2. Predicted gain — direction confirmed with margin

Falsifier: ≥10% either way decides; within 10% validates the threshold as adequate. Measured +14 to +51% grouped — the auto direction is right at 8k, not just adequate.

## 3. Falsifier outcome — CONFIRM on all arms

1. **Decode**: head 25.11/27.25/26.95 vs grouped 37.95/30.97/36.31 t/s (+51.1%/+13.7%/+34.7%). Attn phase: per-head ~63% of decode vs grouped ~50% — position-splitting nearly halves attention time.
2. **Identity**: ids byte-identical all 6 degenerate runs + story + creative head-vs-gqa (online vs two-pass softmax agree here, as in §34).
3. **Auto check**: warmup `--attention auto` run (34.18 t/s, attn 416.4 ms) sits in the grouped arm's band — auto takes the grouped path at 8k as designed. No fidelity run (no numerics changed — shipped paths; the identity lines are the gate, precedented).

## 4. File anchors (current `master`, read but untouched)

- `apps/cli/main.cpp:2055-2068` (the `--attention` flag — the whole experiment needs no code).
- `src/models/qwen2.cpp:1211-1214` (`uses_gqa`: threads>1 AND (Gqa OR (Auto AND position ≥ 256)) — the threshold under test).
- `src/cpu/avx2/attention_avx2.cpp:261+` (grouped online-softmax kernel) vs `:121+` (per-head).
- Grounds: §34 (300-tok A/B), 1568 data shapes, attnsplit (per-head interior mapped).

## 5. What was done (pure measurement — no spike, no branch)

1. ANALYZE: confirmed the flag exists and routes as claimed; §34's scope ends at 300 tok.
2. Gates: build 54/54; `tge_tests` 141/141; `aikit.py check` 2/4 (test + secrets pass; pre-existing fails).
3. Warmup (page cache) + head/gqa interleave ×3 (§6); auto warmup (§3.3); story + creative head/gqa parity (§3.2).
4. Scratch deleted; `git status` clean, stash untouched.

## 6. Measured numbers (0.5B-Q8_0, 7799 tok, ctx 8192, threads 10, 30 tok; tok/s)

| round | per-head | grouped | delta |
|---|---|---|---|
| 1 | 25.11 (attn 752.2, 63%) | 37.95 (attn 395.6, 51%) | +51.1% |
| 2 | 27.25 (attn 692.8, 63%) | 30.97* (attn 417.3, 44%) | +13.7% |
| 3 | 26.95 (attn 695.7, 63%) | 36.31 (attn 405.9, 50%) | +34.7% |

\* R2 grouped hit box load (matvec 489 vs ~340 baseline); still wins the pair. Auto warmup: 34.18 t/s (grouped band).

## 7. Full reimplementation recipe (verification protocol — re-run to re-verify)

Re-run §1; expect grouped ≥ per-head +10% at 8k with identical ids. Revival trigger (the only open question this leaves): the exact threshold value — 256 was never tuned, only its direction validated (300 tok in §34, 8k here). A threshold sweep (64/128/256/512) at a mid ctx (1-2k) would place it precisely; expect a shallow optimum (both paths adequate near the crossover), so this is filed as a micro-tune, not a cycle with teeth.

## 8. Verdict and why

**CONFIRM: position-splitting wins big at long context, and auto already does it.** The 256 threshold's direction is now validated at both ends (300 tok in §34, 8k here) — the remaining unknown is the exact crossover point, a micro-tune explicitly deprioritized. Next-cycle candidates: the 1517 stage-2 skip harness remains the only live structural front; the human-gated port shelf stands.

## 9. Exact commands run and their results

- `generate --model <0.5B-Q8_0> --threads 10 ... --attention auto ...` warmup → auto ≈ grouped band (§3.3).
- `--attention head` vs `--attention gqa` interleave ×3 → §6 table (ids identical all 6).
- Story + creative under both flags → ids identical both prompts.
- build → 54/54; tests → 141/141; `aikit.py check aegean-tge` → 2/4.
- Scratch deleted; no branch, no edits; report committed; no push.

(End of file.)
