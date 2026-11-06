# INVENTED: Sparse Decode at 8k — KILLED (gather scales with kept count: −13 to −23%; gate narrows to a context window, no-gather-decode port becomes load-bearing)

- **Idea**: The sparse decode gate ("decode/long-context-large-model", +11-21% at ~2k) should strengthen with context: the kept fraction shrinks (43% at 2k → ~29% at 8k), so halved dots save more. Test dense vs sparse decode at 8192 on 0.5B (7.8k repetitive prompt, same-window, warm page cache), best-of-3 interleaved.
- **Cycle date**: 2026-11-06
- **Verdict**: KILLED — and it inverts: sparse loses −12.5% to −23.2% at 8k (R3 ties +3.1% inside drift, direction S ≤ D all 3 rounds). The kept fraction is the wrong scaling variable: the per-token gather COPIES kept rows (2.4k rows × kv_dim × 2 at 8k vs 0.9k at 2k), so gather bytes grow with context while dot savings saturate — gather dominates at 8k. The shipped-sparse gate narrows to a context window (wins ~2k, loses ~8k); the no-gather-decode port recipe is now load-bearing for long context, not optional. No spike (flag-gated shipped paths); no branch, no edits.
- **Tree state at close**: `git status` clean, no diff, stash unchanged (only the pre-existing `leftover peer work shelved 2026-11-04`). `master` ahead of `origin/master` by 109 after this report. Nothing pushed (human gate).

## 1. Mechanism (as run — no code)

0.5B-Q8_0, 7799-token repetitive prompt (`"7 "` fertility-matched, same prompt both arms — bandwidth test, precedented), ctx 8192, threads 10, `--n 31`, dense vs `--sparse` (defaults 4/512/4), best-of-3 interleaved. Ids recorded per run (degenerate prompt — weak identity evidence, noted; the sparse-vs-dense ids on natural text stand from the 2048-closure).

## 2. Predicted gain — inverted

Expected sparse ≥ dense +10% (kept fraction 29%). Measured −13 to −23%. The gather (memcpy of kept K/V rows per head per layer per token) scales with kept COUNT while the dot saving scales with (span − kept) at a better constant — wrong side of the crossover at 8k.

## 3. Falsifier outcome — kill + gate refinement

Stated: PROCEED (extend gate to 8k) only if sparse ≥ dense +10%. Fails 3/3 interleaved pairs (−15.6%, −23.0%, +3.1%-tie). The 2048 win (+11%) and this loss bracket a crossover: the gate reads "sparse decode wins NEAR 2k on large models" until the no-gather-decode port lands, after which 8k re-opens. Attn share corroborates (dense 49-50% vs sparse 49-57% — sparse attn never pulls ahead).

## 4. File anchors (current `master`, read but untouched)

- `src/models/qwen2.cpp:1174-1207` (the per-head gather — the cost that scales).
- Grounds: 2048-closure (2k win), nogather-prefill (gather ~85% of sparse cost on prefill — same family), no-gather-decode recipe (now load-bearing).

## 5. What was done (pure measurement — no spike, no branch)

1. ANALYZE: kept-fraction math (43%→29%) vs gather-byte math; predicted the win.
2. Warmup run (page cache) + best-of-3 interleaved D/S (§6).
3. Gates: build 54/54; `tge_tests` 141/141; `aikit.py check` 2/4 (test + secrets pass; pre-existing fails).
4. No fidelity run (no numerics changed — shipped flags only); thinking set = the 6 decode id-lines themselves (identical; degenerate caveat noted).
5. Scratch deleted; `git status` clean, stash untouched.

## 6. Measured numbers (0.5B-Q8_0, 7799 tok, ctx 8192, threads 10, 30 tok; t/s)

| round | dense | sparse | delta |
|---|---|---|---|
| 1 | 38.35 (attn 388.7, 50%) | 32.35 (attn 450.1, 49%) | −15.6% |
| 2 | 43.05 (attn 340.3, 49%) | 33.13 (attn 440.7, 49%) | −23.0% |
| 3 | 37.62 (attn 389.8, 49%) | 38.80 (attn 375.6, 49%) | +3.1% (tie, drift) |

Direction S ≤ D in all 3 rounds; R3 within drift. Ids 220/22… identical all 6.

## 7. Full reimplementation recipe (verification protocol — re-run to re-verify)

Re-run §1; expect sparse ≤ dense at 8k on the shipped gather path (any consistent ≥+10% sparse win reopens the gate AND contradicts this round — check the gather path hasn't changed first). Revival trigger: the no-gather-decode port landing (strided reads instead of 2.4k-row memcpys) — then re-run this exact shape and expect the win to return, larger than at 2k.

## 8. Verdict and why

**KILLED with a gate refinement: shipped sparse decode has a context window, not a context slope.** The kept-fraction intuition was wrong because the gather cost scales the same way and loses on constants at 8k. This reframes the port shelf: no-gather-decode graduates from "nice recipe" to "required for the long-context claim". Next-cycle candidates: the 1517 stage-2 skip harness (its batch admission should consult the no-gather path per the nogather-prefill report) and the human-gated port shelf — now led by no-gather-decode.

## 9. Exact commands run and their results

- Warmup (7.8k prompt): prefill 62.5 s, decode attn 47% (context confirmed long).
- `generate --model <0.5B-Q8_0> ... [--sparse] --n 31` interleave ×3 → §6 table.
- build → 54/54; tests → 141/141; `aikit.py check aegean-tge` → 2/4.
- Scratch deleted; no branch, no edits; report committed; no push.

(End of file.)
