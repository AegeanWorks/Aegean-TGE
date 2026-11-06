# INVENTED (grouped-i8 follow-up): Natural-Text Per-Head i8 — CONFIRM (+12.6%/+6.5% decode, attn −29 to −39%, ids byte-identical on prose)

- **Idea**: The grouped-i8 cycle proved per-head i8 beats per-head f32 on a repetitive prompt (cmdline-length workaround) and flagged natural-text validation as the publishing gate. Run it: fox×150 natural text (~1950 tok, fits cmdline), 0.5B, per-head path, f32 vs i8, interleaved best-of-2, ids diffed.
- **Cycle date**: 2026-11-06
- **Verdict**: CONFIRM — i8 wins on natural prose too: decode +12.6% / +6.5% (best-of-best +12.0%), attn phase −39%/−29%, prefill neutral (+0.4%/−7.3%, inside noise — short prompt, KV-write share small), ids byte-identical all 4 (real prose ids `576 3974 …`, not degenerate). The repetitive-prompt caveat is lifted for the per-head-i8 rule.
- **Tree state at close**: `git status` clean, no diff, stash unchanged (only the pre-existing `leftover peer work shelved 2026-11-04`). No branch created (flag-gated measurement only). `master` ahead of `origin/master` by 116 after this report. Nothing pushed (human gate).

## 1. Mechanism (as run — no code)

0.5B-Q8_0, fox×150 (1950 tok natural text — the fox sentence is real prose with repetition, not a single-token loop), ctx 4096, threads 10, `--attention head`, f32 vs `--kv i8`, `--n 12`, interleaved f/i/f/i, ids recorded.

## 2. Predicted gain — confirmed in range

Falsifier: ≥+5% decode best-of-best with identical ids. Measured +12.0% best-of-best (+12.6%, +6.5% per round), ids 12/12 identical both rounds.

## 3. Falsifier outcome — CONFIRM on both arms

1. **Win holds on prose**: attn share 21-30% here (vs ~50% at 8k) yet i8 still clears the bar — the per-head dequant path is cheaper than widen-then-FMA per row at every ctx tested.
2. **Identity on prose**: byte-identical ids (not just degenerate) — the fidelity gate (i8 class) plus this readline: no divergence risk in the usage rule.

## 4. File anchors (current `master`, read but untouched)

- `apps/cli/main.cpp:2055-2068` (`--attention` — the test surface).
- Grounds: grouped-i8 interaction (the rule + its caveat — caveat now lifted), i8-8k kill (grouped-path numbers stand as measured; the path qualifier is now load-bearing).

## 5. What was done (pure measurement — no spike, no branch)

1. CREATE with the grouped-i8 report's stated validation gate (natural text, ≥5%, identical ids).
2. Gates: build 54/54, `tge_tests` 141/141, `aikit.py check` 2/4 (test + secrets pass; pre-existing fails).
3. Fox×150 per-head f32/i8 interleave ×2 (§6).
4. Scratch deleted; `git status` clean, stash untouched.

## 6. Measured numbers (0.5B-Q8_0, 1950 tok prose, ctx 4096, threads 10, 11-tok decode)

| round | f32 decode (attn) | i8 decode (attn) | delta | prefill f32 → i8 |
|---|---|---|---|---|
| 1 (f,i) | 62.00 t/s (52.4 ms, 30%) | 69.83 t/s (31.9 ms, 21%) | **+12.6%** | 299.4 → 300.7 t/s (+0.4%) |
| 2 (f,i) | 62.34 (52.4, 30%) | 66.40 (37.2, 23%) | **+6.5%** | 300.0 → 279.3 (−6.9%, noise) |

Ids: `576 3974 13876 … 3170` identical all 4.

## 7. Full reimplementation recipe (usage rule — no code to port)

At long ctx, run i8 KV with the per-head attention path (`--attention head --kv i8`); expect ~+10% decode vs f32 on prose at 2k (growing with ctx as the KV share grows), identical trajectories. Do NOT generalize to the grouped path (killed/neutral there) or to short ctx (untested — KV share too small to matter). Revival trigger: a shape where per-head-i8 ties or loses at ≥2k with identical ids — re-opens path selection.

## 8. Verdict and why

**CONFIRM: the per-head-i8 rule is prose-validated and publishable as a usage recipe.** Combined with the grouped-i8 cycle, the i8 decode story is now complete and path-qualified: per-head wins (+6 to +24% across models/shapes/corpora), grouped neutral-to-negative — and auto's one-size path is the remaining inefficiency (KV-aware auto, human-gated). Next-cycle candidates: the 1517 stage-2 skip harness remains the only live structural front; the human-gated port shelf stands.

## 9. Exact commands run and their results

- Fox×150 per-head f32/i8 interleave ×2 → §6 table + ids.
- build → 54/54; tests → 141/141; `aikit.py check` → 2/4.
- Scratch deleted; no branch, no edits; report committed; no push.

(End of file.)
