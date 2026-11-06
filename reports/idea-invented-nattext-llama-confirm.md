# INVENTED (grouped-i8 follow-up): Llama Natural-Text Per-Head i8 — CONFIRM (+6%/+8%, ids identical; transfer matrix complete except one low-priority cell)

- **Idea**: The grouped-i8 cycle proved per-head i8 wins on repetitive text and confirmed it on natural text for 0.5B only, naming Llama-natural transfer as the gate. Run it: Llama-3B, fox×150 natural prose (~1950 tok), --attention head, f32 vs --kv i8, best-of-2 swapped order, ids diffed.
- **Cycle date**: 2026-11-06
- **Verdict**: CONFIRM — i8 wins +6.3% and +8.3% (best-of-best +7.8%) with byte-identical natural-text ids both rounds; attn share falls 20%→14% as in every other cell. The per-head-i8 rule now holds on 5 of 6 model×text cells (0.5B rep/nat, Llama rep/nat, Phi rep); Phi-natural is the single untested cell, explicitly deprioritized (pattern consistent 5/5, low marginal value).
- **Tree state at close**: `git status` clean, no diff, stash unchanged (only the pre-existing `leftover peer work shelved 2026-11-04`). No branch, no edits. `master` ahead of `origin/master` by 116 after this report. Nothing pushed (human gate).

## 1. Mechanism (as run — no code)

Llama-3.2-3B-Q4_K_M, fox×150 (~1950 tok), ctx 4096, threads 10, --attention head, f32 vs --kv i8, --n 31, interleaved f/i then i/f, ids recorded.

## 2. Predicted gain — confirmed in the established band

Falsifier: ≥+5% both rounds + identical ids → CONFIRM transfer; else qualify. Measured +6.3% / +8.3%, ids identical 4/4.

## 3. Falsifier outcome — CONFIRM

1. **Win holds on prose**: best-of-best 8.75 vs 8.12 t/s (+7.8%); attn 736→468 ms (−36%).
2. **Identity on prose**: full natural-text ids identical all 4 runs (not just degenerate).
3. Prefill neutral-to-noisy (−2.8%/+6.5% — short ctx, KV-write share tiny, as expected; not gated).

## 4. File anchors (current `master`, read but untouched)

- `apps/cli/main.cpp:2055-2068` (the `--attention`/`--kv` flags — the whole test surface).
- Grounds: grouped-i8 interaction (the rule + this transfer gate), nattext-i8 (0.5B cell), i8-geometry round-robin (Llama repetitive cell).

## 5. What was done (pure measurement — no spike, no branch)

1. CREATE with the grouped-i8 report's stated transfer gate.
2. Gates: build 54/54; `tge_tests` 141/141; `aikit.py check` 2/4 (test + secrets pass; pre-existing fails).
3. Llama fox×150 head f32/i8 swapped pair (§6) + ids.
4. Scratch deleted; `git status` clean, stash untouched.

## 6. Measured numbers (Llama-3B-Q4_K_M, fox×150, threads 10, 30 tok; per-head)

| round | f32 (attn) | i8 (attn) | delta |
|---|---|---|---|
| 1 (f,i) | 8.12 t/s (735.7 ms, 20%) | 8.63 t/s (496.5, 14%) | **+6.3%** |
| 2 (i,f) | 8.08 (736.3) | 8.75 (468.6) | **+8.3%** |

Ids: `578 3974 13876 38835 …` identical all 4. Transfer matrix now: 0.5B rep ✓ / nat ✓; Llama rep ✓ / nat ✓; Phi rep ✓ / nat — (open, low priority).

## 7. Full reimplementation recipe (verification — re-run to re-verify)

Re-run §1; expect i8 ≥ f32 +5% with identical ids. Revival trigger: an i8 loss on natural prose at ≥2k ctx with per-head attention (would split prose-vs-repetitive, reopening content-dependence — none observed in 6/6 prose runs to date).

## 8. Verdict and why

**CONFIRM: the per-head-i8 rule is prose-validated on the reference model.** The transfer matrix is functionally complete (5/6 cells green, the 6th low-value by pattern consistency). The i8 story for this loop is now closed end to end: fidelity priced, geometry mapped, path qualified, prose confirmed, guidance shippable (KV-aware auto + per-head-i8 usage, both human-gated). Next-cycle candidates: the 1517 stage-2 harness remains the only live structural front; the human-gated port shelf stands.

## 9. Exact commands run and their results

- Llama fox×150 `--attention head` [--kv i8] interleave ×2 → §6 table + ids.
- build → 54/54; tests → 141/141; `aikit.py check` → 2/4.
- Scratch deleted; no branch, no edits; report committed; no push.

(End of file.)
