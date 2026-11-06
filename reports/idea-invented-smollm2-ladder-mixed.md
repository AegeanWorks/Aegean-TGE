# INVENTED: SmolLM2 Coverage + i8 Ladder Bottom Rung — MIXED (runs clean, 98% i8; size trend CONFIRMED in-Qwen on second corpus; cross-arch comparison stays confounded)

- **Idea**: SmolLM2-135M-Q4_K_M never ran in the loop. Smoke-test it, then place the i8 bottom rung: does a 135M model collapse (cliff → size rule breaks downward) or hold (extends the trend)? Added control: 0.5B-Q4_K_M + 1.5B-Q4_K_M on the SAME arch corpus (same window), because cross-arch numbers need a same-corpus anchor and the prior ladder used a different (deleted) corpus.
- **Cycle date**: 2026-11-06
- **Verdict**: MIXED. (1) SmolLM2 loads/runs/generates cleanly (30 layers, 576 dim, 9/3 GQA, 49k vocab; greedy story fluent) — first coverage, no crash. (2) SmolLM2 i8: 195/199 (98.0%), KL 5.9e-4 — holds, no cliff. (3) Size trend CONFIRMED within Qwen on arch: 0.5B 178/199 (89.4%, 21 flips) << 1.5B 196/199 (98.5%, 3 flips). (4) Cross-arch SmolLM2-vs-Qwen comparison stays confounded (different tokenizers → different token streams on the same text); no trend claim made from it. Practical guidance unchanged everywhere (sub-3B already f32/f16; 1.5B stays conditional).
- **Tree state at close**: `git status` clean, no diff, stash unchanged (only the pre-existing `leftover peer work shelved 2026-11-04`). No branch, no edits, no scratch files (all output captured in-tool). `master` ahead of `origin/master` by 116 after this report. Nothing pushed (human gate).

## 1. Mechanism (as run — no code)

`generate` smoke (story +n 4, greedy, fluent stop) + `fidelity --corpus docs/architecture.md --context 512 --tokens 200 --threads 6|1 --gpu off [--kv i8]` on SmolLM2-135M, 0.5B-Q4_K, 1.5B-Q4_K (f32 sanity + i8 each, back-to-back).

## 2. Predicted gain — a coverage verdict, not a speedup

Falsifier: cliff (SmolLM2 i8 collapses ≪90%) would break the size rule downward (REOPEN); holds-or-degrades-smoothly keeps it (CONFIRM/EXTEND within confounds). Measured: holds at 98% — no cliff. Extension proper is blocked by the tokenizer confound (stated, not hidden).

## 3. Falsifier outcome — mixed, with each arm labeled

1. **Smoke**: clean load/run/generate (first coverage — CONFIRM the engine handles the arch).
2. **No cliff**: 98.0% >> any collapse definition — the size rule's low end is ordinary, not broken.
3. **In-Qwen trend on second corpus**: 89.4% → 98.5% with size — CONFIRMS the ladder isn't a diverse-corpus artifact (arch is harsher per-token for 0.5B: 10.6% vs 3.7% flips, worth noting for corpus choice in future pricing).
4. **Cross-arch**: explicitly NOT claimed (different vocabularies → different evaluated token streams; 0.5B's arch harshness may be tokenizer-content interaction, not size alone).

## 4. File anchors (current `master`, read but untouched)

- `src/models/qwen2.cpp` (Phi/fused-qkv handling that also covers SmolLM2's layout — the reason it just works).
- Grounds: i8 size-ladder (bar + conditional keep), phi-i8 (architecture-vs-size precedent for honesty about confounds), i8-8k (corpus/quantizer context).

## 5. What was done (pure measurement — no spike, no branch)

1. ANALYZE: confirmed SmolLM2 absent from all reports; checked the ladder's corpus situation (diverse temp file gone → arch corpus, in-repo, as the common ground).
2. Gates: build 54/54; `tge_tests` 141/141; `aikit.py check` 2/4 (test + secrets pass; pre-existing fails).
3. Smoke + 6 fidelity runs (§6).
4. No thinking-set diff owed (story smoke was the fluency check; fidelity verdicts are the gate).

## 6. Measured numbers (arch corpus, ctx 512, --tokens 200 → 199 eval positions each)

| model (Q4_K_M) | f32 KL | f32 top-1 | i8 KL | i8 top-1 | i8 flips |
|---|---|---|---|---|---|
| SmolLM2-135M | 5.949e-12 | 199/199 | 5.881e-04 | 195/199 (98.0%) | 4 |
| Qwen2.5-0.5B | 1.303e-11 | 199/199 | 3.151e-02 | 178/199 (89.4%) | 21 |
| Qwen2.5-1.5B | 6.171e-12 | 199/199 | 1.598e-03 | 196/199 (98.5%) | 3 |

## 7. Full reimplementation recipe (verification — re-run to re-verify)

Re-run §1; expect the same ordering (0.5B ≪ {1.5B, SmolLM2} on arch-i8) with digit-stable f32 floors. Revival trigger: a sub-0.5B Qwen-family model (same tokenizer, smaller than 0.5B) — the ONLY clean bottom-rung test; none exists in the folder, so the ladder's low end stays at 0.5B until one appears. Do NOT compare across tokenizers to claim size effects.

## 8. Verdict and why

**MIXED, honestly partitioned: coverage CONFIRMED, trend CONFIRMED in-family, cross-arch left open.** The loop gains a new running architecture, a second-corpus confirmation of its most load-bearing conditional (i8 size-dependence is not a diverse-corpus artifact), and an explicit confound boundary instead of a false generalization. Next-cycle candidates: the 1517 stage-2 harness remains the only live structural front; the human-gated port shelf stands.

## 9. Exact commands run and their results

- Smoke: `generate --model <SmolLM2> ... --n 4` → clean (30 layers, 576 dim, 9/3 heads).
- 6× fidelity (0.5B/1.5B/SmolLM2 × f32/i8) → §6 table.
- build → 54/54; tests → 141/141; `aikit.py check` → 2/4.
- No branch, no edits, no scratch files; report committed; no push.

(End of file.)
