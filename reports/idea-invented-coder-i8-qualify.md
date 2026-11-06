# INVENTED: Code-Tuning × i8 Robustness — QUALIFY (same arch/size/tokenizer: coder 3 flips↔9 flips by corpus, instruct flat 3↔3; domain interacts with quantization)

- **Idea**: The i8 size rule was built entirely on instruct-tuned weights. Qwen2.5-Coder-1.5B-Instruct-Q4_K_M is the same architecture, size, quant, and (verified) tokenizer as the instruct 1.5B — differing only in code-tuning. If training doesn't matter, i8 behaves identically; if it does, the size-only rule needs a domain qualifier. Falsifier: |flips diff| ≤ 2 on the same corpus → training-independent (CONFIRM size-only); strong corpus-dependent divergence → QUALIFY.
- **Cycle date**: 2026-11-06
- **Verdict**: QUALIFY — the interaction is real and corpus-dependent. Arch corpus: coder 190/199 (9 flips, KL 1.2e-02) vs instruct 196/199 (3 flips, KL 1.6e-3) — coder 3x worse. Repetitive corpus: coder 199/199 (KL 4.1e-4) vs instruct 196/199 (KL 4.4e-3) — coder 10x BETTER on KL, perfect on top-1. Same tokenizer confirmed byte-identical (promptids match), so this is weights × corpus, not tokenization.
- **Tree state at close**: `git status` clean, no diff, stash unchanged (only the pre-existing `leftover peer work shelved 2026-11-04`). No branch, no edits, no scratch files left (temp corpora deleted; arch is in-repo, repetitive is fox×150 described verbatim). `master` ahead of `origin/master` by 116 after this report. Nothing pushed (human gate).

## 1. Mechanism (as run — no code)

`fidelity --model {Coder-1.5B|Instruct-1.5B}-Q4_K_M --corpus {arch|repetitive-fox} --context 512 --tokens 200 --threads 6 [--kv i8]`; f32 sanity + i8 each (8 runs), plus a tokenizer-identity check (same prompt → same promptids both models).

## 2. Predicted gain — a qualifier, landed

Falsifier: arch 9-vs-3 split + rep inversion (0-vs-3 the other way) rules out both "training-independent" and any simple monotonic story. Verdict is QUALIFY, not kill: the size rule stands within a (training, corpus-class) cell, but any cross-cell comparison must name all three.

## 3. Falsifier outcome — QUALIFY with a mechanism sketch, not a proof

| corpus | instruct f32 → i8 | coder f32 → i8 |
|---|---|---|
| arch | 199 → 196 (3 flips, KL 1.6e-3) | 199 → 190 (9 flips, KL 1.2e-2) |
| repetitive | 199 → 196 (3 flips, KL 4.4e-3) | 199 → 199 (0 flips, KL 4.1e-4) |

Instruct is flat across corpora (3,3); coder swings (9,0). Reading: code-tuning sharpens some features (better on clean repetitive text, worse where the corpus stresses them — exact mechanism unprobed). No fidelity owed beyond these runs (they ARE the fidelity); no thinking-set diff (no generation changes — ids recorded inline).

## 4. File anchors (current `master`, read but untouched)

- Model files: `Qwen2.5-Coder-1.5B-Instruct-Q4_K_M.gguf` vs `qwen2.5-1.5b-instruct-q4_k_m.gguf` (identical geometry/vocab verified at load: 28 layers, 1536 dim, 12/2 heads, vocab 151936).
- Grounds: i8 size ladder + diverse/arch i8 numbers (the rule being qualified), phi-i8 (architecture-vs-training confound precedent — this cycle isolates training).

## 5. What was done (pure measurement — no spike, no branch)

1. CREATE with the stated training-independence falsifier.
2. Gates: build 54/54; tests 141/141; `aikit.py check` 2/4 (test + secrets pass; pre-existing fails).
3. 8 fidelity runs + tokenizer-identity check (§3-4).
4. Scratch corpora deleted; `git status` clean, stash untouched.

## 6. Measured numbers — see §3 table. (All f32 arms 199/199; omitted for brevity — the deltas are the i8 columns.)

## 7. Full reimplementation recipe (verification — re-run to re-verify)

Re-run §1 (any code-heavy doc + any repetitive text, 200 tokens, same flags); expect coder-i8 to trail instruct-i8 on the code-heavy corpus and match-or-beat it on clean repetitive text. Revival trigger: an instruct-vs-coder i8 gap that does NOT vary by corpus in a 3+ corpus sweep — that would promote this from interaction to main effect (unlikely; the current 2-corpus split already shows the dependence).

## 8. Verdict and why

**QUALIFY: i8 robustness = f(size, training, corpus) jointly — the size-only rule is necessary but not sufficient.** Practically: keep the conditional keep as stated (it never claimed cross-training generality), but every future i8 verdict must name its training domain alongside size and corpus — starting with this report. Next-cycle candidates: the 1517 stage-2 harness remains the only live structural front; the human-gated port shelf stands.

## 9. Exact commands run and their results

- 8× `fidelity --model {coder|instruct}-1.5B ... [--kv i8]` on arch + repetitive corpora → §3 table.
- Tokenizer check (same prompt → identical promptids) → confirmed.
- build → 54/54; tests → 141/141; `aikit.py check` → 2/4.
- Scratch deleted; no branch, no edits; report committed; no push.

(End of file.)
