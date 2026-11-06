# INVENTED (coder-i8 follow-up): f16 Training Robustness — CONFIRM (auto's default tier is training-proof: 199/199 both models, KL within 2.2x)

- **Idea**: The coder-i8 cycle proved i8 interacts with training domain (arch: 9 vs 3 flips). kv-auto's DEFAULT long-context tier is f16 — if f16 were ALSO training-sensitive, auto would need a training gate. Same A/B, f16 instead of i8: Coder vs instruct on arch, f32 sanity implied by prior runs (both 199/199 clean).
- **Cycle date**: 2026-11-06
- **Verdict**: CONFIRM — f16 is training-robust: Coder KL 2.582e-06 / 199-199 vs Instruct KL 1.204e-06 / 199-199. Same order, zero flips either way. The training interaction is i8-specific (coarse quantizer meets sharp code-tuned features); f16's finer grid absorbs the same differences without a single flip. kv-auto needs no training gate.
- **Tree state at close**: `git status` clean, no diff, stash unchanged (only the pre-existing `leftover peer work shelved 2026-11-04`). No branch, no edits. `master` ahead of `origin/master` by 116 after this report. Nothing pushed (human gate).

## 1. Mechanism (as run — no code)

`fidelity --model {Coder|Instruct}-1.5B-Q4_K_M --corpus docs/architecture.md --context 512 --tokens 200 --threads 6 --gpu off --kv f16`; top-1 + KL vs the 1.7e-6 f16 price class.

## 2. Predicted gain — a safety gate, passed

Falsifier: any flip gap >1 OR KL >3x between models → REOPEN (auto needs a training gate). Measured: 0 vs 0 flips, KL 2.1x — auto is safe.

## 3. Falsifier outcome — CONFIRM

Both models hold 199/199 with KL in the same class (2.6e-6 vs 1.2e-6, both near the 1.7e-6 precedent). The coder-i8 finding stays fenced to i8: coarse quantization interacts with training; fine quantization does not (at this resolution).

## 4. File anchors (current `master`, read but untouched)

- Same model files + harness as the coder-i8 cycle (the direct comparator).
- Grounds: coder-i8 qualify (the interaction this clears for f16), 2048-closure f16 price (the class that holds).

## 5. What was done (pure measurement — no spike, no branch)

1. CREATE with the stated auto-safety falsifier.
2. Gates: build 54/54; tests 141/141; `aikit.py check` 2/4 (test + secrets pass; pre-existing fails).
3. Two fidelity runs (§3).
4. No thinking-set diff (no numerics changed by ME — flag-selected priced path; the verdicts ARE the numbers).

## 6. Measured numbers (arch corpus, ctx 512, 199 pos, threads 6, f16 KV)

| model | KL mean | worst | top-1 |
|---|---|---|---|
| Coder-1.5B | 2.582e-06 | 4.157e-05 | 199/199 |
| Instruct-1.5B | 1.204e-06 | 1.426e-05 | 199/199 |

## 7. Full reimplementation recipe (verification — re-run to re-verify)

Re-run §1; expect identical top-1 and same-order KL on any long-form corpus. Revival trigger: an f16 flip gap >1 between same-size different-training models on the same corpus — that would reopen auto with a training gate (not observed here).

## 8. Verdict and why

**CONFIRM: kv-auto's f16 default is training-proof, and the training interaction is i8-specific.** This closes the coder thread cleanly: i8 needs its domain named (done last cycle), f16 needs nothing (done this cycle), and auto's tier order (f32 short / f16 long) stands unqualified across everything measured. Next-cycle candidates: the 1517 stage-2 harness remains the only live structural front; the human-gated port shelf stands.

## 9. Exact commands run and their results

- 2× `fidelity --model {coder|instruct}-1.5B ... --kv f16` → §6 table.
- build → 54/54; tests → 141/141; `aikit.py check` → 2/4.
- No branch, no edits; report committed; no push.

(End of file.)
