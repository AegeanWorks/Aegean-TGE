# INVENTED: Fidelity Corpus-Sensitivity Audit — CONFIRM (ordering holds on all 3 corpora; i8 magnitude spans ~10x, no inversion)

- **Idea**: Every priced gate in the library (f16 1.7e-6, i8 3.2e-4-class) was measured on SOME corpus — but is the tier ORDERING corpus-robust, or an artifact of one corpus choice? 3 corpora × 3 tiers on the harness reference config (0.5B-Q8_0, ctx 1024, 200 tokens): arch doc (in-repo, code-heavy), mixed10 (10 fresh topics, harbor→sourdough; temp file, deleted per discipline — reconstruction notes in §7), repetitive fox×150.
- **Cycle date**: 2026-11-06
- **Verdict**: CONFIRM — f32 ≥ f16 > i8 holds on all three corpora with zero inversions. Magnitudes vary as expected (i8 KL 4.4e-3 → 4.3e-2, ~10x; top-1 194 → 175/199): repetitive is kindest, arch harshest, mixed in between. No re-pricing needed — the bars were always corpus-class-specific, and this audit licenses continuing that practice.
- **Tree state at close**: `git status` clean, no diff, stash unchanged (only the pre-existing `leftover peer work shelved 2026-11-04`). No branch, no edits, no scratch files left (corpora deleted after runs; contents reproducible per §7). `master` ahead of `origin/master` by 115 after this report. Nothing pushed (human gate).

## 1. Mechanism (as run — no code)

`fidelity --model <0.5B-Q8_0> --corpus {arch|mixed10|rep2000} --context 1024 --tokens 200 --threads 6 [--kv f16|i8]`; record KL mean/worst + top-1 agreement. 9 runs total, all green exits.

## 2. Predicted gain — a robustness verdict, landed

Falsifier: any gross inversion (a tier beating a strictly finer one by >5 flips) REOPENS corpus-conditional pricing as a bug, not a feature. None observed — closest call is f16-vs-f32, which never drops a single token anywhere.

## 3. Falsifier outcome — CONFIRM on all three corpora

| corpus | f32 (KL / top-1) | f16 (KL / top-1) | i8 (KL / top-1) |
|---|---|---|---|
| arch (docs) | 1.1e-11 / 199 | 1.9e-6 / 199 | 4.3e-2 / 175 |
| mixed10 (fresh) | 1.3e-11 / 199 | 2.2e-6 / 199 | 2.5e-2 / 178 |
| repetitive fox | 2.1e-11 / 199 | 3.9e-7 / 199 | 4.4e-3 / 194 |

## 4. File anchors (current `master`, read but untouched)

- `apps/cli/main.cpp` (`command_fidelity` — the harness under audit; no changes proposed).
- Grounds: every priced gate in reports/ (all cite single-corpus numbers — this audit is their license renewal).

## 5. What was done (pure measurement — no spike, no branch)

1. CREATE with the stated inversion falsifier.
2. Gates: build 54/54; tests 141/141; `aikit.py check` 2/4 (test + secrets pass; pre-existing fails).
3. Built the two constructed corpora (rep2000: fox×150 file; mixed10: 10 fresh paragraphs), ran 3×3 (§3), deleted both.
4. No thinking-set diff (fidelity verdicts are the data; the harness prints no prose to compare).

## 6. Measured numbers — see §3 table. Notable: i8 spread ~10x by corpus (repetitive kindest — low-entropy activations quantize cleanly; arch harshest — code/special tokens flip more); f16 spread ~5x but never drops a token anywhere.

## 7. Full reimplementation recipe (verification — re-run to re-verify)

Rebuild any two contrasting corpora (one code-heavy, one prose-mixed, ~200 tokens each) + the fixed arch doc; run §1; expect the same ordering with i8 magnitudes spanning ~1 order of magnitude. Revival trigger: an inversion beyond noise, or an i8 corpus where 0.5B holds ≥98% (would reopen the size ladder, not just pricing).

## 8. Verdict and why

**CONFIRM: the pricing edifice is corpus-robust where it counts (ordering), corpus-sensitive where expected (magnitudes).** Keep citing per-corpus-class numbers exactly as the reports already do — this audit is the receipt that the practice is sound. Next-cycle candidates: the 1517 stage-2 harness remains the only live structural front; the human-gated port shelf stands.

## 9. Exact commands run and their results

- 9× `fidelity --model <0.5B-Q8_0> --corpus {arch|mixed10|rep2000} ... [--kv f16|i8]` → §3 table.
- build → 54/54; tests → 141/141; `aikit.py check` → 2/4.
- Scratch corpora deleted; no branch, no edits; report committed; no push.

(End of file.)
