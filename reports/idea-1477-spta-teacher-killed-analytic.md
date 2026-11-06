# IDEA-1477: Prefix-Conditioned Teacher Adaptation for Draft Models (SPTA) — KILLED WITHOUT SPIKE (training idea proposed to an inference-only engine; acceptance already saturated without it)

- **Idea ID**: IDEA-1477 (dossier `arxiv-sweep-2026-11-01-03.md`: arXiv:2611.03133 §§3–5/7/9 — prefix-conditioned RLVR then frozen-teacher distillation, matched-optimization margins rescuing stalled pairs; claim: adapting the teacher on random student prefixes before distillation yields stronger students than task-level teacher training under matched cost; distinct from generic draft pairing — the load-bearing piece is prefix conditioning for acceptance, not pairing)
- **Cycle date**: 2026-11-06
- **Verdict**: KILLED on the record, no spike built (dead-idea rule: the mechanism is a training loop and the engine has no training carriers — no gradients, no optimizer, no distillation recipe, no teacher/student machinery, read-only mmap weights by design; and the problem it solves is already closed — speculative acceptance multiplies tokens wherever repetition exists with the current und distilled drafter). No branch, no code touched, nothing to revert.
- **Tree state at close**: `git status` clean (no diff, no untracked), stash unchanged (only the pre-existing `leftover peer work shelved 2026-11-04`, untouched), `master` ahead of `origin/master` by 61 before this report. Nothing pushed (human gate).

## 1. Mechanism (as dossiered)

Distillation recipe, not engine code: run prefix-conditioned RLVR on the teacher (random student prefixes), then frozen-teacher distillation with matched-optimization margins; mmap-first posture kept (no base-weight shifts). Value: small CPU draft models that continue чужд prefixes, lifting speculative acceptance.

## 2. Predicted gain — and why it cannot land here

Paper-side: stronger students than task-level teacher training under matched cost. Against this tree both halves of the transfer fail. The training half has no carrier: adapting a teacher means gradient updates + weight writes, and the engine neither trains nor writes weights (1386 precedent: weights arrive pre-quantized, never produced in-tree; the only train-adjacent strings in `src/`+`include/` are `n_ctx_train`/YaRN metadata reads — header fields, not updates). The acceptance half is already saturated without distillation: the acceptance map (§51, Llama-3B, 34 runs) shows 1.0–5.0 accepted tokens/step wherever repetition exists, and LogitSpec/frequency-abstention stays unopened precisely because "no acceptance deficit exists on record" — the drafter's ceiling is prompt repetition, not prefix-continuation skill. A distilled drafter would multiply the same repetition-bound tokens through the same verifier.

## 3. Falsifier (from the dossier) — fails at the posture, before any run

"Acceptance rate unchanged despite recovery gains" — no recovery gains can exist here (nothing to train, nothing to distill into). "Clean-prompt capability regresses (violates no-retraining posture)" — the mechanism violates the posture by construction (teacher RLVR rewrites teacher weights; the posture is read-only weights), so this arm fires analytically. "Smallest margin vanishes under multi-seed intervals" — no seeds to run: no training loop exists to seed.

## 4. File anchors (current `master`, read but untouched)

- `src/` + `include/` grep for `train|distill|RLVR|gradient|backward|optimizer|finetune` → zero code matches (only `n_ctx_train` YaRN context metadata + a comment mirroring it — reads, not updates).
- `src/models/gguf.cpp` load path (mmap + read-only tensors — the posture the mechanism would need to break) + `reports/idea-1386-ptq-export-killed-analytic.md` (read-only precedent: "weights arrive pre-quantized, the engine never produces a weight file").
- Speculation stack (the saturated alternative): SKIP-locked drafter + fused verifier (`apps/cli/main.cpp`, `src/models/qwen2.cpp`), CoW cross-sequence prefix, measurement-log §51 (acceptance map) + 2026-11-05 SKIP-locked/fused/CoW lock-in.

## 5. What was done (ANALYZE only — no spike per the dead-idea rule)

1. Read the dossier body against the current tree; confirmed the mechanism is a training recipe with an engine-external teacher loop.
2. Carrier greps (§4): training/distillation zero (code); speculation-without-distillation saturated on record.
3. Gates for the record on the unmodified tree: `cmake --build --preset release` clean (54/54); `ctest --preset release` pass; `tge_tests.exe` 141/141; `python aikit.py check aegean-tge` → secrets pass, hygiene + guidelines fail pre-existing, test gate flaked once at ~3 s (the §56/63 race signature — sixth occurrence, no code behind it) with the immediate direct `scripts\test.cmd` rerun rebuilding everything and passing. ctest + 141/141 green in both attempts.
4. No fidelity run owed (no numerics to price — nothing runs); no bench A/B owed (no draft model to pair); no thinking-set diff owed (same binary).

## 6. Measured numbers — none owed (posture kill, saturated alternative)

No benchmark belongs to this kill: a training recipe cannot be A/B-tested by an inference engine, and the acceptance it targets is already measured at ceiling without it (§51: 1.0–5.0/step; LogitSpec unopened for lack of deficit). Gate record: build 54/54, ctest pass, 141/141, aikit 1 + 2 pre-existing fails + 1 flaked-then-green test gate (§5.3).

## 7. Full reimplementation recipe (deliberately NOT recommended — revival is a training project, never an engine spike)

Do not build a drafter-distillation path in TGE: teacher RLVR + distillation is an offline training program producing weight files, and TGE never produces weight files (1386 stands). Revive only as an external program (calibrate → distill → ship a GGUF), gated on the dossier's own falsifier — acceptance gain over the current und distilled drafter at matched cost on the §51 map, with no clean-prompt regression — and only then would the engine see a new file to load, not new code to write. To re-verify this kill: re-run the §4 training greps (still zero) and confirm the §51 no-deficit standing (acceptance still repetition-bound).

## 8. Verdict and why

**KILLED without spike.** A training-loop idea with no training carriers, targeting an acceptance ceiling already measured without it, against a posture (read-only weights) its own falsifier names as a firing condition. The loop's honest output is one line: nothing to train with, nothing to distill into, nothing to gain.

## 9. Exact commands run and their results

- Dossier read + carrier greps (§4: training/distillation zero code).
- `cmake --build --preset release` → 54/54; `ctest --preset release` → pass; `tge_tests.exe` → 141/141.
- `python aikit.py check aegean-tge` → test gate flaked once (~3 s race), direct `scripts\test.cmd` rerun → pass; secrets pass; hygiene + guidelines fail pre-existing. No branch, no edits, no bench, no fidelity, no push.

(End of file.)
