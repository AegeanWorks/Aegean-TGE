# IDEA-1383: Low-bit KV quants + multi-path speculative framework — KILLED WITHOUT SPIKE (falsifier tripped by the int4 verdict; draft half dead three times over)

- **Idea ID**: IDEA-1383 (dossier `scout-2026-11-03-kilo-4target.md`: beellama.cpp KVarN low-bit KV + precision tail; ngram-cache, draft-mtp, draft-dflash, synth-rate calibration. Dossier's own scoping: "MTP paths skipped (Qwen2 lacks blocks)". Falsifier: int4 KL >= 1e-3 with top-1 flips, or decode regresses vs `--kv i8`.)
- **Cycle date**: 2026-11-06
- **Verdict**: KILLED on the record, no spike built (dead-idea rule: every operative half is already measured dead on this tree). No branch created, no code touched, nothing to revert.
- **Tree state at close**: `git status` clean (no diff), stash unchanged (pre-existing peer work only), `master` ahead of `origin/master` by 24 after this report. Nothing pushed (human gate).

## 1. Mechanism (as dossiered, scoped by the dossier itself)

(a) Sub-i8 KV quants (int4/int2 tails with a precision tail) at/near i8-level
fidelity with smaller footprint. (b) Multi-path speculative framework:
ngram-cache second drafter beside prompt-lookup (draft-mtp/dflash paths
explicitly out of scope — no MTP blocks in Qwen2).

## 2. Predicted gain — and why each half is already answered

- (a) int4: measured LAST CYCLE on this tree with an independent implementation
  (report `6b0c84b`): KL **4.641e-3** vs the 1e-3 bar, top-1 **83/86** (3 flips
  vs ~1 allowed). int2 (report `1fc2869`): KL 0.13, 75/86. The dossier's own
  falsifier (int4 KL >= 1e-3 with flips) trips on the int4 verdict run — the
  idea kills itself by its stated bar. No precision-tail variant can recover
  a 4.6x KL gap (tails help recency, the error is structural V heavy-tail per
  stage-1).
- (b) ngram-cache second drafter: the learned-ngram family is measured dead
  three times over — exact trie 3.45→3.18/step (`59e22f6`), rolling-hash killed
  analytic as monotone-weaker (`5e19e9f`), ngram+KV superseded as shipped
  (`031cb29`: spans/bigram/gapped/SKIP saturate the family). A "second drafter
  beside prompt-lookup" is what 1387 spiked and what lost.
- MTP/dflash/synth-rate: out of scope per the dossier (no blocks); calibration
  has no carrier (no calibration flags exist — same finding as the 1385
  auto-calibration note).

## 3. Falsifier (from the dossier) — trips verbatim

"int4 KL >= 1e-3 with top-1 flips": measured 4.641e-3 with 3 flips (worst KL
3.4e-2, max|diff| 1.18). The decode arm is moot — no AVX2 path is owed for a
format that fails numerics.

## 4. File anchors (current `master`, read but untouched)

- `reports/idea-1385-int4kv-killed.md` (`6b0c84b`) — the verdict run this kill
  rests on (env-gated tier, mirror-read vehicle, 86-pos Llama-3B fidelity).
- `reports/idea-1516-kivi-stage2a-killed.md` (`1fc2869`) — int2 bound.
- `reports/idea-1387-trie-drafter-spike-killed.md` (`59e22f6`),
  `idea-1502-rolling-hash-killed-analytic.md` (`5e19e9f`),
  `idea-1542-ngram-kv-superseded.md` (`031cb29`) — the draft family.
- `src/models/qwen2.cpp` — no draft-model KV, no MTP blocks, no calibration
  surface (confirmed across the 1526/1502 cycles).

## 5. What was done (ANALYZE only — no spike per the dead-idea rule)

1. Re-read the dossier section and mapped each operative half to a measured
   verdict on today's tree (§2). Nothing unmeasured remains in scope.
2. Gates for the record: build clean, `ctest` pass, `aikit.py check` 2/4
   (test + secrets pass; hygiene + guidelines fail pre-existing).
3. No thinking-set diff owed: same binary throughout, zero behavior change.

## 6. Measured numbers (cited, not re-run — same tree, prior cycles)

| claim | measurement | bar | report |
|---|---|---|---|
| int4 KV KL | 4.641e-3 | ≤ 1e-3 | `6b0c84b` |
| int4 top-1 | 83/86 (3 flips) | ≥ 98.6% | `6b0c84b` |
| int2 KV KL | 1.285e-1 | ≤ 1e-3 | `1fc2869` |
| learned ngram acceptance (code) | 3.45→3.18 (deterministic) | beat prompt-lookup | `59e22f6` |

## 7. Reimplementation recipe (deliberately NOT recommended — kill-grade)

For the record: an int4/int2 KV tier would repeat the stage-2a vehicle
(env-gated tier, mirror-read fidelity pricing, no AVX2 until KL passes);
a second ngram drafter would repeat the 1387 main.cpp-only spike
(last-resort draft position, acceptance gate). Both vehicles exist in the
reports above; neither outcome would change — the numbers that kill this
idea ARE those vehicles' outputs.

## 8. Verdict and why

**KILLED without spike.** This is the loop working as designed: 1383's
falsifier names the exact measurement that kills it, and the previous cycle
ran that measurement (4.6x over KL, 3 flips). The draft half died three
cycles earlier; the MTP half was scoped out by the dossier itself. Revisit
only if a new sub-8-bit grouping passes tensor evidence first (the stage-1
pattern) — engine work before tensor evidence is what stages 1/2a exist to
prevent.

## 9. Exact commands run and their results

- Build + `ctest` → pass (141/141 on master).
- `aikit.py check` → 2/4 (pre-existing fails). No branch, no edits, no push.
