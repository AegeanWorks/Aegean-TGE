# INVENTED: i8 Flip-Location Probe at 2048 Tokens — FRONT-LOADED, no compounding (2 flips at positions 26/38, 98% of KL mass in Q0)

- **Idea**: Invented follow-up to the i8 long-context verification (`idea-invented-i8-longctx-confirmed.md`): locate WHERE the 2/1999 flips and the KL mass live. Stated falsifiers: flips/KL-mass concentrated in the last quarter (positions ≥1500) → error compounds with context (conditional keep with context caveat); spread uniform or early → clean confirm (no length dependence).
- **Cycle date**: 2026-11-06
- **Verdict**: CLEAN CONFIRM with a stronger shape than asked — error is FRONT-LOADED, not compounding: both flips at positions 26 and 38, 98% of KL mass in Q0 (2.556 of 2.605 total), all top-20 KL positions within 3–89. If anything, long context is safer than short for this tier. Caveat recorded: the corpus repeats one paragraph, so early-vs-late is confounded with novel-vs-repeated content — a non-repetitive long corpus could disentangle this (open, not blocking).
- **Tree state at close**: `git status` shows my report commit plus PRE-EXISTING, UNCOMMITTED peer spike work (`run_spike_mprefix` multi-prefix CoW, +134/−0 and +128/−0, present since two cycles ago, no active process, board silent). My spike (env-gated per-position recorder, 3 hunks) was reverted by exact inverse edits and the resulting `git diff` verified BYTE-IDENTICAL (25,528 bytes) to the peer-baseline patch saved before I touched anything. Branch deleted. `master` ahead of `origin/master` by 41 after this report (reports only). Nothing pushed (human gate).

## 1. Mechanism

Env-gated (`TGE_FID_POS=1`) per-position recorder inside the fidelity loop:
KL + flip flag per position; post-run print of KL mass by quartile, flips by
quartile, flip positions, and top-20 KL positions. One i8 run at 2048 ctx
(1999 positions), 0.5B Q8_0, threads 1, `--gpu off`. No behavior change when
unset (two small vectors, gate-checked).

## 2. Predicted gain — none (diagnostic)

Pass bars: flips/KL spread uniform-or-early → length-independent keep-grade.
Fail bars (not tripped): late-quarter concentration → compounding error,
conditional keep with context caveat + mitigation recipe.

## 3. Falsifier (stated) — passes on the early side

Flips at 26, 38 (both Q0); Q0 holds 2.556/2.605 = 98.1% of KL mass;
Q1–Q3 combined hold 0.049 (1.9%). The feared pattern (late blowup) is
absent — inverted, in fact.

## 4. File anchors (current `master`, untouched except by the reverted spike)

- `apps/cli/main.cpp` (`run_fidelity` loop, per-position KL/top-1).
- Spike (reverted): 3 env-gated hunks (decl, record, report); verified gone
  by byte-identical diff vs pre-spike baseline.

## 5. What was done (temporary spike, fully reverted with verification)

1. Saved peer dirt to a patch file (temp dir, outside repo); created
   `spike-fidpos` (carried peer dirt + mine, never pushed).
2. Added the gated recorder (3 hunks, main.cpp only).
3. Build clean; `ctest` pass; `tge_tests` 141/141.
4. `TGE_FID_POS=1 fidelity --kv i8` at 2048 ctx → §6 (364 s; mean KL
   1.303e-3 reproduces the prior run digit-for-digit — deterministic).
5. Inverse-edited all 3 hunks; `git diff` byte-identical to saved baseline.
6. Deleted branch; rebuilt pristine; re-ran `ctest` + 141/141.
7. `aikit.py check` 2/4 (test + secrets pass; hygiene + guidelines fail
   pre-existing).
8. No thinking-set diff owed: diagnostic read-only when set, zero behavior
   change when unset; same binary throughout for all gated runs.

## 6. Measured numbers (i8, Qwen2.5-0.5B-Q8_0, 1999 positions, ctx 2048)

| slice | KL mass | flips | top-20 KL positions |
|---|---|---|---|
| Q0 (0–499) | 2.556 (98.1%) | 2 (pos 26, 38) | all 20 in pos 3–89 (worst 0.69 at 22) |
| Q1 (500–999) | 0.0220 | 0 | none |
| Q2 (1000–1499) | 0.0128 | 0 | none |
| Q3 (1500–1998) | 0.0146 | 0 | none |

Mean KL 1.303e-3 reproduces the prior run exactly (deterministic harness +
model). Interpretation: error concentrates where content is novel (first
paragraph), then the cache "settles" — consistent with quantization noise
dominating thin-context attention and washing out over redundant history.
Honest confound: the corpus repeats, so novelty and earliness coincide;
a non-repetitive long corpus would separate them. Either reading kills the
compounding hypothesis (late positions are the cleanest, not the dirtiest).

## 7. Full reimplementation recipe (keep-grade diagnostic, not a product)

1. `run_fidelity`: `dump_pos` env gate; `pos_flip` + `pos_kl` vectors;
   record per position inside the KL loop; post-loop print quartile KL
   mass, flips by quartile, flip positions, top-20 KL (selection scan,
   no new includes).
2. Gate: `TGE_FID_POS=1 fidelity --kv i8` on the 86→1999-pos corpus —
   expect flips ⊆ Q0, Q0 mass ≥ 95%, top-20 all < 100.
3. Revert by exact inverse edits; verify `git diff` byte-identical to the
   pre-spike tree before deleting the branch.

## 8. Verdict and why

**CLEAN CONFIRM, stronger than the bar.** The open question from the
long-context verification is answered in the favorable direction: i8 error
does not compound with context length on this workload — it concentrates
in the first ~90 positions and vanishes thereafter (1997/1999 with zero
late flips). The keep-grade for i8 needs no context caveat from this
evidence. The novelty-vs-earliness confound is recorded as the one open
thread, explicitly not a blocker.

## 9. Exact commands run and their results

- Peer-baseline `git diff` saved (310 lines); `git checkout -b
  spike-fidpos`.
- 3-hunk spike; build 54/54; `ctest` pass; 141/141.
- `TGE_FID_POS=1 fidelity --kv i8 ... --tokens 2000` → §6 table (364 s;
  mean reproduces prior run exactly).
- Inverse 3 edits; `git diff` byte-identical to saved baseline (25,528 B).
- `git checkout master`; `git branch -D spike-fidpos`; rebuild; `ctest`
  pass; 141/141; `aikit.py check` 2/4 (pre-existing fails).
- No push.
