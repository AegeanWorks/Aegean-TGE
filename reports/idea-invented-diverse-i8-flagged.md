# INVENTED: i8 KV on Diverse Long-Form Text — CONDITIONAL keep (99.3% on Llama-3B; 94.2% on 0.5B with spread-out flips)

- **Idea**: Invented follow-up ("measure less-assumed"): re-price the kept i8 tier on a NON-repetitive 431-position corpus (10 mixed topics), both models, after the flip-locator showed early-concentration on repetitive text. Stated falsifier: KL > 1e-3 or flips worse than ~1.4% → conditional keep with a usage caveat (not a ship-blocker: the repetitive-context verdicts stand).
- **Cycle date**: 2026-11-06
- **Verdict**: CONDITIONAL KEEP — the falsifier tripped, and the condition is now precise. Llama-3B (reference): KL 1.21e-4, 428/431 (99.3%) — holds, keep stands. Qwen2.5-0.5B: KL 2.22e-2, 406/431 (94.2%) with flips spread across ALL quartiles (4/3/8/10) — fails the bar, uncorrelated with position. Combined reading: i8 degradation tracks per-position difficulty, not length; small models on diverse content are the weak corner. Recommendation recorded: prefer f32/f16 KV for small models on diverse long-form; Llama-3B-class and up: i8 stands as-is.
- **Tree state at close**: `git status` shows my report commit plus the PRE-EXISTING peer spike (`run_spike_mprefix`, +134/+128, no active process, board silent). Mid-cycle I made a real mistake: a bare `git checkout -- apps/cli/main.cpp` (instead of the planned inverse edits) wiped the peer's main.cpp spike ALONG WITH mine. Caught immediately by the byte-compare gate (MISMATCH vs saved baseline). Recovered fully: split the saved baseline patch, dry-ran, re-applied only the peer's hunks, re-verified byte-identical (RESTORED-IDENTICAL). Lesson recorded: never bare-checkout a file containing known peer dirt — inverse-edit, then verify against the saved baseline BEFORE deleting the branch. Branch deleted. `master` ahead of `origin/master` by 43 after this report (reports only). Nothing pushed (human gate).

## 1. Mechanism

Same fidelity harness, new corpus: 10 mixed-topic paragraphs (~12 KB, ~431
usable positions at ctx 512), deterministic. Runs: f32 + i8 on 0.5B-Q8_0
(f32 sanity: KL 9.6e-12, 431/431), i8 on Llama-3B-Q4_K_M, plus a re-applied
flip-locator spike (env-gated per-position KL/flip print, reverted with
verification) for the 0.5B i8 run.

## 2. Predicted gain — none (diagnostic)

Pass bars: i8 KL ≤ 1e-3 with flips ≤ ~1.4%. Trip on EITHER arm → conditional
keep with a usage caveat naming the weak corner.

## 3. Falsifier (stated) — tripped on 0.5B, passed on Llama-3B

- 0.5B i8 diverse: KL 2.224e-2 (22x over), flips 25/431 = 5.8% (4x over).
- Llama-3B i8 diverse: KL 1.211e-4 (8x under), flips 3/431 = 0.7% (2x under).
- Flip locator (0.5B): Q0=4, Q1=3, Q2=8, Q3=10; KL mass Q0 35.6%, Q1 18.6%, Q2 30.8%, Q3 15.0%; top-20 spread pos 15–401. No early concentration (contra the repetitive corpus: 98% in Q0) — difficulty-driven, not length-driven.

## 4. File anchors (current `master`, untouched except by the verified-reverted spike)

- Corpus: temp `fidelity_diverse.txt` (not committed — method artifact).
- Harness: `apps/cli/main.cpp` (`run_fidelity`); KV: `store_kv_q8`.
- Precedents: `idea-invented-i8-longctx-confirmed.md` (repetitive: 2 early
  flips), `idea-invented-flip-locator-frontloaded.md` (spike recipe reused).

## 5. What was done (temporary spike, reverted with verification)

1. Saved peer diff to temp; branched; added env-gated flip locator (3 hunks,
   main.cpp only).
2. Build clean; `ctest` pass; 141/141.
3. f32 baseline + i8 verdict (0.5B) on the diverse corpus; locator run
   (deterministic reproduce: same KL/top-1 as the plain run); Llama-3B i8
   verdict run.
4. Reverted via inverse edits — then caught MY OWN checkout mistake wiping
   the peer's main.cpp spike; restored peer hunks from the saved baseline
   (dry-run first) and re-verified byte-identical full diff.
5. Deleted branch; rebuilt pristine; `ctest` + 141/141.
6. `aikit.py check` 2/4 (test + secrets pass; hygiene + guidelines fail
   pre-existing).
7. Thinking-set N/A beyond runs (no behavior change; same binary paths).

## 6. Measured numbers

| run (ctx 512, 1T, `--gpu off`) | KL mean | worst | max \|diff\| | top-1 | time |
|---|---|---|---|---|---|
| 0.5B f32 diverse (oracle sanity) | 9.6e-12 | 1.4e-10 | 1.1e-4 | 431/431 | 75 s |
| 0.5B i8 diverse | **2.224e-2** | 0.57 | 4.6 | **406/431** | 69 s |
| 0.5B i8 diverse + locator | 2.224e-2 (identical) | 0.57 | 4.6 | 406/431 | 60 s |
| Llama-3B i8 diverse | **1.211e-4** | 1.1e-3 | 0.47 | **428/431** | 569 s |

Reference points: repetitive-corpus i8 was 3.2e-4/31-31 (0.5B) and 1.3e-3/1997-1999 (Llama-3B). The diverse corpus is ~70x (0.5B) harsher in KL but only mildly harsher on Llama-3B (same 1e-4 scale).

## 7. Full reimplementation recipe (keep-grade diagnostic + conditional rule)

1. Corpus: ~10 mixed-topic paragraphs, ~12 KB, natural prose (temp file).
2. `fidelity --model <0.5B-Q8_0|Llama-3B-Q4_K_M> --corpus <file> --context
   512 --tokens 600 --threads 1 --gpu off [--kv i8]`; expect §6 numbers.
3. Optional locator: 3-hunk env-gated per-position print (see flip-locator
   report); verify revert by byte-compare against a pre-spike saved diff.
4. Usage rule (the deliverable): i8 KV unconditional for Llama-3B-class;
   for sub-1B models on diverse long-form content prefer f32/f16, or accept
   ~94% teacher-forced agreement. No engine change.

## 8. Verdict and why

**CONDITIONAL KEEP.** The i8 tier's keep-grade survives where it matters
(reference model, all corpora: 99.3–100%) and carries a now-precise caveat
where it doesn't (sub-1B + diverse long-form: 94.2%, flips spread, no
length-dependence to exploit, no single passage to blame). The alternative
readings were checked and closed: length compounding is refuted (flips
spread, mass not late-weighted — opposite of the repetitive-corpus pattern,
so the driver is content difficulty, confirmed on two corpora); the
measurement itself is deterministic (bit-identical rerun).
Model-size dependence (≈180x KL gap 0.5B→3B) is the finding: redundancy,
not context handling, is what protects the tier.

## 9. Exact commands run and their results

- Peer baseline saved; branch created.
- Diverse corpus written (10 paragraphs, 12,200 bytes).
- Build 54/54; tests pass; 141/141.
- f32 diverse (0.5B): KL 9.6e-12, 431/431.
- i8 diverse (0.5B): KL 2.224e-2, 406/431; locator rerun identical +
  quartile/flip table (§3).
- i8 diverse (Llama-3B): KL 1.211e-4, 428/431 (569 s).
- Inverse-edit revert; byte-compare caught MY checkout mistake wiping peer
  main.cpp; restored peer hunks from saved baseline (dry-run first);
  re-verified byte-identical; deleted branch; rebuilt; `ctest` pass.
- `aikit.py check` → 2/4 (pre-existing fails). Reports-only commit; no push.
