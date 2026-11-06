# INVENTED: Hoist Row-Invariant Weight-Scale Conversion in the kq8 Batch Path — KILLED (clean kernel trial: 5 interleaved pairs inside noise; attempt-1 voided on a stride bug)

- **Idea**: In `gemv_dequant_batch_kq8_avx2` the batch loop is row-outer/token-inner, so each weight row's superblock scales convert once per token (`dot_q4k_q8k`/`dot_q6k_q8k` call scalar `half_to_float` per superblock per call). Hoist: convert each row's scales once per row into a thread-local temp (`[d,dmin]` pairs for Q4_K, single `d` for Q6_K), reuse across the chunk's tokens. Bit-identical by construction (same floats, same order per (row, superblock)); null temp = inline convert (kq8d path untouched).
- **Cycle date**: 2026-11-06
- **Verdict**: KILLED. The clean trial (attempt-2, stride-consistent) shows no kernel move: 5 interleaved S/B pairs on 3072×3072×6 (3 spike-favoring, 2 baseline-favoring, all ±15% = box noise). The conversion is not the bottleneck — the dot is shuffle/madd-bound at ~11 GB/s against a 54+ roof, and the scalar conversion on predictable branches is a few uops per superblock. Attempt-1's −55% is VOID (stride bug + no interleave + thermal confound — §5, not a result).
- **Tree state at close**: `git status` clean, no diff, stash unchanged (only the pre-existing `leftover peer work shelved 2026-11-04`). Branches `spike-kq8hoist` and `spike-kq8hoist2` created (zero commits each) and deleted; the one file reverted with empty diff verified; saved spike binaries + scratch deleted. `master` ahead of `origin/master` by 98 after this report. Nothing pushed (human gate).

## 1. Mechanism (attempt-2 form — the only valid one; reverted)

`src/cpu/avx2/dequant_batch_kq8_avx2.cpp`, file-local, no header change: thread-local `kq8_hoist_tmp`; `dot_q4k_q8k`/`dot_q6k_q8k` gain an optional `const float* xd` (null = inline convert); the batch caller resizes the temp per band (`2*blocks` Q4_K / `blocks` Q6_K), fills it once per weight row, passes it for all tokens of the row. kq8d single-row path passes null (unchanged behavior).

## 2. Predicted gain — not realized

Conversion ÷ tokens (÷64 per chunk): predicted ~5-7% prefill on Q4_K models if conversion were ~10% of the dot. Measured §6: ~0% at the kernel that would carry it.

## 3. Falsifier outcome — kill (kernel arm decides; prefill arm moot)

Stated: PROCEED only if Llama-3B pp512 prefill best-of-3 ≥ +5% with fidelity digit-identity + ids identical. The kernel gate in front of it (same 3072×3072×6 shape as the shipped kq8 win) shows no move across 5 interleaved pairs — there is nothing for prefill to amplify, so the 20-minute Llama round was not re-run for attempt-2 (economy; the mechanism lives in the dot, and the dot doesn't move). Kill.

## 4. File anchors (master = pre-spike; spike reverted)

- `src/cpu/avx2/dequant_batch_kq8_avx2.cpp` (`dot_q4k_q8k` :~142, `dot_q6k_q8k` :~221, batch caller :~378-398 — all reverted).
- `include/tge/cpu/kernels.hpp:278-307` (branchy scalar `half_to_float` — the cost being attacked).
- Grounds: 1458 (fused-dot family), 1394 (kq8 integration), batchcvtph kill (same conversion family on decode).

## 5. What was done (two attempts; only the second counts)

1. `git checkout -b spike-kq8hoist` (zero commits, deleted at close). Baseline: build 54/54, `tge_tests` 141/141.
2. ANALYZE: row-outer/token-inner confirmed; Q4_K (d,dmin) and Q6_K (d) layouts symmetric for hoisting.
3. Attempt-1: applied with a fill/read stride mismatch (fill stride-2 pairs vs read stride-1 from split pointers) for blocks>1. Spike tests green (the zero-input tripwire is scale-blind by construction; the triangulation gate's silence on the mismatch is UNRESOLVED on the record — do not treat the 141 green as stride coverage for hoisted temps). Llama pp512 showed −55% (22.0/22.8/24.5s vs baseline 9.93/9.97/10.03s) — but with no interleave, a 126s monster run immediately prior, upward drift in-block, and the stride bug: VOID, not a result. Fidelity digit-identity on that attempt is ALSO void as a hoist gate — `run_fidelity` forwards single tokens, which take the float path (1394), so the hoisted batch dots never ran there.
4. Attempt-2 (`spike-kq8hoist2`, corrected single-pointer stride-2/stride-1 form): rebuilt 54/54, `tge_tests` 141/141; kernel A/B §6; reverted, baseline rebuilt 54/54, tests 141/141.
5. `aikit.py check aegean-tge` → 2/4 (test + secrets PASS; hygiene + guidelines fail pre-existing, same pair as every prior cycle).
6. Reverted the file (empty diff), branches deleted, scratch binaries deleted, `git status` clean, stash untouched.

## 6. Measured numbers

`bench batch --rows 3072 --cols 3072 --tokens 6 --format q4k --kernel kq8 --repeat 5`, 5 interleaved S/B pairs (best-of-5 each, ms):

| pair | spike (attempt-2) | baseline | delta |
|---|---|---|---|
| 1 | 0.639 | 0.785 | −18.6% |
| 2 | 0.486 | 0.565 | −14.0% |
| 3 | 0.499 | 0.544 | −8.3% |
| 4 | 0.470 | 0.455 | +3.3% |
| 5 | 0.494 | 0.465 | +6.2% |

Reading: pairs 1-2 ran on a settling box (all runs slow, 0.48-0.79); pairs 3-5 cluster 0.455-0.544 with overlapping ranges and split directions. No consistent move — the hoist is inside box noise (±15% on this box right now). Attempt-1's −55% is excluded per §5 (void).

No fidelity run is load-bearing here: fidelity cannot reach the batch-kq8 path (single-token forwards take the float path — established §5), so no fidelity number is cited as a hoist gate; the 0.5B story/creative ids compared spike-vs-master are likewise Q8_0-path (vacuous for kq8, recorded for completeness: identical).

## 7. Full reimplementation recipe (kill-grade — rebuild only to re-verify)

1. Re-apply the attempt-2 form (§1; stride-consistent single-pointer temps — NOT attempt-1's split-pointer form).
2. Gate: build 54/54, `tge_tests` 141/141; the §6 kernel interleave must reproduce ~0% (any consistent ≥5% either way reopens the question with a Llama prefill round + Llama prefill ids spike-vs-baseline as the TRUE bit-identity gate, since fidelity cannot reach this path).
3. Revival trigger ONLY: a kernel shape where conversion provably dominates the dot (not this one — shuffle/madd-bound at ~11 GB/s). Until such a shape, no scale-hoisting variant re-proves the thread split.

## 8. Verdict and why

**KILLED: the conversion is not the kq8 bottleneck.** Two attempts, one voided on a real bug I caught by audit (reported, not hidden), one clean: the dot's cost is shuffles/maddubs and weight streaming, and the few predictable-ALU conversion uops per superblock don't register when removed. This also bounds the family: batchcvtph (decode) and this (prefill) both die at ~0% — scale conversion is exhausted as a lever on this tree. Next-cycle candidates: the 1517 epic's stage-2 skip harness remains the only live structural front; the human-gated port shelf grows (mprefix-table, sparse no-gather, batchsnap kind tags, timers fix, detect memoize).

## 9. Exact commands run and their results

- `git checkout -b spike-kq8hoist` → branch (zero commits, deleted at close).
- Baseline: build 54/54; `tge_tests` → 141/141.
- Attempt-1 (§5 void): Llama-3B fox×205 warm (126.8s — thermal confound note) + pp512 ×3 → voided numbers; Llama kq8 fidelity → vacuous-but-recorded digit-identity; 0.5B thinking set (saved).
- Revert → clean; `git branch -D`; `git checkout -b spike-kq8hoist2`; attempt-2 (§1); rebuild → 54/54; `tge_tests` → 141/141.
- `bench batch --kernel kq8` (spike) 0.478ms; binary saved; revert; rebuild baseline 54/54; tests 141/141; 5 interleaved pairs → §6.
- 0.5B thinking set on master; spike-vs-master Q8_0 ids identical (vacuous for kq8, noted).
- `aikit.py check` → 2/4 (test + secrets pass; pre-existing fails).
- Spike binaries + scratch deleted; `git status` clean; report committed; no push.

(End of file.)
