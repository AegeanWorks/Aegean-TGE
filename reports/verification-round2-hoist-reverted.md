# HOIST RE-VERIFY — FAILED AGAIN, STAYS REVERTED (3/3 losses, stable baseline)

- **Date**: 2026-11-07 (second verification round; see `verification-round-hoist-reverted.md` for round 1).
- **Verdict**: REVERT STANDS. The exact port code (extracted verbatim from `cb2a330`, never retyped) loses 3/3 interleaved pairs against the pre-hoist baseline on today's tree. The re-application was fully reverted (empty diff verified), rebuilt 54/54, tests 145/145. Nothing pushed (human gate).
- **Tree state at close**: clean, no diff, stash untouched. `master` ahead of `origin/master` by 166 after this report.

## 1. What was done

Re-applied `cb2a330`'s function + dispatch + include verbatim (function bytes extracted from the commit blob, not retyped; dispatch + include by hand). Build 54/54, tests 145/145. Interleaved A/B P,B,B,P,P,B (0.5B-Q8_0, 3900-id prompt, ctx 8192, threads 10, `--kv i8`, n=12), pre-all-ports baseline binary `tge-baseline.exe` vs re-applied port binary.

## 2. Measured numbers (decode attn ms, ids identical every run)

| pair | baseline | port | delta |
|---|---|---|---|
| 1 (P,B) | 82.1 | 94.7 | +15.4% |
| 2 (B,P) | 84.9 | 96.8 | +14.0% |
| 3 (P,B*) | 81.7 | 104.5 | +27.9% (*port run showed all-phase inflation = transient contention) |

Baseline runs: 82.1 / 84.9 / 81.7 — rock-stable across the whole session. Port runs: 94.7 / 96.8 / 104.5 — never near baseline, both orders. This is not box noise (the baseline distribution does not overlap the port distribution); it is a consistent ~15% regression of this code on this tree state.

## 3. Why this matters (and what changed)

The code is byte-identical to the landed port, yet the sign flipped between the landing window and now. The rest of the tree changed (table, S1/S2, sparse×2, kv-auto, guards, 1714) — none touch the decode path — and the TU grew (sparse kernels), which can shift codegen/inlining/alignment of the hoisted loop. The effect was evidently never robust: it depended on window conditions, not just arithmetic (which remains bit-identical — ids match everywhere). Round 1 blamed the hot box; round 2 controls for that (stable baseline, order balance) and the loss persists.

## 4. Standing decision

Hoist stays reverted. The spike recipe stands as valid methodology with a real but fragile effect; any future port must (a) reproduce on the CURRENT tree, (b) best-of-5 order-balanced, (c) include a layout-sensitivity check (e.g., function placement / alignment probes), not just timing. No further hoist cycles without (c) — timing alone cannot distinguish a layout artifact from a win.

## 5. Exact commands run and their results

- Baseline saved (current master binary); `git cherry-pick -n cb2a330` → conflict (sparse kernels share the TU) → aborted, tree restored clean.
- Function extracted verbatim from commit blob + hand-applied dispatch/include; build 54/54; tests 145/145.
- 6-run A/B (§2 table); ids identical all runs.
- `git checkout -- src/cpu/avx2/attention_avx2.cpp` → empty diff; rebuild 54/54; tests 145/145; report committed; no push.

(End of file.)
