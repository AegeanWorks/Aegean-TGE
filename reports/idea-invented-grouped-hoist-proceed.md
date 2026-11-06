# INVENTED: Grouped i8 Widen-Hoist — PROCEED (−22%/-22%/-18% attn, bit-identical; hoist redundant per-head dequant into per-token L1 scratch)

- **Idea**: The grouped-i8 path re-widens each shared K/V row once per query-head (verified by code read: widen calls sit inside the per-head token loop). Hoist: token-outer loop, widen each row ONCE into L1-resident scratch, reuse across the group's heads. Predicted ~20% attn win on 0.5B (grouped, G=7): removes ~6/7 of the widen work plus the associated re-reads.
- **Cycle date**: 2026-11-06
- **Verdict**: PROCEED — falsifier cleared all arms: best-of-3 interleaved attn −22.6% / −21.9% / −18.4%; ids byte-identical all runs; fidelity digit-identical (KL 4.310e-02, 175/199 both); thinking set identical spike-vs-master. Porting (hoisted grouped-q8 path as default, env gate removed) is a human-gated follow-up, not this cycle: zero spike code survives.
- **Tree state at close**: `git status` clean, no diff, stash unchanged (only the pre-existing `leftover peer work shelved 2026-11-04`). Spike branch `spike-grouped-hoist` created (zero commits — all work uncommitted) and deleted; the one spike file reverted with empty diff verified; saved binaries + scratch deleted. `master` ahead of `origin/master` by 115 after this report. Nothing pushed (human gate).

## 1. Mechanism (as spiked, reverted)

`src/cpu/avx2/attention_avx2.cpp` only, env-gated (`TGE_GROUPED_HOIST=1`), q8-only: token-outer loop over the grouped span; per token widen K/V rows once into a thread-local scratch (`2*head_dim` floats); per head: q8_dot-equivalent dual-accumulation over pre-widened+scaled rows (same lane split, same order), online-softmax rescale + V accumulate from scratch; per-head online state in true locals (`lmax[64]`/`lsum[64]`, copied to `args.maxima/sums` at end) + per-head accumulator rows (zeroed upfront, as the default does). Unscaled accumulators left for the caller's chunk merge — identical contract to the default path.

## 2. Predicted gain — landed inside the estimated band

Falsifier: ≥10% attn win on 0.5B-i8 grouped at 8k + bit-identical ids + i8-class fidelity. Measured −18 to −23% (attn), +0% matvec (untouched, as expected), ids identical, fidelity digit-identical.

## 3. Falsifier outcome — all arms pass, PROCEED

1. **Perf**: best-of-3 interleaved (env off/on): 174.6→135.1, 178.4→139.3, 173.4→141.5 ms.
2. **Identity**: ids byte-identical on all 6 prefill runs; fidelity KL/max/top-1 digit-identical (4.310e-02 / 1.890e-01 / 175/199).
3. **No collateral**: thinking set identical spike-vs-master (env off = default path); `tge_tests` 141/141 with the spike compiled in.

## 4. File anchors (master = pre-spike; spike reverted)

- `src/cpu/avx2/attention_avx2.cpp` (`attention_group_avx2`: default per-head-inside loop; spike inserted the token-outer q8 branch at function top — reverted).
- Grounds: grouped-i8 thermal + single-thread rounds (the 6-7% gap this beats by removing the widen redundancy instead of the loop order), int8qk kill (same dequant-cost family, different lever), no-gather-prefill (the other half of the i8-attention story).

## 5. What was done (temporary spike, fully reverted — with one self-caught bug)

1. `git checkout -b spike-grouped-hoist` (zero commits, deleted at close). Baseline: build 54/54, `tge_tests` 141/141.
2. ANALYZE: confirmed per-(head,token) widening by code read; designed token-outer + per-head online state; predicted ~20% from widen share.
3. Attempt-1: per-head online state in `args.maxima/sums` (memory, aliasing-unsafe) → bit-identical but 2x SLOWER (loop-carried state through L1 each token). Diagnosed by contradiction (less work, more time ⇒ new bottleneck introduced), not by guessing.
4. Attempt-2: true locals + end writeback → −22% attn, still bit-identical. (Lesson recorded: hoist the state with the loop, or don't hoist at all.)
5. Measured (§6): best-of-3 interleave + ids + fidelity pair + thinking set.
6. `aikit.py check aegean-tge` → 2/4 (test + secrets pass; pre-existing hygiene + guidelines fail, same as every prior cycle).
7. Reverted the file (empty diff), `git checkout master`, branch deleted, rebuilt master 54/54, tests 141/141, master thinking set: identical.

## 6. Measured numbers (0.5B-Q8_0, 7799 tok, ctx 8192, threads 10, --kv i8, --n 12; decode attn ms)

| round | default grouped-i8 | hoisted | delta |
|---|---|---|---|
| 1 | 174.6 | 135.1 | −22.6% |
| 2 | 178.4 | 139.3 | −21.9% |
| 3 | 173.4 | 141.5 | −18.4% |

Fidelity (arch corpus, ctx 512, 199 pos, i8): KL 4.310e-02 / worst 1.890e-01 / max 6.131e+00 / 175-199 — digit-identical both binaries.

## 7. Full reimplementation recipe (keep-grade — human-gated commit)

1. `src/cpu/avx2/attention_avx2.cpp`: add the token-outer q8 branch (§1 verbatim — struct field for the enable flag or unconditional; the recipe's tested form is env-gated `TGE_GROUPED_HOIST`, port decides the final switch shape).
2. Invariants the porter must preserve (or the bit-identity breaks): widen-then-scale per element EXACTLY as `q8_dot_avx2`/V-arm do it (same 8-lane widens, same per-32 scales); dual-accumulator lane split identical to q8_dot; per-head online update order token-ascending with the same exp/rescale/sum sequence; accumulators zeroed per head BEFORE the token loop; NO finalize (caller merges); `maximum`/`sums` live state must be true locals during the loop, written back once (attempt-1's failure mode is the documented anti-pattern).
3. Gate the port: build 54/54, `tge_tests` 141/141, rerun the §6 A/B (expect ≥15% attn win with identical ids), fidelity digit-identity vs master, thinking set identical, `aikit.py check`.

## 8. Verdict and why

**PROCEED: the grouped-i8 widen redundancy is real, measured, and removable — with a complete, tested recipe.** This resolves the mechanism hunt two cycles deep (q8micro closed arithmetic, single-thread closed the algorithm, this closes the scheduling: hoist the widening, keep the loop). It upgrades no-gather-decode's port case (both halves of the i8 path now have validated recipes) and sharpens the auto-routing question (per-head i8 already wins; hoisted grouped-i8 should win bigger — a follow-up A/B, named not built). Next-cycle candidates: the 1517 stage-2 harness can now consult a complete i8-attention story; the human-gated port shelf leads with this recipe.

## 9. Exact commands run and their results

- `git checkout -b spike-grouped-hoist` → branch (zero commits, deleted at close).
- Baseline: build → 54/54; tests → 141/141.
- Spike attempt-1 (§5.3: bit-identical, 2x slow) → diagnosed → attempt-2 (locals) → −22% + bit-identical.
- 0.5B 8k i8 off/on best-of-3 → §6 table + ids; fidelity pair → digit-identical (both); thinking set both binaries → identical.
- `aikit.py check` → 2/4 (test + secrets pass; pre-existing fails).
- Revert file → clean; `git checkout master`; branch deleted; rebuild 54/54; tests 141/141; master thinking set → identical; scratch deleted; report committed; no push.

(End of file.)
