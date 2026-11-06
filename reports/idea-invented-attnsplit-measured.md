# INVENTED: Attention Sub-Phase Split Diagnostic — MEASURED (dots ~47% / softmax 3% / V-sum ~50%, flat; no lever, close the line)

- **Idea**: Attention is a black-box phase in every report (one `attn` number). Split the per-head AVX2 kernel into QK dots / softmax / V-sum with temp thread-local counters (ratios only, no reset hook) and run a prefill-heavy and a decode-heavy shape single-threaded (threads=1 forces the instrumented per-head path; the grouped GQA path uses online softmax and can't split the same way). Name a ≥50% bucket with ceiling math, or report honestly.
- **Cycle date**: 2026-11-06
- **Verdict**: MEASURED, line closed. Prefill-heavy (2.6k ctx): dots 48%, softmax 3%, vsum 50%. Decode-heavy (520 ctx + 199 tok): dots 46%, softmax 3%, vsum 51%. Flat, shape-independent, no ≥50% bucket: the dots run at shipped AVX2-FMA rate, softmax is noise (approx-exp pointless), the V-sum is a single streaming pass (nothing to fuse). No sub-5% lever exists inside the kernel — do not re-split without a new kernel shape.
- **Tree state at close**: `git status` clean, no diff, stash unchanged (only the pre-existing `leftover peer work shelved 2026-11-04`). Spike branch `spike-attnsplit` created (zero commits) and deleted; all three spike files reverted with empty diff verified. `master` ahead of `origin/master` by 108 after this report. Nothing pushed (human gate).

## 1. Mechanism (as spiked, reverted)

`include/tge/cpu/kernels.hpp`: 3 getter decls; `src/cpu/avx2/attention_avx2.cpp`: thread-local accumulators + 3 `steady_clock` pairs around the QK-dots loop / softmax block / V-sum loop of `attention_head_avx2`; `apps/cli/main.cpp`: one lifetime-ratio `attnsplit:` print after the decode phases. Grouped GQA kernel untouched (online softmax fuses the sections; threads=1 routes everything through the per-head path, and shares are thread-count-independent).

## 2. Predicted gain — diagnostic, landed as attribution

Expected: a ≥50% bucket naming the next kernel lever. Got flat thirds (halves): no lever clears the bar, so the finding is a closed line, not a queued spike.

## 3. Falsifier outcome — flat shares honestly reported, no lever with ceiling

Stated: name a ≥50% bucket + ceiling math, or report honestly. Honest outcome: 88-94% of the attn phase attributed to the three buckets (remainder: instrumentation + max/inverse/score-loop fixed costs, clock overhead ~0.1-6% depending on span mix — no lever there either). Per-bucket ceilings: dots already vectorized (int-dot retry killed this loop); softmax 3% (an infinitely fast exp saves 3%); V-sum single streaming pass (already minimal). Revival trigger: a new kernel shape (different dataflow, not a faster op in the same dataflow).

## 4. File anchors (master = pre-spike; spike reverted)

- `src/cpu/avx2/attention_avx2.cpp:121-288` (per-head kernel, the three sections).
- `src/cpu/avx2/attention_avx2.cpp:261+` (grouped kernel — deliberately uninstrumented, online formulation).
- `apps/cli/main.cpp` (phases + temp print site).
- Grounds: the phasesplit recipe pattern, int8qk kill (dots), tiled-attn kills (traffic).

## 5. What was done (temporary spike, fully reverted)

1. `git checkout -b spike-attnsplit` (zero commits, deleted at close). Baseline: build 54/54, `tge_tests` 141/141.
2. ANALYZE: two kernel paths mapped; grouped excluded by construction (online softmax); threads=1 routing verified as the per-head forcing function.
3. Applied §1 (4 small regions). Rebuilt 54/54, `tge_tests` 141/141.
4. Split runs (§6): 2.6k-prefill shape + 520+199tok decode-heavy shape, both threads=1.
5. `aikit.py check aegean-tge` → 2/4 (test + secrets PASS; hygiene + guidelines fail pre-existing, same pair as every prior cycle).
6. Reverted all three files, `git checkout master`, branch deleted, rebuilt master 54/54, `tge_tests` 141/141.
7. No thinking set owed (timers-only diagnostic, normsplit/timers precedent); no fidelity owed (no numerics changed). No scratch files written.

## 6. Measured numbers (0.5B-Q8_0, threads=1)

| shape | attn phase | dots | softmax | vsum | attributed |
|---|---|---|---|---|---|
| prefill-heavy (2600 tok, n=8) | 1980.9 + 142.6 ms | 887.9 (48%) | 48.9 (3%) | 929.2 (50%) | ~88% |
| decode-heavy (520 tok, n=199) | 88.8 + 703.4 ms | 342.4 (46%) | 21.9 (3%) | 382.4 (51%) | ~94% |

## 7. Full reimplementation recipe (diagnostic instrument — rebuild only to re-verify)

Re-apply §1; gate on build 54/54, `tge_tests` 141/141, and the §6 table reproducing within box noise (flat thirds both shapes). No port exists (nothing to keep). Revisit ONLY on the §3 trigger.

## 8. Verdict and why

**MEASURED: attention's interior is mapped and holds no lever.** Dots and V-sum split the phase evenly with both already at their practical forms (vectorized FMAs; single streaming pass), softmax is 3% noise. This closes the last unexamined kernel interior on the hot path — every other bucket (norm/rope/silu, timers, matvec family) was split in prior cycles with the same honest outcome pattern. Next-cycle candidates: the 1517 stage-2 skip harness remains the only live structural front; the human-gated port shelf stands.

## 9. Exact commands run and their results

- `git checkout -b spike-attnsplit` → branch (zero commits, deleted at close).
- Baseline: `cmake --build --preset release` → 54/54; `tge_tests.exe` → 141/141.
- Spike (§1); rebuild → 54/54; tests → 141/141.
- 0.5B fox×200 `--threads 1` + fox×40 `--threads 1 --n 200` → §6 split rows.
- `aikit.py check` → 2/4 (test + secrets pass; pre-existing fails).
- Revert 3 files → clean; `git checkout master`; branch deleted; rebuild 54/54; tests 141/141; report committed; no push.

(End of file.)
