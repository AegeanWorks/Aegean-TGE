# IDEA-1519: Linear-Recurrence Attention with Streaming MXFP4 Experts — KILLED WITHOUT SPIKE (triple no-carrier)

- **Idea ID**: IDEA-1519 (dossier `goal-sweep-2026-11-03.md`: KDA/delta-rule linear recurrence + MLA cached KV, MXFP4 streaming experts, cache-only draft routing on CPU; falsifier: port the cache policy onto one tge MoE file — fail on decode t/s + GB/s miss or non-reproducing bit-identical A/B)
- **Cycle date**: 2026-11-06
- **Verdict**: KILLED on the record, no spike built (dead-idea rule: all three mechanisms lack carriers, and the falsifier's staging target — "one tge MoE file" — does not exist). No branch created, no code touched by this cycle, nothing of mine to revert.
- **Tree state at close**: `git status` shows my report commit plus PRE-EXISTING, UNCOMMITTED peer spike work (`run_spike_mprefix` multi-prefix CoW driver + table, appeared mid-previous-cycle, additive-only +262/−0, no active process, board silent). I touched none of it: not staged, not committed, not pushed, not reverted. `master` ahead of `origin/master` by 40 after this report (reports only). Nothing pushed (human gate).

## 1. Mechanism (as dossiered)

Replace softmax attention layers with KDA/delta-rule linear recurrence plus
MLA cached KV, stream MXFP4 experts from disk with cache-only draft routing.

## 2. Predicted gain — and why no part attaches here

- Linear-recurrence/KDA attention: zero matches in `src/` (`linear_recurrence|
  kda|delta_rule|recurrence|Mamba` — attention is exclusively softmax
  dot-product: AVX2 `q8_dot`/`dot_avx2`, scalar oracle, group/head paths).
- MXFP4 streaming experts: zero matches (`mxfp4|MXFP4|fp4`); quants on record
  are Q8_0/Q4_K/Q6_K/f16/f32/i8. No MoE file, model, or execution path exists
  (1382 + 1518 kills stand: dense-only engine, biggest working set 2.23 GB
  vs 16 GiB RAM).
- Cache-only draft routing: presumes a draft MODEL with its own KV cache;
  this tree's drafters are KV-free token-id retrieval (1502/1526/1542 kills).
  Nothing to route, no cache to keep draft-only.

## 3. Falsifier (from the dossier) — unstageable, which is the kill

Port onto "one tge MoE file" and measure decode/GB/s/bit-identity: there is
no MoE file, no MoE loader, no MoE execution — the experiment cannot be
staged on either side.

## 4. File anchors (current `master`, read but untouched)

- `src/cpu/avx2/attention_avx2.cpp`, `src/models/qwen2.cpp`
  (`attention_gqa`, `attention_batch`) — softmax-only attention.
- `reports/idea-1382-moe-streaming-killed-analytic.md`,
  `reports/idea-1518-moe-streaming-killed-analytic.md`,
  `reports/idea-1526-draft-kv-refresh-killed-analytic.md` — the no-MoE /
  no-draft-KV precedents.

## 5. What was done (ANALYZE only — no spike per the dead-idea rule)

1. Carrier greps: linear-recurrence (zero), MXFP4 (zero); MoE/draft-KV by
   cited precedent.
2. Gates for the record: build clean (compiles WITH the unrelated peer spike
   present — additive-only, gated behind its own subcommand; default paths
   unaffected), `ctest` pass, `tge_tests` 141/141, `aikit.py check` 2/4
   (test + secrets pass; hygiene + guidelines fail pre-existing).
3. No thinking-set diff owed: same behavior on all standard paths.

## 6. Measured numbers (inventory — the experiment is unstageable)

| check | result |
|---|---|
| linear-recurrence/KDA/Mamba matches in `src/` | 0 |
| MXFP4/FP4 matches in `src/` | 0 |
| MoE files / models / execution paths | 0 / 0 / 0 |
| draft models with KV caches | 0 (retrieval drafters only) |

## 7. Reimplementation recipe (deliberately NOT recommended — kill-grade)

Would require a linear-recurrence attention subsystem, an MXFP4 quant format
+ loader, an MoE model family with execution, and a draft model with its own
KV — i.e. a different engine three times over. Do not build any of it.
Revisit only with an MoE workload on this box, which is itself a killed idea
(1382).

## 8. Verdict and why

**KILLED without spike.** Three mechanisms, three missing carriers, one
unstageable falsifier. The dossier describes a different project's machine
(KDA + MXFP4 + MoE); nothing in it attaches to a dense softmax AVX2 engine
serving sub-4B GGUFs.

## 9. Exact commands run and their results

- Carrier greps (linear/MXFP4) → zero matches both.
- `cmake --build --preset release` → 54/54 (with peer spike present, inert);
  `ctest` → pass; `tge_tests` → 141/141.
- `aikit.py check` → 2/4 (pre-existing fails). No branch, no edits by me,
  peer spike left exactly as found, no push.
