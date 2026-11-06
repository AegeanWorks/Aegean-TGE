# INVENTED (phasesplit follow-up): No-Gather Strided Prefill Attention — PROCEED (attn ~50% of dense at identical ids + digit-identical fidelity; gather fully removed)

- **Idea**: Execute the named phasesplit next spike: teach the AVX2 attention kernel the static sink/stride/window pattern directly (read strided K/V rows in place via a `kept` row map) instead of gathering the subset into compacted scratch per (position, head). Same subset, same order, same ops — bit-identical by construction; the 22:1 gather/build split said the gather is ~85% of sparse attention cost.
- **Cycle date**: 2026-11-06
- **Verdict**: PROCEED — the falsifier is cleared with margin: 2k prefill attn at ~50-52% of dense (recipe demanded within 10% of dense), sparse ids byte-identical across all 9 runs + Llama round, fidelity digit-identical (KL 2.703e-03 both). Porting (kept-field + kernel branches + env-gate removal, §7) is a human-gated follow-up, not this cycle: zero spike code survives.
- **Tree state at close**: `git status` clean, no diff, stash unchanged (only the pre-existing `leftover peer work shelved 2026-11-04`). Spike branch `spike-nogather-prefill` created (zero commits) and deleted; all three spike files reverted with empty diff verified; saved spike binary + scratch deleted. `master` ahead of `origin/master` by 102 after this report. Nothing pushed (human gate).

## 1. Mechanism (as spiked, reverted)

- `include/tge/cpu/kernels.hpp`: `AttentionArgs.kept` (`const int*`, null = dense; spike-only field).
- `src/cpu/avx2/attention_avx2.cpp` (`attention_head_avx2` only; GQA untouched — sparse never routes there): QK-dot and weighted-V loops compute `src = kept ? kept[token] : token`, `row = src*kv_stride` (q8 scale offsets follow `src` too, though sparse is f32-only).
- `src/models/qwen2.cpp` (`attention_head`): when the sparse condition holds and `TGE_SPARSE_NOGATHER=1`, skip the gather/compact block, pass `kept.data()` + compacted span with native `kv_stride`. Default path (env off) untouched.
- Attempt-1 crashed (access violation): `kept` was block-local while the kernel runs after the block (the gather path copies to thread-local scratch, so nothing dangles there — the no-gather path only borrows). Fixed by function-scope `kept` (attempt-2, all numbers below). Also: checkpoint printfs poisoned one timing run (29 s of stderr writes) — removed before measuring.

## 2. Predicted gain — landed above the bar

Recipe demanded prefill attn within 10% of dense (i.e., recover the gather loss). Measured ~50% of dense: the strided reads cost ~nothing over the halved dots (streaming + L2 cover them), so nearly the full subset win lands.

## 3. Falsifier outcome — all arms pass, PROCEED

1. **Attn phase**: 0.5B 2k prefill, 3 interleaved D/G/N rounds: nogather/dense = 52%/52%/50% (recipe: within 10% — cleared by 5x). Llama-3B single round: 41% (8187.7 vs 19935.8).
2. **Identity**: sparse ids byte-identical D=G=N on all 9 0.5B runs + Llama round (8/8 each).
3. **Fidelity**: 799-pos sparse run KL 2.703e-03 / worst 0.2161 / maxdiff 5.957 / 785-799, DIGIT-IDENTICAL gather vs no-gather (same subset/order/ops, as constructed). Short-corpus control (399 pos, all spans ≤516) correctly showed the dense floor on both paths.

## 4. File anchors (master = pre-spike; spike reverted)

- `src/models/qwen2.cpp:1174-1241` (sparse gather block + `uses_sparse` + per-head routing — the consult site).
- `src/cpu/avx2/attention_avx2.cpp:121-258` (dense kernel, three token loops — the two patched).
- `include/tge/cpu/kernels.hpp:207-229` (struct — the field site).
- Grounds: sparse phasesplit recipe (the named next spike + falsifier), 1568 Arm 1 (gather-loss baselines reproduced: +24-32% here vs +35% there), 2048-closure (decode-side posture unchanged).

## 5. What was done (temporary spike, fully reverted)

1. `git checkout -b spike-nogather-prefill` (zero commits, deleted at close). Baseline: build 54/54, `tge_tests` 141/141.
2. ANALYZE: gather site bounded (one block, f32-only, scalar/GQA excluded); kernel loops share `row=token*stride`.
3. Implemented §1. Rebuilt 54/54, `tge_tests` 141/141. Smoke: gather path fine; no-gather attempt-1 crashed → diagnosed use-after-free (§1) → attempt-2 fixed.
4. Measured: 0.5B fox×200 D/G/N interleaved ×3 (§6); Llama fox×205 D/G/N ×1 (§6); fidelity gather + no-gather (§6); thinking set on spike (saved).
5. `aikit.py check aegean-tge` → 2/4 (test + secrets PASS; hygiene + guidelines fail pre-existing, same pair as every prior cycle).
6. Reverted all three files, `git checkout master`, branch deleted, rebuilt master 54/54, `tge_tests` 141/141, master thinking set: story + creative ids byte-identical vs spike.
7. Deleted spike binary + scratch; `git status` clean, stash untouched.

## 6. Measured numbers

0.5B-Q8_0 fox×200 (2600 tok), 10 threads, --context 4096, --n 8, prefill attn ms (best of the phase prints):

| round | dense | gather | nogather | N/D |
|---|---|---|---|---|
| 1 | 3018.5 | 3736.2 (+23.8%) | 1573.7 | 52% |
| 2 | 3475.5 | 4317.7 (+24.2%) | 1790.9 | 52% |
| 3 | 3338.5 | 4406.7 (+32.0%) | 1681.6 | 50% |

Box drift noted (matvec +46% R1→R3 in the gather arm); N ran LAST each round yet fastest — direction robust to drift. Prefill wall t/s: R1 177/168/197, R2 159/150/185, R3 154/129/182 (D/G/N).

Llama-3B-Q4_K_M fox×205 (2665 tok), one round: attn D 19935.8 / G 21989.9 (+10.3%) / N 8187.7 (**-58.9%**); wall 36.4/32.1/43.4 t/s; ids 8/8 identical all arms.

Fidelity (0.5B-Q8_0, arch, ctx 1024, 799 pos, sparse 512/4): KL **2.703e-03**, worst **0.2161**, maxdiff **5.957**, top-1 **785/799** — identical all four numbers gather vs no-gather.

## 7. Full reimplementation recipe (keep-grade port — human-gated commit, not this cycle)

1. `kernels.hpp`: keep the `kept` field (rename per convention); `attention_avx2.cpp`: keep the two `src`-mapping sites (QK + V; q8 scale offsets already follow `src`); `qwen2.cpp`: keep the env-gated branch but REPLACE the env gate with a permanent path (or a `--sparse-nogather` flag defaulting on — the recipe's tested shape), with `kept` at function scope (the §1 lifetime lesson — do not reintroduce the block-local form).
2. Hazards for the porter: (a) `AttentionGroupArgs` has NO kept field — sparse must keep routing to the per-head loop (assert `!uses_sparse` on the GQA path or add the field there too); (b) scalar oracle ignores kept (fine — sparse never routes to Scalar, keep both gates); (c) `kept` is rebuilt per (position, head) — hoisting one index list per query across the group is the named follow-up (uses the same recipe pattern, own falsifier).
3. Gate the port: build 54/54, `tge_tests` 141/141, rerun the §6 0.5B interleave (expect N/D ≈ 50%, ids identical), one Llama pair, fidelity digit-identity vs gather, story/creative ids identical vs pre-port binary, `aikit.py check`.

## 8. Verdict and why

**PROCEED: the gather is dead weight and the strided kernel removes it cleanly.** ~50% of dense prefill attn at identical ids and digit-identical fidelity, on both models, with the 1568 gather-loss baselines reproduced on the way in. The attempt-1 crash is on the record as a spike bug with a two-line fix, not a mechanism flaw. This completes the sparse program's prefill half (decode half: no-gather recipe pending the same human gate). Next-cycle candidates: the 1517 stage-2 skip harness (its batch admission can now consult the no-gather path — flag the composition in the epic's stage-2 design) and the human-gated port shelf.

## 9. Exact commands run and their results

- `git checkout -b spike-nogather-prefill` → branch (zero commits, deleted at close).
- Baseline: build → 54/54; `tge_tests` → 141/141.
- Spike (§1); rebuild → 54/54; tests → 141/141.
- Smoke gather (585 tok): attn 149.1 ms, ids recorded. Smoke no-gather: CRASH (use-after-free, §1) → fixed → clean run, ids match gather.
- 0.5B fox×200 D/G/N interleave ×3 → §6 table (ids 8/8 identical all 9).
- Fidelity sparse gather → KL 2.703e-03; fidelity sparse no-gather → identical; short-corpus control → dense floor both.
- Llama fox×205 D/G/N ×1 → §6 Llama row (ids 8/8 identical).
- Thinking set (spike, saved); `aikit.py check` → 2/4 (test + secrets pass; pre-existing fails).
- Revert 3 files → clean; `git checkout master`; branch deleted; rebuild 54/54; tests 141/141; master thinking set → ids identical; scratch deleted; report committed; no push.

(End of file.)
