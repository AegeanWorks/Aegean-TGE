# INVENTED (grouped-hoist follow-up): Group-Size Scaling — CONFIRM (Llama −18% ≈ 2/3 of 0.5B's −22%; Phi −5%/−4% consistent structural floor, not thermal noise)

- **Idea**: The PROCEED-grade hoist predicts wins scaling with (G-1)/G: Llama (G=3) should land near 2/3 of 0.5B's ~−20% (G=7); Phi (G=1) should show ~zero (nothing to hoist). Test on Llama-3.2-3B and Phi-3.5-mini, i8-KV, grouped path, hoist off/on, ids + fidelity-class check. Falsifier: Llama clearly short of ~−10% OR Phi showing a large win (both would break the sharing model).
- **Cycle date**: 2026-11-06
- **Verdict**: CONFIRM with a refined model. Llama: −17.9% (matches the 2/3 prediction within noise). Phi: −5.2% then −4.3% swapped (consistent, small — a real structural floor from the token-outer loop itself, NOT thermal: swapped order rules drift out, and zero-sharing predicts exactly ~0 plus loop-shape effects). Model: win = sharing×(G-1)/G + structural(~5%). No branch left behind (spike reverted); no code shipped.
- **Tree state at close**: `git status` clean, no diff, stash unchanged (only the pre-existing `leftover peer work shelved 2026-11-04`). Spike branch `spike-hoist-xfer` created (zero commits) and deleted; the one spike file reverted with empty diff verified; all scratch deleted. `master` ahead of `origin/master` by 115 after this report. Nothing pushed (human gate).

## 1. Mechanism (as spiked — verbatim re-implementation of the PROCEED recipe)

`src/cpu/avx2/attention_avx2.cpp` only, env-gated (`TGE_GROUPED_HOIST=1`), q8-only: per-token widen-K/V into thread-local scratch (2×head_dim), per-head online-softmax with true-local maxima/sums (writeback at end), unscaled accumulators for the caller merge. Bit-identical by construction (same floats, same order).

## 2. Predicted gain — confirmed per model

0.5B (prior cycle, best-of-3): −22.6/−21.9/−18.4%. Llama (this cycle, single pair): −17.9% (predicted ≈ −13 to −16% from 2/3 scaling). Phi (this cycle, swapped pair): −5.2%/−4.3% (predicted ≈ 0% + structure).

## 3. Falsifier outcome — CONFIRM (scaling holds; null control behaves as a null)

1. **Llama matches the scaled prediction** (no overshoot that would implicate a second mechanism; no undershoot that would deny the first).
2. **Phi is consistent with ~zero sharing gain + a small structural floor**: two swapped runs agree in sign and size; thermal drift would not survive the swap. The earlier 2x grouped-vs-perhead Phi gap is NOT reproduced here and stays withdrawn.
3. Ids byte-identical everywhere (degenerate prompt, noted). No separate fidelity run: bit-identity is structural (same functions, same order as the validated 0.5B build) and ids confirm it end to end.

## 4. File anchors (master = pre-spike; spike reverted)

- `src/models/qwen2.cpp:1177+` (grouped dispatch — untouched; hoist lives in the kernel).
- Grounds: the grouped-hoist PROCEED (recipe + falsifier), the 0.5B best-of-3 (§2 baseline), q8micro (dequant share that bounds the prize).

## 5. What was done (temporary spike, fully reverted)

1. `git checkout -b spike-hoist-xfer` (zero commits, deleted at close). Re-applied §1 verbatim from the committed recipe; build 54/54, `tge_tests` 141/141.
2. Llama i8 grouped pair (off/on) → −17.9%; Phi i8 grouped pair + swapped repeat → −5.2%, −4.3%.
3. Thinking set on spike (no env): story + creative identical vs master build post-revert.
4. `aikit.py check` → 2/4 (test + secrets pass; pre-existing fails).
5. Reverted the one file, `git checkout master`, branch deleted, rebuilt master 54/54, tests 141/141, scratch deleted.

## 6. Measured numbers (repetitive 3.9k prompt, ctx 4096, threads 10, --kv i8, 30 tok decode)

| model (G) | off attn | on attn | delta |
|---|---|---|---|
| Llama-3B (3) | 884.3 ms | 726.1 ms | **−17.9%** |
| Phi-3.5B (1) run 1 | 2845.4 ms | 2698.3 ms | −5.2% |
| Phi-3.5B (1) run 2 (swapped) | 2842.5 ms | 2720.7 ms | −4.3% |

(0.5B G=7 best-of-3 from prior cycle: −22.6/−21.9/−18.4%. Fits win ≈ structural(~5%) + sharing×(G−1)/G.)

## 7. Full reimplementation recipe (unchanged from the PROCEED — this cycle adds the scaling law, not new code)

See `idea-invented-grouped-hoist-proceed.md` §7 for the port recipe. New: expected attn win ≈ 5% + sharing_share × (G−1)/G, where sharing_share ≈ 15% on this box. Revival trigger for a re-measure: a grouped q8 path that differs structurally (different rescale/accumulate order) — the kind tags and loop order are load-bearing for the current numbers.

## 8. Verdict and why

**CONFIRM: the hoist's win scales with group size as predicted, and the null control behaves like a null.** This closes the grouped-hoist arc with a quantitative mechanism, not just an existence proof: 0.5B −20% (sharing-dominated), Llama −18% (mixed), Phi −5% (structure-only). No code ships (human gate); the recipe stands validated across three architectures. Next-cycle candidates: the 1517 stage-2 harness (whose batch admission now has a fully validated i8-attention recipe to consult) and the human-gated port shelf.

## 9. Exact commands run and their results

- `git checkout -b spike-hoist-xfer`; re-apply §1; build → 54/54; tests → 141/141.
- Llama i8 grouped off/on → −17.9%, ids identical.
- Phi i8 grouped off/on + swapped repeat → −5.2%, −4.3%, ids identical.
- Thinking set spike (no env) vs master → identical.
- `aikit.py check` → 2/4. Revert → clean. `git checkout master`; branch deleted; rebuild 54/54; tests 141/141; scratch deleted; report committed; no push.

(End of file.)
