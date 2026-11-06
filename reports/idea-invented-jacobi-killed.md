# INVENTED: Jacobi Parallel Decoding — KILLED analytic (structurally dominated by shipped speculative: same verify primitive, worse accept policy, same guess family)

- **Idea**: Jacobi (parallel) decoding as an alternative multiplier: fill an N-token window with guesses (repeat-last / prompt-lookup, no draft model), run the whole window through one chunked forward, replace each position with its argmax, iterate to fixed point (which equals the greedy AR solution by construction). Each iteration reuses the shipped verifier's chunked forward; convergence typically takes 2-4 iterations for small N.
- **Cycle date**: 2026-11-06
- **Verdict**: KILLED analytic — structurally dominated by the shipped speculative path on all three arms, with no spike warranted (a spike would reimplement `verify_draft_logits` + an accept loop to re-prove parity-or-worse). No branch, no edits, nothing to revert.
- **Tree state at close**: `git status` clean, no diff, stash unchanged (only the pre-existing `leftover peer work shelved 2026-11-04`). `master` ahead of `origin/master` by 101 after this report. Nothing pushed (human gate).

## 1. Mechanism (as proposed, not built)

Per window: guess N tokens → `prefill_run` chunked forward → argmax-replace all → repeat until no position changes → emit window, advance. Convergence detection + fallback to AR on non-convergence. No draft model required (the literature's headline advantage over model-based speculative decoding).

## 2. Predicted gain — none available (domination, not a race)

The win condition (K iterations × chunk cost < N AR tokens with identical output) is unreachable here for structural reasons (§3), not measurement reasons.

## 3. Falsifier outcome — kill on all three arms (verified on today's tree)

1. **Same primitive, same floor cost.** One Jacobi iteration over an N≤6 window IS the shipped verify call: `verify_draft_logits` = one `prefill_run(tokens, count, ...)` (`qwen2.cpp:2089`, window capped by `kSpeculativeBatch = 6` in `qwen2.hpp:129`). Per-iteration cost ≥ shipped verify cost by construction — Jacobi cannot undercut the primitive it shares.
2. **Worse accept policy.** Shipped emits up to 6+1 tokens per verify (accept longest prefix + correct + advance, fused steady state chains the bonus) and never re-verifies a stale suffix. Jacobi emits its window only at convergence: K=1 is full-accept (identical ceiling to shipped's 6.00/step on locked repetition — already achieved, nothing above it exists), and every K>1 pays full-window re-verifies exactly where shipped corrects-and-advances. Larger windows (N=32 via direct `prefill_run`) don't escape: the per-iteration wall scales with the chunk while convergence stays guess-bound.
3. **Same guess family, moot advantage.** Jacobi's guesses (repeat-last / prompt-lookup) come from the shipped drafter chain's own family (SKIP/locks/span/bigram/gapped, `main.cpp` + `prompt_lookup.h`); no draft model exists on this tree (1526 killed), so the literature's "no draft model needed" differentiator buys nothing here — and where guesses are poor (prose), the shipped path correctly ABSTAINS to AR at zero cost while Jacobi would burn K iterations of waste. A strictly-better prose guess source would be a DRAFTER improvement, ported into the shipped accept machinery — not a Jacobi revival.

## 4. File anchors (current `master`, read but untouched)

- `src/models/qwen2.cpp:2077-2091` (`verify_draft_logits` = one chunked `prefill_run`; the operation a Jacobi iteration duplicates).
- `include/tge/models/qwen2.hpp:129` (`kSpeculativeBatch = 6` window cap).
- `apps/cli/main.cpp` (speculative chain: SKIP/locked → lookup/span/bigram/gapped drafts, d0 pre-check, verify, accept-prefix-correct-advance, fused chaining).
- Grounds: 1526 (no draft model), 1542/1387 (ngram/trie killed — guess family bounded), SKIP-locked 6.00/step ceiling, 1568 (speculative +1.66-2x repetitive / parity prose — the shape Jacobi would have to beat).

## 5. What was done (ANALYZE only — no spike per the dead-idea rule)

1. Specified Jacobi against the current tree and checked each arm against the shipped path (carrier reads above — the domination is structural, not empirical).
2. Checked the N-scaling escape (N=32 direct `prefill_run`): per-iteration wall scales with the chunk, convergence stays guess-bound, KV/scratch machinery assumes ≤7-token batches — no escape, worse machinery bill.
3. Gates on the unmodified tree: `cmake --build --preset release` 54/54; `tge_tests.exe` 141/141; `aikit.py check` 2/4 (test + secrets pass; hygiene + guidelines fail pre-existing, same pair as every prior cycle).
4. No fidelity run owed (no numerics exist to price — nothing built); no bench A/B owed (a spike would duplicate the shipped verify primitive to re-prove parity-or-worse); no thinking-set diff owed (same binary, zero behavior change).

## 6. Measured numbers (none — domination proof, not a race)

No new numbers: the kill rests on primitive identity (§3.1: same `prefill_run` call), policy ordering (§3.2: re-verify-stale vs correct-and-advance), and source bounds (§3.3: guess family shared). The applicable recorded numbers are the shipped path's own (6.00/step ceiling locked, parity prose) — Jacobi's ceiling on each branch is ≤ those by construction.

## 7. Full reimplementation recipe (deliberately NOT recommended — rebuild only to re-verify the kill)

To reproduce the negative result (do not ship): implement the §1 loop on `verify_draft_logits` with N≤6, convergence check + AR fallback, and run story (expect K≥2 waste vs abstain-to-AR parity) + locked-repetitive (expect ≤6.00/step, matching shipped, never above). Expected: parity-or-worse everywhere with extra machinery. Revisit ONLY if (a) a prose guess source strictly better than the shipped drafter chain appears (then it's a drafter port, same accept machinery), or (b) hardware arrives where chunked-forward cost is sublinear enough that K×C(N) < N with K≥2 while the shipped verifier stays 6-capped — a regime change, not a code change.

## 8. Verdict and why

**KILLED analytic: Jacobi is the shipped speculative path with a worse accept policy and the same guesses.** The literature advantage (no draft model) is moot where drafters are already model-free; the per-iteration advantage (parallel window) is the verifier the tree already runs. This closes the last unexamined corner of the multiply-tokens family: drafter-based (ships) > Jacobi (dominated) > draft-model-based (no model, killed). Next-cycle candidates: the 1517 stage-2 skip harness remains the only live structural front; the human-gated port shelf stands.

## 9. Exact commands run and their results

- Carrier re-reads (`verify_draft_logits`, `kSpeculativeBatch`, speculative chain, drafter family) — §3 arms verified on today's tree.
- `cmake --build --preset release` → 54/54; `tge_tests.exe` → 141/141; `aikit.py check aegean-tge` → 2/4 (test + secrets pass; pre-existing hygiene + guidelines fails).
- No branch created, no edits made, no benchmarks beyond the gate suite, report committed, no push.

(End of file.)
