# WINNER PORT: Grouped-i8 Widen-Hoist — LANDED, keep-grade (−16% attn, identical ids)

- **Source idea**: `reports/idea-invented-grouped-hoist-proceed.md` (+ scaling confirm `idea-invented-grouped-hoist-scaling-confirm.md`).
- **Port commit**: `cb2a330` (`perf(aegean-tge): hoist per-token widen out of grouped q8 attention loop`), 1 file, +153, reports excluded. No push (human gate).
- **Verdict**: WINNER AGAIN on remeasure — kept in tree as the default q8 grouped path. i8-class numerics, zero behavior change anywhere else.

## 1. What landed

`src/cpu/avx2/attention_avx2.cpp` only: new `attention_group_q8_hoisted_avx2` + q8 dispatch at the top of `attention_group_avx2` (q8 and group ≤ 64 take the hoist; half/float and over-wide groups keep the proven loop untouched). Token-outer over the grouped span; per token, widen+scale each shared K/V row once into thread-local scratch (`2*head_dim` floats); per head, q8_dot-equivalent dual-accumulator dot over the pre-widened rows (same lane split, same order) and online-softmax rescale + V accumulate from scratch; per-head state in stack locals (`maxima[64]/sums[64]`), written back once; accumulators zeroed per head up front; no finalize (caller chunk-merge contract unchanged).

Bit-identity required two exact-order fixes beyond the spike recipe, both in scalar tails that only execute when `head_dim % 32 != 0` (dead code on all production dims, but the port must be exact for any dim): the dot tail is `(query * widened) * scale` and the beta-branch V tail is `(beta * widened) * scale` — the spike's `query * (widened * scale)` form rounds differently and moved fidelity by one top-1 position (921 vs 920/1023). After the fix, i8 fidelity is digit-identical to the pre-port binary twice over.

## 2. Remeasure (this tree, pre-port binary `tge-baseline.exe` vs port)

- 0.5B-Q8_0, ~3.9k prompt, ctx 8192, threads 10, `--kv i8`, n 12, decode attn ms, interleaved best-of-3: baseline 74.7 / 80.9 / 86.9 → best 74.7; port 63.1 / 64.8 / 62.6 → best 62.6. **−16.2% best-of-3** (every pair a win: −15.5 / −19.9 / −28.0); all six runs byte-identical ids (`72030 72030 14 16 17 15 ...`). A later throttled-box pair (+3.7%, all phases inflated both binaries) is recorded as box-variance evidence, not signal.
- Llama-3.2-3B-Q4_K_M spot (2k prompt, ctx 4096, `--kv i8`, n 8): attn 108.5 → 97.1 ms (**−10.5%**), ids identical (`14924` x8). Smaller group (G=4 vs G=7) → smaller win, as the model predicts.
- Fidelity (`--corpus docs/architecture.md`, 1023 positions): i8 port 2.913e-02 / 920-1023 — digit-equal to baseline run twice; f32 port 9.551e-12 / 1023-1023 — the program floor, path untouched.
- Thinking set (0.5B-Q8_0, f32): story + haiku ids byte-identical to pre-port.
- Gates: build 54/54, `tge_tests` 141/141 (port + post-commit master), `aikit.py check aegean-tge` 2/4 (test + secrets pass; hygiene + guidelines fail pre-existing, unrelated single-kernel change, no user-facing surface).

## 3. Why it stays (keep-grade)

Re-measured a clean win in the predicted direction with the predicted scaling (bigger group → bigger win), zero transcript movement across two models, digit-identical fidelity, and a 3-line-exact arithmetic port. No env gate was kept: the default path IS the hoisted path for q8 now, because identity is proven, not assumed.

## 4. Next winners in the queue (one at a time)

mprefix-table (1.86x prefill-side), 1517-S1 populate port + stage-2 skip, no-gather sparse, kv-auto, GPU readback fix. Each gets the same treatment: implement, remeasure against a saved baseline, keep only on a repeat win.

(End of file.)
