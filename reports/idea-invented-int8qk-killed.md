# INVENTED: Integer QK Dots for i8-KV Attention — KILLED (never beats widen-then-FMA: per-chunk reduction + query quant ≥ the maddubs saving at head_dim ≤128)

- **Idea**: The i8-8k finding named the cost: i8-KV attention loses on dequant compute (`q8_dot_avx2` widens every key row to float, then FMAs). Alternative: quantize each head's query ONCE to flat int8 (max-abs/127, symmetric clamp) and run the dots in integer (sign-extend both sides, madd to int32, one float scale per 32-chunk) — ~4-5x fewer dot uops, query-quant amortized over the span.
- **Cycle date**: 2026-11-06
- **Verdict**: KILLED on the stated falsifier (attn ≥20% better than i8-float-dot at 8k). Measured attn: int8qk 601/527/534 ms vs i8-float 544/663/528 ms — never better on a clean pair (R1 +10.5%, R3 +1.1%; R2 −20% is box-load artifact on the f32/i8 arms, noted not claimed). The maddubs saving is eaten by (a) the per-32-chunk int→float reduction (extract/shuffles/cvt, inherent to per-chunk k-scales), (b) the query quant itself, (c) only 2-4 chunks per head at production head_dims — the reduction tax dominates exactly where the dot is short.
- **Tree state at close**: `git status` clean, no diff, stash unchanged (only the pre-existing `leftover peer work shelved 2026-11-04`). Spike branch `spike-int8qk` created (zero commits) and deleted; all three spike files reverted with empty diff verified; scratch deleted. `master` ahead of `origin/master` by 107 after this report. Nothing pushed (human gate).

## 1. Mechanism (as spiked, reverted)

- `include/tge/cpu/kernels.hpp`: `AttentionArgs.query_q8/scale` (null = float path).
- `src/cpu/avx2/attention_avx2.cpp`: `q8_int_dot_avx2` (flat-Q8 query × flat-Q8 keys, madd_epi16 chain, per-chunk float scale combine, scalar tail); QK-loop dispatch on `query_q8 != null`.
- `src/models/qwen2.cpp` (`attention_head`, q8-KV arm, `TGE_INT8_QK=1`): symmetric max-abs query quant into thread-local scratch, passed per head.

## 2. Predicted gain — not realized

Expected ≥20% attn win at 8k from ~4x fewer dot uops. Measured: no win on any clean pair; the integer path costs ≥ the widen path.

## 3. Falsifier outcome — kill

Stated: PROCEED only with attn ≥20% better than i8-float-dot + numerics within class. Perf arm fails outright. Numerics arm PASSES and is recorded (fidelity arch/199-pos: int8qk KL 5.039e-02 vs i8 KL 4.310e-02 = 1.17x, top-1 175/199 both — the query quant adds ~nothing on top of i8-KV; note the arch corpus is brutal for i8 generally, 88% both — corpus effect, not the spike's). A numerics-clean, perf-dead end.

## 4. File anchors (master = pre-spike; spike reverted)

- `src/cpu/avx2/attention_avx2.cpp:59-84` (`q8_dot_avx2` — the widen-then-FMA being attacked).
- `src/models/qwen2.cpp:1155-1162` (q8-KV args block — the hook site).
- Grounds: i8-8k kill (named the dequant cost), kq8 71%-of-pinned (integer-dot precedent that works where chunks are long).

## 5. What was done (temporary spike, fully reverted)

1. `git checkout -b spike-int8qk` (zero commits, deleted at close). Baseline: build 54/54, `tge_tests` 141/141.
2. ANALYZE: confirmed no integer path exists (widen-then-FMA only); quantizer file-local (duplicated flat variant in spike); q8 attention arm reachable via `--kv i8`.
3. Applied §1. Rebuilt 54/54, `tge_tests` 141/141.
4. Fidelity: baseline-i8 vs int8qk, same arch corpus → §3 (numerics pass).
5. 0.5B 7.8k f32/i8/int8qk interleave ×3 (§6; ids identical all 9 — degenerate prompt, recorded with the caveat).
6. Thinking set on spike (saved); `aikit.py check` 2/4 (test + secrets pass; pre-existing fails).
7. Reverted all three files, `git checkout master`, branch deleted, rebuilt master 54/54, tests 141/141, master thinking set: story + creative ids identical vs spike.
8. Scratch deleted; `git status` clean, stash untouched.

## 6. Measured numbers (0.5B-Q8_0, 7799 tok, ctx 8192, 10 threads, 30-tok decode; attn ms)

| round | f32 | i8 (widen-FMA) | int8qk | qk vs i8 |
|---|---|---|---|---|
| 1 | 418.5 | 544.0 | 601.1 | +10.5% |
| 2 | 481.0* | 662.7* | 527.3 | −20.4%* |
| 3 | 400.1 | 528.2 | 533.7 | +1.0% |

\* R2 f32/i8 arms hit box load (matvec 605/839 vs ~350 baseline); qk ran clean after. Artifact favors qk — excluded from the verdict; R1+R3 (clean) both fail it.

## 7. Full reimplementation recipe (kill-grade — rebuild only to re-verify)

Re-apply §1; gate on build 54/54, tests 141/141, fidelity within-class (§3 numbers), and the §6 interleave reproducing ~0-or-worse. Revival trigger: head_dims ≫ 128 (reduction tax amortizes over many chunks) or per-chunk-scale-free i8 KV layout (single scale per row removes the per-chunk float combine) — neither exists on this tree; until one does, integer QK dots stay killed.

## 8. Verdict and why

**KILLED: the integer advantage dies in the per-chunk reduction.** maddubs wins the dot uops, then hands it back in extract/shuffle/cvt per 32 elements plus query quant — at 2-4 chunks per head the tax ≥ the saving. Combined with the i8-8k kill, the attention-i8 line is now fully mapped: bytes win, dequant compute loses, integer re-dots don't recover it. The KV-format guidance stands amended (memory play only, speed tax grows with context). Next-cycle candidates: the 1517 stage-2 skip harness remains the only live structural front; the human-gated port shelf stands.

## 9. Exact commands run and their results

- `git checkout -b spike-int8qk` → branch (zero commits, deleted at close).
- Baseline: build 54/54; `tge_tests` → 141/141.
- Spike (§1); rebuild → 54/54; tests → 141/141.
- Fidelity i8 (KL 4.310e-02, 175/199) vs int8qk (KL 5.039e-02, 175/199) → numerics pass.
- 0.5B 7.8k f32/i8/int8qk interleave ×3 → §6 table (ids identical all 9).
- Thinking set (spike, saved); `aikit.py check` → 2/4 (test + secrets pass; pre-existing fails).
- Revert 3 files → clean; `git checkout master`; branch deleted; rebuild 54/54; tests 141/141; master thinking set → identical; scratch deleted; report committed; no push.

(End of file.)
