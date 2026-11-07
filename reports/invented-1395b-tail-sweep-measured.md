# INVENTED-1395b: exact-tail length sweep (N=32/128 vs 64) — MEASURED: monotonic KL gains to 50x, decode parity at all N

- **Idea ID**: INVENTED-1395b (invented follow-up to Cycle 2 IDEA-1395 slice; measurement-log checked — no prior tail-N work; beellama ladder fixes its tail at 1024, untested on this tree).
- **Cycle date**: 2026-11-07
- **Verdict**: MEASURED — KL improves monotonically with N (flat 2.133e-4 → N32 3.048e-5 → N64 2.087e-5 → N128 **4.307e-6**); top-1 196/197/199/199. Decode at N=128 vs flat: best-of-3 warmed reverse-order 8.08 vs 8.02 t/s (+0.7% → parity), attn-phase ms indistinguishable. Neither falsifier arm tripped. Spike fully reverted; only this report committed.
- **Tree state at close**: `master`, `git status` clean, spike branch created and deleted with nothing committed on it, stash untouched (pre-existing peer stash 2026-11-04 only). Final gates on master: build clean, `tge_tests` 141/141. Nothing pushed (human gate).

## 1. Mechanism

Same ring as Cycle 2 (`TGE_KV_TAIL=N`, exact fp32 copy at store, per-token branch to exact rows for absolute positions `>= span - N` in all 4 kernels). Only the env value changed (32, 128; 64 carried from Cycle 2). Bytes scale linearly: N=32 ~7.3 MB, N=64 ~14.7 MB, N=128 ~29.4 MB resident (28 layers × N × kv_dim 1024 × 4 B × 2, Llama-3B).

Predicted gain: KL(32) between KL(64) and flat if error mass is recency-concentrated; KL(128) ≤ KL(64) marginally. Decode cost ≈ +0.3-1.4% at any N (attn is 4-11% of decode; tail swaps q8 rows for fp32 rows on N positions).

## 2. Falsifier (stated at CREATE)

Non-monotonic KL vs N (indicates a ring-aliasing bug, not physics), or N=128 decode regresses > 5% vs flat i8 best-of-3. Neither tripped: monotonic gains hold (7.0×, then 1.5×, then 4.8× per step), decode +0.7%.

## 3. File anchors

Identical to `idea-1395-tail-slice-measured.md` §3/§7 (5 files, same branches — re-spike from the filed recipe, no recipe changes needed). No new code paths; N flows through `kv_tail_` everywhere.

## 4. What was done (temporary re-spike, fully reverted)

1. `git checkout -b spike-1395b-tailsweep` (no commit ever made on it); re-applied the 5-file Cycle-2 recipe verbatim (only `spike-1395b` comment tags differ).
2. `cmake --build --preset release` clean; `tge_tests` **141/141**.
3. Fidelity, same protocol as Cycle 2 (Llama-3B-Q4_K_M, `docs/architecture.md`, ctx512, 200 tok, 1 thread, `--kv i8`): N=32 → KL **3.048e-5**, worst 3.993e-4, top-1 197/199 (213 s); N=128 → KL **4.307e-6**, worst 4.530e-5, top-1 199/199 (254 s).
4. Thinking-set gate: fox-loop ids identical at N=128 to flat and N=64 (`578 39935 35308 13` ×4).
5. Decode best-of-3, warmed reverse-order, ~1441-ctx prompt, default threads, N=128 vs flat: TAIL128 **8.03/8.08/7.53** vs OFF **7.97/7.53/8.02** → best 8.08 vs 8.02 (+0.7%, parity); attn ms 391-429 vs 395-424.
6. `python aikit.py check aegean-tge` → 2/4 (test + secrets pass; hygiene + guidelines fail pre-existing on master, same record as Cycles 1-2).
7. Reverted all 5 files, deleted branch, rebuilt master clean, `tge_tests` 141/141, `git status` clean.

## 5. Measured numbers (with Cycle-2 points for the curve)

| N (tail) | mean KL | worst KL | top-1 | resident extra |
|---|---|---|---|---|
| 0 (flat i8) | 2.133e-4 | 1.362e-3 | 196/199 | — |
| 32 | 3.048e-5 (7.0×) | 3.993e-4 | 197/199 | ~7.3 MB |
| 64 | 2.087e-5 (1.5×) | 3.287e-4 | 199/199 | ~14.7 MB |
| 128 | **4.307e-6** (4.8×) | 4.530e-5 | 199/199 | ~29.4 MB |

Decode @~1441 ctx, N=128: 8.08 vs 8.02 t/s best-of-3 (+0.7%, noise → parity). Surprise: the 64→128 step (4.8×) beats the 32→64 step (1.5×) — error mass is NOT concentrated in the most recent 64 at ctx 512 on this corpus; mid-distance positions (64-128 back) carry material error. Saturation not reached at 128. Follow-up: confirm at ctx 2048 (where the tail is a smaller fraction) before sizing a default.

## 6. Verdict

Keep-grade curve, no integration proposed yet: N=32 already buys 7× KL for 7 MB; N=128 buys 50× for 29 MB, both at decode parity. Default-N selection needs the ctx-2048 point + a second corpus (single-corpus risk noted: all KL numbers here are architecture.md, 199 positions).

## 7. Reimplementation recipe

Apply `idea-1395-tail-slice-measured.md` §7 verbatim; set `TGE_KV_TAIL` to 32/128. Price with §4 protocol. Kill on §2 falsifier.

## 8. Citation verification

No new external citations (invented idea; carrier = Cycle-2 recipe + beellama ladder already verified in Cycle 2). Internal anchors = Cycle-2 report §3 (verified on `master` before re-spiking; recipe applied without modification — build broke nowhere, confirming the recipe's precision).

## Commands run and their results

- `git checkout -b spike-1395b-tailsweep` → switched; 15 re-applied edits across the same 5 files; `cmake --build --preset release` → 54/54 clean, first try (recipe exact); `tge_tests` → 141/141.
- `tge fidelity` (protocol §4) N=32 → KL 3.048e-5, 197/199, 213 s; N=128 → KL 4.307e-6, 199/199, 254 s.
- `tge generate` fox `--print-ids` N=128 → identical loop ids; ~1441-ctx best-of-3 §4 numbers above.
- `python aikit.py check aegean-tge` → 2/4, pre-existing fails.
- `git checkout -- .` + `git checkout master` + `git branch -D spike-1395b-tailsweep` → master, clean (verified); rebuild + `tge_tests` 141/141.
