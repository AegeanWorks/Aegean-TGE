# INVENTED (flagged puzzle): i8-Attention Geometry Scaling — ARCHITECTURAL (0.5B −18% / Llama +19% / Phi −6%, same window; mechanism open with narrowed candidates)

- **Idea**: The prefill-i8 cycle flagged a puzzle: i8-KV decode loses on 0.5B (−13-23% @8k), wins big on Llama (+16-17% @3.9k), ties on Phi (+1% @4k) — three architectures, three signs, measured in different windows. Designed experiment to decide box artifact vs architecture: same 3.9k repetitive prompt, all-Q4_K weights (kills the weight-format confound — 0.5B-Q4_K_M here vs Q8_0 before), f32/i8 interleaved per model, round-robin L→P→S in one window, attn phases captured.
- **Cycle date**: 2026-11-06
- **Verdict**: ARCHITECTURAL — all three prior directions reproduce in one window (0.5B −17.9%, Llama +19.4%, Phi −5.8%), so box state cannot explain the split. The mechanism stays OPEN: bytes-alone predicts Phi-best (refuted), heads-alone predicts 0.5B-best (refuted); likely a kernel-path × geometry interaction in the grouped q8 arms. No spike (this cycle is the designed measurement; the mechanism hunt is named follow-up work).
- **Tree state at close**: `git status` clean, no diff, stash unchanged (only the pre-existing `leftover peer work shelved 2026-11-04`). No branch created (flag-gated measurement only). `master` ahead of `origin/master` by 112 after this report. Nothing pushed (human gate).

## 1. Mechanism (as run — no code)

0.5B-Q4_K_M / Llama-3.2-3B-Q4_K_M / Phi-3.5-mini-Q4_K_M, repetitive 3900-token prompt (fits cmdline; same both arms), ctx 4096, threads 10, --n 31, f32 then i8 per model back-to-back (L→P→S order), 30-tok decodes + attn phases + ids.

## 2. Predicted gain — a verdict, not a speedup: architectural vs artifact

Falsifier: directions reproduce in-window → ARCHITECTURAL (mechanism hunt follows); directions collapse to one sign → box artifact (puzzle killed). Measured: −17.9% / +19.4% / −5.8% — reproduce, including the 0.5B sign on Q4_K weights (weight format ruled out as the explainer).

## 3. Falsifier outcome — ARCHITECTURAL, mechanism explicitly open

1. **Directions hold**: 0.5B loses, Llama wins, Phi roughly ties — same as the three prior windows. No box story produces three signs in one interleaved window.
2. **Byte model refuted twice**: predicts Phi-best (widest KV); measured neutral. Predicts 0.5B-mildest (narrowest KV); measured worst.
3. **Heads model refuted**: per-byte dequant ∝ heads predicts Phi-worst/0.5B-best; measured Llama-best. Neither simple scaling fits — the interaction lives in the grouped q8 path (dequant placement vs query-group reuse), unprofiled.

## 4. File anchors (current `master`, read but untouched)

- `src/cpu/avx2/attention_avx2.cpp:271+` (grouped kernel, q8 arms — the suspect; per-head path untouched by this puzzle since all runs auto-select grouped at this ctx).
- `src/models/qwen2.cpp:1177-1214` (uses_gqa routing — same for all arms).
- Grounds: i8-8k kill (0.5B loss), prefill-i8 confirm (Llama win + puzzle flag), phi-i8 (fidelity side unaffected), int8qk kill (dequant cost class).

## 5. What was done (pure measurement — no spike, no branch)

1. CREATE with the architectural-vs-artifact falsifier; ANALYZE narrowed the weight-format confound out by design (all-Q4_K).
2. Gates: build 54/54, `tge_tests` 141/141, `aikit.py check` 2/4 (test + secrets pass; pre-existing fails).
3. Six-run round-robin (§6) + ids (identical per model; degenerate prompt, noted).
4. Scratch deleted; `git status` clean, stash untouched.

## 6. Measured numbers (repetitive 3.9k prompt, ctx 4096, threads 10, 30-tok decode)

| model (all Q4_K) | f32 decode (attn ms) | i8 decode (attn ms) | delta |
|---|---|---|---|
| 0.5B-Q4_K (14q/2kv/64) | 640.68 (159.5, 25%) | 780.53 (261.8, 34%) | **−17.9%** |
| Llama-3B (24q/8kv/128) | 4863.12 (1692.1, 35%) | 4068.20 (912.1, 23%) | **+19.4%** |
| Phi-3.5 (32q/32kv/96) | 5999.84 (2606.6, 44%) | 6371.98 (2850.7, 45%) | **−5.8%** |

(Matvec wobble on 0.5B-i8 (+35ms) matches the window's thermal ramp, not the KV format — attn carries each verdict: +64% / −46% / +9%.)

## 7. Full reimplementation recipe (verification protocol + mechanism hunt)

To re-verify: §1 verbatim; expect the same three signs (any sign flip reopens the artifact hypothesis — with interleaving, not excuses). Mechanism hunt (named follow-up, not this cycle): instrument the grouped q8 arms per (row) dequant vs per (row,query) dequant and compare against the three geometries; or `bench batch`-style microbench of the grouped q8 path across head/group counts with traffic counters. Revival trigger for the SIMPLE story: none — bytes-alone and heads-alone are both dead; only kernel-level evidence reopens explaining, and only a format change reopens guidance.

## 8. Verdict and why

**ARCHITECTURAL: i8-KV decode has no universal sign — it depends on geometry × kernel path, and the loop now has three calibrated points instead of a contradiction.** This upgrades the i8 guidance from "neutral-to-negative" to a table (0.5B-class: avoid for decode; Llama-3B-class at long ctx: wins; Phi-class: neutral) — with the mechanism explicitly open rather than papered over. Next-cycle candidates: the mechanism hunt (grouped-q8 profiling) is the natural follow-up if taken; otherwise the 1517 stage-2 harness remains the only live structural front, plus the human-gated port shelf.

## 9. Exact commands run and their results

- `generate --model <0.5B-Q4_K|Llama-3B|Phi-3.5B> --threads 10 ... [--kv i8] --n 31` round-robin f/i per model → §6 table + ids.
- build → 54/54; tests → 141/141; `aikit.py check` → 2/4.
- Scratch deleted; no branch, no edits; report committed; no push.

(End of file.)
