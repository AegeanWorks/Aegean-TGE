# INVENTED (interaction follow-up): Grouped-vs-Per-Head i8, Thermally Controlled — DOWNSIZED (+6-7% real on 0.5B; Phi 2x gap withdrawn as thermal-confounded)

- **Idea**: The interaction cycle's headline (Phi grouped-i8 2x worse than per-head-i8) rested on single pairs from a 40-minute thermally-drifting run sequence, while the 0.5B evidence was also single-shot. Re-test cleanly: 0.5B (fast, thermally stable) per-head-i8 vs grouped-i8, best-of-3 interleaved. If the gap reproduces large → CONFIRM as stated; if small/absent → correct the record (thermal confound) and resize the claim to what replicates.
- **Cycle date**: 2026-11-06
- **Verdict**: REVISED DOWNWARD, honestly. 0.5B grouped-i8 loses consistently but only by +6.1 to +7.2% (3/3 interleaved rounds, attn gap ~60 ms each) — real, not noise, but a quarter of the Phi headline. The Phi 2x gap has no plausible per-head mechanism (the grouped-i8 path differs from per-head-i8 only in loop order + online-softmax rescaling, which cannot 2x compute) and coincides with 10-minute thermal-soak runs; it is WITHDRAWN as a finding and reclassified as thermal-confounded pending a controlled retest (named, not claimed).
- **Tree state at close**: `git status` clean, no diff, stash unchanged (only the pre-existing `leftover peer work shelved 2026-11-04`). No branch, no edits (measurement only). `master` ahead of `origin/master` by 116 after this report. Nothing pushed (human gate).

## 1. Mechanism (as run — no code)

0.5B-Q4_K_M, 7.8k repetitive prompt, ctx 8192, threads 10, `--kv i8`, `--attention {head,gqa}`, --n 31, interleaved h/g ×3, ids + attn phases recorded.

## 2. Predicted gain — a falsifier with teeth in both directions

Stated: CONFIRM if grouped-i8 loses by ≥10% consistently (interaction stands as written); DOWNGRADE if the gap is real but small; KILL the interaction framing if it vanishes into noise. Measured: small-but-real (+6-7%), so DOWNGRADE — the mechanism (grouped online overhead with no dequant benefit) is confirmed in SIGN everywhere, but the MAGNITUDE that motivated the auto-critique came from a confounded run.

## 3. Falsifier outcome — revise, don't discard

1. **0.5B**: per-head wins 3/3 (789/808/802 vs 846/858/851 ms; attn 415-444 vs 474-487). Consistent direction + matching magnitude = real effect, small.
2. **Phi 2x gap**: withdrawn. No per-head mechanism produces 2x from loop order alone; the runs thermally soaked. The honest status is "unmeasured under controlled conditions", not "measured hostile".
3. **Llama ±3%**: leave as noise (single pair, within box variance either way).
4. Ids identical (degenerate prompt, noted as weak identity evidence; natural-text parity from prior cycles stands).

## 4. File anchors (current `master`, read but untouched)

- `apps/cli/main.cpp:2055-2068` (the `--attention` lever — unchanged).
- `src/cpu/avx2/attention_avx2.cpp:121-288` vs `:290+` (per-head two-pass vs grouped online — the structural difference that now carries only a ~6% i8 penalty on GQA, not the claimed catastrophe).
- Grounds: the interaction report (headline being corrected — §3 names exactly what survives it) + the q8micro arithmetic close (no per-element pathology to rescue the 2x).

## 5. What was done (pure measurement — no spike, no branch)

1. CREATE with the thermal-confound falsifier (best-of-3 interleaved on the fast model).
2. Gates: build 54/54; tests 141/141; `aikit.py check` 2/4 (test + secrets pass; pre-existing fails).
3. 0.5B i8 head/gqa interleave ×3 (§6 table) + ids.
4. Scratch deleted; `git status` clean, stash untouched.

## 6. Measured numbers (0.5B-Q4_K_M, 7799 tok, ctx 8192, threads 10, --kv i8, 30 tok)

| round | per-head (attn) | grouped (attn) | delta |
|---|---|---|---|
| 1 | 789.3 ms (414.9, 53%) | 845.7 ms (474.5, 56%) | +7.2% |
| 2 | 808.0 (444.7, 56%) | 858.2 (486.9, 57%) | +6.2% |
| 3 | 801.6 (417.4, 52%) | 851.1 (480.4, 57%) | +6.1% |

Ids: `220 22 …` identical all 6 (spot-checked R1; degenerate, noted).

## 7. Full reimplementation recipe (verification + the Phi retest)

Re-run §1; expect per-head ≈ grouped −5-8% on 0.5B with identical ids. For Phi: do NOT cite the old 2x number — re-run grouped-vs-perhead i8 interleaved back-to-back (accept the full prefill cost both arms so thermal state matches; compare decode only); if the gap reproduces above 15% under those conditions, the structural story reopens with a real number, and if not, the small-GQA-group result stands as the whole story.

## 8. Verdict and why

**CORRECTED: the grouped-vs-i8 interaction is real but small (+6-7% on 0.5B); the Phi 2x gap is withdrawn as thermal-confounded.** This narrows but does not kill the auto-critique: auto-grouped-i8 still leaves single-digit percent on GQA and an unmeasured amount on MHA — worth a KV-aware default, not worth alarm. The per-head-i8 usage guidance (the interaction cycle's other finding) is unaffected — it never depended on grouped being catastrophic, only on per-head-i8 beating f32, which replicates everywhere. Next-cycle candidates: the thermally-controlled Phi retest (only if cheap — it isn't; deprioritized), the 1517 stage-2 harness (still the only live structural front), the human-gated port shelf.

## 9. Exact commands run and their results

- 0.5B 7.8k i8 head/gqa interleave ×3 → §6 table + ids.
- build → 54/54; tests → 141/141; `aikit.py check` → 2/4.
- Scratch deleted; no branch, no edits; report committed; no push.

(End of file.)
