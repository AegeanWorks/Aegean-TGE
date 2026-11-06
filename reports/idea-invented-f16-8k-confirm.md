# INVENTED: f16 KV at 8k — CONFIRM (+11 to +19% decode; the auto rule scales with context as predicted)

- **Idea**: kv-auto sends ≥512 ctx to f16 (validated at 2048: +12%), but never ran at 8k where halved KV bytes should matter more. f16's dequant (single cvt) is cheap unlike i8's (widen+scale), so unlike i8-8k (killed: −13-23%), f16 should GROW its win with context. Test f32 vs f16 decode at 8192 on 0.5B, best-of-3 interleaved.
- **Cycle date**: 2026-11-06
- **Verdict**: CONFIRM — f16 wins +15.8%/+19.4%/+10.9% (attn −25%, matvec flat as expected, ids identical all 6). The auto rule scales exactly as predicted (bigger KV share → bigger f16 win: +12% at 2k → +15% at 8k). No spike (flag-gated priced tiers, precedented); no branch, no edits.
- **Tree state at close**: `git status` clean, no diff, stash unchanged (only the pre-existing `leftover peer work shelved 2026-11-04`). `master` ahead of `origin/master` by 116 after this report. Nothing pushed (human gate).

## 1. Mechanism (as run — no code)

0.5B-Q8_0, repetitive 7.8k prompt (bandwidth test, same both arms), ctx 8192, threads 10, --n 31, f32 vs `--kv f16`, interleaved best-of-3, ids + attn phases recorded.

## 2. Predicted gain — confirmed at the top of the range

Falsifier: ≥+5% with identical ids (else reopen auto as ctx-capped). Measured +11 to +19% — the win grows with the KV share as predicted.

## 3. Falsifier outcome — CONFIRM cleanly

All three interleaved pairs favor f16 by double digits; attn 345-360 ms → 262-269 ms (−25%); matvec unchanged (316-332 both arms — f16 touches KV only); ids byte-identical.

## 4. File anchors (current `master`, read but untouched)

- `src/models/qwen2.cpp` (kv_half path — the priced tier under test).
- Grounds: kv-auto (rule validated here at a new ctx), i8-8k kill (the contrast case — dequant cost decides), fidelity (f16 1.7e-6 class stands).

## 5. What was done (pure measurement — no spike, no branch)

1. CREATE with the stated falsifier (auto extends to 8k or gets a ctx cap).
2. Gates: build 54/54, `tge_tests` 141/141, `aikit.py check` 2/4 (test + secrets pass; pre-existing fails).
3. Warmup + best-of-3 interleaved f32/f16 (§6).
4. No fidelity run owed (f16 priced tier, no numerics changed); thinking set = the 6 decodes' ids (identical).
5. Scratch deleted; `git status` clean, stash untouched.

## 6. Measured numbers (0.5B-Q8_0, 7799 tok, ctx 8192, threads 10, 30 tok)

| round | f32 (attn) | f16 (attn) | delta |
|---|---|---|---|
| 1 | 40.72 t/s (359.6) | 47.16 t/s (265.2) | +15.8% |
| 2 | 40.05 (360.0) | 47.80 (261.7) | +19.4% |
| 3 | 41.43 (344.8) | 45.95 (268.9) | +10.9% |

## 7. Full reimplementation recipe (verification — re-run to re-verify)

Re-run §1; expect f16 ≥ f32 +5% with identical ids at any ctx ≥2k (larger ctx → larger win, bounded by attn's share of decode). Revival trigger: an f16 loss at any ctx with identical setup — reopens auto with a ctx cap.

## 8. Verdict and why

**CONFIRM: kv-auto's f16 tier scales with context as designed.** Combined with the i8-8k kill (dequant-heavy loses) and prefill-i8 confirm (writes win), the KV-format guidance is now complete across tiers × paths × context: f32 short/no-think, f16 decode at long ctx, i8 prefill at long ctx + memory pressure. Next-cycle candidates: the 1517 stage-2 harness remains the only live structural front; the human-gated port shelf stands.

## 9. Exact commands run and their results

- Warmup (7.8k) + f32/f16 interleave ×3 → §6 table.
- build → 54/54; tests → 141/141; `aikit.py check` → 2/4.
- Scratch deleted; no branch, no edits; report committed; no push.

(End of file.)
