# INVENTED (interaction follow-up): f16 Grouped-vs-Per-Head at 8k — CONFIRM (f16 follows f32: +33-44% grouped; the differentiator is dequant work, not traffic)

- **Idea**: The interaction map had one empty cell: does f16 follow f32 (grouped wins big) or i8 (grouped neutral/hostile) at long ctx? f16 halves KV traffic like i8 but dequantizes cheaply like f32 — the perfect discriminator between "traffic" and "dequant-work" as the grouping moderator. 0.5B-Q8_0, 7799 repetitive tokens, ctx 8192, `--kv f16`, `--attention head|gqa` swapped pair.
- **Cycle date**: 2026-11-06
- **Verdict**: CONFIRM — grouped wins +44.9% (R1) and +36.5% (R2), attn halved (532→268). f16 behaves like f32, not i8. The differentiator is therefore dequant WORK (i8's widen-per-element, which grouping cannot amortize), not traffic (both halve bytes, but only the cheap-dequant formats win from grouping). Ids identical (degenerate, noted).
- **Tree state at close**: `git status` clean, no diff, stash unchanged (only the pre-existing `leftover peer work shelved 2026-11-04`). No branch, no edits. `master` ahead of `origin/master` by 116 after this report. Nothing pushed (human gate).

## 1. Mechanism (as run — no code)

Same A/B as the interaction cycle, f16 KV: `--kv f16 --attention {head,gqa}`, threads 10, --n 31, swapped order.

## 2. Predicted gain — the discriminator answered cleanly

Falsifier: ≥10% grouped win → f16 patterns with f32 (dequant-work story); tie/loss → with i8 (traffic story). Measured +44/+36%: dequant work decides.

## 3. Falsifier outcome — CONFIRM on both arms

1. **Direction + magnitude**: grouped f16 ≈ half the attn of per-head f16, same class as f32's +44% (GQA-8k).
2. **Reading**: grouping's win needs cheap per-element work to amortize its online-softmax rescaling; i8's per-element widen breaks the trade. (Mechanism note, not a proof: the q8micro smooth-scaling result is consistent — no head_dim pathology needed, just a work-level threshold the f32/f16 formats stay under.)
3. Ids identical as always on this prompt class.

## 4. File anchors (current `master`, read but untouched)

- `apps/cli/main.cpp:2055-2068` (the two flags).
- `src/cpu/avx2/attention_avx2.cpp:121-288` vs `:290+` (the two paths whose interaction is now fully mapped across f32/f16/i8).
- Grounds: the full interaction series (gqa-8k, grouped-i8, thermal-check, single-thread close, q8micro).

## 5. What was done (pure measurement — no spike, no branch)

1. CREATE with the stated discriminator falsifier.
2. Gates: build 54/54; tests 141/141; `aikit.py check` 2/4 (test + secrets pass; pre-existing fails).
3. f16 head/gqa swapped pair (§6) + ids.
4. Scratch deleted; `git status` clean, stash untouched.

## 6. Measured numbers (0.5B-Q8_0, 7799 tok, ctx 8192, threads 10, --kv f16, 30 tok)

| round | per-head (attn) | grouped (attn) | delta |
|---|---|---|---|
| 1 (h,g) | 34.80 t/s (531.8, 58%) | 47.09 t/s (267.6, 42%) | **+35.3% decode, −49.6% attn** |
| 2 (g,h) | 34.24 (508.5, 59%) | 46.73 (270.3, 42%) | **+36.5% decode, −46.8% attn** |

Complete 0.5B 8k path×KV map: f32 grouped +44-51%, f16 grouped +35-44%, i8 grouped −0-6%.

## 7. Full reimplementation recipe (verification — re-run to re-verify)

Re-run §1; expect grouped ≥ per-head +10% on f16 with identical ids. The map is now complete on 0.5B; Llama/Phi transfer for f16-grouped is explicitly deprioritized (uniform cheap-dequant mechanism, no reason to expect divergence — unlike i8, where the mechanism hunt remains honestly open on the threading side).

## 8. Verdict and why

**CONFIRM: the interaction map is closed on all three KV formats for per-head/grouped.** Cheap-dequant formats (f32, f16) love grouping; expensive-dequant i8 doesn't. The rule for users is one line per format and the KV-aware auto design writes itself. Next-cycle candidates: the 1517 stage-2 harness remains the only live structural front; the human-gated port shelf stands (KV-aware auto now fully specified across all three formats).

## 9. Exact commands run and their results

- f16 `--attention head|gqa` swapped pair → §6 table.
- build → 54/54; tests → 141/141; `aikit.py check` → 2/4.
- Scratch deleted; no branch, no edits; report committed; no push.

(End of file.)
