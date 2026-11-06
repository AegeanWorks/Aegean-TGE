# INVENTED: MHA-vs-GQA Bandwidth-Model Test (Phi vs Llama, Short + 8k Ctx) — MEASURED, model refined (short confirms weights; 8k refutes bytes: decode is compute-bound)

- **Idea**: Phi-3.5-mini is full MHA (32/32 heads, 3.4x the KV bytes/token of GQA Llama-3B at matched hidden size) — a natural experiment for the loop's bandwidth model of decode. Two-regime quantitative prediction: short ctx (weights dominate) → Phi/Llama ≈ 1/1.19 file-size ratio ≈ 0.84x; 8k ctx (KV dominates) → per-token bytes (1.88+2.0 vs 6.4+2.4 GB) → ≈ 0.44x.
- **Cycle date**: 2026-11-06
- **Verdict**: MEASURED with model correction. Short ctx: 0.76x vs 0.84x predicted — CONFIRMS the weights-dominated regime (MHA penalty negligible when KV is small). 8k: 0.77x vs 0.44x predicted — REFUTES the byte model: both engines are compute-bound (attn wall 3-4x the traffic floor), so extra KV bytes cost a fraction, not the whole. The MHA tax is real but modest: +32% short (weights + structure), +31% wall / +44% attn at 8k. No spike (pure measurement, precedented); no branch, no edits.
- **Tree state at close**: `git status` clean, no diff, stash unchanged (only the pre-existing `leftover peer work shelved 2026-11-04`). `master` ahead of `origin/master` by 107 after this report. Nothing pushed (human gate).

## 1. Mechanism (as run — no code)

Story prompt (natural text, --n 31... 30 decoded) best-of-3 interleaved Llama/Phi, threads 10; 8k repetitive prompt (`"7 "` fertility-matched: 7799 vs 7800 tokens, ctx 8192) Llama/Phi pair + Llama repeat, threads 10, --n 31, ids recorded. Same-window, warm page cache.

## 2. Predicted gain — a model test, split verdict

Short: 0.84x ± 20% → measured 0.76x (best-of-best 9.36/12.32). Long: 0.44x ± 25% ([0.33, 0.55]) → measured 0.77x (3.11/4.06). Short confirms; long refutes decisively (outside even with ±20% box error).

## 3. Falsifier outcome — refine, don't discard

1. **Short-ctx arm passes**: 0.76x vs 0.84x — weights dominate, MHA compute overhead ≈ small.
2. **Long-ctx arm fails the byte model**: 0.77x vs 0.44x. Traffic-floor math: Llama attn 139 ms/tok vs 34 ms floor; Phi 200 ms vs 116 ms floor — both ~2-4x compute-dominated, so 3.4x KV bytes cost ~+44% attn, not ~+240%.
3. The loop's "decode is bandwidth-bound" shorthand needs the compute correction (consistent with kq8-71% and shuffle-bound findings: this box's decode is dot-bound first, traffic-bound second).

## 4. File anchors (current `master`, read but untouched)

- Model files (reference + new): sizes 2.0 vs 2.4 GB; configs (Llama 28×24/8×128; Phi 32×32/32×96).
- Grounds: 2048-closure (decode composition), kv-auto (tier rule untouched), phi-i8 (Phi runs correctly).

## 5. What was done (pure measurement — no spike, no branch)

1. ANALYZE: byte/token arithmetic for both models, both regimes; predictions locked BEFORE measuring (§2).
2. Gates: build 54/54; `tge_tests` 141/141; `aikit.py check` 2/4 (test + secrets pass; pre-existing fails).
3. Story best-of-3 interleaved (§6); 8k fertility check (matched to 1 token); 8k pair + Llama repeat (Phi R2 skipped: 10 min/point, falsification already decisive — even ±20% keeps 0.77 outside [0.33, 0.55]; stated, not hidden).
4. Retrospective box-timeout note: the first 8k batch exceeded the 30-min tool cap (runs are 7-10 min each on 3B-class) — partial files inventoried, complete runs kept, no number invented.
5. Scratch deleted; `git status` clean, stash untouched.

## 6. Measured numbers

Story (natural text, 30 tok, threads 10): Llama 11.40/12.32/10.21 t/s; Phi 8.26/9.36/8.63 t/s → **0.76x** (best-of-best).

8k repetitive (7799/7800 tok, ctx 8192, threads 10, 30 tok):
| run | prefill | decode | attn share |
|---|---|---|---|
| Llama R1 | 19.0 t/s (410 s) | 4.06 t/s | 4163.8 ms (56%) |
| Phi R1 | 12.9 t/s (607 s) | 3.11 t/s | 6005.2 ms (62%) |
| Llama R2 | 19.0 t/s (411 s) | 4.12 t/s | 4083.6 ms (56%) |

**Ratio: 3.11/4.06 = 0.77x** (predicted 0.44x). Prefill also informative (19.0 vs 12.9 t/s = 0.68x — MHA KV-write traffic bites prefill harder than decode, still far from the byte ratio).

## 7. Full reimplementation recipe (verification protocol — re-run to re-verify)

1. Same prompts both models (natural story + fertility-matched repetitive long prompt, ctx 8192).
2. Expect: short ≈ weight ratio (±20%), 8k ≈ 0.7-0.8x regardless of KV-byte ratio (compute-bound regime on AVX2 CPU).
3. Revival trigger: a box where attn wall approaches the traffic floor (roof-rate streaming with free dots — e.g., a dot-offload accelerator); there the byte model should take over and the ratio should fall toward 0.44x. Until then, quote compute, not bytes.

## 8. Verdict and why

**MEASURED: MHA costs real but modest money, and the currency is compute, not bytes.** +32% short (weights + structure), +31% wall / +44% attn at 8k — a far cry from the 2.3x the byte model feared, because this box's decode is dot-bound first. The practical upshot for the loop: GQA's win is confirmed but bounded (~1/3, not ~1/2+), and every future "bytes saved" claim must clear the compute-dominance check first (the i8-8k and int8qk kills were early instances of the same lesson). Next-cycle candidates: the 1517 stage-2 skip harness remains the only live structural front; the human-gated port shelf stands.

## 9. Exact commands run and their results

- Gates: build → 54/54; tests → 141/141; `aikit.py check` → 2/4.
- Story Llama/Phi interleave ×3 → §6 story row.
- Fertility check (`"7 "`×3900 → 7799/7800) → 8k prompt shape.
- 8k Llama/Phi/Llama → §6 table (Phi R2 skipped per §5).
- Scratch deleted; no branch, no edits; report committed; no push.

(End of file.)
