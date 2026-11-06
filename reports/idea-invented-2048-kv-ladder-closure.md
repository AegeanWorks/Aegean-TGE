# INVENTED: 2048-History KV Ladder + Head-to-Head Closure — MEASURED (§45 suspension lifted; §29 i8 headline corrected; §41 mechanism confirmed)

- **Idea**: Invented measurement-closure round (no code — precedented by §§43/45/53): finish the suspended §45 2048-history matrix on a healthy box. Same file (Llama-3.2-3B Q4_K_M), same 6 threads, warm page cache, `--prefetch`, adjacent runs: TGE f32/f16/i8 @ ~2050 history + llama-bench cd8427d @ 2048. Closes three open items: (a) §45's suspended f16/i8-at-2048 + parity check, (b) the §29 i8 +32% headline vs today's moved baseline, (c) the TGE/llama 2048 ratio vs the §43 band.
- **Cycle date**: 2026-11-06
- **Verdict**: MEASURED/CLOSED. Health gate held end to end (wall 57.66 → 57.89 → 60.07 → 65.39 read GB/s; story 10.73 t/s with baseline-identical ids) — zero voids, the suspension is lifted, not extended. Ladder at 2048 (decode means): f32 8.26 → i8 9.00 (+9%) → f16 9.26 (+12%): f16 is the best 2048 card, i8 beats f32 but loses to f16; the §29 +32% i8 headline does not replicate (it measured a slower pinned-era f32 baseline) and is corrected here. TGE/llama @2048: prefill 60% (above the §43 52–56% band — kq8 + healthy box), decode 52% (below the 56–70% band — exactly the §41 mechanism: freed bandwidth favors bandwidth-bound llama over unpack-ALU-bound TGE). All three KV ids byte-identical (62 ids).
- **Tree state at close**: `git status` clean (no diff, no untracked), stash unchanged (only the pre-existing `leftover peer work shelved 2026-11-04`, untouched). No branch was created (no code exists to spike — measurement-only round); nothing to revert. `master` ahead of `origin/master` by 54 before this report. Nothing pushed (human gate).

## 1. Mechanism (measurement design, no code)

2050-token fox-sentence prompt (8999+ chars → exactly 2050 tokens on the Llama tokenizer, verified in-run), `--context 4096`, `--n 63` (62 emitted after EOS), `--threads 6`, `--prefetch` (fault-in untimed, §36 precedent), three KV arms back-to-back plus interleaved repeats with ids; llama-bench `-p 2048 -n 64 -t 6 -r 3` adjacent (built-in prompt — token-count-matched per the §36 precedent, not text-matched). Health gate per the §45 rule (wall 48–59+ AND story 9–11 together, re-checked between cells; collapse voids the cell).

## 2. Predicted gain — stated expectations going in (the falsifier's arms)

(a) i8 +32% decode vs f32 at 2048 (replicate §29: 5.80 → 7.63). (b) f16 +14% (replicate §29 paired: 6.61 vs 5.80). (c) TGE/llama in the §43 band (prefill 52–56%, decode 56–70%). Stated falsifier: any cell whose health gate fails is voided, not claimed; if i8 ≤ f32 at 2048 paired, the §29 claim is load narrative and gets corrected; if ratios leave the §43 band, the mechanism (not the numbers) must explain it.

## 3. Falsifier outcome — one arm corrects, one confirms, one splits

(a) FIRES (correction): i8 decode +9–11% over f32 (8.94/9.06 vs 8.09/8.42), not +32% — and i8 loses to f16 (−3.4%). The §29 headline measured a pinned-era f32 baseline on a mid-load box (5.80 t/s); today's f32 baseline is structurally faster (unpinned decode pools §28ab, physical-core default §69) on a healthy box, so the gap compressed. (b) HOLDS: f16 +8.8–15.7% (9.16/9.36), inside the §29 +14% band modulo box noise. (c) SPLITS with mechanism: prefill 53.9/89.16 = 60% (above band — kq8 flip + healthy box), decode 8.26/16.02 = 52% (below band — §41's freed-bandwidth mechanism, now confirmed on a healthy box rather than inferred from a quiet one).

## 4. File anchors (no code this round — references only)

- `docs/measurement-log.md` §§29 (i8 +32% precedent), 41 (TGE unpack-ALU-bound vs llama bandwidth-bound), 43 (128/512 re-baselined band), 45 (suspended 2048 matrix + health-gate rule), 61–62 (kq8 default-switch + post-flip head-to-head).
- Reference model: `Llama-3.2-3B-Instruct-Q4_K_M.gguf`; llama pin: `llama.cpp-cd8427d` (`llama-bench`/`llama-cli` built MinGW-GCC 16.2, §28al/§53).
- Prompt: fox sentence ×205 (token count verified 2050 in-run, §5).

## 5. What was done (measurement-only — no spike, precedented)

1. Gates first: `cmake --build --preset release` clean (54/54); `ctest --preset release` pass; `tge_tests.exe` 141/141; `python aikit.py check aegean-tge` 2/4 (test + secrets pass; hygiene + guidelines fail pre-existing).
2. Health gate: `bench mem` wall read 57.66 + story 10.73 t/s with ids `11 304 … 11 1070` (exact §46 baseline match) — both halves green.
3. TGE f32 @2050 → llama-bench @2048 → wall re-check → TGE f16 @2050 → TGE i8 @2050 → wall → f32 rerun (drift + ids) → f16 ids → i8 ids + closing wall (65.39).
4. No thinking-set diff owed: no code delta exists to move outputs (same-binary precedent §§43/45); ids parity across KV arms measured instead (§6).
5. No branch, no edits, nothing to revert; tree verified clean + stash unchanged at close.

## 6. Measured numbers (2050-tok prompt, 62–63 tok decode, 6 threads, warm cache)

| cell | prefill | decode | decode phases (matvec/attn/norm) | ids |
|---|---|---|---|---|
| TGE f32 R1 | 53.6 t/s | 8.09 t/s | 78% / 20% / 2% | — |
| TGE f32 R2 | 54.2 t/s | 8.42 t/s | 78% / 21% / 2% | periodic `578 4062 …` (62) |
| TGE f16 R1 | 53.7 t/s | 9.36 t/s | 85% / 13% / 2% | — |
| TGE f16 R2 | — | 9.16 t/s | 85% / 13% / 2% | identical to f32 (62/62) |
| TGE i8 R1 | 53.9 t/s | 8.94 t/s | 83% / 15% / 2% | — |
| TGE i8 R2 | — | 9.06 t/s | 83% / 15% / 2% | identical to f32 (62/62) |
| llama-bench pp2048 / tg64 | 89.16 ± 0.91 | 16.02 ± 0.13 | — | — |

Reading, in falsifier order. Prefill is KV-blind (53.6–53.9 all three — matvec-bound, as constructed). Decode ladder: f32 8.09/8.42 → i8 8.94/9.06 (+9–11%) → f16 9.16/9.36 (+9–16%): f16 best, i8 middle — the §41 512-history pattern ("f16 best, i8 worst") extended to 2048 with i8 promoted to middle. The attention share tells the mechanism: f32 20–21% (GQA at 2000+, first history where attention is not noise — §45's 19% note now at 8 t/s, not 3) → f16 13% → i8 15%: the KV-read saving shows exactly where the ladder says it should. Head-to-head: prefill 60% (above band), decode 52% (below band). Wall rose monotonically across the window (57.66 → 65.39) while the ladder order never flipped and f32 drifted +1–4% between its own runs — ratios are window-paired, absolutes unclaimed beyond the window. Single runs per cell (R2 repeats for all three TGE arms, not best-of-3): stated as a limitation; the wall stability + tight llama band (±1%) + R1/R2 agreement is the load control.

## 7. Full reimplementation recipe (measurement protocol — rerun to re-verify)

1. Health gate: `bench mem` wall read ≥48 AND Llama-3B story (`--prompt "Once upon a time" --n 15 --threads 6 --prefetch`) 8–12 t/s with ids `11 304 … 11 1070`; abort/void on either half failing.
2. Prompt: fox sentence ×205 (`--context 4096 --n 63 --threads 6 --prefetch`), expect exactly 2050 prompt tokens; run f32, f16 (`--kv f16`), i8 (`--kv i8`) back-to-back with wall re-checks; expect §6 rows within box noise (±20% absolutes, ladder order f32 < i8 ≤ f16).
3. llama-bench `-p 2048 -n 64 -t 6 -r 3` adjacent; expect prefill ratio ~55–65%, decode ratio ~50–60% (band updated by this round: prefill band widens upward with kq8, decode band widens downward per §41).
4. Parity: `--print-ids` on all three KV arms; expect byte-identical 62 ids. Retry trigger: any ladder inversion on a re-run with wall + story both green → the inversion is evidence, not noise, and reopens the KV-format question.

## 8. Verdict and why

**MEASURED/CLOSED.** The §45 suspension is lifted (health held throughout, zero voids), the 2048 KV ladder is now f16 > i8 > f32 with ids parity, the §29 +32% i8 headline is corrected to +9–11% (baseline moved, not the kernel), and the §41 freed-bandwidth mechanism graduates from quiet-box inference to healthy-box confirmation. Standing guidance sharpened: at 2048 history the KV tier of choice is f16 (+12%, ids-identical, halves the KV 448 → 224 MiB); i8 stays a long-context/multi-session memory card (per the size-ladder conditional: ≥3B diverse holds, sub-3B prefers f32/f16), not a head-to-head lever.

## 9. Exact commands run and their results

- `cmake --build --preset release` → 54/54; `ctest --preset release` → pass; `tge_tests.exe` → 141/141.
- `python aikit.py check aegean-tge` → 2/4 (pre-existing hygiene + guidelines fails).
- Health: wall 57.66 + story 10.73 t/s (baseline ids) → green.
- TGE f32 R1 53.6/8.09; llama 89.16±0.91/16.02±0.13; f16 R1 53.7/9.36; i8 R1 53.9/8.94 (walls 57.89/60.07 between).
- f32 R2 54.2/8.42 + ids; f16 R2 9.16 + identical ids; i8 R2 9.06 + identical ids; closing wall 65.39.
- No branch, no edits, no fidelity (no new numerics — KV tiers already priced), no bench-kernel A/B (end-to-end round), no push.

(End of file.)
