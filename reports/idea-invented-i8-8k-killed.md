# INVENTED: i8 KV at 8k Context — KILLED (i8 loses −13 to −23%: dequant compute dominates the byte saving, worse with context)

- **Idea**: The naive bandwidth math says i8 KV should WIN at long context: decode is KV-read-bound, attention is ~38% of decode at 8k (vs ~20% at 2k), and i8 halves K/V bytes. The only i8-vs-f32 decode numbers on record stop at 2048 (+3.8%, neutral, drift-contaminated). Run the A/B at 8192 where the KV share doubles.
- **Cycle date**: 2026-11-06
- **Verdict**: KILLED, with a signed measurement: i8 LOSES −12.5% to −23.2% at 8k (attn phase +33-42% worse). The byte saving is real but the per-row i8 dequant in attention costs more compute than the traffic saves — and the gap WIDENS with context (neutral at 2k → −17% at 8k), the opposite of the naive prediction. No spike (flag-gated measurement, precedented); no branch, no edits, nothing to revert.
- **Tree state at close**: `git status` clean, no diff, stash unchanged (only the pre-existing `leftover peer work shelved 2026-11-04`). `master` ahead of `origin/master` by 106 after this report. Nothing pushed (human gate).

## 1. Mechanism (as run — no code)

0.5B-Q8_0, 7799-token repetitive prompt (`"7 "`×3900, 15.6KB text — fits the Windows cmdline limit that blocks an 8k natural-text prompt; same prompt both arms so bandwidth, not content, is compared; AR decode, no speculation), `--context 8192`, threads 10, `--n 31`: f32 vs `--kv i8`, best-of-3 interleaved. Repetition is deliberate and precedented (2048-closure used repetitive corpora for KV shapes).

## 2. Predicted gain — inverted

Expected ≥+5% i8 win (halved K/V bytes on a KV-bound phase). Measured −13 to −23%. The model's error: traffic isn't the binding constraint in the i8 attention kernel — per-row dequant compute is.

## 3. Falsifier outcome — kill, with a signed direction

Stated: PROCEED only if i8 ≥ +5% at 8k. All three interleaved pairs go the other way (−16.8%, −12.5%, −23.2%), and the attn phase gap (+33-42%) names the mechanism. Ids identical 6/6 (degenerate prompt — weak identity evidence on its own, but the i8 price is established separately; recorded as a caveat, not a gate).

## 4. File anchors (current `master`, read but untouched)

- `src/cpu/avx2/attention_avx2.cpp` (i8 K/V dequant arms — the per-row cost that eats the byte saving).
- Grounds: 2048-closure (+3.8% neutral at 2k), kv-auto (tier policy stands — this round doesn't move it), diverse/i8 ladders (fidelity side unaffected).

## 5. What was done (pure measurement — no spike, no branch)

1. ANALYZE: confirmed attn share doubles with context (20%→38%), i8 halves the bytes, 2048 showed only +3.8% (neutral) — the 8k shape unmeasured.
2. Solved the 8k-prompt delivery problem (cmdline limit → short-token repetition; `--tokens` id-list has the same limit; no `--prompt-file` exists).
3. Gates: `cmake --build --preset release` 54/54; `tge_tests.exe` 141/141; `aikit.py check` 2/4 (test + secrets pass; hygiene + guidelines fail pre-existing, same pair as every prior cycle).
4. Warmup + best-of-3 interleaved f32/i8 (§6). Scratch deleted; `git status` clean, stash untouched.

## 6. Measured numbers (0.5B-Q8_0, 7799 tok ctx 8192, 10 threads, 30-tok decode)

| round | f32 t/s (attn ms) | i8 t/s (attn ms) | delta |
|---|---|---|---|
| 1 | 36.90 (405.4, 50%) | 30.70 (547.7, 57%) | −16.8% |
| 2 | 35.77 (413.7, 50%) | 31.31 (537.9, 57%) | −12.5% |
| 3 | 34.54 (419.6, 49%) | 26.54 (597.0, 53%) | −23.2% |

Box drift noted (both arms crept run-over-run); pairs adjacent, direction consistent 3/3. No fidelity run owed (no numerics changed — flag-gated priced path); no separate thinking set (the 31-token decodes ARE the outputs; ladder precedent).

## 7. Full reimplementation recipe (kill-grade — rebuild only to re-verify)

Re-run §1 (any long repetitive prompt filling the context; `--tokens` for <2k, short-token text above that). Expect i8 ≈ −15% ± 8% on 0.5B with the attn phase carrying the gap. Revival trigger: an i8 attention kernel whose per-row dequant costs less than the byte saving at long ctx (measure the kernel, not the bytes); until then the KV-format guidance gains a context qualifier — i8 is a MEMORY play (4x smaller cache, longer contexts fit), not a decode-speed play, and the speed tax grows with context (neutral at 2k → ~−17% at 8k).

## 8. Verdict and why

**KILLED: the byte saving loses to dequant compute, increasingly with context.** This sharpens rather than overturns the standing guidance: kv-auto's policy stands, i8's fidelity price stands — but no one should expect i8 decode to get FASTER at long context. The mechanism (per-row dequant in attention) is named, so the next move if any is a kernel change, not a re-measurement. Next-cycle candidates: the 1517 stage-2 skip harness remains the only live structural front; the human-gated port shelf stands.

## 9. Exact commands run and their results

- Prompt-shape scouting (`"7 "` fertility: 2 tokens per unit; `--tokens` list limits confirmed) → 7799-token prompt.
- Warmup run → prefill 62.5 s, decode attn 47% (premise confirmed live).
- `generate --model <0.5B-Q8_0> --threads 10 ... [--kv i8] --n 31` interleaved ×3 → §6 table.
- `cmake --build --preset release` → 54/54; `tge_tests.exe` → 141/141; `aikit.py check aegean-tge` → 2/4 (test + secrets pass; pre-existing hygiene + guidelines fails).
- Scratch deleted; no branch, no edits; report committed; no push.

(End of file.)
