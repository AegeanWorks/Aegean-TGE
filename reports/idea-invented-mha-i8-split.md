# INVENTED: MHA i8 at 4k — SPLIT (decode KILLED at +1%: no MHA flip; prefill +17% twice, PROCEED-grade observation for kv-write savings)

- **Idea**: The 0.5B-GQA i8-8k kill blamed per-row dequant compute. Phi-3.5-mini is full MHA with 3.4x the KV bytes/token — if the byte side ever wins, it wins here. Test i8 vs f32 decode at 4k (cheaper than 8k, still long-context).
- **Cycle date**: 2026-11-06
- **Verdict**: SPLIT. (a) Decode: KILLED — i8 +0.7%/+1.0% across two swapped-order rounds (falsifier needed ≥5%): MHA does not flip the i8 decode verdict; dequant cost scales with the bytes it saves. (b) Prefill: +16.6%/+17.7% i8, twice, both orders — PROCEED-grade observation (halved KV-write traffic on the write-heavy prefill path), with a confirmation recipe, not a claim: needs best-of-3 + transfer checks before it becomes guidance.
- **Tree state at close**: `git status` clean, no diff, stash unchanged (only the pre-existing `leftover peer work shelved 2026-11-04`). No branch created (flag-gated measurement only). `master` ahead of `origin/master` by 111 after this report. Nothing pushed (human gate).

## 1. Mechanism (as run — no code)

Phi-3.5-mini-Q4_K_M (`--kv i8` vs default f32), repetitive 3900-token prompt (fits cmdline; same both arms), ctx 4096, threads 10, --n 31, two rounds swapped order (f/i, i/f), ids recorded.

## 2. Predicted gain — split outcome

Decode: expected a possible flip (MHA bytes). Measured +1%: no flip. Prefill (un-falsified, observed): +17% twice — the KV-write half of prefill traffic (3.0 GB saved of ~5.4 GB total DRAM per full prefill by byte count) lands where the K/V-read half doesn't, because writes are pure traffic while reads pair with the dequant compute.

## 3. Falsifier outcome

- **Decode arm**: KILL. Falsifier required i8 ≥ +5%; measured +0.7% (R1: 6558.5→6527.0) and +1.0% (R2: 6589.6→6524.9). Attn phase explains the miss: i8 saves read bytes but pays per-row dequant; net ~zero even at 3.4x KV.
- **Prefill arm**: no falsifier was stated (observed, not hypothesized) — recorded as a PROCEED-grade observation with its own confirmation recipe (§7), NOT as a verdict. If the confirmation fails, this half dies with it; nothing here is guidance yet.

## 4. File anchors (current `master`, read but untouched)

- `src/cpu/avx2/attention_avx2.cpp` (i8 dequant arms — the decode cost).
- Grounds: i8-8k kill (GQA shape: −13 to −23%), kv-auto (tier policy untouched), phi-i8 (architecture holds — fidelity side unaffected).

## 5. What was done (pure measurement — no spike, no branch)

1. CREATE with a stated decode falsifier (≥5%); prefill recorded opportunistically from the same runs.
2. Gates: build 54/54, `tge_tests` 141/141, `aikit.py check` 2/4 (test + secrets pass; pre-existing fails).
3. Two swapped-order rounds (§6) + ids (identical all 4: degenerate prompt, noted).
4. Scratch deleted; tree verified clean.

## 6. Measured numbers (Phi-3.5-mini-Q4_K_M, 3900 tok, ctx 4096, threads 10)

| round | f32 prefill | i8 prefill | f32 decode (attn) | i8 decode (attn) |
|---|---|---|---|---|
| 1 (f,i) | 176.77 s (22.1 t/s) | 151.11 s (25.8 t/s, **+16.6%**) | 6558.5 ms (2957.3) | 6527.0 ms (2847.0, +0.7%) |
| 2 (i,f) | 175.25 s (22.3 t/s) | 148.87 s (26.2 t/s, **+17.7%**) | 6589.6 ms (2951.7) | 6524.9 ms (2827.9, +1.0%) |

Ids: identical all 4 runs (`29896 29871 13 13 …` — degenerate prompt, weak identity evidence; i8 pricing on natural text stands separately).

## 7. Full reimplementation recipe (two halves)

- **Decode KILL (closed)**: re-run §1; expect i8 ≈ f32 ± noise. Revival trigger: an i8-KV format whose dequant is sublinear in bytes (see int8qk kill for why the obvious version fails).
- **Prefill observation (OPEN — confirmation recipe, not guidance)**: re-run §1 with best-of-3 + one Llama-3B transfer round (same shape, GQA control: expect a smaller win, ~5-8%, since its KV writes are 3.4x smaller); require direction + ≥5% on both models with ids identical; then, and only then, promote to "i8 prefill wins on long ctx" guidance. If either arm fails, kill the observation with it and note the box-drift alternative (these two rounds, though swapped-order, ran in one thermal window).

## 8. Verdict and why

**SPLIT: the MHA decode flip does not exist (+1%), but MHA prefill shows a +17% i8 win twice that the loop did not predict.** The decode arm closes the "bigger KV → i8 wins" story (dequant scales with the bytes it saves — the int8qk theorem, now cross-architectural). The prefill arm opens a narrower, cheaper question (KV-write traffic on the write-heavy path) with a falsifier already attached. Next-cycle candidates: the prefill-i8 confirmation round is the cheapest live item on the board; the 1517 stage-2 harness remains the only live structural front; the human-gated port shelf stands.

## 9. Exact commands run and their results

- `generate --model <Phi-3.5B-Q4_K_M> --threads 10 --gpu off --context 4096 [--kv i8] --prompt <3900-rep> --n 31` ×4 (f,i,i,f) → §6 table + ids.
- build → 54/54; tests → 141/141; `aikit.py check` → 2/4.
- Scratch deleted; no branch, no edits; report committed; no push.

(End of file.)
