# INVENTED (mha-i8 follow-up): Prefill-i8 at Long Context — CONFIRM (+12 to +25% on Phi best-of-3 and Llama ×2; KV-write savings land where K/V-reads don't)

- **Idea**: The mha-i8 split cycle killed i8 for DECODE (+1%) but recorded a +17% i8 PREFILL win twice as an un-gated observation. Confirm it with the stated recipe: best-of-3 on Phi + one Llama transfer pair, same 3.9k repetitive shape (bandwidth test; prefill traffic is content-independent to first order — matvec + KV writes dominate).
- **Cycle date**: 2026-11-06
- **Verdict**: CONFIRM. Phi prefill best-of-3: f32 176.8/175.2/173.3 s vs i8 151.1/148.9/151.1 s (best-of-best +14.1%; round deltas +14.5/+15.1/+12.8%). Llama transfer pair + repeat (swapped order): +17.5%, +19.9% (and a +24.7% prefill outlier in i2 — same direction, noted). Ids identical everywhere (degenerate prompt, noted caveat). Guidance: `--kv i8` wins prefill on long contexts via halved KV-write traffic — the write side has no dequant counterpart, so unlike decode reads there is no compute tax to offset the bytes.
- **Tree state at close**: `git status` clean, no diff, stash unchanged (only the pre-existing `leftover peer work shelved 2026-11-04`). No branch created (flag-gated measurement only). `master` ahead of `origin/master` by 111 after this report. Nothing pushed (human gate).

## 1. Mechanism (as run — no code)

Phi-3.5-mini-Q4_K_M + Llama-3.2-3B-Q4_K_M, repetitive 3900-token prompt, ctx 4096, threads 10, --n 31: f32 vs `--kv i8`, Phi best-of-3 (R1 f/i, R2 i/f, R3 f/i), Llama two rounds (f/i, i/f), ids recorded.

## 2. Predicted gain — confirmed above the bar on both models

Required: direction holds + ≥5% on best-of-3 (Phi) and the Llama transfer. Measured: Phi +14.1% best-of-best (all three rounds +12-17%), Llama +17.5%/+19.9% (+24.7% once). Cleared with margin, order-independent.

## 3. Falsifier outcome — CONFIRM prefill; decode puzzle flagged, not pursued

1. **Prefill**: the stated gate, passed on both models.
2. **Decode (out of scope, recorded)**: Phi ~+1% (dead), Llama +16-17% i8 (attn nearly halved: 1670→910 ms) vs 0.5B −13-23% — three architectures, three signs (0/+/−). No model reconciles them yet (KV width? GQA grouping? head_dim?); this is a NAMED next cycle ("i8-attn geometry scaling"), explicitly not answered here. Do not cite the prefill win as covering decode — the two directions are independent measurements.

## 4. File anchors (current `master`, read but untouched)

- `src/models/qwen2.cpp` KV store paths (f32 vs i8 write path — the mechanism).
- Grounds: mha-i8-split (the observation + its confirmation recipe — executed verbatim), i8-8k kill (decode side stays killed), kv-auto (tier policy untouched).

## 5. What was done (pure measurement — no spike, no branch)

1. CREATE with the mha-i8 recipe's falsifier (best-of-3 Phi + Llama transfer, ≥5% both).
2. Gates: build 54/54, `tge_tests` 141/141, `aikit.py check` 2/4 (test + secrets pass; pre-existing fails).
3. Phi R3 + Llama pair + Llama swapped repeat (§6); ids all identical (degenerate, noted).
4. Scratch deleted; `git status` clean, stash untouched.

## 6. Measured numbers (repetitive 3.9k prompt, ctx 4096, threads 10)

| run | model | f32 prefill | i8 prefill | delta |
|---|---|---|---|---|
| Phi R1 (f,i) | 176.77 s | 151.11 s | **+14.5%** |
| Phi R2 (i,f) | 175.25 s | 148.87 s | **+15.1%** |
| Phi R3 (f,i) | 173.27 s | 151.09 s | **+12.8%** |
| Llama R1 (f,i) | 114.50 s | 94.45 s | **+17.5%** |
| Llama R2 (i,f) | 114.75 s | 91.94 s | **+19.9%** |

Phi decode (context): f32 6558-6589 ms vs i8 6525-6524 (+0.7-1.0% — the killed arm, unchanged). Llama decode (out of scope, flagged): f32 4853/4833 vs i8 4137/4155 (+15-17%).

## 7. Full reimplementation recipe (usage rule — no code to port)

To re-verify: §1 on any long prompt (≥2k tokens so KV writes matter); expect i8 prefill ≥ f32 by ~10-25% on 3B-class (scales with KV bytes/token: MHA ≥ GQA). Natural-text transfer still open (repetitive prompt used for cmdline length; traffic argument says it transfers — verify once on prose before putting it in user-facing guidance). Revival trigger: none needed — this is confirmed; the OPEN question is the decode-geometry puzzle (§3.2), which owns the next cycle if taken.

## 8. Verdict and why

**CONFIRM: i8 KV wins prefill on long contexts via halved KV writes — no dequant tax on the write path.** The asymmetry with decode (reads pay dequant, writes don't) is the whole story, and it upgrades the i8 guidance from "memory only" to "memory + prefill speed on long ctx, decode neutral-to-negative". The decode-geometry puzzle (0/+/− across architectures) is flagged with numbers, not solved. Next-cycle candidates: that puzzle, if taken, needs a designed experiment (not a re-run); otherwise the 1517 stage-2 harness remains the only live structural front, plus the human-gated port shelf.

## 9. Exact commands run and their results

- Phi R3 pair (f/i) + Llama pairs (f/i, i/f) → §6 table + identical ids.
- build → 54/54; tests → 141/141; `aikit.py check` → 2/4.
- Scratch deleted; no branch, no edits; report committed; no push.

(End of file.)
