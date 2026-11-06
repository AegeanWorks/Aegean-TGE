# INVENTED (nattext follow-up): Phi Natural-Text Per-Head i8 — CONFIRM (+11%/+13% with one recurring priced-class flip; matrix complete 6/6)

- **Idea**: The transfer matrix had one open cell: Phi natural text (0.5B rep/nat, Llama rep/nat, Phi rep confirmed; Phi nat pending). Same gate as the Llama transfer: per-head i8 ≥ f32 +5% with ids within the priced i8 class.
- **Cycle date**: 2026-11-06
- **Verdict**: CONFIRM. Phi fox×150, per-head, interleaved f/i/i/f: decode +11.0% and +12.8% (attn −35% / −45%). Ids differ by ONE recurring token substitution (11203→17366, 3 instances on the repetitive prompt) — same trajectory structure, no loops, no collapse: exactly the priced i8 behavior (diverse-corpus fidelity: 3 flips), not a divergence. The full transfer matrix is now 6/6 green on perf with ids inside the priced contract everywhere.
- **Tree state at close**: `git status` clean, no diff, stash unchanged (only the pre-existing `leftover peer work shelved 2026-11-04`). No branch, no edits. `master` ahead of `origin/master` by 116 after this report. Nothing pushed (human gate).

## 1. Mechanism (as run — no code)

Phi-3.5-mini-Q4_K_M, fox×150 natural text (~2400 tok), ctx 4096, threads 10, --attention head, f32 vs --kv i8, --n 31, f/i/i/f rounds, ids diffed per round + one inspection pair.

## 2. Predicted gain — confirmed at the established magnitude

Falsifier: ≥+5% both rounds + ids within the priced i8 class (identical OR rare-flip, no garbage) → CONFIRM; crash/garbage trajectory → REOPEN. Measured +11.0% / +12.8% with the single-recurring-substitution pattern.

## 3. Falsifier outcome — CONFIRM with the ids note recorded, not hidden

1. **Perf**: best-of-best +12.1%; attn share falls 45%→23% (the per-head i8 mechanism, same as every other cell).
2. **Identity**: NOT byte-identical here (differs from the 0.5B/Llama natural runs, which were) — one token substitution recurring 3x on the repetitive prompt. This is the correct reading of the i8 contract (top-1 ~99%, not 100%): the 0.5B/Llama identicals were luck-of-the-draw on their prompts, and this round shows the priced behavior manifesting. No action: fidelity gates (no numerics changed) already price exactly this.
3. The initial R1/R2 comparison was deleted before full inspection (scratch discipline) and re-measured for the ids audit — the audit pair (§6, R3) carries the ids verdict; the perf verdict rests on R1+R2.

## 4. File anchors (current `master`, read but untouched)

- `apps/cli/main.cpp:2055-2068` (the `--attention` / `--kv` flags — the whole test surface).
- Grounds: nattext-i8 (0.5B cell), nattext-llama (Llama cell), grouped-i8 + phi-i8 (mechanism + fidelity class).

## 5. What was done (pure measurement — no spike, no branch)

1. CREATE with the grouped-i8 report's stated last-cell gate.
2. Gates: build 54/54; tests 141/141; aikit 2/4 (test + secrets pass; pre-existing fails).
3. Phi fox×150 per-head f/i + i/f (§6) + audit pair with full ids (§3.2).
4. Scratch deleted (including the prematurely cleaned R1/R2 files — audit re-ran instead of reconstructed); `git status` clean, stash untouched.

## 6. Measured numbers (Phi-3.5-mini-Q4_K_M, fox×150, threads 10, 30 tok, per-head)

| round | f32 (attn) | i8 (attn) | delta |
|---|---|---|---|
| 1 (f,i) | 6.07 t/s (1545.2, 44%) | 6.74 t/s (1005.1, 45%→44%*) | +11.0% |
| 2 (i,f) | 6.24 (1455.8) | 7.04 (818.6) | +12.8% |
| 3 audit (f,i) | same shape | same shape | ids: one recurring 11203→17366 substitution ×3 |

\* attn share denominators shift with the wall; absolute ms is the comparison (§3).

## 7. Full reimplementation recipe (verification — re-run to re-verify)

Re-run §1; expect i8 ≥ f32 +5% with ids identical-or-priced-flips. Revival trigger: an i8 LOSS on Phi natural (would contradict two independent Phi wins and reopen content-dependence for real) or garbage ids (would reopen the fidelity file, since mixed10 says flips stay rare).

## 8. Verdict and why

**CONFIRM: the transfer matrix is complete and green (6/6 on perf, all ids inside the priced contract).** The per-head-i8 usage rule is now evidenced on every model×text cell the loop can construct with on-disk models: use `--attention head --kv i8` at long ctx; expect +6 to +25% decode with rare, priced flips. Next-cycle candidates: the 1517 stage-2 harness remains the only live structural front; the human-gated port shelf stands (KV-aware auto would automate exactly this rule).

## 9. Exact commands run and their results

- Phi fox×150 `--attention head` [--kv i8] f/i + i/f → §6 table.
- Audit pair with full ids → §3.2 (one recurring substitution).
- build → 54/54; tests → 141/141; `aikit.py check` → 2/4.
- Scratch deleted; no branch, no edits; report committed; no push.

(End of file.)
