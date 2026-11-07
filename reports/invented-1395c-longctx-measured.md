# INVENTED-1395c: exact tail at long context + multi-domain corpus — MEASURED: ordering holds, saturation begins as tail fraction drops

- **Idea ID**: INVENTED-1395c (invented follow-up to Cycles 2-3; long-ctx fraction + second-domain check. Long corpus = report-0.2-verification.md + report-0.1.md + architecture.md + benchmarks.md concatenated in temp, 50,547 B — verification prose + tables, i.e. 2nd/3rd/4th domains vs the architecture-only corpora of Cycles 2-3).
- **Cycle date**: 2026-11-07
- **Verdict**: MEASURED — KL ordering holds at ctx 2048 (flat 2.554e-4 → N64 2.363e-5, 10.8× → N128 1.097e-5, 23.3× vs flat, 2.2× vs N64); top-1 315/319 → 319/319 → 319/319. The 64→128 step shrinks to 2.2× (vs 4.8× at ctx 512) — saturation begins as the tail fraction drops (40% max span at 320 pos vs 64% at 199). Decode at N=128 vs flat: best-of-3 warmed reverse-order 8.45 vs 8.73 t/s (−3.2%, inside ±20% box noise → parity). Falsifier arms (KL inversion at long ctx, corpus-specific no-gain, >5% decode regress) did not trip. Spike fully reverted; only this report committed.
- **Tree state at close**: `master`, `git status` clean, spike branch created and deleted with nothing committed on it, stash untouched (pre-existing peer stash 2026-11-04 only). Final gates on master: build clean, `tge_tests` 141/141. Nothing pushed (human gate).

## 1. Mechanism

Same ring as Cycles 2-3 (`TGE_KV_TAIL=N`, filed recipe `idea-1395-tail-slice-measured.md` §7, re-spike byte-identical). Only the evaluation regime changed: ctx 2048, 319-399 positions of multi-domain text, so the N=128 tail covers 32-40% of max span instead of 64%.

Predicted: KL ordering preserved, absolute gains shrink (more q8 bulk); top-1 stays perfect; decode parity improves in relative terms (fixed-N tail traffic vs growing bulk).

## 2. Falsifier (from CREATE)

KL ordering inverts at long ctx (recency effect was a short-ctx artifact), or multi-domain corpus shows no gain (corpus-specific), or N=128 decode regresses > 5% vs flat best-of-3. None tripped.

## 3. File anchors

Identical to `idea-1395-tail-slice-measured.md` §3/§7 (same 5 files; re-spike needed no recipe changes — third clean first-try build).

## 4. What was done (temporary re-spike, fully reverted)

1. `git checkout -b spike-1395c-longctx` (no commit ever made on it); re-applied the 15-edit Cycle-2/3 recipe verbatim.
2. `cmake --build --preset release` clean first try; `tge_tests` **141/141**.
3. Long corpus built in temp (outside repo, never committed).
4. Fidelity (Llama-3B-Q4_K_M, ctx 2048, 1 thread, `--kv i8`): flat@400pos KL **2.620e-4**, 395/399, 475 s; flat@320pos KL **2.554e-4**, 315/319, 343 s; N64@320pos KL **2.363e-5**, 319/319, 338 s; N128@320pos KL **1.097e-5**, worst 8.854e-5, 319/319, 364 s. (One flat@400 attempt exceeded the 600 s tool timeout and was rerun at 320; a prior-session background job was lost on resume and rerun foreground.)
5. Thinking-set gate: fox-loop ids identical at N=128 (`578 39935 35308 13` ×4, same as flat/N64); architecture-excerpt free-run diverges per tier (flat `683 374 7559 520...` vs N64 `...83 713 5640...` vs N128 `683 374 539 6982 1618 13 1102...`) — expected lossy-KV behavior (KIVI/VNNI precedent), teacher-forced agrees 319/319.
6. Decode best-of-3, warmed reverse-order, ~1441-ctx prompt, default threads, N=128 vs flat: TAIL128 **7.45/8.12/8.45** vs OFF **8.73/8.45/6.77** → best 8.45 vs 8.73 (−3.2%, parity); attn ms 381-427 vs 395-479 (indistinguishable).
7. `python aikit.py check aegean-tge` → 2/4 (test + secrets pass; hygiene + guidelines fail pre-existing on master, same record as Cycles 1-3).
8. Reverted all 5 files, deleted branch, rebuilt master clean, `tge_tests` 141/141, `git status` clean.

## 5. Measured numbers (with short-ctx points for the fraction curve)

| regime | flat KL | N64 KL | N128 KL | tail fraction @max span |
|---|---|---|---|---|
| ctx512/199pos arch-only (C2/C3) | 2.133e-4 | 2.087e-5 (10.2×) | 4.307e-6 (4.8× over 64) | 64% |
| ctx2048/320pos multi-domain (this cycle) | 2.554e-4 | 2.363e-5 (10.8×) | 1.097e-5 (2.2× over 64) | 40% |

Top-1: 196→199/199 short; 315/315→319/319/319 long. Decode N=128 best-of-3: short-ctx +0.7%, long-protocol −3.2% — both inside box noise → parity in both regimes.

## 6. Verdict

Keep-grade with a sizing rule emerging: the tail buys ~10× (N=64) to ~25-50× (N=128) KL at decode parity across both regimes; the marginal 64→128 step halves as the tail fraction drops (4.8× → 2.2×). Default-N selection still needs the ctx-2048 point with span ≫ 128 (this cycle reached span 320; push to 1000+ when a faster harness or longer tool budget allows) — recorded as the open follow-up, not a new spike today.

## 7. Reimplementation recipe

`idea-1395-tail-slice-measured.md` §7 verbatim. Long-ctx protocol: concatenate domain docs to temp (never commit), `--context 2048 --tokens 320 --threads 1 --kv i8`, flat + N=64 + N=128; warmed reverse-order decode best-of-3 at N=128 vs flat (`--n 32`, ~1441-ctx fox prompt, default threads). Kill on §2 falsifier.

## 8. Citation verification

No new external citations (invented idea; carrier = Cycle-2 recipe + Cycle-2 beellama verification). Temp corpus files are repo docs (`docs/report-0.2-verification.md`, `docs/report-0.1.md`, `docs/architecture.md`, `docs/benchmarks.md`) — internal, unmodified. Internal anchors = Cycle-2 report §3 (third byte-identical re-spike, first-try clean build).

## Commands run and their results

- `git checkout -b spike-1395c-longctx` → switched; 15 re-applied edits; `cmake --build --preset release` → 54/54 clean; `tge_tests` → 141/141.
- Temp corpus concat → 50,547 B.
- `tge fidelity` long-ctx: flat@400 KL 2.620e-4 (475 s); flat@320 KL 2.554e-4 (343 s); N64@320 KL 2.363e-5 (338 s); N128@320 KL 1.097e-5 (364 s). One 600 s timeout (flat@400 first attempt) → reran at 320.
- `tge generate --print-ids` fox N=128 → identical loop; architecture excerpt N=128 → diverged string in §4.5.
- Warmed reverse-order decode best-of-3 N=128 vs flat → §4.6 numbers.
- `python aikit.py check aegean-tge` → 2/4, pre-existing fails.
- `git checkout -- .` + `git checkout master` + `git branch -D spike-1395c-longctx` → master, clean (verified); rebuild + `tge_tests` 141/141.
