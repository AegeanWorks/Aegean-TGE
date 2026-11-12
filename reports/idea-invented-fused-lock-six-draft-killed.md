# INVENTED: fused chain with the 6-draft lock drafter — KILLED analytically (the bonus row is structural; the two ways to build it cost more than the 1.17x token gain)

- **Idea**: `skip_locked_draft` proposes period-1/2 cycles at the **6.00 accepted/step ceiling** in the non-fused path, versus 5.05/step for the span drafter in the fused chain. If the fused chain could draft six and still chain, a full-accept step would emit 7 tokens instead of 6 (~1.17x at the same step cost), on top of the fused chain's 1.27x advantage over the correction-forward structure.
- **Cycle date**: 2026-11-11
- **Verdict**: **KILLED — no spike.** The two structures that could carry six drafts both cost more than the tokens they add:
  1. **Seven verify rows** (six drafts + bonus row): `kSpeculativeBatch` would go to 7. The 12-row line measured the row-cost at 1.018x (6 threads) and 1.089x (10 threads) median end-to-end, with per-step cost growing ~linearly in rows; at 7 rows a step costs ~1.20x for 7/6 = 1.17x tokens, i.e. ~0.98 net before any benefit. The wider-batch shape is already priced dead (`idea-invented-spec12-wide-verify-killed.md`, `...-10t-killed.md`).
  2. **Correction forward per step** (the non-fused 6-draft structure): measured in this cycle at **49.16/49.76/49.97 t/s** (fused, 22 steps, 5.05/step) vs **38.22/38.69/37.69 t/s** (`--no-fuse`, 19 steps, 6.00/step) on a locked 400-token prompt, n=128, f16, 10 threads. The lock drafter does achieve 6.00/step, but the extra 1-row correction forward costs 27% — more than the 1.17x token gain.
  - The fused chain's whole advantage is that the sixth row *is* the bonus re-fed for its KV row, so it has no correction forward. Six drafts plus chaining requires either that row or that forward; there is no third structure inside the current scheduler.
- **Tree state at close**: no code touched (analysis + measurement only); `git status` clean; stash holds only the pre-existing 2026-11-04 peer entry; `tge_tests` **171/171**, `python aikit.py check aegean-tge` **4/4**. No branch, nothing pushed.

## 1. Mechanism (current shapes, read from the tree)

- **Fused greedy steady state** (`apps/cli/main.cpp:788-922`): `feed[0] = history.back()` (the bonus token from the previous step), `feed[1..5]` = five span drafts; one `verify_draft(feed, 6, ...)` produces the KV rows for all six, so the next step's bonus needs no separate forward. Emits `accepted + 1` tokens; full accept = 6 tokens, `position = vpos + 7`.
- **Non-fused greedy path** (`:924-1070`): `skip_locked_draft(history, draft, 6)` first (lock path), else `prompt_lookup_draft(history, draft, 6)`; `feed[0..5] = draft[0..5]`; `verify_draft(feed, 6, ...)`; accept loop `accepted < 6`; then a correction `forward(corrected, ...)` writes the bonus KV row. Emits up to 7 tokens but pays a 1-row forward per step.
- The lock drafter is correct and cheap (O(8) checks), and the verifier still decides; the difference is purely structural.

## 2. Measured structure costs (this cycle)

`Llama-3.2-3B-Instruct-Q4_K_M.gguf`, 400-token periodic prompt, `--n 128 --kv f16 --threads 10 --gpu off --speculative`, greedy:

| structure | t/s | steps | accepted/step | tokens emitted |
|---|---:|---:|---:|---:|
| fused chain (default) | 49.16 / 49.76 / 49.97 | 22 | 5.05 | 132 |
| `--no-fuse` (6-draft lock + correction forward) | 38.22 / 38.69 / 37.69 | 19 | **6.00** | 132 |

The 6-draft lock structure reaches the 6.00 ceiling and cuts steps 22 -> 19, but is **27% slower** because of the correction forward. Any fused 6-draft form must add either that forward or a 7th row.

## 3. Falsifier (stated before the analysis)

**PROCEED** if a fused 6-draft variant can be built without a 7th row or a correction forward, or if the 7-row form beats the fused 5+bonus chain by >= 1.10x in the existing 12-row cost model. **KILL** otherwise.

Neither holds: the scheduler has no third structure, and the row-cost model (1.018x/1.089x at 12 rows, ~linear in rows) puts a 7-row step at ~1.20x for a 1.17x token gain.

## 4. File anchors (current HEAD, read-before-write verified)

- `apps/cli/main.cpp:788-922` — fused greedy chain (feed[0] = bonus, five drafts, `position = vpos + 7`).
- `:930-955` — lock-first drafting (`skip_locked_draft(history, draft, 6)`), `first_ok` pre-check.
- `:976-1022` — 6-row verify and `accepted < 6` loop.
- `:1037-1039` — the correction `forward` that the fused chain avoids.
- `include/tge/models/qwen2.hpp` — `kSpeculativeBatch = 6`.
- `reports/idea-invented-spec12-wide-verify-killed.md`, `...-10t-killed.md` — the row-cost evidence.

## 5. What would make it real

Only a scheduler that writes the bonus KV row *without* a model call: e.g., a verify that also computes row 6 for the bonus (7 rows, priced dead), or a KV-copy trick where the bonus row is produced as part of the same matvec (the bonus embedding is not known until row 5's argmax — it would need a second dependent pass, i.e. the correction forward). Recorded so the idea is not re-attempted without a new mechanism.

## 6. Reimplementation recipe (re-run the measurement)

1. No code: `tge generate --model Llama-3.2-3B-Instruct-Q4_K_M.gguf --prompt "<fox x40>" --n 128 --kv f16 --threads 10 --gpu off --speculative [--no-fuse]`.
2. Read `decode:`, `spec:`. Expect fused ~49-50 t/s (22 steps, 5.05/step) and `--no-fuse` ~38 t/s (19 steps, 6.00/step).
3. Do not build a fused 6-draft variant: the 7-row form is the already-priced wider-batch shape, and the correction-forward form is the measured 27%-slower non-fused structure.

## 7. Exact commands run and results

```text
tge generate ... --n 128 --speculative            -> 22 steps, 5.05/step, 132 emitted, 49.16/49.76/49.97 t/s
tge generate ... --n 128 --speculative --no-fuse  -> 19 steps, 6.00/step, 132 emitted, 38.22/38.69/37.69 t/s
tge_tests.exe                                     -> 171 cases, 0 failed
git status --short                                -> clean (no code touched)
python aikit.py check aegean-tge                  -> 4/4 gates passed
```

No push, no merge, no force-push. The committed tree is unchanged except this report.
