# INVENTED: fused six-draft (7-row) lock chain revisited at 8k — KILLED for the long-context revival, but it wins on true period-1 locks at short context (2k lock 1.09x, 4k fox 1.06x)

- **Idea**: The previous cycle killed a fused chain with six drafts plus the bonus row (7-row verify) analytically: seven rows cost ~1.20x a six-row step for 7/6 = 1.17x tokens, i.e. ~0.98 net. That kill used the 2k row-cost model (spec12 12-row line: 1.74-1.99x the 6-row step, ~linear in rows). This cycle's 8k measurement showed the opposite scaling pressure: a six-row verify shares each KV row across six queries, so per-row attention at 8k is 39-40 ms versus 94-106 ms for a single-row decode token. Hypothesis: at 8k the seventh row may cost < 1.17x a six-row step (attention amortization), reviving the six-draft fused chain where the one-step token gain is realised.
- **Cycle date**: 2026-11-12
- **Verdict**: **KILLED for the 8k revival; the short-context exception is real and measured.** Llama-3.2-3B-Instruct-Q4_K_M, f16 KV, 10 threads, `--gpu off --speculative`:
  - **8k lock-1** (`a` + ` a`x8192 = 8,193 tokens): base 8.51 t/s (3 steps, 4.33 accepted/step) -> spike 7.95 t/s (3 steps, 4.67/step) = **0.934x**. The model's period-1 lock decays at 8k, so the seventh row mostly adds cost.
  - **8k fox-4** (compact sentence x2048): base 9.00 -> spike 8.73 = **0.970x**; both accepted 3.00/step, so the extra draft never changed acceptance.
  - **2k fox-4** (fox x205, n=128, interleaved pairs): base 28.73/30.42 vs spike 28.97/28.10 -> **0.965x median**; the spike cut steps 22 -> 19 (accepted 4.95 -> 5.79/step) and still lost, because at 2k the seventh row's matvec/LM-head cost exceeds the 14% step reduction.
  - **2k lock-1** (interleaved pairs, n=32): base 16.77/19.09/18.88 vs spike 19.33/19.66/20.66 -> **1.09x median**; acceptance goes 5.25 -> 6.00/step and steps 4 -> 3. This is the case the analytic kill missed: a true period-1 token lock accepts the sixth draft every step.
  - **4k fox-4** (fox x410, interleaved pairs): base 21.03/22.38 vs spike 23.32/22.49 -> **1.056x median**, 6 -> 5 steps, 4.83 -> 5.20 accepted/step.
  - **Conclusion**: the seventh row pays ~1.10-1.12x a six-row step at every context; it wins only when the sixth draft is accepted nearly every step (>= 5.8 accepted/step), which happens on a genuine period-1 lock at 2k and partially at 4k, but not at 8k where lock acceptance decays to 4.33-4.67/step. No context crossover exists: attention amortization does not make the extra row free, and acceptance quality falls faster than row cost rises.
- **Tree state at close**: spike reverted from `include/tge/models/qwen2.hpp` and `apps/cli/main.cpp`; `git status` clean; stash holds only the pre-existing 2026-11-04 peer entry; rebuild clean, `tge_tests` **171/171**, `python aikit.py check aegean-tge` **4/4**. No branch, no spike commit, nothing pushed.

## 1. Mechanism (as spiked)

The shipped fused greedy chain (`apps/cli/main.cpp:788-922`) builds `feed[0]` = bonus id and `feed[1..5]` = five prompt-lookup drafts, verifies six rows, and emits `accepted + 1` tokens (up to 6). A fully accepted step chains without a correction forward. The spike widened the fused chain only:

1. `Qwen2::kSpeculativeBatch` 6 -> 7 (`include/tge/models/qwen2.hpp:180`): the verify cap (`src/models/qwen2.cpp:2715`) and the `verify_logits_` scratch (`:603`) grow automatically.
2. The non-fused greedy and sampled paths are **pinned to 6**: they fill `feed[0..5]` and now call `verify_draft(..., 6, ...)` / `verify_draft_logits(..., 6, ...)` with literals. Without this pin they would verify `feed[6]` uninitialized (the load-bearing trap recorded by the spec12 recipe).
3. Fused path: `draft6[6]` from `prompt_lookup_draft(history, draft6, 6)`, `feed[0..6]`, accept loop `< 6`, penalized-row widening over 7 rows, full-accept `position = vpos + 8` (drafts at vpos+1..6, corrected at vpos+7), chain condition `accepted == 6`.

The six-draft drafter is the shipped span matcher asked for one more continuation; no new drafting logic. The verifier is unchanged arithmetic: same per-head schedule (ADR-002), same accept rule. Output ids are identical by construction; the spike only changes how many rows are verified per step.

## 2. Measured results

All runs: Llama-3.2-3B-Instruct-Q4_K_M, `--kv f16 --threads 10 --gpu off --speculative`; prompt and `--n` per row. "accepted/step" is the CLI `spec` line.

| workload | context | n | base t/s | spike t/s | ratio | base steps (acc/step) | spike steps (acc/step) |
|---|---:|---:|---:|---:|---:|---|---|
| fox-4 x205 (interleaved, 2 pairs) | 2560 | 128 | 28.73 / 30.42 | 28.97 / 28.10 | **0.965** | 22 (4.95) | 19 (5.79) |
| lock-1 ` a`x2048 (interleaved, 3 pairs) | 2560 | 32 | 16.77 / 19.09 / 18.88 | 19.33 / 19.66 / 20.66 | **1.09** | 4 (5.25) | 3 (6.00) |
| fox-4 x410 (interleaved, 2 pairs) | 4608 | 32 | 21.03 / 22.38 | 23.32 / 22.49 | **1.056** | 6 (4.83) | 5 (5.20) |
| fox-4 compact x2048 | 8704 | 16 | 9.00 | 8.73 | **0.970** | 4 (3.00) | 4 (3.00) |
| lock-1 ` a`x8192 | 8704 | 16 | 8.51 | 7.95 | **0.934** | 3 (4.33) | 3 (4.67) |

Per-step cost: the two 8k rows are the cleanest pricing because the step counts are equal. Lock-1 8k: 3 steps, +1 accepted token total, +6.6% wall -> the seventh row costs ~1.07x a step and buys 1/15 of a token. Fox-4 8k: equal steps and equal acceptance -> the spike pays ~3% for one extra row with no output change. The 2k fox-4 row shows the same thing from the other side: 19 steps versus 22 should be a 13.6% win, but per-step cost rises enough to leave 0.965x.

The lock-1 2k row is the exception: 6.00 accepted/step means every step emits 7 tokens versus the base's mixed 5-draft/6-draft steps that average 5.25 and occasionally leave the fused chain, and the measured wall drops 9%. The prior analytic kill used fox-4 at 2k (a period-4 text, where the sixth draft is not reliably accepted), so it correctly predicted the fox but missed the true period-1 cell.

## 3. Falsifier (stated before the spike)

**REVIVE/KEEP** at 8k if the spike is >= 1.10x the base on the 8k lock workload with equal ids, or >= 1.05x on the 8k fox workload. **KILL** otherwise. Measured 0.934x (lock) and 0.970x (fox): the revival fails by margin in both cells. The short-context lock win (1.09x) does not meet the 8k bar and is recorded as a bounded exception for the reimplementation recipe, not a ship recommendation.

## 4. File anchors (current HEAD, read-before-write verified)

- `include/tge/models/qwen2.hpp:180` — `kSpeculativeBatch = 6` (spike raised to 7).
- `src/models/qwen2.cpp:603` — `verify_logits_` scratch sized by `kSpeculativeBatch`.
- `src/models/qwen2.cpp:2697-2727` — `verify_draft` / `verify_draft_logits`; the cap is `count > kSpeculativeBatch`.
- `apps/cli/main.cpp:788-922` — fused greedy chain (feed[0] = bonus, five drafts, 6-row verify, `position = vpos + 7` on full accept).
- `apps/cli/main.cpp:937-1070` — non-fused greedy path that must stay pinned to 6 rows.
- `apps/cli/main.cpp:1081-1180` — sampled path that must stay pinned to 6 rows.
- `reports/idea-invented-fused-lock-six-draft-killed.md` — the analytic kill this cycle re-prices.
- `reports/idea-invented-spec-at-8k-measured.md` — the row-amortization evidence that motivated the revival.
- `reports/idea-invented-spec12-wide-verify-killed.md` / `...-10t-killed.md` — the 12-row step-cost data.

## 5. Reimplementation recipe (exact)

1. `include/tge/models/qwen2.hpp:180`: `static constexpr int kSpeculativeBatch = 7;` (scratch and the verify cap follow automatically).
2. Pin the non-fused sites to literal 6 (`apps/cli/main.cpp`): both calls in the non-fused greedy block (`verify_draft(feed, 6, position, predicted)` and `verify_draft_logits(feed, 6, position)`) and the sampled call (`verify_draft_logits(feed, 6, position)`). The arrays may stay `[kSpeculativeBatch]`; only the passed count matters.
3. Fused block (`:788-922`): `int draft6[6]; if (!prompt_lookup_draft(history, draft6, 6))`; `feed[0] = history.back(); for (i=0;i<6;++i) feed[i+1] = draft6[i];`; penalized widening loop `< 7` with `reserve(size+7)`; accept loop `accepted < 6`; full-accept `position = vpos + 8` and `accepted == 6`; the partial path stays `position = vpos + accepted + 2`.
4. Build + `tge_tests` (expect 171/171) before any benchmark.
5. A/B: preserve the pre-spike binary, build the spike, run alternating pairs with `--kv f16 --threads 10 --gpu off --speculative`.
   - 2k lock: `--prompt ("a" + (" a" * 2048)) --context 2560 --n 32` -> expect base 5 drafts/mixed 5.25 accepted and ~17-19 t/s, spike 6.00 accepted and ~19-20.7 t/s (1.03-1.15x).
   - 2k fox: `fox x205 --context 2560 --n 128` -> expect base ~29-30 t/s, spike ~28 t/s (0.965x).
   - 4k fox: `fox x410 --context 4608 --n 32` -> expect spike 1.0-1.11x (median 1.056).
   - 8k lock: `("a" + (" a" * 8192)) --context 8704 --n 16` -> expect base 8.5, spike 7.9 (0.934x).
   - 8k fox: compact `("The fox runs. " * 2048) --context 8704 --n 16` -> expect 9.0 vs 8.7 (0.970x).
6. Revert both files; rebuild and retest; the tree must be clean.

Do not promote this spike. The only positive cell (period-1 lock at 2k) is exactly where the non-fused lock path already reaches 6.00 accepted/step; the fused 7-row form saves its correction forward, but the same +9% is available from any workload-aware switch, and the 8k decay makes a context-dependent ship rule unjustified for one 2k cell.

## 6. Exact commands run and results

```text
cmake --build --preset release                         -> clean (56/56)
build\release\bin\tge_tests.exe                        -> 171 cases, 0 failed
copy build\release\bin\tge.exe %TEMP%\kilo\tge_base_fused6.exe   (pre-spike baseline)

fox x205, ctx 2560, n 128:  base 28.73/30.42  spike 28.97/28.10  -> 0.965x
lock-1 x2049, ctx 2560, n 32: base 16.77/19.09/18.88  spike 19.33/19.66/20.66 -> 1.09x
fox x410, ctx 4608, n 32:   base 21.03/22.38  spike 23.32/22.49  -> 1.056x
fox compact x8193, ctx 8704, n 16: base 9.00  spike 8.73 -> 0.970x
lock-1 x8193, ctx 8704, n 16:    base 8.51  spike 7.95 -> 0.934x

git checkout -- include/tge/models/qwen2.hpp apps/cli/main.cpp   -> clean
cmake --build --preset release                         -> clean; tge_tests -> 171/171
python aikit.py check aegean-tge                       -> 4/4 gates passed
```

No push, no merge, no force-push. The committed tree is unchanged except this report.
