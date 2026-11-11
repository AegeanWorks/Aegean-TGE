# INVENTED: 12-row speculative verify (double-width draft batch) — KILLED (steps halve 11 -> 6, throughput flat at ~1.02x)

- **Idea**: The shipped speculative stepper verifies 5 drafts + 1 bonus in one 6-row batch and accepts 5.09 drafts/step on repetitive text, emitting at the 6-token/step ceiling. Widen the batch to 11 drafts + bonus: the prompt-lookup matcher already finds a long continuation, so the step count should halve (11 -> 6 steps for 63 tokens) and halve the number of weight passes. If verify cost is dominated by streaming the 2.0 GiB of weights, the extra rows are nearly free and the run should gain 1.3-1.5x.
- **Cycle date**: 2026-11-11
- **Verdict**: **KILLED.** The mechanism works exactly as designed — 11 -> **6 steps**, accepted/step **5.09 -> 10.17**, ids bit-identical on periodic, code-echo and prose prompts — but end-to-end throughput is flat: periodic 44.89/39.38/36.15/38.43 -> 42.01/37.70/39.02/42.38 t/s (ratios 0.936, 0.957, 1.079, 1.103; median **1.018x**), code-echo median ~1.03x, prose 1.00x. The 12-row verify costs ~2x the 6-row verify, so per-token cost is unchanged: verification at M=6 is already compute-bound in the batch GEMM + LM-head path, not weight-stream-bound as the prediction assumed.
- **Tree state at close**: spike reverted from `include/tge/models/qwen2.hpp` and `apps/cli/main.cpp`; `git status` clean; stash holds only the pre-existing 2026-11-04 peer entry; rebuild clean, `tge_tests` **171/171**, `python aikit.py check aegean-tge` **4/4**. No branch, no spike commit, nothing pushed.

## 1. Mechanism

Shipped state: `Qwen2::kSpeculativeBatch = 6` (`include/tge/models/qwen2.hpp:180`), used as both the verify-row count and the size of `verify_logits_` (`src/models/qwen2.cpp:602-604`, `2713-2727`). The fused greedy steady state (`apps/cli/main.cpp:788-922`) builds `feed[0]` = bonus id and `feed[1..5]` = five prompt-lookup drafts, verifies all six in one `prefill_run`, then accepts the longest matching prefix and emits `accepted + 1` tokens. Full accept chains without a correction forward; `position = vpos + 7` and the next step's `feed[0]` is the freed bonus.

Spike: raise the batch to 12 and draft 11.

1. `kSpeculativeBatch` 6 -> 12 (buffers and the verify cap grow automatically; `verify_logits_` becomes 12 x vocab).
2. `prompt_lookup_draft` cap 6 -> 12; the deep pass asks for 11 continuations, the 5-deep fallback pass and the bigram/gapped helpers (still capped at 6) are untouched.
3. Fused path: `draft5[11]`, fill `feed[1..11]`, penalized-row widening over 12 rows, accept loop `< 11`, full-accept position `vpos + 13`, `accepted == 11` chain condition.
4. The non-fused greedy path and the sampled path stay 6-wide: their `feed[]` arrays are pinned to `[6]` and their verify calls pass the literal `6`. **This is load-bearing**: with only the constant raised, those blocks call `verify_draft(feed, kSpeculativeBatch, ...)` with `feed[6..11]` uninitialized, so every such site must be pinned to its fill count.

## 2. Predicted gain (with the math)

Periodic 1,200-token prompt, 63 generated tokens, 6 threads:

- 6-row: 11 steps at 5.09 accepted/step; measured 1.40-1.74 s wall -> ~128-158 ms/step, ~22-26 ms/token.
- 12-row: 6 steps at ~10.2 accepted/step. If the verify were weight-stream-bound, a 12-row batch would cost ~1.15-1.3x a 6-row batch (the 2.0 GiB weight stream plus 12 rows of attention and 12 x 128k LM-head logits), so wall would fall to ~0.55-0.65x (1.5-1.8x speedup).
- The actual verify costs scale with rows: 12-row step ~250-280 ms, i.e. ~2.0x the 6-row step, cancelling the step reduction. The weight stream was already amortized at M=6: a 6-row verify costs ~1.8x a single-row forward on this box, so the marginal cost of rows 7-12 is compute (batch Q4_K/Q6_K GEMM + the 128,256-row LM head), not bytes.

## 3. Falsifier (stated before the spike)

**PROCEED** if median end-to-end wall gains >= 1.15x over the 6-row binary on the periodic and code-echo workloads, with bit-identical greedy ids and no prose regression. **KILL** otherwise.

Measured periodic median 1.018x (best 1.103x), code median ~1.03x, prose 1.00x; ids identical everywhere. Kill.

## 4. File anchors (current HEAD, read-before-write verified)

- `include/tge/models/qwen2.hpp:180` — `kSpeculativeBatch = 6`; `:517` — `verify_logits_`.
- `src/models/qwen2.cpp:602-604` — `verify_logits_.resize(kSpeculativeBatch * vocab)`; `:2713-2727` — `verify_draft_logits` (`count > kSpeculativeBatch` check, `prefill_run` with all logits).
- `apps/cli/main.cpp:788-922` — fused greedy steady state (drafts, feed, penalized rows, accept loop, full-accept position math).
- `:924-1070` — non-fused greedy path (6-deep, `skip_locked_draft`/`prompt_lookup_draft` with 6); `:1081-1210` — sampled path (5 drafts + sixth slot, 6-row verify).
- `:2080-2118` — `prompt_lookup_draft` (cap `need > 6`, span/bigram/gapped passes).
- `reports/idea-invented-sampled-sixth-draft-keep.md`, `reports/idea-invented-spec-equivalence-sweep.md` — the 6-row basis and workload numbers.

## 5. What was done (temporary spike, fully reverted)

Built both binaries from the same source tree with only the listed edits (baseline via stash -> build -> copy, spike via restore -> build); ran alternating pairs; reverted with `git checkout --` and rebuilt.

## 6. Measured numbers

`Llama-3.2-3B-Instruct-Q4_K_M.gguf`, `--prompt <case> --n 64 --kv f32 --threads 6 --gpu off --speculative --print-ids`, greedy. "spec6" = shipped HEAD binary, "spec12" = spike binary, same source otherwise.

### Periodic text (fox sentence x120 = 1,200 prompt tokens)

| pair | spec6 t/s | spec12 t/s | ratio | spec6 steps / acc/step | spec12 steps / acc/step | ids |
|---:|---:|---:|---:|---|---|---|
| 1 | 44.89 | 42.01 | 0.936 | 11 / 5.09 | 6 / 10.17 | identical |
| 2 | 39.38 | 37.70 | 0.957 | 11 / 5.09 | 6 / 10.17 | identical |
| 3 | 36.15 | 39.02 | 1.079 | 11 / 5.09 | 6 / 10.17 | identical |
| 4 | 38.43 | 42.38 | 1.103 | 11 / 5.09 | 6 / 10.17 | identical |
| **median** | 38.9 | 40.5 | **1.018** | | | 4/4 |

Per-step cost: spec6 127.6-158.4 ms/step (5.7 tokens/step, 22.3 ms/token), spec12 248-278 ms/step (11.7 tokens/step, 21.4 ms/token). The step count halves; the per-token cost does not move.

### Code-echo text (`int sum = 0; for (...)` x20)

| pair | spec6 | spec12 | ratio | steps (6 -> 12) | ids |
|---:|---:|---:|---:|---|---|
| 1 | 45.55 | 43.29 | 0.950 | 11 -> 6 | identical |
| 2 | 38.81 | 43.01 | 1.108 | 11 -> 6 | identical |
| **median** | 42.2 | 43.2 | **1.029** | | 2/2 |

### Free prose (dragon, 24-token prompt) — control

| pair | spec6 | spec12 | ratio | steps | ids |
|---:|---:|---:|---:|---|---|
| 1 | 11.52 | 11.58 | 1.005 | 2 / 2 | identical |
| 2 | 11.55 | 11.48 | 0.994 | 2 / 2 | identical |

Both arms take the non-fused path here; the widened constant changes only buffer sizes, and the pinned 6-wide calls keep behavior identical.

## 7. Why it failed

- **Verification at M=6 is not weight-stream-bound.** A 6-row verify costs ~1.8x a single-row forward, already amortizing the 2.0 GiB stream; rows 7-12 add batch-GEMM and LM-head compute (12 x 128,256 logits per verify) almost linearly. The marginal token beyond the first six is compute, not bytes.
- The LM head is the largest per-row cost: 12 rows of a 308 MiB Q6_K projection per verify. Doubling rows doubles that, which is why the per-token time is flat.
- The step-count mechanism is confirmed (11 -> 6, 10.17 accepted/step) and ids are exact, so this is an arithmetic-cost ceiling, not a correctness/coupling failure.
- Consequence: widen the batch only if the verifier gets cheaper per row (e.g. a cheaper LM-head path for draft rows, or accepting fewer logits rows). Changing the row count alone cannot pay.

## 8. Full reimplementation recipe (rebuild only to re-verify the kill)

Do not ship. To reproduce:

1. `include/tge/models/qwen2.hpp:180`: `kSpeculativeBatch = 12`.
2. `apps/cli/main.cpp` `prompt_lookup_draft` cap: `need > 6` -> `need > 12`; update its comment ("deep draft first").
3. Fused block (`:788`): `int draft5[11]`; `prompt_lookup_draft(history, draft5, 11)`; `for (index < 11) feed[index + 1] = draft5[index]`; penalized widening `reserve(size + 12)` and `index < 12`; `while (accepted < 11 && ...)`; `if (accepted == 11)` with `position = vpos + 13`.
4. Pin the other two paths to 6 or they verify uninitialized rows: non-fused gate `position + 6 <= max_context()`, `int feed[6]`/`int predicted[6]`, `verify_draft(feed, 6, ...)`, `verify_draft_logits(feed, 6, ...)`, widened `+6`/`index < 6`; sampled gate `position + 6 <= max_context()`, `int feed[6]`, `verify_draft_logits(feed, 6, position)`.
5. Build + `tge_tests` 171/171, then compare against a binary built from the unmodified tree with alternating pairs on the periodic and code-echo prompts at `--n 64 --kv f32 --threads 6 --gpu off --speculative`, checking ids with `--print-ids`. Expected: steps halve, throughput ~1.0x, ids identical. Any median below 1.15x is the recorded kill.

## 9. Exact commands run and results

```text
cmake --build projects\aegean-tge\build\release --config Release   -> clean (56/56)
projects\aegean-tge\build\release\bin\tge_tests.exe                -> 171 cases, 0 failed
# baseline binary from the stashed tree -> build\idea\tge-spec6-base.exe
tge generate --model Llama-3.2-3B-Instruct-Q4_K_M.gguf --prompt <p> --n 64 \
  --kv f32 --threads 6 --gpu off --speculative --print-ids          -> §6 tables
git checkout -- include/tge/models/qwen2.hpp apps/cli/main.cpp      -> reverted
cmake --build ... -> clean; tge_tests -> 171/171
python aikit.py check aegean-tge                                    -> 4/4 gates passed
```

No push, no merge, no force-push. Spike binaries live only under `build/idea/` and `build/release/bin/`; the committed tree is unchanged except this report.
