# INVENTED: 12-row speculative verify re-priced at the default 10-thread decode — KILLED again (median 1.089x at 10 threads; the 6-thread kill was not thread-limited)

- **Idea**: `idea-invented-spec12-wide-verify-killed.md` killed the 11-drafts + bonus verify at 6 decode threads: steps halve (11 -> 6) but the 12-row verify costs ~2x the 6-row verify, so throughput is flat. That kill is only valid if the row cost is thread-invariant. The 12-row step is compute-heavy (batch GEMM + 12 x 128,256 LM-head logits), while 10 decode threads give ~1.67x more compute than 6 on this 6P+4E box. Re-price the same spike at the tree's default 10-thread decode: if the 12-row step falls from 2.0x to <=1.5x of the 6-row step, the halved step count should win >= 1.15x.
- **Cycle date**: 2026-11-11
- **Verdict**: **KILLED at 10 threads too.** Periodic prompt: 8 alternating pairs, **8/8 positive**, ratios 1.109/1.252/1.168/1.076/1.101/1.076/1.077/1.069, **median 1.089x** — real but under the pre-stated 1.15x bar. Code-echo: 0.960/1.093 (median ~1.03), prose: 0.979/1.004-in-noise (non-fused path is pinned to 6 rows, so prose is behaviorally identical). Per-step cost falls from **1.99x (6 threads) to 1.74x (10 threads)** — the interaction is confirmed and directionally right, just not enough. Ids bit-identical in every id-checked run.
- **Tree state at close**: spike reverted from `include/tge/models/qwen2.hpp` and `apps/cli/main.cpp`; `git status` clean; stash holds only the pre-existing 2026-11-04 peer entry; rebuild clean, `tge_tests` **171/171**, `python aikit.py check aegean-tge` **4/4**. No branch, no spike commit, nothing pushed.

## 1. Mechanism

Identical to the first cycle's spike (its §8 recipe rebuilt verbatim): `kSpeculativeBatch` 6 -> 12, `prompt_lookup_draft` cap 12 with 11 drafts in the fused steady state, 12-row feed/penalized widening, accept loop `< 11`, full-accept `position = vpos + 13`. The non-fused greedy block and the sampled block are **pinned to `feed[6]` / `verify_draft(..., 6, ...)`** so they never verify uninitialized rows (the load-bearing trap from the first cycle). The only change this cycle is the decode thread count: `--threads 10` instead of `--threads 6`.

## 2. Predicted gain (with the math)

At 6 threads: 6-row spec 11 steps, 12-row spec 6 steps; step costs measured ~128 ms vs ~255 ms (ratio 1.99) -> throughput ratio 11/6 / 1.99 = 0.92 (measured 1.018 median; the step-count part and the cost part nearly cancel).

If the verify cost splits into a weight-stream part (thread-invariant) and a compute part (scales with threads), 10 threads should reduce the 12-row/6-row step ratio. Measured across the two cycles:

| thread count | 6-row step | 12-row step | ratio | tokens/step 6-row -> 12-row | predicted throughput ratio |
|---:|---:|---:|---:|---:|---:|
| 6 | ~128 ms | ~255 ms | 1.99 | 6.0 -> 11.2 | 11/6 / 1.99 = 0.92 |
| 10 | ~164 ms | ~285 ms | 1.74 | 6.0 -> 11.2 | 11/6 / 1.74 = 1.05 |

The measured 10-thread median is 1.089x, in the direction the model predicts, but the pre-stated bar was 1.15x.

## 3. Falsifier (stated before the spike)

**PROCEED** if median end-to-end decode at `--threads 10` gains **>= 1.15x** over the 6-row binary on the periodic prompt with identical ids and no code/prose regression. **KILL** otherwise.

Measured periodic median 1.089x (8 pairs, 8/8 positive); kill.

## 4. File anchors (current HEAD, read-before-write verified)

- `include/tge/models/qwen2.hpp:180` — `kSpeculativeBatch`.
- `apps/cli/main.cpp:788-922` — fused steady state (drafts/feed/penalized rows/accept/position).
- `:924-1070` — non-fused greedy (pinned to 6); `:1081-1210` — sampled (pinned to 6).
- `:2080-2118` — `prompt_lookup_draft` cap.
- `src/models/qwen2.cpp:2713-2727` — `verify_draft_logits` (`count > kSpeculativeBatch` check).
- `src/core/dispatch.cpp:77-94` — `resolve_decode_threads()` defaults to one worker per physical core (10 on this box, H3 +40% over 6).
- `reports/idea-invented-spec12-wide-verify-killed.md` — the 6-thread numbers and the rebuild recipe reused here.

## 5. What was done (temporary spike, fully reverted)

1. Re-applied the spec12 edits from the first report's §8: constant 6 -> 12, `prompt_lookup_draft` cap 12, 11 drafts in the fused path, 12-row widening, accept `11`, full-accept `vpos + 13`.
2. Re-applied the pinning of the non-fused and sampled paths to 6 rows (their `feed[]` arrays and verify calls), including the widened-history block inside the non-fused penalized branch.
3. Built; `tge_tests` 171/171. Reused `build/idea/tge-spec6-base.exe` (built from the same HEAD code during the first cycle; only docs commits have landed since) as the 6-row baseline.
4. Ran 8 alternating periodic pairs at `--threads 10`, plus 2 code-echo and 2 prose control pairs, then a phase snapshot; reverted both files; rebuilt; 171/171; gates 4/4; clean tree.

## 6. Measured numbers

`Llama-3.2-3B-Instruct-Q4_K_M.gguf`, `--prompt <case> --n 64 --kv f32 --threads 10 --gpu off --speculative`, greedy. spec6 = shipped 6-row binary, spec12 = spike.

### Periodic text (fox x120 = 1,200 prompt tokens), 8 pairs

| pair | spec6 t/s | spec12 t/s | ratio |
|---:|---:|---:|---:|
| 1 | 34.17 | 37.90 | 1.109 |
| 2 | 27.90 | 34.93 | 1.252 |
| 3 | 31.42 | 36.71 | 1.168 |
| 4 | 33.36 | 35.91 | 1.076 |
| 5 | 33.61 | 36.99 | 1.101 |
| 6 | 36.80 | 39.58 | 1.076 |
| 7 | 35.15 | 37.86 | 1.077 |
| 8 | 36.76 | 39.28 | 1.069 |
| **median** | 33.49 | 37.03 | **1.089** |

Steps: 11 -> 6; accepted/step 5.09 -> 10.17; ids `9D-93-29…` identical.

### Code-echo control

| pair | spec6 | spec12 | ratio | ids |
|---:|---:|---:|---:|---|
| 1 | 44.17 | 42.40 | 0.960 | identical |
| 2 | 38.44 | 42.01 | 1.093 | identical |

### Prose control (non-fused path, pinned to 6 rows)

| pair | spec6 | spec12 | ratio |
|---:|---:|---:|---:|
| 1 | 13.59 | 13.31 | 0.979 |
| 2 | 13.65 | 12.62 | 0.925 |

Behaviorally identical path; the spread is box noise, which bounds how much weight a single 1.05x reading deserves.

### Phase snapshot (one pair, default 10 threads)

| arm | wall | decode matvec | attn | norm | steps |
|---|---:|---:|---:|---:|---:|
| spec6 | 1545.9 ms (40.75 t/s) | 1196.4 ms (78%) | 238.1 ms (16%) | 97.2 ms (6%) | 11 |
| spec12 | 1608.3 ms (39.17 t/s) | 1282.8 ms (80%) | 201.1 ms (13%) | 109.9 ms (7%) | 6 |

The 12-row arm spends ~7% more total matvec time (more rows) but halves the step count; the snapshot pair landed at 0.96x, inside the same noise the 8-pair median averages out.

## 7. Why it still fails

- The 12-row step's compute penalty only fell from 1.99x to 1.74x when going from 6 to 10 threads. The weight-stream fraction of the verify is larger than the thread-scaling model assumed: attention (16%) and matvec reads do not scale with workers, and the LM head's 12 rows add 12 x 308 MiB of reads per step on top of the 6-row step's 6.
- Halving steps (11 -> 6) buys 1.83x in step count; the 1.74x step cost cancels it to 1.05x, and the measured 1.089x includes scheduling slack. To reach 1.15x the row-cost ratio would need <= 1.59x, which needs either >= 16 threads (unresolved by the earlier thread sweeps) or cheaper verify rows (fewer LM-head logits per draft row, e.g. a partial-vocabulary accept check).
- The code-echo control confirms the effect is workload-specific (median ~1.03x), so a default change would not pay across workloads even if the periodic shape clears the bar.

## 8. Full reimplementation recipe (rebuild only to re-verify the kill)

Identical to `idea-invented-spec12-wide-verify-killed.md` §8 (constant + four fused call sites + pinning of the two 6-row paths). The only addition is the measurement condition: run both binaries with `--threads 10` (the box's default decode setting) and compare periodic, code-echo and prose prompts with `--print-ids`. Expected: periodic median ~1.09x (8/8 positive), code ~1.0x, prose ~1.0x, ids identical. Any median below 1.15x is the recorded kill; the interaction table in §2 is the reason a future retest should vary the LM-head row count, not the batch width alone.

## 9. Exact commands run and results

```text
cmake --build projects\aegean-tge\build\release --config Release   -> clean (56/56)
projects\aegean-tge\build\release\bin\tge_tests.exe                -> 171 cases, 0 failed
# baseline: build\idea\tge-spec6-base.exe (same HEAD code, 6-row)
tge generate --model Llama-3.2-3B-Instruct-Q4_K_M.gguf --prompt <p> --n 64 \
  --kv f32 --threads 10 --gpu off --speculative [+ --print-ids]    -> §6 tables
git checkout -- include/tge/models/qwen2.hpp apps/cli/main.cpp      -> reverted
cmake --build ... -> clean; tge_tests -> 171/171
python aikit.py check aegean-tge                                    -> 4/4 gates passed
```

No push, no merge, no force-push. Spike binaries live only under `build/release/bin/` and `build/idea/`; the committed tree is unchanged except this report.
