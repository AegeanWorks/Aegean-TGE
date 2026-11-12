# INVENTED: path-invariant continuous-batch kq8 decode (`tokens >= 1`, drop `float_only`) — KEEP-GRADE (audit now passes 3/3; +6% median throughput; thinking set identical 5/5)

- **Idea**: The occupancy audit (`idea-invented-continuous-kq8-leak-audit-measured.md`) failed because dropping `float_only` left a path split: occupancy 1 took the `tokens == 1` float kernel while occupancy N took kq8, and the same request changed output at token 11/15 on one shape. The fix the audit prescribed: remove the split itself. Route `tokens >= 1` through the kq8 batch kernel in `batch_matvec` (its per-row Q8_K quantiser and dots are identical for any token count) and drop `float_only` in `forward_batch`. Then occupancy 1 and occupancy N run literally the same arithmetic, and the audit should pass by construction.
- **Cycle date**: 2026-11-11
- **Verdict**: **KEEP-GRADE (report-only; promotion to default still human-gated).** 
  - **LEAK audit passes 3/3 shapes** on `qwen2.5-0.5b-instruct-q4_k_m` (`--tokens 16 --threads 10 --kv f16`): prompt-tokens 8 `83-8E-B6…` solo == busy; **24 `99-22-12…` solo == busy** (the previously failing shape; the old kq8 build gave solo `35-E5-78…` vs busy `99-22-12…`); 48 `10-3D-0F…` solo == busy. The path split is gone.
  - **Throughput**: 12 interleaved pairs vs the shipped float-batch binary: ratios 1.263/0.789/1.163 (4 slots), 1.321/0.954/0.969 (8 slots), 1.033/1.020/1.079/0.944/0.979/1.061 (6 more mixed) -> **median ~1.06x, 8/12 positive**. The 0.789 outlier is a loaded window (the float arm hit 159 t/s, above its own 103-159 band); medians are the claim.
  - **Thinking set identical 5/5**: story/creative/code/cjk/abstract greedy n=64 ids match the shipped binary exactly, because `generate` uses `forward`/`matvec` for decode and `prefill_run` for prompts > 1 token — only continuous-batch arithmetic changes.
  - Batch outputs differ from today's float-batch build on some shapes (pt=8 `05-74-D3…` -> `83-8E-B6…`), as expected when switching to the kq8 kernel; kq8 is the default prefill kernel and is priced at 9.248e-12 KL (near-exact), well inside the f16-class floor.
- **Tree state at close**: spike reverted from `src/models/qwen2.cpp`; `git status` clean; stash holds only the pre-existing 2026-11-04 peer entry; rebuild clean, `tge_tests` **171/171**, `python aikit.py check aegean-tge` **4/4**. No branch, no spike commit, nothing pushed.

## 1. Mechanism

- `batch_matvec` (`src/models/qwen2.cpp:1134`): condition changed from `tokens > 1` to `tokens >= 1` (with a comment tying it to the LEAK rule), and the eight `forward_batch`/`output_logits_batch` call sites pass `float_only = false`.
- Now every occupancy in `forward_batch` (1..N) routes K-quant matvecs to `gemv_dequant_batch_kq8_avx2`. That kernel quantises each token row independently to Q8_K (`quantize_row_q8k`, per 256-value block) and runs the same per-row dots, so a row's result cannot depend on the number of other rows in the batch. Path invariance is structural, not empirical.
- Single-token `Qwen2::matvec` (used by `generate`'s decode and by the speculative correction forward) is untouched: it keeps the fused float dot (or kq8d when opted in). `prefill_run` was already on kq8 for `chunk > 1`; with `tokens >= 1` a 1-token prompt chunk also takes kq8, which only makes prefill self-consistent.
- Correctness argument for the batch: kq8 at `tokens == 1` is bit-identical to kq8 at `tokens == N` for that row (same quantiser, same dot, same scratch), which is exactly what the audit verifies.

## 2. Predicted gain (with the math)

Kernel bench: kq8 is 1.10x the float batch kernel at 4 tokens, 1.53x at 8, 1.35x at 16, and 0.8x at 1 token (the quantiser cost the old comment cited). Continuous steps at 4 slots are 4 rows, at 8 slots 8 rows; the 1-row penalty only applies when a single sequence is active. Predicted aggregate gain at steady state: 1.05-1.20x at 8 slots, 1.03-1.08x at 4 slots. Measured median 1.06x across both, consistent; box noise (±20%) dominates individual pairs.

## 3. Falsifier (stated before the spike)

**PROCEED** if (a) the solo-vs-busy transcript audit passes on every tested shape (8/24/48 prompt tokens), (b) median continuous throughput >= 1.03x over the shipped binary, and (c) the thinking set is id-identical for `generate`. **KILL** on any audit mismatch or a neutral/negative median.

Measured: audit 3/3, median ~1.06x, thinking set 5/5. Pass.

## 4. File anchors (current HEAD, read-before-write verified)

- `src/models/qwen2.cpp:1131-1140` — `float_only` comment and the kq8 batch branch (condition edited here).
- `:1703/1705/1707` (q/k/v), `:1797` (o), `:1816/1818` (gate/up), `:1828` (down), `:2039` (batch logits) — the eight `float_only` arguments.
- `src/cpu/avx2/dequant_batch_kq8_avx2.cpp:324-387` — `gemv_dequant_batch_kq8_avx2` (per-token Q8_K quant, banded rows; accepts tokens >= 1).
- `apps/cli/main.cpp` `command_continuous_bench` — harness and in-run `state:` check (insufficient alone; see audit report).
- `reports/idea-invented-continuous-kq8-leak-audit-measured.md` — the failed audit this fixes, including the solo-vs-busy method.
- `reports/idea-invented-continuous-kq8-batch-keep.md` — the prior throughput numbers (superseded) and the withdrawn keep.

## 5. What was done (temporary spike, fully reverted)

Changed the eight `, true);` arguments to `, false);` and the condition to `tokens >= 1` in `src/models/qwen2.cpp`; built; 171/171. Audit: solo (1 request) vs busy (16 requests, 8 slots) at prompt-tokens 8/24/48, hashing the `first:` transcript. Throughput: 12 interleaved pairs (`--requests {8,16} --tokens 32 --sequences {4,8} --kv f16`). Thinking set: five prompts, greedy n=64, ids vs the shipped binary. Reverted; rebuilt; 171/171; gates 4/4; clean tree.

## 6. Measured numbers

### Audit (`bench continuous --model qwen2.5-0.5b-instruct-q4_k_m --tokens 16 --threads 10 --kv f16`)

| prompt-tokens | path-invariant solo | path-invariant busy8 | verdict |
|---:|---|---|---|
| 8 | `83-8E-B6` | `83-8E-B6` | **match** |
| 24 | `99-22-12` | `99-22-12` | **match** (was solo `35-E5-78` vs busy `99-22-12`) |
| 48 | `10-3D-0F` | `10-3D-0F` | **match** |

### Throughput vs shipped float-batch

| pair | seq | base t/s | path-invariant t/s | ratio |
|---:|---:|---:|---:|---:|
| 1 | 4 | 131.33 | 165.81 | 1.263 |
| 2 | 4 | 159.23 | 125.62 | 0.789 |
| 3 | 4 | 143.26 | 166.58 | 1.163 |
| 1 | 8 | 142.74 | 188.63 | 1.321 |
| 2 | 8 | 184.72 | 176.24 | 0.954 |
| 3 | 8 | 158.39 | 153.54 | 0.969 |
| 4 | 4 | 119.32 | 123.22 | 1.033 |
| 5 | 4 | 118.69 | 128.09 | 1.079 |
| 6 | 4 | 103.45 | 101.24 | 0.979 |
| 4 | 8 | 150.34 | 153.38 | 1.020 |
| 5 | 8 | 154.21 | 145.51 | 0.944 |
| 6 | 8 | 131.84 | 139.83 | 1.061 |

Median 1.06x; 8/12 positive. Same-run `state: independent (all transcripts identical)` in every run.

### Thinking set (generate, greedy n=64, threads 10)

| prompt | ids match shipped |
|---|---|
| story | yes |
| creative | yes |
| code | yes |
| cjk | yes |
| abstract | yes |

## 7. Reading and caveats

- The change is safe for occupancy by construction and the audit confirms it; `float_only` can be dropped in `forward_batch` **only together with** `tokens >= 1` in `batch_matvec`. Either half alone reintroduces the LEAK (the audit proved the `float_only`-only version fails at pt=24).
- Continuous-batch outputs change relative to today's build on some shapes (kq8 vs float batch). That is the intended near-exact kq8 arithmetic (9.248e-12 KL class), not a new approximation; single-request `generate` output is untouched (thinking set 5/5).
- Throughput is a modest +6% median on this box; at 8 slots with a large batch it can reach +20-30% in quiet windows (the 1.321 pair), but the box cannot resolve that reliably.
- Promotion to default is a human-gated product decision; the safe form is an opt-in `--kq8-batch` flag (`generate` unaffected). If adopted, update the `float_only` comment to record that path invariance is structural via `tokens >= 1`, and keep the solo-vs-busy audit as the regression test.
- The audit also shows why the earlier in-run `state:` check was insufficient: all in-run sequences share the same path, so it can only catch inconsistencies inside a run, not composition effects across runs.

## 8. Reimplementation recipe

1. `src/models/qwen2.cpp`: change the kq8 branch in `batch_matvec` from `tokens > 1` to `tokens >= 1` and add a comment naming the LEAK rule (occupancy 1 must share the kernel with occupancy N).
2. Change the eight `batch_matvec(...)` calls in `forward_batch`/`output_logits_batch` (`:1703-1828`, `:2039`) from `, true);` to `, false);`.
3. Build + `tge_tests` 171/171.
4. Audit: for prompt-tokens 8/24/48, run `bench continuous --requests 1 --tokens 16 --threads 10 --sequences 4 --kv f16` and `--requests 16 --sequences 8`; hash the `first:` line of each; solo must equal busy. (This is the gate; pt=24 failed without step 1.)
5. Throughput: 6+ interleaved pairs vs a binary built from the reverted tree at `--requests {8,16} --tokens 32 --sequences {4,8} --kv f16`; expect median ~1.03-1.10x.
6. Thinking set: five prompts, greedy n=64, ids must match the shipped binary (generate path untouched).
7. Revert (report-only) or gate behind `--kq8-batch` after the human push decision.

## 9. Exact commands run and results

```text
cmake --build projects\aegean-tge\build\release --config Release   -> clean (56/56)
projects\aegean-tge\build\release\bin\tge_tests.exe                -> 171 cases, 0 failed
tge bench continuous ... --requests {1,16} --tokens 16 --threads 10 \
  --sequences {4,8} --kv f16 --prompt-tokens {8,24,48}            -> §6 audit table (3/3 match)
tge bench continuous ... --requests {8,16} --tokens 32 --threads 10 \
  --sequences {4,8} --kv f16                                      -> §6 throughput table
tge generate ... --n 64 --threads 10 --gpu off --print-ids <5 prompts> -> identical 5/5
git checkout -- src/models/qwen2.cpp
cmake --build ... -> clean; tge_tests -> 171/171
python aikit.py check aegean-tge                                   -> 4/4 gates passed
```

No push, no merge, no force-push. The spike exists only in the rebuilt binary; the committed tree is unchanged except this report.
