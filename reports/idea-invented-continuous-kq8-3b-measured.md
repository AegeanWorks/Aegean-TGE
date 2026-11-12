# MEASURED: path-invariant kq8 continuous decode on the reference 3B model — 1.35-1.71x (median ~1.6x), audit clean, generate parity

- **Question**: The path-invariant kq8 change (`tokens >= 1` + drop `float_only`) was priced on `qwen2.5-0.5b-instruct-q4_k_m` only: +6% median, audit 3/3 shapes, thinking set identical. The reference model for the loop is **Llama-3.2-3B-Instruct-Q4_K_M**, where decode is weight-bandwidth-bound and the kq8 batch kernel's advantage over the float batch GEMM is much larger at continuous-batch token counts. Re-run the occupancy audit, throughput pairs, and generate parity on the reference model.
- **Cycle date**: 2026-11-11
- **Verdict**: **MEASURED — much stronger on 3B.** `bench continuous`, 10 threads, f16 KV:
  - **4 slots, 8 requests x 16 tokens**: 20.19/17.42/17.40/17.02 -> 30.75/29.86/28.45/28.90 t/s, ratios **1.523/1.714/1.635/1.698** (median **~1.67x**, 4/4).
  - **8 slots, 12 requests x 8 tokens**: 22.14/18.46 -> 29.84/29.42, ratios **1.348/1.594** (median ~1.47x).
  - **Occupancy audit passes**: prompt-tokens 24 and 48, solo vs busy8, both arms -> `2E-7E-48…` and `45-B1-96…` identical across solo/busy and across float/kq8.
  - **Generate parity**: 16-token greedy ids identical (`D0-25-F3…`) on float vs kq8-path-invariant builds; `state: independent (all transcripts identical)` in every continuous run.
- **Tree state at close**: spike reverted from `src/models/qwen2.cpp`; `git status` clean; stash holds only the pre-existing 2026-11-04 peer entry; rebuild clean, `tge_tests` **171/171**, `python aikit.py check aegean-tge` **4/4**. No branch, no spike commit, nothing pushed.

## 1. Mechanism

Same spike as `reports/idea-invented-continuous-kq8-path-invariant-keep.md`: eight `float_only` arguments in `forward_batch`/`output_logits_batch` set to `false`, and `batch_matvec`'s kq8 condition changed from `tokens > 1` to `tokens >= 1`. All occupancies in the batch path then run `gemv_dequant_batch_kq8_avx2`, whose per-row Q8_K quantisation makes a row's result independent of batch composition.

Why 3B gains so much more than 0.5B: at 4 tokens/step the kq8 kernel is 1.10x the float batch kernel on 0.5B but the 3B decode is far more weight-bandwidth-bound; the float path's dequantise-to-f32 scratch and f32 GEMM cost scales with rows x cols while the integer path streams packed rows. The batch kernel bench at 4 tokens (1.10x) understates the model-level effect for the larger model because the float path's scratch round trip grows with the weight matrix and the box's page-cache behaviour.

## 2. Predicted gain (with the math)

Weight-stream model: 3B Q4_K_M = 2.02 GiB/token, 6 threads previously ~18.7 GB/s. At 4 slots the batch does one weight pass for 4 rows; the float path additionally writes/reads a dequantised f32 scratch of `rows x cols x 4 B`, which for the 8192x3072 FFN matrices is 96 MiB per matvec — the integer path removes that traffic. Prediction from the 0.5B result (+6%) scaled by the 3B's lower compute/byte ratio was "larger, 1.2-1.5x"; measured 1.5-1.7x.

## 3. Falsifier (stated before the run)

**PROCEED** if the solo-vs-busy audit passes on every clean shape, generate ids match, and the 3B continuous median is >= 1.10x. **KILL** on any audit mismatch.

Measured: audit clean (24/48), generate identical, medians 1.67x (4 slots) and 1.47x (8 slots). Pass.

## 4. File anchors (current HEAD, read-before-write verified)

- `src/models/qwen2.cpp:1131-1140` — `batch_matvec` `float_only` guard and kq8 branch.
- `:1703-1828`, `:2039` — the eight `forward_batch`/`output_logits_batch` arguments.
- `src/cpu/avx2/dequant_batch_kq8_avx2.cpp:324-387` — kq8 batch kernel (per-token Q8_K, banded rows).
- `apps/cli/main.cpp` `command_continuous_bench` — harness; the `state:` check is in-run only (solo-vs-busy is the real audit).
- `reports/idea-invented-continuous-kq8-path-invariant-keep.md` — the 0.5B result and recipe this re-measures.
- `reports/idea-invented-continuous-kq8-leak-audit-measured.md` — the audit method.

## 5. Measured numbers

`bench continuous --model Llama-3.2-3B-Instruct-Q4_K_M.gguf --threads 10 --kv f16`.

| config | pair | float-batch t/s | path-invariant t/s | ratio |
|---|---:|---:|---:|---:|
| 4 slots, 8 req, 16 tok | 1 | 20.19 | 30.75 | 1.523 |
| | 2 | 17.42 | 29.86 | 1.714 |
| | 3 | 17.40 | 28.45 | 1.635 |
| | 4 | 17.02 | 28.90 | 1.698 |
| 8 slots, 12 req, 8 tok | 1 | 22.14 | 29.84 | 1.348 |
| | 2 | 18.46 | 29.42 | 1.594 |

Audit (solo vs busy8, `--tokens 8`): pt=24 `2E-7E-48` in all four arm/occupancy combinations; pt=48 `45-B1-96` in all four. Generate (n=16, 3B, greedy): `D0-25-F3` on both builds.

## 6. Reading and caveats

- This is the strongest continuous-batch result in the library so far and it lands on the reference model: +47-71% aggregate decode at 4 slots, +35-59% at 8. The 0.5B's +6% was a poor proxy; continuous batching on 3B is where the change matters.
- All parity evidence holds: occupancy audit 2/2 shapes on 3B (plus 3/3 on 0.5B), generate thinking-set ids identical (1/1 spot on 3B, 5/5 on 0.5B), in-run transcripts identical in every run.
- Promotion to default is still human-gated. If promoted, the invariant to preserve is "kernel chosen by availability, never by occupancy"; the regression set is the solo-vs-busy audit (pt 24/48), the generate parity check, and `tge_tests`.
- The earlier 0.5B-only caveat in the prior keep report is superseded: the change is worth much more on the reference model, and the audit/parity still hold.

## 7. Reimplementation recipe

1. Apply the path-invariant change (`reports/idea-invented-continuous-kq8-path-invariant-keep.md` §8): eight `float_only` flips + `tokens >= 1`.
2. Build + `tge_tests` 171/171. Build a float-batch baseline binary from the reverted tree.
3. Audit on 3B: `bench continuous --requests {1,12} --tokens 8 --threads 10 --sequences {4,8} --kv f16 --prompt-tokens {24,48}`; hash the `first:` line; solo must equal busy on both arms.
4. Throughput: 4 pairs at `--requests 8 --tokens 16 --sequences 4`, 2 pairs at `--requests 12 --tokens 8 --sequences 8`; expect 1.35-1.71x.
5. Generate parity: `generate --n 16 --threads 10 --gpu off --print-ids` on the 3B; ids must match the shipped binary.
6. Revert (report-only) or gate behind `--kq8-batch` after the human decision.

## 8. Exact commands run and results

```text
cmake --build projects\aegean-tge\build\release --config Release   -> clean (56/56)
projects\aegean-tge\build\release\bin\tge_tests.exe                -> 171 cases, 0 failed
tge bench continuous ... --requests {1,12} --tokens 8 --threads 10 \
  --sequences {4,8} --kv f16 --prompt-tokens {24,48}              -> audit clean
tge bench continuous ... --requests {8,12} --tokens {16,8} --threads 10 \
  --sequences {4,8} --kv f16                                      -> §5 throughput table
tge generate ... --n 16 --threads 10 --gpu off --print-ids         -> ids identical (D0-25-F3)
git checkout -- src/models/qwen2.cpp
cmake --build ... -> clean; tge_tests -> 171/171
python aikit.py check aegean-tge                                   -> 4/4 gates passed
```

No push, no merge, no force-push. Baseline binary lives at `build/idea/tge-cont-float-base.exe`; the committed tree is unchanged except this report.
