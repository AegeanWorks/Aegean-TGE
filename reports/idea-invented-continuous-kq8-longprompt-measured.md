# MEASURED: path-invariant kq8 continuous decode at 56-token prompts on 3B — 1.41-1.58x (median 1.57x), audit clean

- **Question**: The 3B continuous result (`idea-invented-continuous-kq8-3b-measured.md`) used a 24-token prompt, so admission and KV-cache costs were minimal. Real continuous loads carry longer prompts (chat system prompts, RAG contexts). Re-price at `--prompt-tokens 56` with 6 slots on the reference model, and re-run the solo-vs-busy audit at that shape.
- **Cycle date**: 2026-11-11
- **Verdict**: **MEASURED — the win holds at longer prompts.** Llama-3.2-3B-Instruct-Q4_K_M, `bench continuous --requests 6 --tokens 8 --threads 10 --sequences 6 --kv f16 --prompt-tokens 56`:
  - float-batch: 13.91/13.94/12.21 t/s -> path-invariant kq8: 22.01/19.64/19.17 t/s, ratios **1.582/1.409/1.570**, **median 1.57x** (3/3).
  - **Occupancy audit passes at 56 prompt tokens**: solo vs busy6, both arms -> `45-B1-96…`, identical across all four arm/occupancy combinations.
  - `state: independent (all transcripts identical)` in every run.
- **Tree state at close**: spike reverted from `src/models/qwen2.cpp`; `git status` clean; stash holds only the pre-existing 2026-11-04 peer entry; rebuild clean, `tge_tests` **171/171**, `python aikit.py check aegean-tge` **4/4**. No branch, no spike commit, nothing pushed.

## 1. Mechanism

Same path-invariant spike (`tokens >= 1` in `batch_matvec`, eight `float_only` flips). This cycle only changes the workload: 56 prompt tokens instead of 24, 6 slots instead of 4/8, 8 decode tokens per request. At 56 tokens the prefill admission per request is larger and the KV stores/reads are longer, so the kernel's share of the step should shrink; the measurement asks whether the +1.5-1.7x survives.

## 2. Predicted gain (with the math)

At 24 prompt tokens the measured 3B 4-slot ratio was 1.67x and the 8-slot 1.47x. Longer prompts add prefill and attention work per request that is identical in both arms, so the ratio should compress toward the decode-kernel share: prediction 1.2-1.5x. Measured 1.41-1.58x — inside, slightly above the midpoint.

## 3. Falsifier (stated before the run)

**PROCEED** if the audit passes at the 56-token shape and the median ratio is >= 1.15x; **KILL** on any solo-vs-busy mismatch or a median below 1.10x.

Measured: audit 4/4 combinations identical, median 1.57x. Pass.

## 4. File anchors (current HEAD, read-before-write verified)

- `src/models/qwen2.cpp:1131-1140` — `batch_matvec` kq8 guard/condition.
- `:1703-1828`, `:2039` — eight `float_only` arguments.
- `src/cpu/avx2/dequant_batch_kq8_avx2.cpp:324-387` — kq8 batch kernel.
- `apps/cli/main.cpp` `command_continuous_bench` — harness (`first:` line, `state:` check).
- `reports/idea-invented-continuous-kq8-3b-measured.md` — the 24-token 3B result this extends.
- `reports/idea-invented-continuous-kq8-leak-audit-measured.md` — the audit method.

## 5. Measured numbers

| pair | float-batch t/s | path-invariant t/s | ratio |
|---:|---:|---:|---:|
| 1 | 13.91 | 22.01 | 1.582 |
| 2 | 13.94 | 19.64 | 1.409 |
| 3 | 12.21 | 19.17 | 1.570 |
| **median** | 13.91 | 19.64 | **1.570** |

Audit at `--prompt-tokens 56`: float solo/busy6 `45-B1-96`, kq8 solo/busy6 `45-B1-96`.

## 6. Reading

- The continuous kq8 win is not an artefact of tiny prompts: at 56 prompt tokens with 6 slots it is still ~1.57x on the reference 3B. Combined with the 24-token result (1.67x at 4 slots, 1.47x at 8), the change is robust across prompt length and slot count.
- Parity evidence now covers audit shapes 8/24/48/56 on 0.5B/3B, generate ids on both models, and in-run transcripts in every run. Nothing in the evidence blocks promotion; only the human gate remains.
- The gain's source is unchanged: removing the dequantise-to-f32 scratch and f32 GEMM from the continuous matvec, which at 3B is the dominant per-token cost.

## 7. Reimplementation recipe

1. Apply the path-invariant change (`reports/idea-invented-continuous-kq8-path-invariant-keep.md` §8).
2. Build + 171/171; build the float-batch baseline from the reverted tree.
3. `bench continuous --model Llama-3.2-3B-Q4_K_M --requests 6 --tokens 8 --threads 10 --sequences 6 --kv f16 --prompt-tokens 56`, alternating arms -> expect 1.41-1.58x.
4. Audit: same command with `--requests 1` vs `--requests 8`, hash the `first:` line; solo must equal busy.
5. Revert (report-only) or promote after the human decision.

## 8. Exact commands run and results

```text
cmake --build projects\aegean-tge\build\release --config Release   -> clean (56/56)
projects\aegean-tge\build\release\bin\tge_tests.exe                -> 171 cases, 0 failed
tge bench continuous ... --requests 6 --tokens 8 --threads 10 --sequences 6 \
  --kv f16 --prompt-tokens 56                                     -> §5 table
tge bench continuous ... --requests {1,8} ... --prompt-tokens 56   -> audit 45-B1-96 in all four runs
git checkout -- src/models/qwen2.cpp
cmake --build ... -> clean; tge_tests -> 171/171
python aikit.py check aegean-tge                                   -> 4/4 gates passed
```

No push, no merge, no force-push. Baseline binary lives at `build/idea/tge-cont-float-base.exe`; the committed tree is unchanged except this report.
