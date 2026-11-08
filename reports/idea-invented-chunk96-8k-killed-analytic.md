# INVENTED: 96-token prefill chunks above 4096 — KILLED ANALYTICALLY (per-token kq8 throughput is best at 64)

- **Idea**: Above 4096 prompt tokens the engine uses 64-row chunks because 128 was slower at 8192. A 96-row chunk is a middle point and is a multiple of the batch kernel's 6-token group (16 groups exactly, no 4-row tail), so it might amortize weight passes better than 64 without the 128-token attention/working-set penalty.
- **Cycle date**: 2026-11-08
- **Verdict**: **KILLED on today's tree before a spike.** The per-token integer prefill kernel is fastest at 64 tokens: best-of-3 `bench batch --format q4k --kernel kq8` gives **305.3 / 292.8 / 290.4 GFLOPS** at 64 / 96 / 128 on 3072x3072, and **341.4 / 303.3 / 301.7** on 8192x3072. Total matvec time is proportional to `1 / GFLOPS(chunk)`, so 96 is 4% (attention projections) to 11% (FFN) slower than 64 on the dominant phase; the only upside is fewer attention key-row loads, and the 128-token retest already failed to turn that into a win at 8161.
- **Tree state**: no spike, no branch, no edits. The report is the only change.

## 1. Mechanism and the math

Prefill processes a prompt in `C`-token chunks. Each chunk invokes the batch matvec once per weight matrix, and the kernel's per-token efficiency depends on `C` (activation rows resident, weight stream amortized, group tails). For `N` prompt tokens:

```text
total matvec  =  (N / C) calls * (C * per_token_work / GFLOPS(C))
              =  N * per_token_work / GFLOPS(C)
```

So the fastest chunk is the one with the highest per-token GFLOPS, not the one with the fewest calls. The shipped rule uses `C = 128` up to 4096 and `C = 64` above, because at 8192 the 128-token attention working set outweighed the matvec difference. A 96-token middle point would have to beat 64 on GFLOPS and attention jointly.

## 2. Current-tree probe (no spike)

`tge bench batch --rows R --cols C --tokens T --repeat 10 --format q4k --kernel kq8`, best of three processes; box noise ~+/-10% on this probe, but the ordering is stable across shapes:

| shape | 64 tokens | 96 tokens | 128 tokens |
|---|---:|---:|---:|
| 3072x3072 (attention projections) | **305.3 GFLOPS** | 292.8 | 290.4 |
| 8192x3072 (FFN) | **341.4 GFLOPS** | 303.3 | 301.7 |

Per-token throughput is monotonically worse above 64. The 96-token chunk is not a sweet spot; it is already on the 128 side of the curve.

## 3. The attention side does not rescue it

The only gain from a larger chunk is fewer key-row loads in block prefill (one K/V row load feeds the chunk's queries). The 2026-11-08 retest of 128 vs 64 at 8161 tokens (post value-tile) measured total ratios 1.085 / 0.832 / 0.743 / 1.164 (median 0.958), but the identical 64-row arm's matvec drifted 152-289 s (1.90x). The isolated kernel showed 128 does not reduce per-token matvec work. 96 sits between 64 and 128 on both the matvec curve and the attention working set, so the expected value is between two results that are already neutral-to-negative and inside the box drift.

## 4. Falsifier (for any future retest)

- **KEEP** if 8161-token f16 total prefill is >=5% faster than the 64-row rule in at least 4 of 6 alternating pairs with identical ids and unchanged fidelity.
- **KILL** otherwise.

The current probe predicts the falsifier fails before the benchmark: the dominant phase loses 4-11% per token.

## 5. Full reimplementation recipe (if a future box or kernel changes the curve)

1. `src/models/qwen2.cpp`, `prefill_run`: change
   `const int chunk_limit = count <= 4096 ? capacity : std::min(capacity, 64);`
   to `... : std::min(capacity, 96);`.
   Buffers are already sized for 128 rows (`kPrefillChunk = 128`), so no allocation change is needed.
2. Rebuild, `tge_tests` (162/162 expected; chunk size is numerical-order-neutral per token).
3. A/B at 8161 tokens (`--context 8192 --kv f16 --n 2`) with alternating arm order; compare total prefill, matvec and attention phases, and ids. Predict no win from the probe above.
4. If it somehow wins, also check 2049/4081 (the <=4096 path is untouched) and the 8192 chunk-selection unit if one is added.

## 6. Reopen triggers

- A new batch kernel whose per-token GFLOPS improves with chunk size (e.g., a prepacked layout that amortizes better at 96/128).
- A quiet box (identical-arm matvec spread <10%) with at least six 8161-token pairs, if the attention side is re-opened by a layout change.
- Do not retest on this kernel/shape: 64 is the measured maximum of the per-token throughput curve.
