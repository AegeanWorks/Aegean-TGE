# TGE vs llama.cpp on Llama-3.2-3B-Q4_K_M — 2026-11-07

- **Purpose**: current side-by-side CPU comparison requested after the idea cycles. No engine code changed for this measurement; the tree is the post-cycle-10 baseline.
- **Verdict**: TGE is behind llama.cpp on this box in the stable runs — pp512 ~58% and tg128 ~55–63%, pp2048 ~85–103% and tg32 ~47–59%. The box was heavily loaded and the same-window pairs were too noisy to refine those ratios; the numbers are directional, not a tight benchmark.
- **Tree state**: clean; report-only commit; nothing pushed.

## 1. Setup

| item | value |
|---|---|
| Model | `Llama-3.2-3B-Instruct-Q4_K_M.gguf`, 1.87 GiB, 3.21 B params |
| CPU | Intel Core i5-14400F, 6 P-cores used, 16 GiB single DIMM |
| TGE | `aegean-tge` `5588448`, release build, GCC 16.2, `--threads 6 --gpu off` |
| llama.cpp | `C:\Users\Yunus-Emre\.llama-bench\build-release\bin\llama-bench.exe`, build `cd8427d` (1), GCC 16.2 |
| KV | llama default f16 K+V; TGE default f32, plus TGE f16 for a like-for-like KV comparison |
| Tests | `llama-bench -p 512 -n 128 -r 3`; `-p 2048 -n 32 -r 3`; TGE `generate --tokens <id prompt> --n 128/32`, 3 runs each |

The TGE prompts are fixed token-id lists of exactly 512/2048 tokens, so the prefill token counts match llama-bench's pp512/pp2048. TGE `decode` reports 127 tokens for `--n 128` (the sampler's first token is outside the timed decode loop); llama reports 128. The difference is <1%.

## 2. Results

### pp512 / tg128

| test | llama.cpp | TGE f32 | TGE f16 |
|---|---:|---:|---:|
| pp512 | **73.37 ± 0.80 t/s** | 41.7 ms-median (best 42.7) | 42.9 ms-median (best 43.4) |
| tg128 | **9.97 ± 1.08 t/s** | 5.45 ms-median (best 5.65) | 5.41 ms-median (best 6.27) |
| ratio, best | 1.00 | 0.582 / 0.567 | 0.592 / 0.629 |

### pp2048 / tg32

| test | llama.cpp | TGE f32 | TGE f16 |
|---|---:|---:|---:|
| pp2048 | **38.02 ± 5.81 t/s** | 34.1 ms-median (best 34.3) | 32.5 ms-median (best 39.1) |
| tg32 | **9.84 ± 0.45 t/s** | 4.59 ms-median (best 5.07) | 4.77 ms-median (best 5.78) |
| ratio, best | 1.00 | 0.902 / 0.515 | 1.028 / 0.587 |

Raw TGE runs:

```text
pp512  f32: 41.6, 42.7, 41.7 t/s | tg128 f32: 5.45, 5.65, 4.86 t/s
pp512  f16: 43.4, 42.3, 42.9 t/s | tg128 f16: 6.27, 4.27, 5.41 t/s
pp2048 f32: 34.1, 33.8, 34.3 t/s | tg32  f32: 5.07, 4.59, 3.74 t/s
pp2048 f16: 39.1, 32.5, 24.1 t/s | tg32  f16: 4.77, 2.91, 5.78 t/s
```

llama-bench raw:

```text
pp512  73.37 ± 0.80 | tg128 9.97 ± 1.08
pp2048 38.02 ± 5.81 | tg32  9.84 ± 0.45
```

## 3. Noise and fairness

The box was not quiet. A same-window paired check (`llama-bench -p 512 -n 32 -r 1` → TGE f32/f16, order reversed on round 2) produced:

```text
round1: llama pp512=46.57 tg32=5.77 | TGE f32 pp=22.70 tg=5.60 | TGE f16 pp=42.50 tg=5.75
round2: llama pp512=52.31 tg32=8.59 | TGE f32 pp=38.30 tg=3.42 | TGE f16 pp=35.60 tg=3.76
```

llama.cpp itself swung from 73 to 46–52 t/s pp512 and 9.97 to 5.77–8.59 t/s decode between runs. Round 1's f16 pair is near parity (42.50 vs 46.57 prefill, 5.75 vs 5.77 decode); round 2 is much worse for TGE. The variance is larger than the gap being measured, so no single paired round is trustworthy. The stable table above is the best available evidence, with the ±20% caveat from the project measurement log (this session was worse than ±20%).

TGE's 512 numbers are also depressed relative to its own other runs: an earlier 401-token prompt measured 52.5 t/s prefill and 7.28 t/s 2048 decode under a lighter load. That supports treating the 512 row as load-limited.

## 4. Interpretation

- **Prefill**: TGE reaches ~58% of llama at 512 and ~85–103% at 2048 in the best runs. The 2048 result is consistent with the historical "71% prefill" note and shows that TGE's chunked prefill (64-token chunks) scales better at longer prompts.
- **Decode**: TGE is ~47–59% of llama in the stable long runs; the best 2k decode (5.78 t/s f16) is ~59% of llama's 9.84. The gap is not explained by KV type: TGE f16 is close to llama's default f16, and still behind.
- **KV type**: TGE f32 vs f16 at 512 is within noise; f16 helps decode at 2048 on the best run (5.78 vs 5.07), consistent with the log's f16 memory case.
- **Startup**: not compared here. TGE's tokenizer/state-cache and no-init work is in the report library but is not in this baseline binary (spikes reverted), so this table is the shipped/default engine, not the optimized bundle.

## 5. What would make the comparison decisive

Run on a quiet box with ABBA ordering:

1. `llama-bench -p 512 -n 128 -r 3` and `-p 2048 -n 32 -r 3`;
2. TGE f32 and f16, same prompts/n, 3 runs each, interleaved `L-T-T-L` per round;
3. report best-of-3 and the median of paired ratios, not cross-session absolutes;
4. keep the same 6-thread, no-GPU, f16-KV-for-llama setup.

Until then, the honest summary is: **TGE is behind llama.cpp on this box, with a large decode gap and a prefill gap that narrows at 2048; the current box load makes the exact percentages unreliable.**

## 6. Exact commands

```text
llama-bench -m Llama-3.2-3B-Instruct-Q4_K_M.gguf -t 6 -ngl 0 -p 512  -n 128 -r 3
llama-bench -m Llama-3.2-3B-Instruct-Q4_K_M.gguf -t 6 -ngl 0 -p 2048 -n 32  -r 3
tge generate --model Llama-3.2-3B-Instruct-Q4_K_M.gguf --tokens <512 ids>  --n 128 --threads 6 --gpu off --kv f32|f16
tge generate --model Llama-3.2-3B-Instruct-Q4_K_M.gguf --tokens <2048 ids> --n 32  --threads 6 --gpu off --kv f32|f16
```

No code changes, no push, no merge.
