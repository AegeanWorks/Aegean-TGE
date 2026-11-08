# TGE vs llama.cpp — 2026-11-08 re-comparison (shipped baseline and the validated stack)

- **Purpose**: Fresh side-by-side against llama.cpp `cd8427d` after the 48-cycle idea loop. The shipped tree is unchanged by the loop (all spikes reverted), so the comparison has two TGE arms: the **shipped baseline** and the **best measured configuration** (block prefill attention + kq8d LM head/down + full q8 value tail + 4 GQA decode chunks), all behind env switches and reverted after measurement.
- **Verdict**: llama.cpp remains **faster at prefill** in this session (TGE 0.77–0.80x at 512, 0.49–0.64x at 2048). On decode, the shipped engine is 0.56–0.70x at 512 and 0.25–0.43x at 2048; the **validated stack reaches llama parity at 512 (0.98–1.06x)** and is 1.5x over the shipped baseline. The box was heavily loaded and the 2048 decode sample is only 31 tokens, so 2048 decode is unresolved.
- **Tree state**: measurement-only; spikes reverted; 145/145 tests; `aikit.py check` 4/4; no push, no merge.

## 1. Setup

| item | value |
|---|---|
| Model | `Llama-3.2-3B-Instruct-Q4_K_M.gguf` (1.87 GiB) |
| CPU | i5-14400F, 6 threads, no GPU |
| TGE | `tge` at HEAD (shipped baseline); stack = `TGE_ATTN_BLOCK=on`, `TGE_KQ8D_HEAD=1`, `TGE_KQ8D_SCOPE=down`, `TGE_Q8_FULLTAIL=on`, `TGE_GQA_CHUNKS=4` |
| llama.cpp | `llama-bench` build 1, commit **cd8427d**, GCC 16.2, `-t 6 -ngl 0 -r 1` |
| KV | llama default f16; TGE f32 |
| Prompts | TGE fixed 512/2048-token id prompts; llama-bench internal prompts |
| Protocol | ABBA per test: round 1 llama → base → stack, round 2 stack → base → llama |

**Load warning**: llama itself measured pp512 41.8 / pp2048 29.2 t/s here, versus 73.4 / 38.0 in the 2026-11-07 session — the machine was ~40% slower. Only same-window ratios are meaningful.

## 2. Results

### pp512 / tg128

| round | llama pp | llama tg | base pp | base tg | stack pp | stack tg |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 41.79 | 5.54 | 33.6 | 3.88 | 35.7 | **5.87** |
| 2 | 37.69 | 4.34 | 28.9 | 2.44 | 27.0 | 4.24 |
| **best** | **41.79** | **5.54** | 33.6 | 3.88 | 35.7 | 5.87 |

Same-window ratios: base/llama 0.804/0.700 (R1), 0.767/0.562 (R2); stack/llama 0.854/**1.060** (R1), 0.716/0.977 (R2).

### pp2048 / tg32

| round | llama pp | llama tg | base pp | base tg | stack pp | stack tg |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 29.22 | 6.29 | 17.3 | 2.68 | 18.6 | 2.40 |
| 2 | 28.81 | 5.83 | 14.0 | 1.46 | 17.1 | 2.72 |
| **best** | **29.22** | **6.29** | 17.3 | 2.68 | 18.6 | 2.72 |

Same-window ratios: base/llama 0.592/0.426 (R1), 0.486/0.250 (R2); stack/llama 0.636/0.382 (R1), 0.594/0.467 (R2). The 31-token decode sample is too short for the stack/base comparison (0.90x and 1.86x in opposite directions).

### Stack vs shipped baseline (best-of)

| test | shipped | stack | ratio |
|---|---:|---:|---:|
| pp512 | 33.6 | 35.7 | 1.063x |
| tg128 | 3.88 | 5.87 | **1.513x** |
| pp2048 | 17.3 | 18.6 | 1.075x |
| tg32 | 2.68 | 2.72 | 1.015x |

The prefill ratios at 512/2048 are ~1.06–1.08x, consistent with block attention being neutral below ~2k; its 1.6–2.3x gain appears at 8k (cycles 33/36). The 512 decode gain (1.5x) is the kq8d head+down, q8 tail and 4-chunk changes.

## 3. Reading

- **Prefill remains the gap.** llama is 1.25–1.6x faster at 512 and 1.6–2.1x at 2048 on this box. The report-only block attention targets long prompts, not these.
- **Decode at 512 is now at parity with llama** using the validated stack, and the shipped engine is ~30–45% behind. At 2048 the box could not resolve decode; the shipped engine's best (2.68 t/s) and the stack's (2.72) are equal, both far below llama's 6.29 in a loaded window.
- **Speculative mode is a separate, workload-specific lever**: 1.95–2.15x on periodic 4096 prompts and 4.98x on a code prompt (cycles 43/45), with neutral free-prose behavior and a kernel-path equivalence caveat. llama was run with default settings (no draft), so that comparison is not apples-to-apples and is excluded from the table.
- Absolute numbers from 2026-11-07 are not comparable; both engines are ~40% slower today.

## 4. Exact commands

```text
llama-bench -m Llama-3.2-3B-Instruct-Q4_K_M.gguf -t 6 -ngl 0 -p 512  -n 128 -r 1
llama-bench -m ... -t 6 -ngl 0 -p 2048 -n 32 -r 1
tge generate --model <gguf> --tokens <512|2048 ids> --n 128|32 --threads 6 \
  --gpu off --kv f32 --context 1024|2560
# stack arm adds:
TGE_ATTN_BLOCK=on TGE_KQ8D_HEAD=1 TGE_KQ8D_SCOPE=down TGE_Q8_FULLTAIL=on TGE_GQA_CHUNKS=4
```

No push, no merge, no force-push. No code changed in the committed tree.
