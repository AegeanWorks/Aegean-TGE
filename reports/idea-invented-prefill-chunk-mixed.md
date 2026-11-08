# INVENTED: Prefill chunk-size sweep — MIXED (128 wins at 2048, ties at 8192; keep 64)

- **Idea**: The prefill loop is fixed at a 64-token chunk (`kPrefillChunk`), last tuned at 510 tokens. Each chunk streams the full weight set, so a larger chunk halves weight passes and amortizes per-call overhead. Sweep 16/32/64/128/256 at 2048 tokens and validate the best at 8180 tokens.
- **Cycle date**: 2026-11-08
- **Verdict**: **MIXED — keep 64 as the default.** Chunk 128 is faster at 2048 in 3/3 alternating pairs (ratios 1.142, 1.376, 2.590; median 1.376) and was the best size in both sweep rounds, but at 8180 tokens it is a **tie**: a controlled reverse pair measured 926,314 ms (128) vs 921,603 ms (64) = **0.995x**, with prefill matvec/attention phases within 3%. The 8k target is 69% attention, and chunk size only amortizes the 28% matvec phase, so the 2048 gain does not carry.
- **Tree state**: spike reverted from `src/models/qwen2.cpp` (chunk env override + allocation sizing) and `apps/cli/main.cpp` (temporary `@file` token helper); 145/145 baseline tests rebuilt and green; `aikit.py check` 4/4; no branch, no spike commit; nothing pushed.

## 1. Mechanism

`prefill_run` processes the prompt in `capacity`-row chunks, where `capacity` is the batch buffer size derived from `kPrefillChunk = 64`. Each chunk runs a full forward over all layers, so the model weights are streamed once per chunk:

```text
weight passes = ceil(prompt_tokens / chunk)
2048 tokens, chunk 64  → 32 passes
2048 tokens, chunk 128 → 16 passes
8180 tokens, chunk 64  → 128 passes
8180 tokens, chunk 128 → 64 passes
```

The spike adds `TGE_PREFILL_CHUNK` (clamped 8–256) and sizes `logits_batch_`, `scores_` and the batch buffers from it. The chunk loop picks it up through the existing `capacity` derivation. No arithmetic changes; only the grouping order changes.

Already in place: the prefill K-quant matvec uses the kq8 batch kernel whenever `kq8_` is on (default) and `tokens > 1`; the single-token output projection is the only float-path matvec, and it is cycle 29's head tier. So the prefill-side integer matvec part of the idea needed no work.

## 2. Falsifier

- **KEEP a larger chunk** if it is ≥10% faster at 2048 **and** ≥5% faster at 8180 in a drift-controlled reverse pair, with identical ids.
- **KEEP 64** if the 8180 gain disappears.

Measured: 2048 gains do not survive at 8180; keep 64.

## 3. Measured numbers

### 2048-token prefill sweep (one run each direction, 6 threads, f32 KV)

| chunk | round 1 | round 2 | note |
|---:|---:|---:|---|
| 16 | 17.0 t/s | 20.9 t/s | noisy |
| 32 | 11.8 t/s | 18.2 t/s | noisy |
| 64 | 18.7 t/s | 20.4 t/s | default |
| **128** | **28.2 t/s** | **22.6 t/s** | best both rounds |
| 256 | 21.5 t/s | 17.4 t/s | mixed |

### 2048-token alternating 64-vs-128 pairs

| pair | 64 | 128 | ratio |
|---:|---:|---:|---:|
| 1 | 17.8 t/s | 24.5 t/s | 1.376 |
| 2 | 10.5 t/s | 27.2 t/s | 2.590 |
| 3 | 19.0 t/s | 21.7 t/s | 1.142 |
| **median** | — | — | **1.376** |

All three favor 128, but the spread (1.14–2.59) shows the box drift; the median is the usable estimate. Generated first tokens are identical across every chunk size.

### 8180-token prefill

Pair A — 64 then 128:

| chunk | prefill | matvec | attention | decode matvec (control) |
|---:|---:|---:|---:|---:|
| 64 | 1,062,028 ms (7.7 t/s) | 312,805 ms | 728,721 ms | 9,477 ms |
| 128 | 848,172 ms (9.6 t/s) | 253,375 ms | 573,200 ms | 1,153 ms |

Both matvec **and** attention improved ~25%, and decode matvec was 8x slower in the first arm — the window moved, not the kernel. Not usable as evidence.

Pair B — 128 then 64 (controlled reverse):

| chunk | prefill | matvec | attention |
|---:|---:|---:|---:|
| 128 | 926,314 ms (8.8 t/s) | 269,660 ms | 634,789 ms |
| 64 | 921,603 ms (8.9 t/s) | 262,461 ms | 640,963 ms |

128 = **0.995x** — a tie. The ids were identical in pair A.

## 4. Why the 2048 gain does not carry

- At 8180 tokens, prefill is **attention-dominated**: 635–641 s attention vs 262–270 s matvec (≈69% vs 29%).
- Chunk size only changes how often weights are re-read inside the matvec phase. Halving 128 weight passes to 64 can at most move the 29% matvec component, and that component is compute-bound (kq8 integer dots), so weight-read savings are a fraction of it.
- The 2048 window is matvec-heavier and benefits from fewer per-chunk call setups; that is a real but context-dependent gain, not a default change.

## 5. Disposition

- **Keep `kPrefillChunk = 64`.** Larger chunks add memory (scores/batch buffers scale linearly: 128 rows at context 8192 is ~100 MB) with no gain at the long-context target.
- No code change is warranted. The prefill-side integer matvec was already shipped; the only remaining prefill lever is attention at long context, which chunking does not address.
- If a future server workload is 2048-token prompts only, chunk 128 is a legitimate opt-in preset; it should not be the default.

## 6. Exact commands

```text
cmake --build --preset release                         → 54/54
.\build\release\bin\tge_tests.exe                      → 145/145
TGE_PREFILL_CHUNK=16|32|64|128|256 tge generate --model Llama-3B \
  --tokens <2048 ids> --n 8 --threads 6 --gpu off --context 2560
TGE_PREFILL_CHUNK=64|128 tge generate --model Llama-3B \
  --tokens @ids8180.txt --n 8 --threads 6 --gpu off --context 8192
git checkout -- src/models/qwen2.cpp apps/cli/main.cpp
python aikit.py check aegean-tge                        → 4/4
```

No push, no merge, no force-push. Spike code is not in the tree or in any commit.
