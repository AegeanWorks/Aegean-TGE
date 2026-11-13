# MEASURED: 128-row prompt chunks above 4096 are still a loss at 8k (0.95-0.99x) — the block kernel's K/V reuse is not the binding term

- **Idea**: `Qwen2::prefill_run` uses 128-row prompt chunks up to 4096 tokens and 64 above (`src/models/qwen2.cpp:2406`). That rule was measured before the query-block attention kernel and its value tile landed ("64 was 7-11% faster in both orders"). The block kernel loads each K/V row once per chunk, so a 128-query chunk should halve the K/V row loads per token versus 64 — exactly the kind of reuse the old rule may have under-priced at 8k, where attention is now 66-69 s of a ~205 s prefill. Re-test 128 vs 64 above 4096 on the current tree.
- **Cycle date**: 2026-11-13
- **Verdict**: **KILLED — 64 stays; the reopen does not hold.** Llama-3.2-3B-Instruct-Q4_K_M, 8,193-token prompt, `--context 8704 --n 16 --kv f16 --threads 10 --gpu off`, interleaved pairs with the pre-spike binary preserved:
  - **pair 1**: chunk-64 prefill **213.4 s** (matvec 132.1, attn 69.9) vs chunk-128 **224.5 s** (137.9, 74.6) -> **0.95x** for 128. Matvec differs 4.4%, so this pair is drift-contaminated.
  - **pair 2**: chunk-128 **229.7 s** (140.5, 75.1) vs chunk-64 **227.5 s** (138.8, 73.7) -> **0.99x** for 128. Matvec within 1.2%, so this is the clean reading.
  - **The attention phase did not fall** (74.6-75.1 s for 128 vs 69.9-73.7 s for 64), despite the 2x K/V-row reuse. Attention is compute-bound in the dot/softmax work, not in K/V row loads — the same conclusion as the tiled-attention ceiling kill (attention ran at ~4.4 GB/s against a ~55 GB/s roof). Halving a non-binding traffic term buys nothing, and the larger chunk pays slightly more in matvec/working-set cost.
  - **The old rule's direction is confirmed, with a smaller magnitude**: 128 is 1-5% slower above 4096 on the current tree, rather than the recorded 7-11%. The rule stays at 64; the comment's margin should be read as ~1-5% today, not 7-11%.
- **Tree state at close**: spike reverted from `src/models/qwen2.cpp`; `git status` clean; stash holds only the pre-existing 2026-11-04 peer entry; rebuild clean, `tge_tests` **171/171**, `python aikit.py check aegean-tge` **4/4**. No branch, no spike commit, nothing pushed.

## 1. Mechanism (as spiked)

`prefill_run` walked the prompt in chunks of `count <= 4096 ? capacity : min(capacity, 64)`. The spike replaced the expression with `capacity` (128 rows), so every prompt chunk above 4096 was 128 rows. Nothing else changed: the block attention args take `tokens = chunk` and the block kernel tiles the query block in 8-query value tiles; both chunk sizes execute the same math, so this is a pure scheduling/working-set experiment (ids identical by construction; no fidelity run owed).

## 2. Results

| pair | arm | prefill | matvec | attn | norm | decode |
|---|---|---:|---:|---:|---:|---:|
| 1 | chunk 64 | 213.4 s | 132.1 s | 69.9 s | 9.6 s | 5.89 t/s |
| 1 | chunk 128 | 224.5 s | 137.9 s | 74.6 s | 10.0 s | 5.74 t/s |
| 2 | chunk 128 | 229.7 s | 140.5 s | 75.1 s | 12.0 s | 5.53 t/s |
| 2 | chunk 64 | 227.5 s | 138.8 s | 73.7 s | 12.6 s | 5.67 t/s |

Ratios (128 / 64): **0.95x** (pair 1, drift), **0.99x** (pair 2, matvec matched). Attention phase is flat-to-worse for 128 in both pairs; matvec is flat-to-worse. The box slowed across the sequence (matvec 132 -> 140 s), which is why pair 1 is not admissible and pair 2 is quoted as the result.

## 3. Reading

- **Prompt-chunk reuse is not the lever at 8k.** The block kernel already amortizes each K row over the chunk's queries; doubling the chunk does not reduce the attention phase because the phase is compute-bound, not row-load-bound. This is the third independent confirmation of that ceiling (tiled-attention kill, value-tile work, now chunk size).
- **The 64-row rule is correct but its margin is smaller than recorded**: 1-5% on today's tree at 8k, not 7-11%. The comment at `src/models/qwen2.cpp:33-38` is left as-is per the loop's no-code rule; a future doc pass could update the margin.
- **What this closes**: the "maybe the block kernel changes the chunk trade" question. It does not. 8k prefill's 66-75 s attention phase needs a different kernel (lossy sparse or block-sparse, already priced), not a scheduler knob.
- **No decision change**: 128 rows up to 4096, 64 above. No user-facing flag exists for this, so no recommendation changes.

## 4. Falsifier (stated before the runs)

**REOPEN the 128-row rule above 4096** if 128 is >= 1.05x the 64-row binary at 8k with a lower attention phase in a matvec-matched pair. **KILL** otherwise. Measured 0.95x (drift) / 0.99x (clean) with attention flat or higher: kill.

## 5. File anchors (current HEAD, read-before-write verified; spike reverted)

- `src/models/qwen2.cpp:33-38` — the rule and its comment ("128 rows up to 4096 and 64 above; 64 was 7-11% faster").
- `src/models/qwen2.cpp:2406` — the chunk-limit expression the spike replaced.
- `src/models/qwen2.cpp:2407-2412` — the chunk loop and `chunk_limit` use.
- `src/models/qwen2.cpp:1580-1615` — block attention dispatch (tokens = chunk; 8-query value tile in the kernel).
- `src/cpu/avx2/attention_avx2.cpp:660-864` — block kernel (row load once per key, tile accumulators).
- `reports/idea-invented-tiled-attn-ceiling-killed.md` — the compute-bound attention ceiling this result reproduces.
- measurement-log — the original chunk-size round (pre-block-attention).

## 6. Reimplementation recipe

1. Preserve the pre-spike binary: `copy build\release\bin\tge.exe %TEMP%\kilo\tge_base_chunk64.exe`.
2. Edit `src/models/qwen2.cpp:2406` to `const int chunk_limit = capacity;`, build (`cmake --build --preset release`), run `tge_tests` (171/171).
3. Interleave arms at 8k:
   ```powershell
   tge generate --model Llama-3.2-3B-Instruct-Q4_K_M.gguf --prompt ("The fox runs. " * 2048) \
     --context 8704 --n 16 --kv f16 --threads 10 --gpu off
   ```
   Expect 128 rows: prefill 224-230 s, attn 74.6-75.1 s; 64 rows: 213-228 s, attn 69.9-73.7 s; ratio 0.95-0.99x for 128. Compare only pairs with matvec within ~3%.
4. Revert the file, rebuild, retest, confirm clean tree.
5. No code ships.

## 7. Exact commands run and results

```text
cmake --build --preset release                    -> clean (56/56)
build\release\bin\tge_tests.exe                   -> 171 cases, 0 failed (spike build)

8k (compact fox x2048, ctx 8704, n 16, f16, 10 threads, gpu off), interleaved base/spike/spike/base:
  chunk 64 : 213.4 s (matvec 132.1, attn 69.9) ; 227.5 s (matvec 138.8, attn 73.7)
  chunk128 : 224.5 s (matvec 137.9, attn 74.6) ; 229.7 s (matvec 140.5, attn 75.1)
  ratio 128/64: 0.95x (drift), 0.99x (matvec-matched)
git checkout -- src/models/qwen2.cpp              -> clean
cmake --build --preset release -> clean ; tge_tests -> 171/171 ; git status -> clean
python aikit.py check aegean-tge -> 4/4 gates passed
```

No push, no merge, no force-push. The committed tree is unchanged except this report.
