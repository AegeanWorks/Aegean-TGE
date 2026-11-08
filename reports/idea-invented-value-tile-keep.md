# INVENTED: 8-query value-accumulation tile in block prefill attention — KEEP (attention 1.32–2.41x, ids identical)

- **Idea**: `attention_block_head_avx2` was already reusing each K/V row across the whole 128-query prompt chunk, but its value pass still laid every query's accumulator out in `args.out`: 128 x head_dim floats (64 KiB at head_dim 128, 32 KiB at 64), past the 32-48 KiB L1. Every key iteration therefore read and wrote the full accumulator set from L2. Keep a small tile of query accumulators in L1 instead.
- **Cycle date**: 2026-11-08
- **Verdict**: **KEEP as the default block-prefill kernel.** The value pass now runs in 8-query tiles whose accumulators (8 x head_dim, 2-4 KiB) stay in an aligned stack buffer, then copies each tile to the output. Attention is faster in **10/10 pairs**: median **1.690x / 1.576x** at a 2,049-token prompt (f16 / f32) and **2.09x / 2.006x** at 4,081 tokens. Total prefill median is 1.14x at 2k and 1.31x at 4k; the untouched matvec phase drifted 0.71-1.34x across the same pairs, so the attention phase is the attributable number. Generated ids are identical 10/10 and fidelity is unchanged.
- **Tree state**: kept in `src/cpu/avx2/attention_avx2.cpp`; a temporary `TGE_GQA_CHUNKS` decode override that a prior session had left in `src/models/qwen2.cpp` was dropped before the commit (inert, no report or ledger trace, defaulted to the shipped chunk rule). Tests 146/146, fidelity unchanged, `aikit.py check aegean-tge` 4/4. No push, no merge, no force-push.

## 1. Mechanism

The block kernel runs in three passes over a prompt chunk: QK scores, causal softmax, then value accumulation. Pass 2 was:

```text
for each key k:
    load V[k] once
    for each query q that attends k:
        out[q] = fma(weight[q,k], V[k], out[q])
```

`out` is the caller's `batch_attn_` buffer, so a 128-query chunk keeps 128 x head_dim accumulators live for the whole pass (64 KiB at head_dim 128). That set does not fit L1, so each of the chunk's keys causes a read and a write of all of it.

The shipped pass 2 processes eight queries at a time:

```text
for each tile of 8 queries:
    zero tile_acc[8 x head_dim]            (2-4 KiB, L1-resident)
    for each key k:
        load V[k] once per tile
        for each query q in the tile that attends k:
            tile_acc[q] = fma(weight[q,k], V[k], tile_acc[q])
    memcpy each query's tile row to out
```

Key order per query is unchanged, the AVX2 FMA width and tail handling are unchanged, and the tile copy is a byte move, so the output is bit-identical. The trade is deliberate: each value row is re-read once per tile (16 tiles at 128 queries, 512 B each, ~8 KiB of row traffic per key) instead of once per key, in exchange for keeping the accumulators in L1. At head_dim 64 the tile is 2 KiB; at 128 it is 4 KiB.

Scope: `attention_block_head_avx2` only, i.e. f32 and f16 prefill KV. The i8 prefill path, sparse attention, the scalar mode and every decode kernel are untouched. The stack buffer is sized for head_dim <= 256 with a heap fallback above it, matching `widen_row`'s existing bound.

## 2. Falsifier

- **KEEP** if the attention phase is consistently faster with identical ids and no fidelity movement.
- **KILL** if the tile is neutral or slower in the clean pairs, or if any id moves.

Measured: attention faster in 10/10 pairs; ids identical 10/10; fidelity unchanged.

## 3. Measured numbers

Model: `Llama-3.2-3B-Instruct-Q4_K_M.gguf` (1.87 GiB), `--threads 6 --gpu off`, n=2. Pairs alternate arm order to cancel warm-up drift; the prompt is the same sentence repeated, so token count is exact.

### 2,049-token prompt, context 4096

| KV | pair | arm order | attention baseline → tile (ms) | ratio | prefill baseline → tile (t/s) | ratio | matvec ratio | ids |
|---:|---:|---|---:|---:|---:|---:|---:|---|
| f16 | 1 | baseline → tile | 13,918.5 → 8,234.8 | 1.690x | 21.5 → 27.5 | 1.304x | 1.260 | identical |
| f16 | 2 | tile → baseline | 21,030.5 → 10,590.4 | 1.986x | 18.3 → 22.6 | 1.240x | 1.147 | identical |
| f16 | 3 | baseline → tile | 16,985.3 → 12,878.6 | 1.319x | 23.2 → 18.2 | 0.785x | 0.714 | identical |
| f32 | 1 | baseline → tile | 12,506.2 → 9,421.8 | 1.327x | 24.0 → 26.8 | 1.118x | 1.089 | identical |
| f32 | 2 | tile → baseline | 16,453.9 → 10,439.6 | 1.576x | 24.0 → 25.2 | 1.050x | 0.971 | identical |
| f32 | 3 | baseline → tile | 16,956.3 → 10,305.7 | 1.645x | 21.8 → 25.2 | 1.156x | 1.087 | identical |

Median attention ratio: **1.690x (f16)**, **1.576x (f32)**. Median total prefill: 1.240x (f16), 1.118x (f32). Pair 3's tile arm ran in a 40% slower matvec window, which is why its total ratio is below 1 despite a 1.32x attention win; the phase split shows the tile was still faster where it acts.

### 4,081-token prompt, context 8192

| KV | pair | arm order | attention baseline → tile (ms) | ratio | prefill baseline → tile (t/s) | ratio | matvec ratio | ids |
|---:|---:|---|---:|---:|---:|---:|---:|---|
| f16 | 1 | baseline → tile | 62,435.6 → 25,920.0 | 2.409x | 20.7 → 31.9 | 1.546x | 1.342 | identical |
| f16 | 2 | tile → baseline | 46,649.7 → 26,292.4 | 1.774x | 30.2 → 34.1 | 1.128x | 0.946 | identical |
| f32 | 1 | baseline → tile | 38,843.0 → 23,130.2 | 1.679x | 33.6 → 39.7 | 1.180x | 1.036 | identical |
| f32 | 2 | tile → baseline | 51,557.0 → 22,099.1 | 2.333x | 28.0 → 40.4 | 1.443x | 1.201 | identical |

Median attention ratio: **2.09x (f16)**, **2.006x (f32)**. Median total prefill: 1.31x across the four 4k pairs. The tile's own attention time grows from 8.2-12.9 s at 2k to 22.1-26.3 s at 4k (2.0-3.2x for twice the tokens), while the un-tiled arm grows from 12.5-21.0 s to 38.8-62.4 s (2.9-4.4x).

### Numerics

The block-attention differential test (`attention_block_matches_head_path`, 4 queries x 2 heads x 32 dims, partial tile) passes; it is part of the 146-case suite. Fidelity at 399 positions is exactly the pre-change values: f32 mean KL 1.020e-11, worst 2.330e-10, top-1 399/399; f16 mean 3.252e-07, worst 2.709e-05, top-1 399/399. Generated ids are identical in all 10 model pairs.

## 4. Interpretation

- The win is cache residency, not arithmetic: pass 2 does the same FMAs in the same order. The old layout made the chunk's live accumulator set 64 KiB and paid an L1 miss on every key; the tile makes it 4 KiB and pays extra value-row reads instead.
- The gain is larger at 4k because there are four times as many key visits for the same tile size, so the removed accumulator traffic dominates the added row traffic by more.
- The tile is a pure kernel change with no new flag: it is bit-identical, and the differential test and fidelity both confirm it. There is nothing to price or document as a trade-off.
- The matvec phase is untouched and moved by up to +34%/-29% across pairs, so the total-prefill ratios understate or overstate the effect depending on the window. The attention-phase ratio is the stable signal (10/10 pairs).

## 5. Limits and disposition

- **f32/f16 prefill only.** The i8 block path is not used (i8 stays per-token), and decode does not run this kernel.
- **Tile size 8.** It is 2 KiB at head_dim 64 and 4 KiB at 128. A 16-query tile would still fit L1 at both widths (4-8 KiB) but halves the value-row re-streams; it is untested and would need the same pair protocol. Recorded as a follow-up lever, not a claim.
- **Heap fallback above head_dim 256** is unexercised by the Qwen2/Llama heads in this project (64/128); it mirrors `widen_row`'s existing fallback.
- **Shipped as the default** because it is arithmetic-neutral. No CLI flag, no ADR needed: the contract (ids, fidelity) is unchanged.

## 6. Exact commands

```text
cmake --preset release; cmake --build --preset release    → 54/54
.\build\release\bin\tge_tests.exe                         → 146/146
.\build\release\bin\tge.exe fidelity --model Llama-3.2-3B-Instruct-Q4_K_M.gguf \
  --corpus docs\architecture.md --context 512 --tokens 400 --threads 6 \
  --gpu off --kv f32|f16                                  → 1.020e-11 / 3.252e-07, 399/399
.\build\release\bin\tge.exe generate --model Llama-3.2-3B-Instruct-Q4_K_M.gguf \
  --prompt "<sentence x 128|255>" --n 2 --threads 6 --gpu off \
  --kv f32|f16 --context 4096|8192 --print-ids            → pairs in §3
python aikit.py check aegean-tge                          → 4/4
```

Baseline and experiment binaries were built from the same tree separated only by the `attention_avx2.cpp` patch; the pair runner alternated their order. No push, no merge, no force-push.
