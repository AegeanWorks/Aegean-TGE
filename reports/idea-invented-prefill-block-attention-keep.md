# INVENTED: Query-block prefill attention — KEEP (attention 1.63–1.88x, 8k prefill 1.20–1.44x, ids identical)

- **Idea**: Prefill attention is per token × per head: every token re-reads its whole causal K/V span, so the last 64-token chunk alone reads ~64x more K/V bytes than necessary. Add a query-block kernel that loads each K/V row once and reuses it across the chunk's 64 queries, with per-head parallelism and the same causal softmax.
- **Cycle date**: 2026-11-08
- **Verdict**: **KEEP as an opt-in prefill kernel (f32 KV).** At 8180 tokens, attention is **1.88x / 1.63x** faster and total prefill **1.44x / 1.20x** faster in the two drift-controlled pairs, with generated ids **identical** on 2048 and 8180 prompts and 3/3 short prompts × 64 tokens. At 2048 the block path is neutral (attention is ~20% and the KV working set is cache-resident).
- **Tree state**: spike reverted from `include/tge/cpu/kernels.hpp`, `src/cpu/avx2/attention_avx2.cpp`, `src/models/qwen2.cpp`, `tests/test_attention.cpp` and `apps/cli/main.cpp` (temporary `@file` helper); 145/145 baseline tests rebuilt and green; `aikit.py check` 4/4; no branch, no spike commit; nothing pushed.

## 1. Mechanism

`attention_batch` split the chunk over tokens and called `attention_head` per (token, head). For a token at position p, the head scans keys 0..p and reads the K and V rows each time. With a 64-token chunk, each row is read up to 64 times per head.

The spike adds `attention_block_head_f32_avx2(args, head)`:

1. **QK pass**: for each key position k, load the key row once, then dot it with every query in the block whose position ≥ k (`first_query = max(0, k - start)`). Scores land in the existing per-(token, head) slices.
2. **Softmax**: per query, over its causal prefix, using the same max/`exp_avx2`/inverse form as `attention_head`.
3. **Value pass**: for each key position k, load the value row once and FMA it into every query's accumulator with that query's weight.

`attention_batch` dispatches the block kernel under `TGE_ATTN_BLOCK` (default on in the spike) when the KV tier is f32, the attention mode is not Scalar, and sparse attention is off; heads are spread over the prefill pool.

A unit test (`attention_block_matches_head_path`) compares the block kernel against `attention_head_avx2` on 4 queries × 2 heads × 32 dims and passes at 1e-4.

## 2. Falsifier

- **KEEP** if 8180-token prefill is ≥1.15x faster in a drift-controlled pair with identical ids; the unit test must match the per-token path.
- **KILL** if it is neutral or ids move.

Measured: 1.44x and 1.20x; ids identical.

## 3. Measured numbers

### 8180-token prefill, f32 KV, 6 threads

Pair 1 — per-token then block:

| phase | per-token | block | ratio |
|---|---:|---:|---:|
| prefill total | 682,663 ms (12.0 t/s) | 474,956 ms (17.2 t/s) | **1.437x** |
| matvec | 194,132 ms | 207,202 ms | 0.937x |
| attention | 471,945 ms | 251,221 ms | **1.879x** |
| norm | 14,293 ms | 14,967 ms | 0.955x |

Pair 2 — block then per-token:

| phase | block | per-token | ratio |
|---|---:|---:|---:|
| prefill total | 501,742 ms (16.3 t/s) | 600,890 ms (13.6 t/s) | **1.197x** |
| matvec | 235,496 ms | 178,568 ms | 0.758x |
| attention | 249,160 ms | 406,894 ms | **1.633x** |
| norm | 15,202 ms | 13,918 ms | 0.915x |

The matvec phase (untouched by the kernel) moved 6% slower in pair 1 and 32% slower in pair 2 for the block arm, so the total-prefill ratios understate the effect. Attention alone is 1.63–1.88x, consistent across opposite arm orders.

### 2048-token prefill

| pair | per-token | block | ratio |
|---:|---:|---:|---:|
| 1 | 28.5 t/s | 18.3 t/s | 0.642 |
| 2 | 38.5 t/s | 39.0 t/s | 1.013 |

Neutral overall (pair 1's first arm ran in a slower window; pair 2 is a tie). At 2048 the KV working set fits in cache, so the block kernel's traffic saving does not pay.

### Ids

- 2048-token prompt, `--n 8`: **identical** in both pairs.
- 8180-token prompt, `--n 8`: **identical**.
- Three short prompts, `--n 64`: 64/64, 64/64, 64/64 tokens identical.

## 4. Why it wins

- The per-token schedule reads each K/V row once per query per head. At 8k that is roughly 65 MB of K/V per token per layer, and the total prefill traffic is in the tens of TB — far beyond L3, so it runs at memory speed.
- The block schedule reads each row once per 64-query block, cutting K/V traffic ~64x and turning attention into a compute-bound kernel. The compute itself is unchanged.
- Below ~2k the working set is cache-resident and the extra loop structure is neutral, so the gain appears only at long context — exactly the target.

## 5. Limits and disposition

- **f32 KV only.** `kv_half_`/`kv_q8_` and sparse attention still use the per-token path; f16/i8 prefill would need their own widen paths in the block kernel.
- **Sequential head parallelization only.** Each head's block runs as one task; a head × key-range split with online-softmax merge could use more parallelism but is not needed at 24 heads / 6 workers.
- Recommend enabling the block kernel for f32 prefill above ~2048 positions, leaving the per-token path for short prompts. Ship behind a flag first (the spike's `TGE_ATTN_BLOCK`), then consider default-on after a broader ids corpus.
- Report-only; integration is the human-gated decision.

## 6. Exact commands

```text
cmake --build --preset release                         → 54/54
.\build\release\bin\tge_tests.exe                      → 146/146 with the block unit test; 145/145 after revert
TGE_ATTN_BLOCK=0 tge generate --model Llama-3B --tokens @ids8180.txt --n 8 \
  --threads 6 --gpu off --kv f32 --context 8192
tge generate ... (same, block on)
tge generate --prompt "<three prompts>" --n 64 --threads 6 --gpu off --print-ids
git checkout -- include/tge/cpu/kernels.hpp src/cpu/avx2/attention_avx2.cpp \
  src/models/qwen2.cpp tests/test_attention.cpp apps/cli/main.cpp
python aikit.py check aegean-tge                        → 4/4
```

No push, no merge, no force-push. Spike code is not in the tree or in any commit.
