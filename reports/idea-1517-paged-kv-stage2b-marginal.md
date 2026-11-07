# IDEA-1517: Paged KV stage 2b (row-pointer hoist) — MARGINAL (recovers to within 2%, but no speed win)

- **Idea ID**: IDEA-1517. Stage 2b replaces the per-row block-table resolution that cost 6–15% in stage 2 with a persistent row-pointer table built once at load. The attention inner loop then does one `key_rows[token]` / `value_rows[token]` load instead of table arithmetic per token per head.
- **Cycle date**: 2026-11-07
- **Verdict**: **MARGINAL — close as a speed idea.** The hoist recovers the stage-2 loss: 8×32 continuous 0.937x → **0.981x** median (12 pairs), 2048 decode 0.887x → **0.980x** median (3 pairs), both within the 2% noise budget. But there is no positive margin (continuous is 1.9% below baseline), and the paged path adds +8 ms load and ~14.7 MB of row pointers per sequence. Its value is memory management (more slots/eviction/prefix sharing), not throughput; do not ship it as a speed optimization.
- **Tree state at close**: spike reverted from `include/tge/models/qwen2.hpp`, `include/tge/cpu/kernels.hpp`, `src/models/qwen2.cpp`, `src/cpu/scalar/attention_scalar.cpp`, `src/cpu/avx2/attention_avx2.cpp`; 145/145 baseline tests rebuilt and green; no branch, no spike commit; nothing pushed.

## 1. Mechanism

Stage 2 (killed: `idea-1517-paged-kv-stage2-killed.md`) wired 16-token block storage and resolved each row in the attention loop:

```text
physical = block_table[position / 16]
row      = (physical * 16 + position % 16) * kv_stride
```

That is ~1.8M resolutions per 2048 decode token (28 layers × 2048 positions × 32 heads) and cost 6–15%.

Stage 2b builds **persistent row pointers** at load:

```text
kv_key_rows_  [sequence][layer][kv_head][position] -> &k_cache_[...]
kv_value_rows_[sequence][layer][kv_head][position] -> &v_cache_[...]
```

one entry per (sequence, layer, KV head, position), 8 bytes each. At 1 sequence / Llama-3B / 4096 context: 28 × 8 × 4096 = 917,504 pointers per table, 7.3 MB per table, 14.7 MB for K+V. At 4 sequences: 58.8 MB.

`AttentionArgs` and `AttentionGroupArgs` gained `key_rows` / `value_rows`. The scalar and AVX2 f32 per-head and group kernels read:

```text
key   = key_rows  ? key_rows[token]  : keys  + token * stride
value = value_rows? value_rows[token]: values+ token * stride
```

The block table is still used for **writes** (`kv_slot`) and to build the pointer table; attention never consults it. f16/i8 stay on the contiguous path. Prefix/sparse paths are bypassed when paged, as in stage 2.

## 2. Predicted gain / falsifier

Stage 2 measured 0.937x continuous and 0.887x 2048 decode. The hoist removes the per-head table arithmetic and reduces the resolution count from 1.8M to 7.3M **once at load** (for one sequence; 29M for four). Expected recovery to within 2%.

Falsifier:

- **PROCEED** only if 8×32 continuous is at least baseline and 2048 decode is within 2%;
- f32/f16/i8 fidelity and ids unchanged.
- **KILL/close** if it still misses the 2% budget or shows no positive margin.

Measured: within the 2% budget, but no positive margin.

## 3. Measured numbers

Llama-3.2-3B-Q4_K_M, 6 threads, `--gpu off`, paged f32 (`TGE_PAGED_KV=contig`), baseline is the pre-cycle contiguous binary. All transcripts identical.

### 3.1 8×32 continuous, 4 slots (12 pairs)

First 8 pairs baseline→paged:

```text
1.002, 0.897, 1.208, 1.001, 0.937, 0.884, 1.059, 0.875   median 0.969
```

Four alternating-order pairs (BS, SB, BS, SB):

```text
0.961, 1.134, 0.840, 1.027   median 0.994
```

All 12 ratios sorted: `0.840, 0.875, 0.884, 0.897, 0.937, 0.961, 1.001, 1.002, 1.027, 1.059, 1.134, 1.208`; **median 0.981** (1.9% below baseline, inside the box's ±20% band). All first transcripts identical and `state: independent`.

### 3.2 2048-token decode (3 pairs)

| order | baseline | paged 2b | paged/base |
|---|---:|---:|---:|
| BS | 5.03 t/s | 3.98 t/s | 0.791 |
| BS | 5.12 t/s | 5.02 t/s | 0.980 |
| SB | 5.68 t/s | 5.86 t/s | 1.032 |

Median **0.980** (2% below baseline). The order bias is visible: the paged run is always second after a ~45 s prefill in the BS pairs; the single SB pair is faster. The three-pair median is the honest summary.

### 3.3 Load and memory

One sequence, context 4096, lazy vocabulary, 5 paired launches:

| | median load |
|---|---:|
| baseline | 185.9 ms |
| paged 2b | 193.9 ms |

+8 ms (4.3%) for building 14.7 MB of row pointers. At 4 sequences the tables are 58.8 MB and the build cost scales; continuous loads measured 650–820 ms for both binaries under current box load, so the exact 4-sequence delta is not isolated.

### 3.4 Correctness and fidelity

| check | result |
|---|---|
| 5-token ids | IDENTICAL |
| 2048-token prompt ids | IDENTICAL |
| 8×32 continuous transcripts | IDENTICAL |
| f32 paged fidelity | 1.130e-11 mean KL, 63/63 |
| f16 contiguous fidelity | 2.501e-07, 63/63 |
| i8 contiguous fidelity | 2.373e-04, 61/63 |
| tests | 145/145 spike and after revert |
| project gate | 4/4 after revert (19.4 s) |

The fidelity tiers are exactly the pre-paging values; the pointer table changes addresses only.

## 4. Interpretation

The hoist does what it was designed to do: stage 2's 6–15% regression becomes ~2% (continuous 0.981x, 2048 0.980x). That is within the box's noise budget, so the paged addressing is not a performance dead end. But it is also not a win: the best estimate is parity, with a small load and memory tax. Paging's remaining value is **memory management**, not speed:

- more sequences than fixed slots allow;
- eviction/refill without recomputing the prompt;
- block-granular prefix sharing;
- bounded KV memory under a budget.

Those are feature metrics, not the throughput/fidelity gates this loop uses. Closing IDEA-1517 as a speed idea is the honest disposition; reopening it should be a memory-manager project with a different falsifier (slots admitted under a fixed budget, eviction correctness, prefix hit rate), not another decode-t/s cycle.

## 5. Reimplementation recipe (exact, for the feature version)

### 5.1 Storage and tables

As stage 2 §1: 16-token block arena, `kv_paged_`, `kv_block_tokens_ = 16`, `kv_blocks_per_sequence_`, `kv_block_count_`, `kv_block_tables_`, `kv_slot`, `kv_layer_base`; enable only for f32 (`TGE_PAGED_KV=1` scattered or `contig`).

### 5.2 Persistent row pointers

Add `std::vector<const float*> kv_key_rows_, kv_value_rows_` to `Qwen2`. At load, after the block tables are filled:

```text
for sequence, layer, kv_head, position:
  physical = table[sequence][position / 16]
  row = (layer * block_count + physical) * 16 * kv_dim
        + (position % 16) * kv_dim
        + kv_head * head_dim
  index = ((sequence * block_count + layer) * kv_heads + kv_head) * context + position
  kv_key_rows_[index]   = k_cache_.data() + row
  kv_value_rows_[index] = v_cache_.data() + row
```

Sizes: `sequence_count * block_count * kv_heads * context` pointers each.

### 5.3 Kernel args and loops

- `AttentionArgs`: add `const float* const* key_rows`, `value_rows`.
- `AttentionGroupArgs`: same.
- `attention_head`: set `key_rows = kv_key_rows_.data() + (((sequence * block_count + layer) * kv_heads + kv_head) * context)`; same for values.
- `attention_gqa`: same base `+ start` for the chunk.
- Scalar/AVX2 f32 paths: `const float* key_row = key_rows ? key_rows[token] : keys + token*stride;` and the value analogue. Leave half/q8 on the old path.
- Keep `args.keys`/`values` pointing at a valid base so existing validation and non-paged paths are unchanged.

### 5.4 Guards and gate

- Disable sparse, `shared_prefix_len`, `prime_cache_from_table` and the multi-prefix populate/hit when paged.
- Gate: build + 145/145; 5/2048/continuous id parity; f32 paged + f16/i8 fidelity; 8×32 and 2048 A/B; load delta; `aikit.py check aegean-tge` 4/4.
- Feature version adds slot admission/eviction tests and a fixed-budget memory falsifier.

## 6. Verdict and disposition

**MARGINAL — close as a speed idea.** The row-pointer hoist recovers the stage-2 regression to within the 2% budget (continuous 0.981x, 2048 0.980x) with exact ids and fidelity, but it does not beat the contiguous cache and adds +8 ms load plus 14.7 MB/sequence of pointers. Do not ship it for throughput. Keep the recipe for a future paged-KV **memory manager**, where the falsifier is slots admitted under a fixed KV budget and eviction/prefix correctness, not decode t/s.

## 7. Exact commands run (selection)

```text
cmake --build --preset release                         → 54/54
.\build\release\bin\tge_tests.exe                      → 145/145
TGE_PAGED_KV=contig tge bench continuous ... --requests 8 --tokens 32 --sequences 4
                                                       → 12-pair ratios in §3.1
TGE_PAGED_KV=contig tge generate --model Llama-3B --tokens <2048 ids> --n 8 ...
                                                       → 3-pair ratios in §3.2
TGE_PAGED_KV=1 tge fidelity ... --kv f32|f16|i8        → tiers in §3.4
git checkout -- include/tge/models/qwen2.hpp include/tge/cpu/kernels.hpp \
  src/models/qwen2.cpp src/cpu/scalar/attention_scalar.cpp src/cpu/avx2/attention_avx2.cpp
python aikit.py check aegean-tge                        → 4/4
```

No push, no merge, no force-push. Spike code is not in the tree or in any commit.
