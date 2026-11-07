# IDEA-1517: Paged KV stage 2 (storage wiring) — KILLED (correct, but per-row lookup costs 6–15%; 2% bar fails)

- **Idea ID**: IDEA-1517. Stage 2 wires 16-token block storage, per-sequence block tables and direct attention row addressing into the f32 decode/prefill path. Stage 1 (`idea-1517-paged-kv-stage1-proceed.md`) had priced fragmented reads and translation as free; this cycle tests the same indirection inside the real attention kernels.
- **Cycle date**: 2026-11-07
- **Verdict**: **KILLED on the performance falsifier.** All ids and fidelity tiers are exact, and the storage/addressing works, but the per-row block-table resolution in the attention inner loops costs **6.3% on 8×32 continuous** (median ratio 0.937) and **11% on 2048 decode** (0.929, 0.846; scatter 0.791). The bar was "continuous ≥ baseline, 2048 decode within 2%". The design is correct but not shippable in this form.
- **Tree state at close**: spike reverted from `include/tge/models/qwen2.hpp`, `include/tge/cpu/kernels.hpp`, `src/models/qwen2.cpp`, `src/cpu/scalar/attention_scalar.cpp`, `src/cpu/avx2/attention_avx2.cpp`; 145/145 baseline tests rebuilt and green; no branch, no spike commit; nothing pushed.

## 1. Mechanism (implemented, f32-only)

- **Storage**: 16-token physical blocks. A block stores 16 K rows then 16 V rows (`2 × 16 × kv_dim` floats). The cache is `layers × block_count × 16 × kv_dim` (same total bytes as the contiguous layout). f16/i8 stay on the existing contiguous path.
- **Sequence table**: each sequence has a `std::vector<int>` logical→physical block table. Two mappings were tested:
  - `TGE_PAGED_KV=1`: global permutation `(index * step) % block_count` (scattered);
  - `TGE_PAGED_KV=contig`: `physical = sequence * blocks_per_sequence + logical` (locality-preserving).
- **Writes**: `kv_slot(sequence, layer, position)` resolves `physical = table[position / 16]`, `row = position % 16`, then `(layer * block_count + physical) * 16 * kv_dim + row * kv_dim`. All three store sites (prefill batch, forward, forward_batch) use it.
- **Attention**: `AttentionArgs`/`AttentionGroupArgs` gained `block_table`, `block_tokens`, `start_position`. The scalar and AVX2 f32 per-head and group kernels resolve each row as
  `(physical * block_tokens + position % block_tokens) * kv_stride`.
  Base pointers point at the layer/head origin (no row offset). `kv_layer_base` returns the paged layer arena or the old sequence/layer base.
- **Guards**: static sparse, cross-sequence prefix sharing and the multi-prefix populate/hit paths are disabled when paged (stage 3 work); the same-sequence H1 prefix skip is safe and kept.

## 2. Predicted gain / falsifier

Stage 1 measured:
- fragmented 16-token block reads at 1.00–1.12x contiguous bandwidth;
- block translation 0.1–0.2 ns/position, prefix hash ~8 µs/2k tokens;
- estimated indirection ~0.4% of a 2k decode token.

Falsifier:
- **KEEP** only if 8×32 continuous aggregate is at least the contiguous baseline and 2048 decode is within 2%;
- f32/f16/i8 fidelity and ids unchanged.
- **KILL** on a repeatable regression above 2%.

The regression is 6–15%; kill.

## 3. Measured numbers

Llama-3.2-3B-Q4_K_M, 6 threads, `--gpu off`. Baseline is the pre-cycle binary (contiguous). Paged is f32 (`TGE_PAGED_KV=contig` unless noted). All runs are interleaved baseline→paged.

### 3.1 8×32 continuous, 4 slots (3 pairs)

| pair | baseline aggregate | paged aggregate | ratio | first transcript |
|---:|---:|---:|---:|---|
| 1 | 19.37 t/s | 18.40 t/s | 0.950 | IDENTICAL |
| 2 | 18.73 t/s | 16.88 t/s | 0.901 | IDENTICAL |
| 3 | 17.85 t/s | 16.72 t/s | 0.937 | IDENTICAL |
| **median** | — | — | **0.937** | — |

Continuous batching is 6.3% slower; all transcripts identical, `state: independent`.

### 3.2 2048-token decode (id prompt, 8 generated tokens)

| pair | baseline decode | paged decode | ratio | ids |
|---:|---:|---:|---:|---|
| contig 1 | 7.28 t/s | 6.76 t/s | 0.929 | IDENTICAL |
| contig 2 | 6.30 t/s | 5.33 t/s | 0.846 | IDENTICAL |
| scatter 1 | 6.03 t/s | 4.77 t/s | 0.791 | IDENTICAL |

Median contig ratio **0.887** (11% slower). Prefill was mixed (39.1→43.5, 43.1→39.8 t/s in two runs) and is not used for the verdict; decode is the stable signal. The 2048-token prompt ids are identical before generation.

### 3.3 Shorter contexts

- 401-token prompt: promptids and generated ids identical; one pair measured prefill 52.5→40.2 t/s and decode 4.80→5.92 t/s (noisy, not used).
- 5-token prompt: generated ids identical (`12366 13 578 6864 315 15704 374 22463`).

### 3.4 Fidelity (63 positions, ctx 512)

| KV | mean KL | worst KL | top-1 |
|---|---:|---:|---:|
| f32 paged | 1.130e-11 | 5.531e-11 | 63/63 |
| f16 contiguous | 2.501e-07 | 1.239e-06 | 63/63 |
| i8 contiguous | 2.373e-04 | 1.234e-03 | 61/63 |

Exactly the previously recorded tiers. Storage/addressing does not change arithmetic; the paged path is f32-only, f16/i8 remain on the contiguous layout.

## 4. Why it missed

Stage 1 measured memory bandwidth and a translation loop; it did not execute the row resolution **once per token per head inside the attention kernel**. At 2048 decode, that is:

```text
28 layers × 2048 positions × 32 query heads ≈ 1.8M row resolutions per token
```

Each resolution adds a block-table load, a division/modulo by 16 (shifts/mask), a multiply and a branch. At ~1.8M × a few cycles, plus the lost contiguous-row addressing the compiler could strength-reduce, the cost is milliseconds per token — 6–15% — even though the memory pattern is fine. Contiguous per-sequence blocks did not recover it: the lookup itself, not fragmentation, is the cost. This is the useful negative result: the stage-1 bandwidth probe is necessary but not sufficient.

## 5. What would move it (stage 2b, not attempted)

Hoist row resolution out of the head loop:

1. For each layer and span, build one `const float* key_rows[span]` and `value_rows[span]` (or a small struct with K/V/scales) once per layer, reused by all 32 heads.
2. Pass those pointers to the attention kernels; the inner loop does one pointer load per token instead of table arithmetic.
3. For GQA, build per chunk (or per layer span) and share across the group's heads.
4. Falsifier: recover 2048 decode to within 2% of contiguous and 8×32 continuous to ≥ baseline; if not, paged KV is not viable on this CPU and the contiguous cache stays.

The pointer array is ~16 KB per layer at 2k, L1-resident; the build cost is ~2k resolutions per layer instead of ~65k (32 heads × 2k). This removes the 1.8M-per-token multiplier, which is the measured problem. A second option is a cached block base + row offset per chunk (no pointer array), but it must be per-head-shared to win.

## 6. Reimplementation recipe (exact, for the record — kill-grade as built)

### 6.1 Storage and tables (`qwen2.hpp`/`qwen2.cpp`)

- Add `bool kv_paged_`, `int kv_block_tokens_ = 16`, `int kv_blocks_per_sequence_`, `int kv_block_count_`, `std::vector<std::vector<int>> kv_block_tables_`.
- In `Qwen2::load`, enable paged only for f32 (`!kv_half && !kv_q8`) when `TGE_PAGED_KV` is set.
- `blocks_per_sequence = ceil(context / 16)`; `block_count = sequences * blocks_per_sequence`.
- `cache_elements = layers * block_count * block_tokens * kv_dim`; allocate `k_cache_`/`v_cache_` with that.
- Fill each sequence table with either the global permutation or the contiguous mapping above.
- `kv_slot`/`kv_layer_base` as in §1; replace the three store-site slot formulas.
- Guard `uses_sparse`, `shared_prefix_len`, `prime_cache_from_table`, the batch populate and the cross-sequence hit with `!kv_paged_`.

### 6.2 Kernels

- `AttentionArgs`: add `const int* block_table = nullptr; int block_tokens = 0;`.
- `AttentionGroupArgs`: add the same plus `int start_position = 0;`.
- In scalar and AVX2 f32 per-head/group loops, replace `row = token * stride` with a `row_of(token)` helper (using `position = start_position + token` for groups) and reject block tables on half/q8.
- Set `args.block_table`/`block_tokens` in `attention_head` and `attention_gqa` only when `kv_paged_`; set `start_position` for the GQA chunk.

### 6.3 Gate

- build + 145/145 tests;
- 5/401/2048-token id parity;
- f32 paged fidelity + f16/i8 contiguous fidelity;
- 3 interleaved 8×32 continuous pairs and 2–3 2048 decode pairs;
- `python aikit.py check aegean-tge` 4/4.

## 7. Verdict and disposition

**KILLED on the 2% performance bar.** The stage-2 storage, block tables and direct row addressing are correct and exact (all ids, all fidelity tiers), but the per-row lookup costs 6–15% in the two measured loads. Do not ship this form. The next cycle should implement the row-pointer hoist (stage 2b) before deciding whether paged KV is viable at all; if that still misses 2%, the contiguous cache stays and IDEA-1517 is closed as a CPU-layout dead end.

## 8. Exact commands run (selection)

```text
cmake --build --preset release                         → 54/54
.\build\release\bin\tge_tests.exe                      → 145/145
TGE_PAGED_KV=1 tge generate --model Llama-3B --prompt "The capital of France is" \
  --n 8 --threads 6 --print-ids --text --gpu off        → ids IDENTICAL
TGE_PAGED_KV=1 tge generate --model Llama-3B --tokens <2048 ids> --n 8 ...
                                                       → ids IDENTICAL; decode ratios in §3.2
TGE_PAGED_KV=contig tge bench continuous ... --requests 8 --tokens 32 --sequences 4
                                                       → ratios in §3.1
TGE_PAGED_KV=1 tge fidelity ... --kv f32|f16|i8
                                                       → tiers in §3.4
git checkout -- include/tge/models/qwen2.hpp include/tge/cpu/kernels.hpp \
  src/models/qwen2.cpp src/cpu/scalar/attention_scalar.cpp src/cpu/avx2/attention_avx2.cpp
python aikit.py check aegean-tge                        → 4/4
```

No push, no merge, no force-push. Spike code is not in the tree or in any commit.
