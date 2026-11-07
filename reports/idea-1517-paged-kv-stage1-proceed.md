# IDEA-1517: Paged KV / prefix-hash — stage-1 probe **PROCEED** (fragmentation and indirection are not the blocker)

- **Idea ID**: IDEA-1517 (paged KV with block tables + prefix-hash sharing + chunked prefill; dossier `scout-filing-2026-aegean-tge.md` §IDEA-1517). This cycle prices the two costs that decide whether the block layout is viable before any engine wiring: fragmented 16-token block reads versus one contiguous KV span, and block-table/prefix-hash translation.
- **Cycle date**: 2026-11-07
- **Verdict**: **PROCEED to stage 2 (wiring).** Paged scattered reads ran at 1.00–1.12x the contiguous bandwidth (never slower) across four runs; block translation is 0.1–0.2 ns/position and the 2k-token prefix hash is ~8 µs, together ~0.2–0.4% of a 2k decode token. The remaining risk is engine wiring complexity, not memory-pattern cost.
- **Tree state at close**: probe reverted from `tests/test_block_table.cpp`; 145/145 baseline tests rebuilt and green; no branch, no spike commit; nothing pushed.

## 1. Mechanism / probe design

The foundation already exists in-tree:

- `include/tge/models/block_table.hpp`: `BlockTable` with 16-token blocks, allocate/share/release/refcount, `slot_for(position, block_tokens)`, FNV-1a `hash_tokens`, and `remember_prefix`/`find_prefix`.
- `include/tge/models/kv_block_layout.hpp`: K column-major/V row-major reorder, block gather.

These are unit-tested but not wired into `Qwen2`; the engine still has a contiguous KV cache. The stage-1 probe asks: if the cache were stored as 16-token blocks, would the access pattern and indirection costs kill the idea before the wiring work?

Temporary `paged_kv_stage1_probe` test:

- Llama-3B-like geometry: `kv_dim = 1024` (8 KV heads × 128), context 2048, 16-token blocks → 128 logical blocks per layer;
- 4 layers × 7 repeats = 448 MB read per arm (working set 64 MB contiguous + 128 MB block arena, both above L3);
- contiguous arm: one K/V span per layer, rows interleaved K then V, sequential read;
- paged-seq arm: 128 blocks mapped sequentially (`table[i] = i`);
- paged-scatter arm: 128 blocks mapped by `(i * 73) & 255` into a 256-block arena (a deterministic permutation, 16 MB of address-space gaps per layer);
- each arm read 3 times, best-of-3; single-threaded sum to keep the comparison about memory/access, not threading;
- layout correctness: gather logical block 0 through the table and compare to the contiguous K rows;
- translation arm: `BlockTable::slot_for` + table lookup + row offset for every (layer, position, head), 10 runs;
- prefix hash arm: `BlockTable::hash_tokens` over 2048 tokens, 1000 runs.

## 2. Predicted gain / falsifier

For a 2048-token decode, the KV read is ~448 MB (28 layers × 2 × 2048 × 1024 × 4 bytes). At 7.5–8.3 GB/s single-thread, the probe's arm is ~56–63 ms, deliberately larger than one engine token because it isolates the memory pattern.

Falsifier (stage-1):

- **PROCEED** only if paged-scattered bandwidth is at least 0.9x contiguous, and translation + prefix hash are under 1% of a 2k decode token (~25 ms).
- **KILL** if scattered reads fall below 0.8x contiguous or the indirection exceeds 1%.

Measured: all pass.

## 3. Measured numbers (four independent runs)

| run | contiguous | paged-seq | paged-scatter | paged/contiguous | translate | hash 2k |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 58.2 ms / 8.07 GB/s | 56.8 ms / 8.28 GB/s | 58.3 ms / 8.05 GB/s | 0.998x | 0.1 ns | 7.73 µs |
| 2 | 61.4 ms / 7.65 GB/s | 56.5 ms / 8.31 GB/s | 57.9 ms / 8.11 GB/s | 1.060x | 0.2 ns | 8.00 µs |
| 3 | 62.9 ms / 7.47 GB/s | 56.0 ms / 8.39 GB/s | 56.2 ms / 8.36 GB/s | 1.120x | 0.2 ns | 8.02 µs |
| 4 | 62.6 ms / 7.50 GB/s | 58.8 ms / 7.99 GB/s | 59.4 ms / 7.90 GB/s | 1.054x | 0.2 ns | 7.99 µs |

Reading: the paged scattered arm is never slower than contiguous in these runs (ratio is paged/contiguous; values above 1 mean paged is faster, within the box's ±20% band). 16 MB of address-space gaps per layer did not defeat the hardware prefetcher at 128 KB per block.

Indirection budget for a 2048-token decode:

```text
translation: 28 layers × 2048 positions × 8 heads = 458,752 lookups
            × 0.2 ns ≈ 92 µs
prefix hash: 2048 tokens ≈ 8 µs
total      ≈ 100 µs vs ~25 ms token ≈ 0.4%
```

### Layout correctness

The probe recreates the block layout, gathers logical block 0 through the scattered table, and compares every K value against the contiguous layout. It passes. The existing block-table/KV-layout tests also pass; the temporary probe brought the suite to 146/146, then it was reverted to 145/145.

## 4. Fidelity / ids

**N/A by construction.** The probe is a test-only measurement; `tge.exe`, the model, the cache and all kernels are untouched. No numerics changed, so no fidelity run or thinking-set diff is owed. The gate is the layout equality check + the full suite.

## 5. Gates

- **Build**: `cmake --build --preset release` green (54/54).
- **Tests**: 146/146 with the probe; 145/145 after revert.
- **Project gate**: `python aikit.py check aegean-tge` → **4/4 gates passed** (test 17.5 s, hygiene, guidelines, secrets) after revert.

## 6. Stage-2 wiring recipe (next cycle, not attempted)

1. **Storage**: allocate KV as physical 16-token blocks per layer; a block holds K rows then V rows (`2 × 16 × kv_dim` floats) to match the probe. Keep f32/f16/i8 variants; i8 has the parallel scale blocks.
2. **Sequence table**: per sequence, a `std::vector<int> block_table` of physical block ids; `BlockTable` owns the free list/refcounts. Allocate a block when position crosses a 16-token boundary; prefill a new request by allocating blocks for its prompt, not a fixed `context_` span.
3. **Addressing**: replace `slot = layer_base + position * kv_dim` with `(layer, physical block, row)` in `store_kv`/attention. Pass a row-pointer table to the attention kernel or add a `kv_stride` + block-table accessor; avoid a full gather (`gather_blocks_row_major`) on the hot path — the probe shows direct block reads are fine, while a gather would double traffic.
4. **Prefix sharing**: hash the first 16/32/64 tokens with `hash_tokens`; `find_prefix` returns a shared block; `share` increments the refcount; `release` frees. This extends the shipped multi-prefix CoW table (up to 4 start-0 prefixes) to block granularity.
5. **Chunked prefill**: admit a long prompt in 16-token blocks interleaved with active decodes; the existing `forward_batch` can mix rows once block addressing is in place.
6. **Falsifiers for stage 2**:
   - 8×32 continuous aggregate at least the current baseline (best-of-3, same run);
   - 2048 decode within 2% of the contiguous path;
   - f32/f16/i8 fidelity unchanged (top-1 and KL tiers);
   - free-running ids identical on thinking/CJK/control prompts;
   - block-table refcount invariants on allocate/share/release.

The stage-1 probe removes the "will fragmentation kill it?" uncertainty. The next cycle should wire storage first, not chunked prefill; prefix sharing is the second step.

## 7. Verdict

**PROCEED.** The block-table and prefix-hash costs are negligible, and fragmented 16-token block reads match contiguous bandwidth on this box. The idea is not killed by memory behavior; the remaining work is the storage/addressing wiring, which is large but bounded. Do not ship the probe (test-only); the next cycle is the storage integration.

## 8. Exact commands run (selection)

```text
cmake --build --preset release                         → 54/54
.\build\release\bin\tge_tests.exe                      → 146/146 with probe; 145/145 after revert
[paged-kv] contiguous ... | paged-seq ... | paged-scatter ... | ratios ...
                                                       → four runs in §3
git checkout -- tests/test_block_table.cpp
python aikit.py check aegean-tge                        → 4/4
```

No push, no merge, no force-push. Probe code is not in the tree or in any commit.
