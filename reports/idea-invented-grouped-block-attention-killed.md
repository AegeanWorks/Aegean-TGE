# INVENTED: grouped-head block-prefill attention (share K/V rows across GQA heads) — KILLED (attention 1.43x–1.60x slower, ids identical)

- **Idea**: The query-block prefill kernel is dispatched per query head (`parallel_for(heads)`), so every K/V row is loaded once per query head. GQA heads in the same group share the same K/V rows, so fusing a group inside one kernel task would load each row once per KV head instead of once per query head.
- **Cycle date**: 2026-11-08
- **Verdict**: **KILLED.** Attention is slower in every pair: **1.427x / 1.687x / 1.162x** slower at 2049 tokens (median **1.43x**) and **1.597x** slower at 4081 (grouped 34.3 s vs baseline 21.5 s). Ids are identical and fidelity is unchanged, so the kernel is numerically exact — the loss is scheduling and working-set cost, not correctness.
- **Tree state**: spike reverted from `include/tge/cpu/kernels.hpp`, `src/cpu/avx2/attention_avx2.cpp`, `src/models/qwen2.cpp`, `tests/test_attention.cpp`; branch `idea/grouped-block-attention` deleted; `git status` clean; nothing committed except this report.

## 1. Mechanism

`attention_block_head_avx2` runs one query head over a 128-query chunk. Pass 1 loads each key row and dots it with the chunk queries; pass 2 streams each value row into the 8-query tile accumulators. For Llama-3.2-3B (24 query heads / 8 KV heads) three query heads share one KV head, so each K/V row is loaded **three times per chunk** — once per head task.

The spike refactors the kernel into `attention_block_group_avx2(args, head_base, group)`:

- One K/V row load and one `widen_row` per key, reused for `group` query heads.
- `args.scores` points at `head_base`'s slice; member `g` uses `args.scores + g * context`.
- Query/output offsets for member `g` are `(head_base + g) * head_dim`.
- Pass 2 tiles `group` accumulator sets (one per member) so the value row is loaded once per tile and reused across members.
- `attention_block_head_avx2` becomes `attention_block_group_avx2(args, head, 1)`, so the shipped path is exactly the old code.
- `attention_batch` dispatches `parallel_for(kv_heads)` instead of `parallel_for(heads)`.

Predicted gain: at long context, K/V row loads are the only traffic the group removes. With 3 heads sharing, row bytes drop 3x. If rows were 30% of attention time, the attention cap is ~20% and the total-prefill cap ~4-6%. The task count drops 24 -> 8 on a 6-worker prefill pool, so the critical path rises from 4 head-units to 2 group-tasks unless a group costs less than 2 head-units.

## 2. Falsifier

- **KEEP** if 4081-token f16 attention is >=10% faster in drift-controlled pairs with identical ids and unchanged fidelity.
- **KILL** if attention is neutral or slower at either 2049 or 4081.

Measured: slower in all four paired arms; kill.

## 3. Measured numbers

Model: `Llama-3.2-3B-Instruct-Q4_K_M.gguf`, `--threads 6 --gpu off`, n=2, same-source baseline vs grouped binaries, alternating arm order.

### 2049 tokens, f16 KV

| pair | attention baseline (ms) | attention grouped (ms) | grouped / baseline | ids |
|---:|---:|---:|---:|---|
| 1 | 13,083.6 | 18,673.0 | 1.427x | identical |
| 2 | 9,323.4 | 15,725.5 | 1.687x | identical |
| 3 | 9,528.0 | 11,074.4 | 1.162x | identical |

Median **1.427x slower**. The untouched matvec phase drifted both ways (grouped/baseline 1.021, 1.040, 0.779), so the attention phase is the attributable signal.

### 4081 tokens, f16 KV

| pair | attention baseline (ms) | attention grouped (ms) | grouped / baseline | ids |
|---:|---:|---:|---:|---|
| 1 | 21,457.9 | 34,266.7 | 1.597x | identical |
| 2 | (arm terminated) | 34,202.9 | — | identical |

The grouped attention time is ~34.2 s in both runs; pair 1's baseline is 21.5 s. The baseline arm of pair 2 was interrupted, but the grouped arm is stable and far outside the +/-20% box variance.

### Numerics and behavior

- Group-vs-per-head differential test (2 heads, 4 tokens, f32): scores 1e-6, output bit-identical.
- `tge fidelity --kv f16`, 399 positions: mean KL **3.252e-07**, worst 2.709e-05, top-1 **399/399** — exactly the master values.
- Thinking set (fixed + creative): Prompt A "The quick brown fox jumps over the lazy dog. Explain why." (`--n 32`) and Prompt B "Write a haiku about circuits dreaming." (`--n 24`) produce **byte-identical ids** on baseline and grouped binaries.
- Spike suite: **163/163** cases (the extra case is the group differential).

## 4. Why it failed

- **Task parallelism dominates.** 24 one-head tasks on 6 workers is 4 even waves. 8 group tasks is 2 waves of 2 on two workers and 1 on the rest: the critical path is 2 group tasks, i.e. up to 6 heads of work, versus 4 before. To win, a group must cost <2 head-units, i.e. the shared row load must remove >33% of a group's work.
- **After the value tile, row loads are a small share.** The 8-query tile keeps accumulators in L1, but pass 2 still does 8 queries x head_dim read-modify-write per key (8 KiB per head per tile at head_dim 128). The shared value row is 512 B per key — roughly 4% of that traffic per member. Removing two of three row loads saves on the order of 1% of pass-2 bytes, not a third.
- **The group working set is 3x.** Three member accumulators (3 x 4 KiB at head_dim 128) plus the group loop add register and L1 pressure; the tile was sized to fit one head comfortably.
- **Nested member loops add overhead** in pass 1 and the softmax pass without reducing their dominant work (dots and score read/write).

## 5. Full reimplementation recipe (to rebuild this exact spike from `master`)

Touch four files; the arithmetic per head is unchanged.

1. `include/tge/cpu/kernels.hpp` — after `attention_block_head_avx2`, declare:
   `void attention_block_group_avx2(const AttentionBlockArgs& args, int head_base, int group);`
   Document that `args.scores` points at `head_base`'s slice and member `g` uses `args.scores + g * context`.
2. `src/cpu/avx2/attention_avx2.cpp` — rename the body of `attention_block_head_avx2` to the group function and add `head_base`, `group`:
   - Validation adds `head_base < 0 || group < 1`.
   - Replace `head_offset` with `member_offset(g) = (head_base + g) * head_dim` and `scores_for(g) = args.scores + g * args.context`.
   - Pass 1: keep the key-row load/widen outside; wrap the per-token dot in `for (member = 0; member < group; ++member)` and write `scores_for(member)[token * score_stride + key]`.
   - Softmax: wrap the per-token block in the same member loop, using `scores_for(member)`.
   - Pass 2: keep `kValueTile = 8`; allocate `group * kValueTile * head_dim` floats with a 8192-float (32 KiB) aligned stack buffer and a heap fallback; `member_stride = tile_count * head_dim`; zero each member tile; per key load/widen the value row once, then for each member apply the existing weight/token FMA loop into `tile_acc + member * member_stride`; copy each member's rows to `out + token * query_stride + member_offset(member)`.
   - Add `void attention_block_head_avx2(...) { attention_block_group_avx2(args, head, 1); }`.
3. `src/models/qwen2.cpp` — in `attention_batch` replace
   `pool.parallel_for(heads, 1, [&](int head) { ... head/group ... attention_block_head_avx2(per_head, head); });`
   with
   `const int kv_heads = heads / group; pool.parallel_for(kv_heads, 1, [&](int kv_head) { per_head.kv_head_offset = kv_head * head_dim; per_head.scores = block.scores + (kv_head * group) * context_; attention_block_group_avx2(per_head, kv_head * group, group); });`
4. `tests/test_attention.cpp` — add `attention_block_group_matches_per_head`: same fixture as `attention_block_matches_head_path` (4 tokens, start 4, 2 heads, dim 32), run per-head with `kv_head_offset = 0` and grouped with `group = 2`, assert scores within 1e-6 and output bit-identical.

Rebuild and verify: `cmake --build --preset release`, `tge_tests` (163/163 with the added case), then the falsifier pairs at 2049 and 4081 f16.

## 6. Reopen triggers

- Only with a scheduler that keeps at least `workers` tasks: split each group's token range into pieces, or group only when `kv_heads >= 2 * workers`; measure the split variant, not this one.
- Only if a profile shows K/V row loads are >=20% of post-tile attention bytes. On today's tree the tile makes accumulator traffic dominant and the row-saving cap is a few percent.
- The numerics are free, so a future layout that changes the value pass (e.g., a transposed V tile) may make row reuse matter again; re-price it then.
