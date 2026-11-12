# INVENTED: block-sparse prefill (kept-set walk inside the 128-query block kernel) — KEEP-GRADE (8k prefill 1.27x dense / 1.18-1.24x per-token sparse; spec verify 1.24x; 2k ids bit-identical)

- **Idea**: `reports/idea-invented-sparse-prefill-8k-measured.md` priced `--sparse` prefill at 8k (attention 0.61-0.83x, total 1.06-1.15x) and named the structural cost: with sparse enabled, `attention_batch` skips the 128-query block kernel and falls back to per-(token, head) calls, so the kept-fraction saving is paid for with lost block reuse. Predicted fix: keep the block kernel and walk the causal sink/stride/window kept set inside it — one kept K/V row load feeds every query that keeps it, and rows no query keeps are never loaded. Predicted 0.30-0.35x attention and 1.30-1.40x total prefill at 8k. Spike it, measure, revert, record the exact recipe.
- **Cycle date**: 2026-11-12
- **Verdict**: **KEEP-GRADE (report-only; promotion human-gated).** Llama-3.2-3B-Instruct-Q4_K_M, 8,193-token prompt (`The fox runs. ` x 2048), `--context 8704 --n 16 --kv f16 --threads 10 --gpu off`:
  - **Same-session dense vs block-sparse**: prefill **205.3 s -> 162.4 s (1.265x)**; prefill attention **66.7 s -> 27.1 s (0.406x)**; matvec flat (127.4 vs 124.3 s).
  - **Versus the shipped per-token sparse path** (interleaved pairs): **205.3 -> 166.0 s (1.237x)** and **187.7 -> 159.0 s (1.180x)**; attention **60.3 -> 27.5 s (0.457x)** and **54.0 -> 26.9 s (0.498x)**. Median total **1.21x**, median attention **0.48x**.
  - **Speculation composes**: 8k `--speculative --sparse` prefill **199.8 -> 171.1 s (1.17x)**, decode **14.45 -> 17.93 t/s (1.24x)**, verify attention **392.2 -> 231.3 ms (0.59x)**, same 3 steps / 5.33 accepted/step.
  - **2k control**: block-sparse prefill 37.0 s (attn 2.84 s) vs per-token sparse 38.3-38.9 s (attn 3.31-3.33 s) and dense 36.1 s (attn 3.82 s): attention 0.74x dense / 0.86x per-token, total within noise (attention is only ~10% of 2k prefill).
  - **Ids: byte-identical to the shipped per-token sparse path.** 2k periodic, `--sparse --print-ids`: both produce `578 4062 14198 39935 35308 927 279 16053 5679 13 220 578 4062 14198 39935 35308`. No new numerics: same kept positions, same ascending order.
- **Tree state at close**: spike reverted from `include/tge/cpu/kernels.hpp`, `src/cpu/avx2/attention_avx2.cpp`, `src/models/qwen2.cpp`; `git status` clean; stash holds only the pre-existing 2026-11-04 peer entry; rebuild clean, `tge_tests` **171/171**, `python aikit.py check aegean-tge` **4/4**. No branch, no spike commit, nothing pushed. The spike diff (3 files, +281/-3) was captured to `%TEMP%\kilo\blocksparse_spike.patch` for recipe verification only; it is not in the repo.

## 1. Mechanism (as spiked)

- Shipped state: `attention_batch` (`src/models/qwen2.cpp:1580+`) takes the block kernel only when `!kv_q8_ && !sparse_.enabled && attention_mode != Scalar`. With `--sparse` it falls through to `pool.parallel_for(tokens * heads)` and `attention_head` dispatches the per-token strided kernel (`:1331-1341`). The block kernel loads each K/V row once and dots it with up to 128 queries; the per-token path loads each kept row once per token.
- The spike adds `attention_block_sparse_head_avx2(const AttentionBlockArgs&, int head, int sink, int window, int stride)` — the dense block kernel's structure with a causal kept-set walk:
  - For query token `t`, `span = start + t + 1`. If `span <= sink + window`, the token keeps the dense prefix `[0, span)` — exactly the model's `uses_sparse` boundary, so the hand-off between dense and sparse tokens inside one chunk is seamless.
  - Deeper tokens keep `[0, sink)`, then `sink + i*stride` for the middle, then the `window` tail, identical to `attention_head_static_sparse_*` / `build_sparse_index`.
  - `kept_ordinal(span, key)` maps a resident key to its compact ordinal (or -1). Pass 1 writes compact scores; one key row is loaded and widened once and dotted only with the queries that keep it.
  - `key_touched(key)` prefilters rows no query keeps: `key < last_span && (key < sink || key >= start + 1 - window || ((key - sink) % stride == 0 && key + window < last_span))`. Middle rows that no query keeps in this chunk are never loaded.
  - Per-token softmax runs over the compact kept count with the same vectorized `exp_avx2` block.
  - Pass 2 keeps the 8-query value tile: one kept V row feeds every query in the tile that keeps it, accumulating in ascending key order — the same order as the per-token sparse kernel, hence bit-identical per-token sums.
- Dispatch: drop `!sparse_.enabled` from the block gate and call the block-sparse twin in the per-head lambda when `sparse_.enabled`, passing `sparse_.sink`, `sparse_.window`, `max(2, sparse_.stride)`; otherwise the dense block kernel. i8 stays excluded from sparse as before, and the Scalar oracle is untouched.
- The speculative verifier uses `prefill_run` -> `attention_batch`, so its six rows now also take the block-sparse kernel; that is the 1.24x decode-side composition above.

## 2. Predicted gain and outcome

Prediction was 0.30-0.35x dense attention and 1.30-1.40x total prefill at 8k. Measured **0.406x dense attention** and **1.265x total** (same-session pair), with 1.18-1.24x over the shipped per-token sparse path. The prediction was slightly optimistic on attention (the kept fraction is ~30%, but window rows are kept by up to 512 queries and the per-(key, token) ordinal check has a cost), and slightly pessimistic on total because matvec is flat. The direction and the keeper threshold are met with margin.

## 3. Measured results

All runs: 3B Q4_K_M, `--kv f16 --threads 10 --gpu off`, 8,193-token prompt, n=16. "per-token" = shipped binary, `--sparse`; "block-sparse" = spike binary, `--sparse`; "dense" = shipped binary, no `--sparse`.

### 8k prefill, same-session dense vs block-sparse

| arm | total prefill | prefill attn | matvec | ratio vs dense |
|---|---:|---:|---:|---:|
| dense | 205.3 s | 66.7 s | 127.4 s | — |
| block-sparse | **162.4 s** | **27.1 s** | 124.3 s | **1.265x total / 0.406x attn** |

### 8k prefill, interleaved per-token vs block-sparse

| pair | per-token total / attn | block total / attn | total ratio | attn ratio | matvec (pt / bs) |
|---|---|---:|---:|---:|---|
| 1 (pt -> bs) | 205.3 s / 60.3 s | 166.0 s / 27.5 s | **1.237** | 0.457 | 133.2 / 127.2 s |
| 2 (bs -> pt) | 187.7 s / 54.0 s | 159.0 s / 26.9 s | **1.180** | 0.498 | 122.7 / 121.1 s |
| **median** | **196.5 / 57.2** | **162.5 / 27.2** | **1.21** | **0.48** | |

### 8k `--speculative --sparse` (verify rows through the new kernel)

| arm | prefill | decode | spec line | verify attn |
|---|---:|---:|---|---:|
| per-token | 199.8 s | 14.45 t/s | 3 steps, 5.33/step, 18 emitted | 392.2 ms |
| block-sparse | **171.1 s** | **17.93 t/s** | 3 steps, 5.33/step, 18 emitted | **231.3 ms** |

Prefill **1.17x**, decode **1.24x**, verify attention **0.59x**; acceptance identical (same ids path).

### 2k control (2,051 tokens, n=16)

| arm | total prefill | prefill attn |
|---|---:|---:|
| dense (shipped) | 36.1 s | 3.82 s |
| per-token sparse (shipped) | 38.3-38.9 s | 3.31-3.33 s |
| block-sparse (spike) | 37.0 s | **2.84 s** |

Attention 0.74x dense / 0.86x per-token; total within noise because 2k prefill is 88% matvec.

### Ids (2k periodic, `--sparse --print-ids`)

```text
per-token   : 578 4062 14198 39935 35308 927 279 16053 5679 13 220 578 4062 14198 39935 35308
block-sparse: 578 4062 14198 39935 35308 927 279 16053 5679 13 220 578 4062 14198 39935 35308
```

Byte-identical, as designed: same kept positions, same ascending accumulation order. (Both differ from dense at position 10 — the priced sparse tier's near-tie, unchanged.)

## 4. Falsifier

Stated before the spike: **KEEP** if 8k block-sparse prefill is >= 1.10x the shipped per-token sparse path (or >= 1.20x dense) with attention <= 0.55x per-token, and 2k ids byte-identical to the per-token sparse path. Measured 1.18-1.24x over per-token, 1.265x over dense, attention 0.46-0.50x per-token, ids identical. All arms pass.

## 5. File anchors (current HEAD, read-before-write verified; spike reverted)

- `src/models/qwen2.cpp:1580-1615` — block gate (`!kv_q8_ && !sparse_.enabled` before the spike; the spike drops the sparse term) and the per-head dispatch lambda.
- `src/models/qwen2.cpp:1635-1655` — the per-token fallback sparse prefill uses today.
- `src/models/qwen2.cpp:1329-1342` — per-token sparse dispatch (`uses_sparse` path).
- `src/cpu/avx2/attention_avx2.cpp:660-864` — `attention_block_head_avx2`, the structure the twin copies (widen_row, pass 1, compact softmax, 8-query value tile).
- `src/cpu/avx2/attention_avx2.cpp:263-490` — the per-token strided kernels whose kept-set geometry and order the twin matches.
- `include/tge/cpu/kernels.hpp` — `AttentionBlockArgs` (tokens, start_position, score_stride, context, kv_head_offset) and the new declaration site after `attention_block_head_avx2`.
- `reports/idea-invented-sparse-prefill-8k-measured.md` — the per-token 8k pricing this cycle removes.
- `reports/idea-invented-sparse-8k-reopened.md` — the decode-side 8k result (unchanged by this spike; decode still uses the per-token kernel).

## 6. Reimplementation recipe (exact)

1. **Declaration** (`include/tge/cpu/kernels.hpp`, after `attention_block_head_avx2`):
   `void attention_block_sparse_head_avx2(const AttentionBlockArgs& args, int head, int sink, int window, int stride);`
2. **Kernel** (`src/cpu/avx2/attention_avx2.cpp`, after `attention_block_head_avx2`): copy the dense block kernel and change the key/token loops:
   - Keep the dense kernel's `kStackHeadDim = 256` row buffer and `widen_row` lambda verbatim.
   - Add `kept_ordinal(span, key)`: `span <= sink + window` -> `key < span ? key : -1` (dense prefix; matches `uses_sparse`); else `key < sink` -> `key`; else `window_start = span - window`, if `key < window_start` -> `((key - sink) % stride == 0) ? sink + (key - sink)/stride : -1`; else `key < span` -> `sink + middle_count + (key - window_start)` where `middle_count = (window_start - sink + stride - 1)/stride`; else `-1`.
   - Add `key_touched(key)`: `key < last_span && (key < sink || key >= start + 1 - window || (((key - sink) % stride) == 0 && key + window < last_span))`, with `last_span = start + tokens`.
   - Pass 1: for each `key_index <= last_key` with `key_touched`, load/widen the K row once, then for `token = max(0, key_index - start)..tokens-1` compute `kept_ordinal(start + token + 1, key_index)` and skip negatives; dot and store at `scores[token * score_stride + ordinal]`. Use the dense kernel's f32 `dot_avx2` and f16 two-accumulator loop unchanged.
   - Softmax per token over `kept_count` = `span` when `span <= sink + window`, else `sink + (span - window - sink + stride - 1)/stride + window`; reuse the dense kernel's vectorized max/exp/normalize block with `span` replaced by `kept_count`.
   - Pass 2: keep the `kValueTile = 8` tile; for each touched key, compute `first_query = max(tile_start, key_index - start) <= tile_end`, load/widen the V row once, then for each token in the tile range get the ordinal, skip negatives, read `scores[token * score_stride + ordinal]`, and fmadd. Keep key order ascending. Copy the tile to `out` as the dense kernel does.
3. **Dispatch** (`src/models/qwen2.cpp`, `attention_batch`): the block gate becomes `!kv_q8_ && attention_mode_ != Scalar && tokens > 0` (drop `!sparse_.enabled`). In the `parallel_for(heads)` lambda: if `sparse_.enabled` call `cpu::attention_block_sparse_head_avx2(per_head, head, sparse_.sink, sparse_.window, sparse_.stride < 2 ? 2 : sparse_.stride)`, else `cpu::attention_block_head_avx2(per_head, head)`.
4. **Tests**: build and run the full suite (expect 171/171). For a real port, add a differential test: synthetic f32 and f16 caches, one block of 128 tokens at `start = 2048`, compare the block-sparse output per token against `attention_head_static_sparse_*` on the same rows (expect bit-identical; the per-token kernel is the oracle), plus a block whose first tokens still fit `sink + window` (dense-prefix boundary).
5. **Measure** (spike tree): preserve the pre-spike binary, build the spike, and run the same-session dense and interleaved per-token pairs as in §3. Expect 8k prefill 1.18-1.27x, attention 0.40-0.50x, 2k ids identical, spec+sparse decode ~1.24x.
6. **Boundary cases to keep**: `sparse_.stride < 2` clamps to 2; spans `<= sink + window` use the dense prefix inside the block; keys beyond `last_span` are never touched; i8 never enters either block path; the Scalar oracle keeps its per-head path.

Do not ship without the differential test: the 2k id match is strong evidence, but the real port should pin the f32/f16 inertness and the dense-prefix boundary in the suite.

## 7. Exact commands run and results

```text
cmake --build --preset release                    -> clean (56/56)
build\release\bin\tge_tests.exe                   -> 171 cases, 0 failed

2k ids (fox x205, ctx 2560, n 16, --sparse --print-ids):
  per-token and block-sparse both -> 578 4062 14198 39935 35308 927 279 16053 5679 13 220 578 4062 14198 39935 35308

8k (compact fox x2048, ctx 8704, n 16):
  dense 205.3 s (attn 66.7, matvec 127.4) ; block-sparse 162.4 s (attn 27.1, matvec 124.3)  -> 1.265x / 0.406x
  per-token 205.3 s (attn 60.3) ; block-sparse 166.0 s (attn 27.5)                            -> 1.237x / 0.457x
  block-sparse 159.0 s (attn 26.9) ; per-token 187.7 s (attn 54.0)                            -> 1.180x / 0.498x
8k spec+sparse:
  per-token 199.8 s prefill, 14.45 t/s decode, verify attn 392.2 ms
  block-sparse 171.1 s prefill, 17.93 t/s decode, verify attn 231.3 ms -> 1.17x / 1.24x / 0.59x
2k prefill:
  dense 36.1 s (attn 3.82) ; per-token 38.3-38.9 s (attn 3.31-3.33) ; block-sparse 37.0 s (attn 2.84)
git checkout -- include/tge/cpu/kernels.hpp src/cpu/avx2/attention_avx2.cpp src/models/qwen2.cpp
cmake --build --preset release -> clean ; tge_tests -> 171/171 ; git status -> clean
python aikit.py check aegean-tge -> 4/4 gates passed
```

No push, no merge, no force-push. The committed tree is unchanged except this report.
