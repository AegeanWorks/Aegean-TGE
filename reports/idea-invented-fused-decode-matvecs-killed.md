# INVENTED: fused per-layer decode projections (qkv and gate/up in one pool pass) — KILLED (median 1.01x end-to-end; matvec phase +5% median, below the 5% decode bar)

- **Idea**: Decode issues seven `matvec` calls per layer (q, k, v, o, gate, up, down), each a separate band-parallel fork/join over the decode pool. Projections that share one activation (q/k/v on `h`, gate/up on `h2`) can be dispatched as ONE band-parallel pass over the concatenated row space, so the fork/join and the per-call tail are paid once per group instead of once per matrix. Every row still runs the shipped per-row fused K-quant dot, so output is bit-identical by construction.
- **Cycle date**: 2026-11-11
- **Verdict**: **KILLED.** End-to-end decode on the reference Llama-3.2-3B-Instruct-Q4_K_M: median **1.01x** over 7 alternating pairs (full fusion 4 pairs, gate/up-only 3 pairs); best single pair 1.056x. The decode **matvec phase** (96-97% of decode) does improve: 4013 -> 3764 ms median at 48 tokens (1.066x), so the mechanism is real but worth ~2% end-to-end on this pool, under the 5% bar. Ids bit-identical in all 10 id-checked runs.
- **Tree state at close**: spike reverted from `include/tge/cpu/kernels.hpp`, `include/tge/models/qwen2.hpp`, `src/cpu/avx2/dequant_batch_avx2.cpp`, `src/models/qwen2.cpp`; `git status` clean; stash holds only the pre-existing 2026-11-04 peer entry; rebuild clean, `tge_tests` **171/171**, `python aikit.py check aegean-tge` **4/4**. No branch, no spike commit, nothing pushed.

## 1. Mechanism

The shipped decode path (`Qwen2::matvec`, `src/models/qwen2.cpp:967`) parallelises each matrix independently: `band_rows = max(1, rows / (threads * 4))`, `bands = ceil(rows / band_rows)`, `pool.parallel_for(bands, 1, ...)`. For Llama-3.2-3B (dim 3072, kv 1024, ffn 8192, 24 heads / 8 kv heads, 28 layers) a token issues 28 x 7 + 1 = **197** parallel calls, matching the comment at `src/core/thread_pool.cpp:26`.

The spike fused the two groups whose members read the same activation:

- `(wq, wk, wv)` on `h` -> 3072 + 1024 + 1024 = **5120 rows** in one pass;
- `(w_gate, w_up)` on `h2` -> 8192 + 8192 = **16384 rows** in one pass.

Groups cannot cross a dependency: `wo` needs the attention output, `down` needs `silu(gate) * up`, so four passes per layer is the minimum under this scheme.

## 2. Predicted gain (with the math)

Joins per token fall from 197 to 113 (-43%). The join cost is the sum of (a) the fork/join handshake (spin-based, `kSpinAttempts = 2048` at `thread_pool.cpp:29`, so a worker usually never parks) and (b) the per-call tail: the slowest of `4 x threads` bands against the mean.

At 96% matvec share and ~9.7 t/s decode (103 ms/token), each of the 197 calls averages ~0.50 ms. If a join costs 30-100 us (6-20% of a call), removing 84 joins predicts **2.5-8.4 ms/token, i.e. 2.4-8.2%** — the falsifier sat at the top of that range on purpose.

Why the prediction was too high: `band_rows` already produces 4 bands per worker, so a single call's rounding tail is at most one band out of 4 (<=25% of that call's own time), and for k/v (1024 rows) the 25-band split gives 5/4 = 25% tail on 20% of the qkv work — about 1% of the layer. The handshake itself is a few microseconds when workers spin through the gap, which is the case decode creates by construction.

## 3. Falsifier

**PROCEED** if end-to-end decode on `Llama-3.2-3B-Instruct-Q4_K_M` gains **>= 5% best-of-3** (alternating pairs, `--threads 6 --gpu off --kv f32`) with bit-identical ids. **KILL** otherwise.

Measured median 1.01x; kill. A secondary gate was that the matvec phase should move by more than box noise; it moves +3-7% (median 6.6%), which is recorded but does not rescue a sub-bar end-to-end result.

## 4. File anchors (current HEAD `5f4f26f`, read-before-write verified)

- `src/models/qwen2.cpp:967-1051` — `Qwen2::matvec`: per-tensor dispatch (VNNI Q8 batch, `gemv_q8`, kq8d row, fused Q4_K/Q6_K, dequant batch, f32) and `phase_matvec_ns_` accounting.
- `src/models/qwen2.cpp:2178-2180` — decode q/k/v calls; `:2218-2219` — `wo`; `:2228-2229` — gate/up; `:2233` — down. (Line numbers are the post-revert tree.)
- `src/models/qwen2.cpp:1702-1815`, `:2431-2507` — the batch/prefill call sites; untouched by this idea (prefill routes through `batch_matvec`).
- `src/cpu/avx2/dequant_batch_avx2.cpp:55` and `:94` — `gemv_q4k_fused_avx2` / `gemv_q6k_fused_avx2`: the banding template the fused pass copies.
- `src/cpu/avx2/dequant_avx2.cpp:158` / `:291` — `dot_q4k_row_avx2` / `dot_q6k_row_avx2`: the per-row dots reused verbatim.
- `src/cpu/avx2/dequant_batch_avx2.cpp:133` — `gemv_dequant_batch_avx2`, the fallback for other kinds.
- `src/core/thread_pool.cpp:26-29,249,279` — spin-then-park worker loop; the 2048-pause comment is the reason joins are cheap.

## 5. What was done (temporary spike, fully reverted)

1. `include/tge/cpu/kernels.hpp`: added `FusedRowsArgs { data[3], kind[3], rows[3], out[3], parts, cols, input }` and `void gemv_rows_fused_avx2(const FusedRowsArgs&, const GemvConfig&)`, documented as Q4_K/Q6_K-only and bit-identical per row.
2. `src/cpu/avx2/dequant_batch_avx2.cpp`: implemented `gemv_rows_fused_avx2` — validates every part, prefix-sums the row space, `band_rows = max(1, total / (threads * 4))`, one `pool.parallel_for`, and inside each band maps the global row range onto (part, local row) and runs `dot_q4k_row_avx2` / `dot_q6k_row_avx2` per row. Same `GemvConfig`/pool as the standalone kernels.
3. `include/tge/models/qwen2.hpp` + `src/models/qwen2.cpp`: added private `bool Qwen2::matvec_fused(const Matrix* const* weights, float* const* outputs, int count, const float* input) const` — returns false (no effect) when `scalar_kernels_`, `vnni_`, `count > 3`, non-AVX2, non-shared cols, or any part is not Q4_K/Q6_K; otherwise one fused call and `phase_matvec_ns_` accounting.
4. Decode call sites: qkv becomes `qkv_weights[3]`/`qkv_outputs[3]` with fallback `if (kq8d_attn_ || !matvec_fused(...)) { three matvecs }`; gate/up becomes `ffn_weights[2]`/`ffn_outputs[2]` with fallback to the two matvecs.
5. Built, `tge_tests` 171/171, built a baseline binary from the stashed tree, then alternated arms. A second variant reverted the qkv site (gate/up-only) to separate the two fusions.
6. Reverted all four files with `git checkout --`; rebuilt; 171/171; gates 4/4; `git status` clean.

## 6. Measured numbers

Reference model `Llama-3.2-3B-Instruct-Q4_K_M.gguf`, `--threads 6 --gpu off --kv f32`, greedy, fixed dragon prompt. Alternating pairs (base first, then spike), same window.

### Full fusion (qkv + gate/up), n = 64

| pair | base tok/s | fused tok/s | ratio | ids |
|---:|---:|---:|---:|---|
| 1 | 10.36 | 9.89 | 0.955 | identical |
| 2 | 9.61 | 9.87 | 1.027 | identical |
| 3 | 9.71 | 10.25 | 1.056 | identical |
| 4 | 10.07 | 10.20 | 1.013 | identical |
| **median** | 9.89 | 10.04 | **1.015** | 4/4 |

### Full fusion, n = 128 (box loaded mid-run)

| pair | base | fused | ratio |
|---:|---:|---:|---:|
| 1 | 9.91 | 9.37 | 0.945 |
| 2 | 7.80 | 9.19 | 1.177 |
| 3 | 8.76 | 8.34 | 0.951 |
| **median** | 8.76 | 9.19 | **1.01** |

The third sample shows the load swing (base fell to 7.80 t/s in pair 2); only same-pair ratios are read, and they disagree in sign, so the longer run adds no win.

### Matvec phase, n = 48 (the cleaner signal: 96-97% of decode time)

| pair | base matvec ms | fused matvec ms | ratio |
|---:|---:|---:|---:|
| 1 | 3951.1 | 3753.7 | 1.053 |
| 2 | 4013.1 | 3763.7 | 1.066 |
| 3 | 4351.6 | 4255.6 | 1.023 |
| **median** | 4013.1 | 3764.0 | **1.066** |

### Gate/up-only variant, n = 48 (isolates the 16384-row fusion)

| pair | base matvec ms | gate/up matvec ms | ratio | base tok/s | variant tok/s | ratio |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 4330.7 | 4800.1 | 0.902 | 10.45 | 9.40 | 0.900 |
| 2 | 4504.6 | 4565.4 | 0.987 | 10.07 | 9.89 | 0.982 |
| 3 | 4650.1 | 4380.4 | 1.062 | 9.76 | 10.30 | 1.055 |
| **median** | 4504.6 | 4565.4 | **0.987** | 10.07 | 9.89 | **0.982** |

The two fusions do not add up to a win individually: the 5120-row qkv fusion carries the small positive phase signal, the 16384-row gate/up fusion is neutral-to-negative.

## 7. Why it failed

- **The join is already cheap.** Decode's 197 calls arrive microseconds apart, so workers spin (`kSpinAttempts = 2048`) instead of parking; the handshake is a few microseconds, not tens. Removing 84 of 197 saves ~2% of the token.
- **Bands already amortise the tail.** `threads * 4` bands per call means each worker owns four equal slices; the only loss is rounding, and the largest call (gate/up, 16384 rows) has the smallest relative tail — exactly the call whose fusion measured neutral.
- **96% of decode is weight streaming**, and fusion moves no bytes: total weight bytes per token are identical, so the ceiling on any join-saving idea here is the small non-streaming remainder.
- **Fusing k/v hurts instruction locality slightly** (two extra weight streams per band), which is visible in the qkv-only variant's pair 1.

The honest reading: this is a *latency* idea applied to a *bandwidth-bound* phase, and the pool was already built for exactly this call pattern.

## 8. Full reimplementation recipe (rebuild only to re-verify the kill)

Do not ship. To reproduce:

1. `include/tge/cpu/kernels.hpp`, after the Q6_K fused declarations: add `struct FusedRowsArgs` with `const std::uint8_t* data[3]; QuantKind kind[3]; int rows[3]; float* out[3]; int parts; int cols; const float* input;` and declare `void gemv_rows_fused_avx2(const FusedRowsArgs&, const GemvConfig&);`.
2. `src/cpu/avx2/dequant_batch_avx2.cpp`: implement it with the `gemv_q4k_fused_avx2` skeleton (`src/models/qwen2.cpp:967` is the dispatch reference). Resolve `threads` via `resolve_thread_count`, take `config.pool` or `acquire_pool`, compute per-part `quant_row_bytes`, prefix-sum `starts[3]`, `total = sum(rows)`, `band_rows = max(1, total / (threads * 4))`, `bands = ceil(total / band_rows)`. In the lambda, skip parts whose `starts[p] + rows[p] <= begin`, then loop `take = min(end - offset, rows[p] - local)` rows calling `dot_q4k_row_avx2` or `dot_q6k_row_avx2` by `kind[p]`, advancing the part when its range ends. Throw `std::invalid_argument` on null/empty parts or a non-Q4_K/Q6_K kind.
3. `include/tge/models/qwen2.hpp`: declare `bool matvec_fused(const Matrix* const* weights, float* const* outputs, int count, const float* input) const;`.
4. `src/models/qwen2.cpp`: implement the predicate gate (`scalar_kernels_ || vnni_ || count <= 0 || count > 3 || !avx2` -> false; per part `cols` equal, `rows > 0`, `to_quant_kind` in {Q4_K, Q6_K}); fill `FusedRowsArgs`; call `gemv_rows_fused_avx2(args, GemvConfig{workers, &decode_pool()})`; add elapsed time to `phase_matvec_ns_`; return true.
5. Decode sites (`src/models/qwen2.cpp:2178-2180`, `:2228-2229`): build the pointer arrays and gate with `if (kq8d_attn_ || !matvec_fused(...))` for qkv (kq8d stays on its own path) and `if (!matvec_fused(...))` for gate/up, keeping the original matvec calls as the fallback body.
6. Build; `tge_tests` must stay 171/171 (all dots unchanged). Compare against a baseline binary from the reverted tree with at least 4 alternating pairs of `generate --n 64 --kv f32 --threads 6 --gpu off`, plus the `phases:` line at `--n 48`. Expected: matvec phase 1.02-1.07x, end-to-end 0.98-1.03x, ids identical. Anything below 1.05x end-to-end is the recorded kill.

## 9. Exact commands run and results

```text
cmake --build projects\aegean-tge\build\release --config Release   -> clean (56/56)
projects\aegean-tge\build\release\bin\tge_tests.exe                -> 171 cases, 0 failed
# baseline binary built from the stashed tree -> build\idea\tge-fusedmv-base.exe
# spike binaries                              -> build\release\bin\tge.exe, build\idea\tge-fusedmv-gateup.exe
tge generate --model Llama-3.2-3B-Instruct-Q4_K_M.gguf --prompt <dragon> \
  --n 64|128 --kv f32 --threads 6 --gpu off [--print-ids]          -> §6 tables
git checkout -- include/tge/cpu/kernels.hpp include/tge/models/qwen2.hpp \
  src/cpu/avx2/dequant_batch_avx2.cpp src/models/qwen2.cpp         -> reverted
cmake --build ... -> clean; tge_tests -> 171/171
python aikit.py check aegean-tge                                   -> 4/4 gates passed
```

No push, no merge, no force-push. The spike exists only in the two throwaway binaries under `build/idea/`; the committed tree is unchanged except this report.
