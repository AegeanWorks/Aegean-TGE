# INVENTED: Combined no-init KV + scratch — KEEP (load 175.3 → 8.7 ms, 20.1x)

- **Idea**: Integrate the two no-init reports into the shipping bundle: apply `NoInitAllocator` to the eight KV cache vectors **and** the three large scratch buffers, replacing all eleven load-time `assign(n, 0)` calls with `resize(n)`. This cycle measures the compound result the previous report predicted (~9–10 ms load).
- **Cycle date**: 2026-11-07
- **Verdict**: **KEEP-GRADE.** Llama-3.2-3B-Q4_K_M context-4096 load: **175.3 → 8.7 ms** median (**20.1x**), 10/10 pairs, range 17.8–25.3x. Context-8192 load: 12.8 ms. All prompt/generated ids identical (f32/f16/i8/speculative/continuous), fidelity tiers unchanged, 145/145 tests, 4/4 project gates.
- **Tree state at close**: spike reverted from `include/tge/models/qwen2.hpp` and `src/models/qwen2.cpp`; 145/145 baseline tests rebuilt and green; no branch, no spike commit; saved binaries live outside the tree; pre-existing stash untouched; nothing pushed. (An unrelated uncommitted `.agents/task_ledger.json` in the AIKit root was left untouched.)

## 1. Mechanism

Two previously measured, independently reverted changes are applied together:

| group | members | zeroed bytes at Llama-3B ctx 4096 |
|---|---|---:|
| KV caches | `k_cache_`, `v_cache_`, `k_cache_half_`, `v_cache_half_`, `k_cache_q8_`, `v_cache_q8_`, `k_cache_q8_scales_`, `v_cache_q8_scales_` | 896 MiB (f32 K+V) |
| scratch | `logits_batch_`, `scores_`, `verify_logits_` | 69.5 MB |
| **total** | **11 vectors** | **~965 MB** |

All eleven buffers are written before they are read:

- KV rows: `forward`/`forward_batch` write the current position before attention; attention only reads `0..position`.
- `logits_batch_`: the batched output projection writes one row per prompt token; only those rows are read.
- `scores_`: each attention task writes its own slice before softmax reads it.
- `verify_logits_`: the verifier writes the draft rows it then post-judges.

The change is exactly the union of `idea-invented-kv-noinit-keep.md` §8 and `idea-invented-scratch-noinit-keep.md` §7:

- add `NoInitAllocator<T>` / `NoInitVector<T>` to `qwen2.hpp`;
- change the eleven members to `NoInitVector`;
- change the eleven load-time `assign` calls to `resize`.

`Qwen2::reset()` and all explicit fills are unchanged.

## 2. Predicted gain, with the math

Separate measurements:

```text
KV no-init:      183.2 → 18.9 ms   (saves ~164 ms)
scratch no-init: 177.9 → 167.9 ms  (saves ~10 ms)
```

Arithmetic on the two deltas:

```text
combined load ≈ 18.9 - 10.0 ≈ 8.9 ms
```

The falsifier was set at **≤20 ms and ≥8x** at context 4096, with ids and fidelity unchanged.

## 3. Falsifier

- **KEEP** only if all hold:
  1. context-4096 load is at most 20 ms and at least 8x faster than baseline;
  2. thinking, f16/i8, speculative and continuous ids/transcripts are identical;
  3. f32/f16/i8 fidelity stays in its known tier and the suite is green.
- **KILL** on any id/fidelity divergence, crash, or load above 20 ms.

All passed.

## 4. Measured numbers

Llama-3.2-3B-Q4_K_M, `--vocabulary lazy --context 4096`, 5-token prompt, 6 threads, `--n 1`; baseline and combined launched back-to-back per pair.

| pair | baseline load | combined load | speedup |
|---:|---:|---:|---:|
| 1 | 175.3 | 8.7 | 20.1x |
| 2 | 176.4 | 8.5 | 20.8x |
| 3 | 197.3 | 9.1 | 21.7x |
| 4 | 220.3 | 8.7 | 25.3x |
| 5 | 169.2 | 8.7 | 19.4x |
| 6 | 170.3 | 8.7 | 19.6x |
| 7 | 173.2 | 8.8 | 19.7x |
| 8 | 175.6 | 8.6 | 20.4x |
| 9 | 178.3 | 10.0 | 17.8x |
| 10 | 174.1 | 8.8 | 19.8x |
| **median** | **175.3** | **8.7** | **20.1x** |
| **best** | **169.2** | **8.5** | **19.9x** |

The ~166.6 ms median saving is the one-pass removal of ~965 MB of zero-fill; the residual ~8.7 ms is mmap + tensor-table parse + small buffers. Context 8192 (KV 1.8 GB uninitialized) loads in **12.8 ms** and prefills normally.

## 5. Correctness evidence

| path | comparison | result |
|---|---|---|
| thinking prompt A | promptids + generated ids vs baseline | IDENTICAL |
| thinking prompt B | promptids + generated ids vs baseline | IDENTICAL |
| CJK prompt (96 prompt ids) | promptids + generated ids vs baseline | IDENTICAL |
| `--kv f16` think A | generated ids vs baseline | IDENTICAL |
| `--kv i8` think A | generated ids vs baseline | IDENTICAL |
| `--speculative` story | generated ids vs baseline | IDENTICAL |
| `bench continuous` 8×32 / 4 slots | `first` + `state` | IDENTICAL |
| test suite | spike and after revert | 145/145 |

Fidelity, 63 positions, context 512:

| KV | mean KL | worst KL | top-1 |
|---|---:|---:|---:|
| f32 | 1.130e-11 | 5.531e-11 | 63/63 |
| f16 | 2.501e-07 | 1.239e-06 | 63/63 |
| i8 | 2.373e-04 | 1.234e-03 | 61/63 |

These are byte-for-byte the tiers from the separate no-init cycle; allocation does not move arithmetic.

## 6. Gates

- **Numerics**: no arithmetic changed; fidelity tiers above.
- **Kernel bench**: no kernel touched; no kernel A/B owed.
- **Build**: `cmake --build --preset release` green (54/54).
- **Tests**: 145/145 before and after.
- **Project gate**: `python aikit.py check aegean-tge` → **4/4 gates passed** (test 20.0 s, hygiene, guidelines, secrets) after revert.

## 7. Reimplementation recipe (exact)

Apply both recipes as one change:

1. `include/tge/models/qwen2.hpp`
   - add `<new>`, `<type_traits>`, `<utility>`;
   - add `NoInitAllocator<T>` and `NoInitVector<T>` (default-initializing no-arg `construct`, `::operator new`);
   - change these members to `NoInitVector`:
     `k_cache_`, `v_cache_`, `k_cache_half_`, `v_cache_half_`, `k_cache_q8_`, `v_cache_q8_`, `k_cache_q8_scales_`, `v_cache_q8_scales_`, `logits_batch_`, `verify_logits_`, `scores_`.
2. `src/models/qwen2.cpp` (`Qwen2::load`)
   - KV: replace the f32/f16/i8 `assign(cache_elements, 0)` block with `resize(cache_elements)` for all eight cache vectors and `resize(cache_elements / 32)` for the two scale vectors;
   - scratch: `logits_batch_.resize(min(kPrefillChunk, context_) * vocab_size)`, `verify_logits_.resize(kSpeculativeBatch * vocab_size)`, `scores_.resize(prefill_rows * head_count * context_)`;
   - do not change `logits_`, `x_`, `h_`, `q_`, `k_`, `v_`, `gate_`, `up_`, `attn_`, rope tables, GQA state or any buffer read before written.
3. Invariant: `NoInitVector` is legal only when every row is written before its first read. `reset()` still explicitly fills the caches for sequence reuse.
4. Gate:
   - build + 145/145 tests;
   - interleaved load A/B at context 4096 (target ≤20 ms);
   - thinking/CJK + f16/i8 + speculative + continuous parity;
   - f32/f16/i8 fidelity in class;
   - `python aikit.py check aegean-tge` 4/4.

## 8. Verdict and disposition

**KEEP-GRADE, ship as one bundle.** This supersedes the separate KV and scratch reports for integration: together they turn a ~175 ms context-4096 load into ~8.7 ms (20.1x) with no arithmetic, id or fidelity change. The remaining load is mmap/tensor-table parse and small buffers; further startup work would target those, not the caches.

## 9. Exact commands run (selection)

```text
cmake --build --preset release                         → 54/54
.\build\release\bin\tge_tests.exe                      → 145/145
<baseline> generate --model Llama-3B --prompt "The capital of France is" \
  --n 1 --threads 6 --vocabulary lazy --context 4096    → 169.2–220.3 ms load
<combined> generate ...                                 → 8.5–10.0 ms load
tge generate ... --print-ids --text --gpu off           → thinking/CJK ids IDENTICAL
tge generate ... --kv f16|i8 ... ; tge generate ... --speculative ...
tge bench continuous --requests 8 --tokens 32 --sequences 4
tge fidelity --model Llama-3B --corpus docs/architecture.md \
  --context 512 --tokens 64 --threads 6 --gpu off --kv f32|f16|i8
tge generate ... --context 8192                         → load 12.8 ms
git checkout -- include/tge/models/qwen2.hpp src/models/qwen2.cpp
python aikit.py check aegean-tge                        → 4/4
```

No push, no merge, no force-push. Spike code is not in the tree or in any commit.
