# INVENTED: No-init KV cache — KEEP (load 183 → 19 ms at 4096, ids identical)

- **Idea**: Stop zero-filling the entire KV cache at model load. Give the eight cache vectors an allocator that default-initializes (leaves raw) their elements, and use `resize` instead of `assign(n, 0)`. Every KV element is written before it is read; the zero-fill was pure startup work. New invented direction (not in the dossiers); checked against `docs/measurement-log.md` first — §8/§10 record load and KV allocation costs, but not this allocation policy.
- **Cycle date**: 2026-11-07
- **Verdict**: **KEEP-GRADE**. Llama-3.2-3B-Q4_K_M load at context 4096: 183.2 → 18.9 ms median (**9.7x**); context 2048: 98.3 → 14.7 ms (**6.7x**); prefill unchanged; all ids, fidelity tiers, continuous batching, sessions and speculative decode identical.
- **Tree state at close**: spike reverted from `include/tge/models/qwen2.hpp` and `src/models/qwen2.cpp`; 145/145 baseline tests rebuilt and green; no branch, no spike commit; pre-existing stash untouched; nothing pushed.

## 1. Mechanism

`Qwen2::load` allocated the KV cache with:

```cpp
k_cache_.assign(cache_elements, 0.0F);
v_cache_.assign(cache_elements, 0.0F);
```

and the same for the f16 and i8 forms. `assign` value-initializes every element, so the loader writes the whole cache once before the model has seen a token. At Llama-3B, context 4096:

```text
28 layers × 4096 positions × 1024 kv_dim × 4 bytes × 2 (K and V) = 896 MiB
```

That is ~180 ms of first-touch page faults and zero stores on this box — most of the reported `load` time. The cache is then overwritten position by position as tokens are processed. Attention at position `p` reads rows `0..p`, and the row for `p` is written before the attention call in both `forward` and `forward_batch`; the same is true for f16 and i8. Positions beyond the written span are never read. The zero-fill bought no safety.

The spike:

- adds a stateless `NoInitAllocator<T>` in `qwen2.hpp` whose no-argument `construct` does `::new (p) T` (default-initialization, no zero store) and whose `allocate` uses `::operator new`;
- aliases `NoInitVector<T> = std::vector<T, NoInitAllocator<T>>`;
- changes only the eight KV members (`k_cache_`, `v_cache_`, `k_cache_half_`, `v_cache_half_`, `k_cache_q8_`, `v_cache_q8_`, `k_cache_q8_scales_`, `v_cache_q8_scales_`);
- changes the eight load-time `assign(n, 0)` calls to `resize(n)`.

`Qwen2::reset()` still explicitly fills every cache with zero when a sequence is reset, so sequence reuse keeps its previous semantics. All other buffers keep the ordinary allocator and their zero initialization.

## 2. Predicted gain, with the math

The loader's `load` line scales linearly with context before the change:

| context | baseline `load` (median) | zeroed KV bytes (f32, K+V) |
|---:|---:|---:|
| 128 | 16.5 ms | 28 MiB |
| 512 | 34.1 ms | 112 MiB |
| 2,048 | 98.3 ms | 448 MiB |
| 4,096 | 183.2 ms | 896 MiB |

The slope is ~190 µs per MiB, consistent with first-touch zeroing at ~5.5 GB/s effective. If the zero-fill is the slope and the fixed mmap/tensor-table cost is the ~11 ms intercept, removing the fill should leave ~11–19 ms at every context:

```text
predicted 4096 load ≈ 11 ms (intercept) + small page-table cost ≈ 15–25 ms
```

The falsifier was set at **≥4x load reduction at 4096**, with ids and fidelity unchanged.

## 3. Falsifier

- **KEEP** only if all hold:
  1. context-4096 text-prompt `load` (skipped vocabulary) drops by at least 4x versus baseline;
  2. f32, f16 and i8 generation ids are identical to baseline;
  3. continuous batching, sessions and speculative decode transcripts are identical;
  4. the full suite is green and no fidelity tier moves outside its known class.
- **KILL** on any id divergence, crash, or a prefill regression larger than the box's ±20% band.

All passed.

## 4. Baseline reproduction (pre-spike tree)

Same binary, same model, `--vocabulary lazy`, `--threads 6`, `--n 1`, 5-token prompt. `load` includes mmap, tensor-table parse and KV allocation.

| context | load ms (3 runs) | median |
|---:|---|---:|
| 128 | 16.5, 15.7, 17.7 | 16.5 |
| 512 | 34.1, 39.9, 33.2 | 34.1 |
| 2,048 | 116.0, 102.1, 101.7 / 98.3, 96.9, 99.4 | ~99 |
| 4,096 | 200.9, 199.8, 202.0 / 183.2, 182.3, 184.1 | ~183–202 |

Two baseline sessions were taken (before the spike and after the revert); both show the same linear slope. The lower 183 ms values are used for the A/B below.

## 5. Spike implementation (reverted; recipe in §8)

Files: `include/tge/models/qwen2.hpp`, `src/models/qwen2.cpp`. No kernel, sampler, tokenizer or arithmetic change.

Build green, 54/54 targets. Tests **145/145**, identical count to baseline.

## 6. Measured numbers

### 6a. Load A/B

| context | baseline median | spike median | speedup |
|---:|---:|---:|---:|
| 128 | 16.5 ms | 11.0 ms | 1.5x |
| 512 | 34.1 ms | 12.2 ms | 2.8x |
| 2,048 | 98.3 ms | 14.7 ms | **6.7x** |
| 4,096 | 183.2 ms | 18.9 ms | **9.7x** |

Spike runs at 4096: 18.4, 19.8, 18.9. At 2048: 14.5, 14.7, 15.2. The fixed floor (~11 ms) is the mmap/tensor-table work; the removed part is the zero-fill.

### 6b. Prefill unchanged

5-token prompt at context 4096, same session:

| | baseline | spike |
|---|---:|---:|
| prefill | 418–463 ms (11–12 t/s) | 370–386 ms (13–13.5 t/s) |

The spike is if anything slightly faster because the machine was quieter; either way, no regression. The load saving is not moved into prefill for short prompts: only the used KV rows fault.

### 6c. Fidelity (teacher-forced, 63 positions, ctx 512)

| KV | mean KL | worst KL | max \|logit\| | top-1 |
|---|---:|---:|---:|---:|
| f32 | 1.130e-11 | 5.531e-11 | 5.054e-05 | 63/63 |
| f16 | 2.501e-07 | 1.239e-06 | 9.610e-03 | 63/63 |
| i8 | 2.373e-04 | 1.234e-03 | 2.169e-01 | 61/63 |

These are the known tiers (f16 precedent 1.7e-6, i8 3.2e-4), not allocation artifacts. No NaN/Inf and no wild KL.

### 6d. Behavioural parity

| path | comparison | result |
|---|---|---|
| fixed thinking prompt A | promptids + generated ids vs baseline | IDENTICAL |
| creative thinking prompt B | promptids + generated ids vs baseline | IDENTICAL |
| CJK prompt (96 prompt ids) | promptids + generated ids vs baseline | IDENTICAL |
| `--kv f16` think A | generated ids vs baseline | IDENTICAL |
| `--kv i8` think A | generated ids vs baseline | IDENTICAL |
| `--speculative` story | generated ids vs baseline | IDENTICAL |
| `bench continuous` 8×32 / 4 slots | first transcript + `state` | IDENTICAL |
| `bench sessions` 4×32 / 1 thread | first transcript + `state` | IDENTICAL |
| context 8192, 5-token prompt | load 26.5 ms, prefill 386 ms, no crash | PASS |

### 6e. Whole-startup effect

Text prompt on Llama-3B, context 4096: baseline `load` 183 ms + tokenizer ~110–130 ms ≈ 300–315 ms before the first model token; spike `load` 19 ms + tokenizer ~110–130 ms ≈ 130–150 ms. The tokenizer remains the larger half; the BPE-heap cycle (separate report) addresses pathological prompt encode, not construction.

## 7. Fidelity and gates

- **Numerics**: no arithmetic changed; f32/f16/i8 fidelity re-measured and in class.
- **Kernel bench**: no kernel changed; no kernel A/B owed.
- **Build**: `cmake --build --preset release` green.
- **Tests**: 145/145 on the spike and 145/145 after revert.
- **Project gate**: `python aikit.py check aegean-tge` → **4/4 gates passed** (test 18.1 s, hygiene, guidelines, secrets) after revert.

## 8. Reimplementation recipe (exact)

### 8.1 `include/tge/models/qwen2.hpp`

Add `<new>`, `<type_traits>`, `<utility>` includes and, in `namespace tge::models` before `Qwen2`:

```text
template <typename T>
struct NoInitAllocator {
  using value_type = T;
  NoInitAllocator() noexcept = default;
  template <typename U> NoInitAllocator(const NoInitAllocator<U>&) noexcept {}

  T* allocate(size_t count) {
    return static_cast<T*>(::operator new(count * sizeof(T)));
  }
  void deallocate(T* pointer, size_t) noexcept { ::operator delete(pointer); }

  template <typename U> struct rebind { using other = NoInitAllocator<U>; };

  template <typename U, typename... Args>
  void construct(U* pointer, Args&&... args) const {
    if constexpr (sizeof...(args) == 0) {
      ::new (static_cast<void*>(pointer)) U;   // default-init: no zero store
    } else {
      ::new (static_cast<void*>(pointer)) U(std::forward<Args>(args)...);
    }
  }
};

template <typename T>
using NoInitVector = std::vector<T, NoInitAllocator<T>>;
```

Change only the eight KV members:

```text
NoInitVector<float>          k_cache_, v_cache_;
NoInitVector<uint16_t>       k_cache_half_, v_cache_half_;
NoInitVector<int8_t>         k_cache_q8_, v_cache_q8_;
NoInitVector<float>          k_cache_q8_scales_, v_cache_q8_scales_;
```

Leave `x_`, `h_`, `q_`, `k_`, `logits_`, rope tables, etc. as ordinary vectors.

### 8.2 `src/models/qwen2.cpp` (`Qwen2::load`, around line 511)

Replace each cache `assign(n, 0)` with `resize(n)`:

```text
if (kv_half) {
  k_cache_half_.resize(cache_elements);
  v_cache_half_.resize(cache_elements);
} else if (kv_q8_) {
  k_cache_q8_.resize(cache_elements);
  v_cache_q8_.resize(cache_elements);
  k_cache_q8_scales_.resize(cache_elements / 32);
  v_cache_q8_scales_.resize(cache_elements / 32);
} else {
  k_cache_.resize(cache_elements);
  v_cache_.resize(cache_elements);
}
```

Do not touch `Qwen2::reset()` — its explicit `std::fill(..., 0)` is the safety net for sequence reuse.

### 8.3 Invariant to preserve

Every read of a KV row must be preceded by a write to that row in the same slot. The current design satisfies it:

- `forward` (decode) writes the current position's K/V before calling `attention(0, layer, position, ...)`; attention's span is `0..position`.
- `forward_batch` writes every token's row before the attention pass; a new slot's first row is written at admission.
- Prefix-cache copies read rows that a previous populate wrote.
- Unused positions beyond the written span are never addressed.

If a future path ever reads a speculative/unwritten row, it must zero it explicitly or call `reset()` first. A debug build could add a `TGE_KV_ZERO_INIT=1` branch to restore the old behavior for bisection; the shipped path should not.

### 8.4 Tests and gates

- `cmake --build --preset release`
- `build/release/bin/tge_tests.exe` → 145/145 (existing tests cover attention, KV f16/f32/i8, batch and cache reset).
- `tge fidelity --model Llama-3B --corpus docs/architecture.md --context 512 --tokens 64 --threads 6 --gpu off` and `--kv f16`, `--kv i8`; keep the known tier ranges.
- `tge generate` thinking set with `--print-ids --text`; diff baseline vs candidate.
- `tge bench continuous` 8×32/4 slots and `tge bench sessions` 4×32; compare `first` and `state`.
- `python aikit.py check aegean-tge` → 4/4.

## 9. Verdict and disposition

**KEEP-GRADE.** The zero-fill was startup-only work on memory that is always overwritten before use. Removing it cuts Llama-3B load by up to 9.7x at the default context, leaves prefill and all numerics untouched, and is provably behaviour-identical across f32/f16/i8, batching, sessions and speculation. The only caveat is where the first-touch faults go: for a prompt that fills the cache, the same pages fault during prefill instead of load, and total traffic is actually lower (one write pass instead of two). Short/interactive prompts touch only a small fraction of the cache and get the full saving.

## 10. Exact commands run (selection)

```text
cmake --build --preset release
.\build\release\bin\tge_tests.exe                         # 145/145 spike and baseline
tge generate --model Llama-3B --prompt ... --n 1 --threads 6 --vocabulary lazy --context {128,512,2048,4096}
tge generate --model Llama-3B --prompt <thinking> --n 32 --threads 6 --print-ids --text --gpu off
tge generate ... --kv f16|i8 ... ; tge generate ... --speculative ...
tge bench continuous --model Llama-3B --requests 8 --tokens 32 --threads 6 --sequences 4
tge bench sessions --model Llama-3B --sessions 4 --tokens 32 --threads 1 --context 2048
tge fidelity --model Llama-3B --corpus docs/architecture.md --context 512 --tokens 64 --threads 6 --gpu off [--kv f16|i8]
python aikit.py check aegean-tge                          # 4/4
git checkout -- include/tge/models/qwen2.hpp src/models/qwen2.cpp
git status --porcelain=v1                                 # only ledger/report at close
```

No push, no merge, no force-push. Spike code is not in the tree or in any commit.
