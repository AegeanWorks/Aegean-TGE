# INVENTED: No-init scratch buffers — KEEP (5.6% load standalone; ~2x of the residual after KV no-init)

- **Idea**: Apply the proven `NoInitAllocator` to the three remaining large scratch buffers that are zero-filled at load — `logits_batch_`, `verify_logits_`, `scores_` — and use `resize` instead of `assign(n, 0)`. Every row is written before it is read. Follow-up to `idea-invented-kv-noinit-keep.md`; this cycle prices the scratch share separately.
- **Cycle date**: 2026-11-07
- **Verdict**: **KEEP-GRADE (marginal standalone, strong as a bundle).** Llama-3B context 4096 load: **177.9 → 167.9 ms** median (5.6%), 10/10 pairs faster, 9/10 ≥5%, best 172.0 → 159.1 ms (7.5%). All ids identical. With the no-init KV change, the same scratch zeroing would be ~half of the remaining ~19 ms load (predicted ~9–10 ms total; combined run not measured here).
- **Tree state at close**: spike reverted from `include/tge/models/qwen2.hpp` and `src/models/qwen2.cpp`; 145/145 baseline tests rebuilt and green; no branch, no spike commit; saved binaries live outside the tree; pre-existing stash untouched; nothing pushed.

## 1. Mechanism

`Qwen2::load` zero-fills three large scratch arrays whose contents are always overwritten before use:

| buffer | size at Llama-3B, ctx 4096 | written by |
|---|---:|---|
| `logits_batch_` | 64 × 128,256 × 4 = 32.8 MB | batched output projection, one row per prompt token |
| `scores_` | 64 × 32 heads × 4,096 × 4 = 33.6 MB | per-task attention score slices |
| `verify_logits_` | 6 × 128,256 × 4 = 3.1 MB | speculative verifier rows |
| **total** | **69.5 MB** | — |

At the ~5.5 GB/s effective first-touch rate measured in the KV report, that is ~12.6 ms of load. The spike:

- adds the stateless `NoInitAllocator<T>` (default-initializing `construct`, `::operator new`) and `NoInitVector<T>` alias from the KV report;
- changes only those three members to `NoInitVector<float>`;
- changes their load-time `assign(n, 0)` calls to `resize(n)`.

`logits_` (0.5 MB) and all other buffers keep the ordinary allocator. `Qwen2::reset()` and explicit fills are unchanged.

## 2. Predicted gain, with the math

```text
zeroed bytes = 32.8 + 33.6 + 3.1 = 69.5 MB
effective first-touch zero rate ≈ 5.5 GB/s (KV report)
predicted saving ≈ 69.5 MB / 5.5 GB/s ≈ 12.6 ms
baseline load ≈ 177.9 ms → predicted ≈ 165 ms (≈7%)
```

The falsifier was set at **≥5% median load reduction**, ids identical, tests green.

## 3. Falsifier

- **KEEP** only if all hold:
  1. context-4096 load drops by at least 5% on interleaved pairs;
  2. thinking, speculative and continuous transcripts are identical;
  3. the full suite is green.
- **KILL** on any id divergence, crash, or a median below 5%.

All passed; the measured 5.6% is at the low end of the prediction.

## 4. Measured numbers

Llama-3.2-3B-Q4_K_M, `--vocabulary lazy --context 4096`, 5-token prompt, 6 threads, `--n 1`; baseline and spike launched back-to-back per pair.

| pair | baseline load | no-init scratch | delta | reduction |
|---:|---:|---:|---:|---:|
| 1 | 172.0 | 159.1 | 12.9 | 7.5% |
| 2 | 176.9 | 162.7 | 14.2 | 8.0% |
| 3 | 173.3 | 163.1 | 10.2 | 5.9% |
| 4 | 176.2 | 170.0 | 6.2 | 3.5% |
| 5 | 179.7 | 168.5 | 11.2 | 6.2% |
| 6 | 178.9 | 168.2 | 10.7 | 6.0% |
| 7 | 176.7 | 166.7 | 10.0 | 5.7% |
| 8 | 185.4 | 167.5 | 17.9 | 9.7% |
| 9 | 183.8 | 172.3 | 11.5 | 6.3% |
| 10 | 183.5 | 173.0 | 10.5 | 5.7% |
| **median** | **177.9** | **167.9** | **10.0** | **5.6%** |
| **best** | **172.0** | **159.1** | **12.9** | **7.5%** |

10/10 pairs faster; 9/10 at or above 5%. The measured saving (~10 ms) is slightly below the 12.6 ms prediction, consistent with the zero-fill rate being closer to 7 GB/s on this box.

### Compound with the KV result (predicted, not measured)

The no-init KV report measured 183.2 → 18.9 ms at context 4096. Subtract the same ~10 ms scratch zeroing from that residual:

```text
combined load ≈ 18.9 - 10.0 ≈ 9 ms  (≈20x versus the 183 ms baseline)
```

That estimate is arithmetic on two separately measured deltas; it was not run as a combined spike.

## 5. Correctness evidence

| path | comparison | result |
|---|---|---|
| thinking prompt A | promptids + generated ids vs baseline | IDENTICAL |
| thinking prompt B | promptids + generated ids vs baseline | IDENTICAL |
| CJK prompt (96 prompt ids) | promptids + generated ids vs baseline | IDENTICAL |
| `--speculative` story | generated ids vs baseline | IDENTICAL |
| `bench continuous` 8×32 / 4 slots | `first` transcript + `state` | IDENTICAL |
| test suite | spike and after revert | 145/145 |

Every scratch row is written before read: the batched projection writes each token's logits row, attention writes its score slice before reading it, and the verifier writes the draft rows it then post-judges. The tests and the speculative/continuous runs exercise all three.

## 6. Fidelity and gates

- **Numerics**: allocation only; no arithmetic changed. No fidelity run owed, and the id parity above is exact.
- **Kernel bench**: no kernel touched; no kernel A/B owed.
- **Build**: `cmake --build --preset release` green (54/54).
- **Tests**: 145/145 before and after.
- **Project gate**: `python aikit.py check aegean-tge` → **4/4 gates passed** (test 15.6 s, hygiene, guidelines, secrets) after revert.

## 7. Reimplementation recipe (exact)

### 7.1 `include/tge/models/qwen2.hpp`

Add `<new>`, `<type_traits>`, `<utility>`, then the allocator/alias:

```text
template <typename T>
struct NoInitAllocator {
  using value_type = T;
  NoInitAllocator() noexcept = default;
  template <typename U> NoInitAllocator(const NoInitAllocator<U>&) noexcept {}
  T* allocate(size_t n) { return static_cast<T*>(::operator new(n * sizeof(T))); }
  void deallocate(T* p, size_t) noexcept { ::operator delete(p); }
  template <typename U> struct rebind { using other = NoInitAllocator<U>; };
  template <typename U, typename... Args>
  void construct(U* p, Args&&... args) const {
    if constexpr (sizeof...(args) == 0) ::new (static_cast<void*>(p)) U;
    else ::new (static_cast<void*>(p)) U(std::forward<Args>(args)...);
  }
};
template <typename T> using NoInitVector = std::vector<T, NoInitAllocator<T>>;
```

Change only:

```text
std::vector<float> verify_logits_;   -> NoInitVector<float>
std::vector<float> logits_batch_;    -> NoInitVector<float>
std::vector<float> scores_;          -> NoInitVector<float>
```

### 7.2 `src/models/qwen2.cpp` (`Qwen2::load`)

Replace the three cache-like `assign(..., 0.0F)` calls with `resize(...)`:

```text
logits_batch_.resize(min(kPrefillChunk, context_) * vocab_size);
verify_logits_.resize(kSpeculativeBatch * vocab_size);
scores_.resize(prefill_rows * head_count * context_);
```

Do **not** change `logits_`, `x_`, `h_`, `q_`, `k_`, `v_`, `gate_`, `up_`, `attn_`, rope tables, GQA state, or any buffer that is read before being fully written. `assign(n, value)` on a no-init vector still initializes, so explicit fills remain explicit.

### 7.3 Invariant to preserve

A buffer may use `NoInitVector` only when every element it contributes is written before the first read. For the three above:

- `logits_batch_`: `batch_matvec` writes rows `0..count-1`; only those rows are sampled.
- `scores_`: each attention task writes its own slice before softmax reads it.
- `verify_logits_`: `verify_draft` writes one row per drafted token, then reads those rows.

If a future path reads an unused row (padding, speculative tail, zero-copy shortcut), it must zero that row or use the ordinary vector.

### 7.4 Gate

1. build + 145/145 tests;
2. interleaved load A/B at context 4096 (≥5%);
3. thinking + CJK + speculative + continuous id/transcript parity;
4. `python aikit.py check aegean-tge` 4/4.

## 8. Verdict and disposition

**KEEP-GRADE as a bundle.** Standalone, the scratch no-init saves ~10 ms of a ~178 ms load (5.6%) — real, consistent, and zero-risk, but below the threshold that would justify a standalone feature. Bundled with the no-init KV change, it removes roughly half of the remaining zero-fill and is predicted to bring load to ~9–10 ms at context 4096 (~20x baseline). Ship the two together, not the scratch change alone.

## 9. Exact commands run (selection)

```text
cmake --build --preset release                         → 54/54
.\build\release\bin\tge_tests.exe                      → 145/145
<baseline> generate --model Llama-3B --prompt "The capital of France is" \
  --n 1 --threads 6 --vocabulary lazy --context 4096    → 172.0–185.4 ms load
<spike>    generate ...                                 → 159.1–173.0 ms load
tge generate ... --print-ids --text --gpu off           → thinking/CJK ids IDENTICAL
tge generate ... --speculative --print-ids              → ids IDENTICAL
tge bench continuous --requests 8 --tokens 32 --sequences 4
                                                       → first/state IDENTICAL
git checkout -- include/tge/models/qwen2.hpp src/models/qwen2.cpp
python aikit.py check aegean-tge                        → 4/4
```

No push, no merge, no force-push. Spike code is not in the tree or in any commit.
