# INVENTED: Query-block prefill attention for f16/i8 KV — KEEP f32/f16, i8 marginal (f16 attention 1.47–1.83x, i8 1.02–1.07x)

- **Idea**: Cycle 33's query-block prefill kernel was f32-only. Extend it to f16 and i8 KV so the long-context prefill win is available on the memory-saving tiers: one K/V row is loaded/widened once and reused across the 64-token query block.
- **Cycle date**: 2026-11-08
- **Verdict**: **KEEP for f32/f16; i8 is marginal.** At 8180 tokens, f16 attention is **1.472x / 1.827x** faster and total prefill **1.355x / 1.519x** in the two drift-controlled pairs, with identical ids. i8 attention is only **1.024x / 1.071x** — its 8k K/V working set (~16 MB) nearly fits L3, so re-reads are cache-served and blocking barely helps. Recommend enabling the block kernel for f32/f16 prefill and leaving i8 on the per-token path.
- **Tree state**: spike reverted from `include/tge/cpu/kernels.hpp`, `src/cpu/avx2/attention_avx2.cpp`, `src/models/qwen2.cpp`, `tests/test_attention.cpp` and `apps/cli/main.cpp`; 145/145 baseline tests rebuilt and green; `aikit.py check` 4/4; no branch, no spike commit; nothing pushed.

## 1. Mechanism

`AttentionBlockArgs` now carries `keys_half`/`values_half` (f16) and `keys_q8`/`values_q8`/`keys_q8_scales`/`values_q8_scales` plus `half`/`q8` flags and `scale_stride`. The kernel branches once per row load:

- **f16**: `_mm256_cvtph_ps` widens each 8-value group; two accumulators for the QK dot, one per value store.
- **q8**: the existing `q8_dot_avx2` for scores (per-32 scales at `row*scale_stride + kv_head_offset/32`), `widen_i8 × scale` for values.

Softmax and the row-reuse schedule are unchanged from cycle 33. `attention_batch` builds the args from `k_cache_`, `k_cache_half_` or `k_cache_q8_` and keeps the same head-parallel dispatch; Scalar mode and sparse attention still use the per-token path.

The unit test (`attention_block_matches_head_path`) now runs all three tiers (f32/f16/q8) against `attention_head_avx2` and passes at 1e-4.

## 2. Falsifier

- **KEEP a tier** if 8180-token prefill attention is ≥1.15x faster in drift-controlled pairs with identical ids.
- **Leave the tier** on the per-token path if the gain is inside drift.

Measured: f32/f16 pass; i8 does not.

## 3. Measured numbers

### f16 KV, 8180-token prefill, 6 threads

Pair 1 — per-token then block:

| phase | per-token | block | ratio |
|---|---:|---:|---:|
| prefill total | 544,191 ms (15.0 t/s) | 401,590 ms (20.4 t/s) | **1.355x** |
| attention | 295,657 ms | 200,822 ms | **1.472x** |
| matvec | 228,245 ms | 183,371 ms | 1.245x |
| ids | — | — | IDENTICAL |

Pair 2 — block then per-token:

| phase | block | per-token | ratio |
|---|---:|---:|---:|
| prefill total | 430,927 ms (19.0 t/s) | 654,584 ms (12.5 t/s) | **1.519x** |
| attention | 211,358 ms | 386,294 ms | **1.827x** |
| matvec | 201,338 ms | 249,276 ms | 1.238x |

The untouched matvec also moved ~1.24x in the block arm's favor both times, so the total ratios are optimistic; attention alone is 1.47–1.83x.

### i8 KV, 8180-token prefill

| pair | per-token total | block total | total ratio | per-token attn | block attn | attention ratio |
|---:|---:|---:|---:|---:|---:|---:|
| 1 (off→block) | 564,670 ms | 611,973 ms | 0.923x | 283,548 ms | 276,875 ms | 1.024x |
| 2 (block→off) | 447,475 ms | 393,263 ms | 1.138x | 220,236 ms | 205,558 ms | 1.071x |

Pair 1's matvec drifted 20% against the block arm, which explains the negative total. Attention itself is only 1.02–1.07x: i8 K/V at 8k is 8180 × 128 × 2 bytes × 8 KV heads ≈ 16 MB, close to the 20 MB L3, so the per-token re-reads are mostly cache hits and the block kernel's traffic saving is small.

### Ids

f16 8180 prompt: **identical**; i8 8180 prompt: **identical** (pair 1). Cycle 33 already established f32 parity at 2048, 8180 and 3 × 64-token prompts.

## 4. Interpretation

- The block kernel pays exactly when the K/V working set exceeds cache: f32 (~64 MB) > f16 (~32 MB) > L3 (~20 MB) > i8 (~16 MB). That ordering matches the measured gains (f32 1.6–1.9x, f16 1.5–1.8x, i8 ~1.0x).
- f16 is the practical long-context tier: half the memory of f32 and the same ~1.5–1.8x attention gain.
- i8 prefill should stay per-token: no meaningful win, and the block path adds a third KV branch to maintain.

## 5. Disposition

- Enable the block prefill kernel for **f32 and f16** KV (above ~2048 positions; neutral below), behind a flag first.
- Keep **i8 prefill** on the per-token path.
- Report-only; integration is the human-gated decision.

## 6. Exact commands

```text
cmake --build --preset release                         → 54/54
.\build\release\bin\tge_tests.exe                      → 146/146 with the three-tier block test; 145/145 after revert
TGE_ATTN_BLOCK=0|on tge generate --model Llama-3B --tokens @ids8180.txt \
  --n 8 --threads 6 --gpu off --kv f16|i8 --context 8192
git checkout -- include/tge/cpu/kernels.hpp src/cpu/avx2/attention_avx2.cpp \
  src/models/qwen2.cpp tests/test_attention.cpp apps/cli/main.cpp
python aikit.py check aegean-tge                        → 4/4
```

No push, no merge, no force-push. Spike code is not in the tree or in any commit.
