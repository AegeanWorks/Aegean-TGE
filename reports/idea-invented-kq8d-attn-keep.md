# INVENTED: kq8d on the attention projections — KEEP as an opt-in scope (projection phase 1.31–1.84x, decode ~+5%)

- **Idea**: Extend the kq8d integer decode path to the four attention projections (`wq`, `wk`, `wv`, `wo`, ~14 MiB per layer at 4k context). Cycle 20 sampled this scope only at 63 positions (1.439e-3 KL, 60/63), too noisy to judge; this cycle prices it on the 1,199-position corpus and with a direct projection timer.
- **Cycle date**: 2026-11-08
- **Verdict**: **KEEP as an opt-in f32 decode scope.** Fidelity at 1,199 positions: **mean KL 6.951e-4, top-1 1,173/1,199 (97.83%)** — inside the 1.6e-3 / 97.6% bar. Speed: the kernel is **2.91x / 3.14x** faster on a 151 MB cold matrix; the model's projection phase is **1.84x** at 512 tokens and **1.31x** in a matched-window 1024 pair. Decode gains ~4–7% (projections are only ~10–20% of decode), and the tier stacks with the head+down scopes.
- **Tree state**: spike reverted from `include/tge/models/qwen2.hpp`, `src/models/qwen2.cpp`, `apps/cli/main.cpp` and `tests/test_dequant.cpp`; 145/145 baseline tests rebuilt and green; `aikit.py check` 4/4; no branch, no spike commit; nothing pushed.

## 1. Mechanism

`TGE_KQ8D_SCOPE=attn` passes `allow_kq8d` to the four single-token projection matvecs in `forward` (`wq`, `wk`, `wv` and `wo`), routing Q4_K/Q6_K rows through `gemv_dequant_kq8_row_avx2` (Q8_K activation × integer weights) instead of the float-dequant batch kernel. Prefill, attention itself, and the FFN remain unchanged. A temporary `phase_proj_ns_` counter wraps the four calls so the projection time can be compared directly.

The GGUF types: `attn_q` Q4_K 3072×3072, `attn_k` Q4_K 3072×1024, `attn_v` Q6_K 3072×1024, `attn_output` Q4_K 3072×3072 — all covered by the kq8d branch (confirmed with a one-shot stderr marker in the model run).

## 2. Falsifier

- **KEEP** if 1,199-position fidelity is inside the class bar and the projection phase is ≥1.15x faster in a drift-controlled comparison with identical ids.
- **KILL** if fidelity fails or the projection phase is unchanged.

Measured: fidelity passes; projection phase 1.31–1.84x.

## 3. Measured numbers

### Fidelity, 1,199 positions, f32

| scope | mean KL | worst KL | top-1 |
|---|---:|---:|---:|
| attn (this cycle) | **6.951e-4** | 1.278e-2 | **1,173/1,199 (97.83%)** |
| reference bar | 1.6e-3 | — | 97.6% |

The cycle-20 63-position sample (1.439e-3, 60/63 = 95.2%) was pessimistic; the larger corpus resolves it inside the bar.

### Kernel benchmark — `gemv_dequant_batch_avx2` vs `gemv_dequant_kq8_row_avx2`, 6 threads

| matrix | float | kq8d | speedup |
|---|---:|---:|---:|
| 3072×3072 (6.75 MB, L3-warm) | 0.374 ms | 0.102 ms | 3.66x |
| 3072×3072 (37 MB) | 2.77 ms | 0.69 ms | 3.5–4.0x |
| **8192×32768 (151 MB, L3-defeating)** | **10.85 / 12.40 ms** | **3.73 / 3.95 ms** | **2.91x / 3.14x** |

The 151 MB case is the honest model proxy: even fully DRAM-streamed, the integer row kernel is ~3x faster because it skips the per-row dequantization scratch and math.

### Model projection phase (temporary timer, 6 threads, f32)

| run | baseline proj | attn proj | ratio | note |
|---|---:|---:|---:|---|
| 512 tokens, n=16 | 499 ms | 272 ms | **1.84x** | decode 5.52 → 5.78 t/s (1.047x) |
| 1024 pair 1 | 2,855 ms | 3,186 ms | 0.90x raw | attention control 13% slower in attn arm → ~1.26x normalized |
| 1024 pair 2 | 8,903 ms | 6,782 ms | **1.31x** | attention control matched (2,401 vs 2,383 ms) |

The 2048/n128 pairs from earlier in the cycle were discarded: the unaffected attention phase moved 26–35% between arms, so their raw 1.22x decode ratios were drift, not the projection kernel.

### Ids

Identical in every pair (512 × 16 and 1024 × 32 tokens).

## 4. Interpretation

- The projection phase is only ~10–20% of decode at these contexts (11–20% of matvec), so even a 1.3–1.8x projection speedup moves decode by ~4–7%. That is below the 15% bar on its own but is additive to the head (3.3–3.6x) and down (10–36%) scopes.
- The kernel win is unambiguous when measured on a cold, L3-defeating matrix; the model-level measurement is harder because the projection phase is short and the box drifts.
- Fidelity is better than the earlier small-sample reading suggested: attn projections in Q4_K/Q6_K are a mild error source (6.95e-4 mean KL, 97.83% top-1).

## 5. Disposition

**KEEP as an opt-in f32 decode scope**, report-only. Add it to the staged integration after head and down: `--kernel kq8d=head,down,attn`. Expected stack decode at 4k rises from ~1.29x to ~1.35x. Keep f32-only until f16/i8 stacking is priced; the 1,199-position corpus is the acceptance gate.

## 6. Exact commands

```text
cmake --build --preset release                         → 54/54
.\build\release\bin\tge_tests.exe                      → 145/145
TGE_KQ8D_SCOPE=attn tge fidelity --model Llama-3B \
  --corpus <big corpus> --context 1536 --tokens 1200 --threads 6 --gpu off --kv f32
TGE_KQ8D_SCOPE=attn tge generate --tokens <1024 ids> --n 32 --threads 6 \
  --gpu off --kv f32 --context 1280 --print-ids
git checkout -- include/tge/models/qwen2.hpp src/models/qwen2.cpp \
  apps/cli/main.cpp tests/test_dequant.cpp
python aikit.py check aegean-tge                        → 4/4
```

No push, no merge, no force-push. Spike code is not in the tree or in any commit.
