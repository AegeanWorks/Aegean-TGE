# INVENTED: Selective kq8d on the FFN down projection — KEEP as opt-in f32 decode tier (+10–36%, KL 7.95e-4, top-1 97.74%)

- **Idea**: Reopen the all-matvec kq8d decode kill with a **scoped** wiring: use the integer single-row kernel `gemv_dequant_kq8_row_avx2` only for the FFN **down** projection in decode, leaving attention, gate/up and the output projection on the float dequant path. The all-matvec version failed at mean KL 4.533e-3 / top-1 28/30; the down-only scope is much cheaper numerically.
- **Cycle date**: 2026-11-07
- **Verdict**: **KEEP as an opt-in f32 decode tier.** 399-position fidelity: mean KL **7.954e-4**, top-1 **390/399 (97.74%)** — under the VNNI-class 1.6e-3 bar and at its top-1 precedent. 2048 decode: **4.57 → 5.01 t/s (+9.6%)** and **4.13 → 5.60 t/s (+35.6%)**; ids identical on the 2048 prompt and both thinking prompts. Do not default it; f32 only.
- **Tree state**: spike reverted from `include/tge/models/qwen2.hpp` and `src/models/qwen2.cpp`; 145/145 baseline tests rebuilt and green; no branch, no spike commit; nothing pushed.

## 1. Mechanism

In decode (`tokens == 1`), K-quant matrices normally use `gemv_dequant_batch` (dequantize each row to floats, then dot). The bench-only `gemv_dequant_kq8_row_avx2` keeps the weights Q4_K/Q6_K and quantizes the activation once to Q8_K, then runs integer dots; it measured 2.07–2.80x on the large decode shapes.

The spike added `matvec(..., bool allow_kq8d = false)`. `forward` reads `TGE_KQ8D_SCOPE` and passes `true` only for the selected matrices:

| scope | matrices |
|---|---|
| `down` | FFN `w_down` only |
| `gate` / `up` | one FFN projection |
| `ffn` | gate + up + down |
| `attn` | wq/wk/wv/wo |
| `all` | ffn + attn (output projection remains float) |

The kq8d branch is gated on `!scalar_kernels_` (so the fidelity reference stays float), `Q4_K`/`Q6_K`, and AVX2. Prefill (`batch_matvec`), the output projection and f16/i8 KV paths are untouched.

## 2. Predicted gain / falsifier

- The down projection is the largest FFN matrix and has the strongest kq8d kernel win (2.8x on its shape). Decode reads it every token; a 10–20% decode gain was plausible.
- **KEEP as opt-in** if mean KL ≤1.6e-3 (VNNI-class), top-1 ≥97.6%-equivalent, and ids are identical; otherwise kill.
- **Restrict to f32** if f16/i8 stacking pushes the tier below the bar.

## 3. Measured numbers

### Scope fidelity, 63 positions, f32

| scope | mean KL | worst KL | top-1 |
|---|---:|---:|---:|
| **down** | **1.197e-3** | 9.130e-3 | **61/63** |
| gate (not shown separately) | — | — | — |
| ffn | 1.135e-3 | 7.940e-3 | 60/63 |
| attn | 1.439e-3 | 1.278e-2 | 60/63 |
| all | 2.848e-3 | 1.828e-2 | 58/63 |

Down is the best scope by top-1; `ffn`/`attn`/`all` are worse. The 63-position top-1 is noisy (one flip = 1.6%), so the longer corpus decides.

### Long f32 fidelity, 399 positions

| scope | mean KL | worst KL | top-1 |
|---|---:|---:|---:|
| down | **7.954e-4** | 1.581e-2 | **390/399 (97.74%)** |

This passes the VNNI-class precedent (1.568e-3, 485/497 = 97.6%).

### KV-tier stacking, 63 positions

| KV | mean KL | top-1 |
|---|---:|---:|
| f16 | 1.215e-3 | 60/63 |
| i8 | 1.277e-3 | 59/63 |

The kq8d error stacks with the lossy KV tiers; the tier is therefore **f32-only**.

### 2048-token decode A/B (same binary, env down vs off, interleaved)

| pair | baseline | kq8d down | ratio | ids |
|---:|---:|---:|---:|---|
| 1 | 4.57 t/s | 5.01 t/s | 1.096 | IDENTICAL |
| 2 | 4.13 t/s | 5.60 t/s | 1.356 | IDENTICAL |

Median gain ~1.22x; both pairs faster. The box was loaded; the direction is consistent but the magnitude is not a quiet-box claim.

### Ids

Thinking prompt A and B: generated ids **IDENTICAL** baseline vs down-scope. 2048-token prompt: ids **IDENTICAL**.

## 4. Why this is a keep and the earlier kill stands

The all-matvec kq8d failed at mean KL 4.533e-3 / top-1 28/30 because every projection contributed quantization error. The down projection alone is ~1/3 of the FFN and its error, at 399 positions, is **7.954e-4** with 97.74% top-1 — inside the VNNI-class budget. The earlier kill was correct for its scope; this is the narrow subset that survives.

## 5. Reimplementation recipe (exact)

1. Add `bool allow_kq8d = false` to `Qwen2::matvec` (header + definition).
2. In the K-quant branch, before `kernels().gemv_dequant_batch`:
   ```text
   kq8d_kind = kind == Q4_K || kind == Q6_K
   if (allow_kq8d && !scalar_kernels_ && kq8d_kind && cpu::detect().features.avx2)
       gemv_dequant_kq8_row_avx2(args, config); return;
   ```
3. In `forward`, read a scope (`TGE_KQ8D_SCOPE` for the spike; a `--kernel kq8d=down` flag for production) and pass `true` only for `w_down` (`down`/`ffn`/`all` scopes; production should ship `down` only).
4. Keep all other paths on the float dequant path; do not enable with f16/i8 KV.
5. Gate: 399-position fidelity (mean KL ≤1.6e-3, top-1 ≥97.6%), 2048 decode ≥5% with identical ids, thinking ids, 145/145 tests, `aikit.py check` 4/4.

## 6. Verdict and disposition

**KEEP as an opt-in f32 decode tier.** The down-only kq8d path gives a repeatable 10–36% 2048 decode gain at a fidelity price inside the VNNI-class budget, with identical ids on the measured prompts. It should ship behind an explicit flag (or `--kernel kq8d=down`), not as the default, and must not be combined with f16/i8 KV until those stacks are priced. The full recipe is small: one flag through `matvec`, one branch, one call site.

## 7. Exact commands

```text
cmake --build --preset release                         → 54/54
.\build\release\bin\tge_tests.exe                      → 145/145
TGE_KQ8D_SCOPE=down tge fidelity --model Llama-3B --corpus docs/architecture.md \
  --context 1024 --tokens 400 --threads 6 --gpu off --kv f32
TGE_KQ8D_SCOPE=down tge generate --model Llama-3B --tokens <2048 ids> --n 32 --threads 6 --gpu off
git checkout -- include/tge/models/qwen2.hpp src/models/qwen2.cpp
python aikit.py check aegean-tge                        → 4/4
```

No push, no merge, no force-push. Spike code is not in the tree or in any commit.
