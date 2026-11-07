# INVENTED: kq8d on the LM-head projection — KEEP as opt-in (3.3–3.6x head matvec, +33–39% decode, KL 1.9e-4)

- **Idea**: Route the final vocabulary projection (`token_embd.weight`, 3072×128256 Q6_K, 308 MiB) through the existing integer row kernel `gemv_dequant_kq8_row_avx2` instead of the float-dequant path. The LM head is the largest single matvec read per decode token and was the only projection left entirely on the float path after the kq8d scope work.
- **Cycle date**: 2026-11-08
- **Verdict**: **KEEP as an opt-in decode tier.** Direct head timer: **3.61x and 3.30x** faster (5,568 → 1,542 ms and 4,143 → 1,256 ms over 127 tokens). Decode: **+33% and +39%** in both interleaved pairs, ids identical. 399-position fidelity stays in class on every KV tier: f32 1.874e-4 / 393/399, f16 1.903e-4 / 392/399, i8 4.086e-4 / 395/399. Not defaulted in this report-only cycle; integration is a human-gated decision.
- **Tree state**: spike reverted from `include/tge/models/qwen2.hpp`, `src/models/qwen2.cpp`, `apps/cli/main.cpp` (including the temporary head timer); 145/145 baseline tests rebuilt and green; `aikit.py check` 4/4; no branch, no spike commit; nothing pushed.

## 1. Mechanism

Decode calls `output_logits(hidden)` once per token, which does `matvec(output_, hidden, logits_, q_out)`. For Q6_K that took the generic float path:

```text
gemv_dequant_batch: for each of 128,256 rows
  dequantize 3072 Q6_K values to float (scratch)
  dot with the 3072-float hidden row
```

The spike adds the cycle-19 `allow_kq8d` flag to `matvec` and passes it only from `output_logits` (guarded by `TGE_KQ8D_HEAD`). The kq8d branch quantizes the 3072-float activation once to Q8_K blocks and runs the integer Q6_K × Q8_K dot row by row — no per-row dequantization to float.

Prefill/`batch_matvec`, the attention projections, the FFN and all KV tiers are untouched. Decode is the only affected path.

## 2. Falsifier

- **KEEP as opt-in** if 399-position mean KL ≤1.6e-3 with top-1 ≥97.6%-equivalent on f32 and the stacked tiers, ids identical, and the head matvec is materially faster in a paired A/B.
- **KILL** if fidelity leaves the VNNI-class budget or the direct head timer shows no gain.

Measured: passes fidelity on all three tiers and the head timer shows a 3.3–3.6x gain.

## 3. Measured numbers

### 399-position fidelity (context 1024)

| KV | mean KL | worst KL | top-1 |
|---|---:|---:|---:|
| f32, head off (baseline) | 1.020e-11 | 2.330e-10 | 399/399 |
| f32, head kq8d | **1.874e-4** | 9.005e-4 | **393/399 (98.50%)** |
| f16, head kq8d | **1.903e-4** | 9.408e-4 | **392/399 (98.25%)** |
| i8, head kq8d | **4.086e-4** | 6.551e-3 | **395/399 (98.997%)** |

All tiers are well under the 1.6e-3 VNNI-class mean-KL bar and above its 97.6% top-1 precedent. The head error does not stack badly with i8 KV (2.3e-4 without the head → 4.1e-4 with it).

### Direct head matvec timer (127 decode tokens, 2048 context)

The temporary `phase_head_ns_` counter wraps only the `output_logits` matvec call.

| pair | head off | head on | speedup | head/token off → on |
|---:|---:|---:|---:|---:|
| 1 | 5,568 ms | 1,542 ms | **3.61x** | 43.8 → 12.1 ms |
| 2 | 4,143 ms | 1,256 ms | **3.30x** | 32.6 → 9.9 ms |

Head share of decode matvec fell from 28% → 8% and 25% → 7%. The absolute head time is box-load dependent; the paired within-run ratio is the evidence.

### Decode A/B (2048 prompt, n=128, interleaved order)

| pair | head off | head on | ratio | ids |
|---:|---:|---:|---:|---|
| 1 | 2.27 t/s | 3.02 t/s | **1.330x** | IDENTICAL |
| 2 | 3.45 t/s | 4.81 t/s | **1.394x** | IDENTICAL |

Both pairs faster; ids identical over 127 generated tokens. A separate 512-token `--n 32` check also produced identical ids.

## 4. Why it wins

- The head is **308 MiB of Q6_K read per token**. The float path dequantizes all 128,256 rows into scratch before each dot; the integer path keeps the weights quantized and spends one activation quantization (12 Q8_K blocks) per token.
- The observed float-path rate was ~7 GB/s effective (44 ms for 308 MiB); kq8d reached ~25 GB/s. The dequantization work, not the weight bytes, was the limiting factor.
- The Q6_K activation quantization error is small: 1.9e-4 mean KL and 98.5% top-1 against the float reference at 399 positions.

## 5. Reimplementation recipe (exact)

1. Add `bool allow_kq8d = false` to `Qwen2::matvec` (header + definition).
2. In the K-quant branch, before `kernels().gemv_dequant_batch`:
   ```text
   kq8d_kind = kind == Q4_K || kind == Q6_K
   if (allow_kq8d && !scalar_kernels_ && kq8d_kind && cpu::detect().features.avx2)
       gemv_dequant_kq8_row_avx2(args, config); return;
   ```
3. In `output_logits`, pass the flag (production: a CLI flag such as `--head-kq8d`, or auto-on for K-quant heads).
4. Leave `batch_matvec`/prefill float; leave the GPU Q8_0 path untouched.
5. Gate: 399-position fidelity on f32/f16/i8 (mean KL ≤1.6e-3, top-1 ≥97.6%-equivalent), ids identical, head timer ≥2x, 145/145 tests, `aikit.py check` 4/4.

## 6. Disposition

**KEEP as an opt-in decode tier, report-only.** The LM head is the strongest measured decode win in the library: 3.3–3.6x on the head matvec and +33–39% end-to-end decode at a 1.9e-4 KL price, with identical ids. It should ship behind an explicit flag first; a default-on decision needs a broader transcript sample and the usual human gate. It combines orthogonally with the down-only kq8d tier (different matrix), but that combination is not measured here.

## 7. Exact commands

```text
cmake --build --preset release                         → 54/54
.\build\release\bin\tge_tests.exe                      → 145/145
TGE_KQ8D_HEAD=0|1 tge fidelity --model Llama-3B --corpus docs/architecture.md \
  --context 1024 --tokens 400 --threads 6 --gpu off --kv f32|f16|i8
TGE_KQ8D_HEAD=0|1 tge generate --model Llama-3B --tokens <2048 ids> --n 128 \
  --threads 6 --gpu off
git checkout -- include/tge/models/qwen2.hpp src/models/qwen2.cpp apps/cli/main.cpp
python aikit.py check aegean-tge                        → 4/4
```

No push, no merge, no force-push. Spike code is not in the tree or in any commit.
