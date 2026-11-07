# INVENTED: AVX-VNNI Q8 KV attention dot — KILLED analytic (13% attention ceiling; QK-only 2x saves ~3%)

- **Idea**: Quantize the attention query to Q8_K and compute the QK dot as integer Q8×Q8 with AVX-VNNI (`vpdpbusd`), replacing the current i8-KV path's widen-to-float + FMA. Value accumulation stays float. Predicted 2x on the QK dot.
- **Cycle date**: 2026-11-07
- **Verdict**: **KILLED without a spike.** At 2048-token decode with i8 KV, attention is only **13%** of token time (1.194 s of 8.904 s); the QK dot is roughly half of attention, so even a 2x QK kernel saves ~3% of decode — below the 5% bar — before adding query-quantization error on top of the existing i8 KV tier (3.2e-4 KL). The ceiling fails on arithmetic; per the loop rule, no spike for a dead idea.
- **Tree state**: no code changes; project tree clean; 145/145 tests and 4/4 gates unchanged; nothing pushed.

## 1. Mechanism

The i8 KV form stores K/V as signed int8 with one float scale per 32 values. `attention_head_avx2` currently widens every int8 to float and FMA-accumulates. The proposed kernel would:

1. quantize the query head (head_dim values) to int8 with per-32 or per-head scales;
2. run the QK dot as integer `vpdpbusd` (unsigned query × signed key, or `maddubs` with a sign mask);
3. apply `qscale * kvscale` to the int32 sum;
4. keep the softmax and value accumulation as they are (value is a scalar weight × int8 value, not a dot).

The speedup applies only to the QK dot, not the value pass.

## 2. Falsifier / ceiling

- **PROCEED** only if a 2x QK dot can save ≥5% of decode, i.e. `attention_share × QK_share × 0.5 ≥ 0.05`.
- At `attention_share = 0.13` and `QK_share ≈ 0.5`, the saving is 0.0325 (3.25%) → **kill**.
- Reopen if a long-context measurement shows attention ≥25% of decode (QK-only 2x then ≈6%) or QK ≥70% of attention.

## 3. Measured phase shares (Llama-3.2-3B-Q4_K_M, 2048-token prompt, 15 decoded tokens, 6 threads)

| KV | decode matvec | attention | norm | attention time / total |
|---|---:|---:|---:|---:|
| f32 | 80% | **18%** | 2% | 0.724 / 3.928 s |
| f16 | 87% | **12%** | 1% | 1.180 / 10.014 s |
| i8 | 86% | **13%** | 1% | 1.194 / 8.904 s |

The box was loaded (i8 1.68 t/s absolute), so the shares are the evidence, not the absolute times. Attention is the largest non-matvec phase, but 13% leaves too little room for a QK-only kernel.

## 4. Why not spike it

- **Ceiling:** a free QK dot saves at most ~6.5% of i8 decode (half of 13%); a realistic 2x QK saves ~3.2%.
- **Quality:** query quantization adds error to a KV tier already at 3.2e-4 KL; the VNNI activation-quantization precedent (ADR-0006) shows the error is real and the end-to-end gain can vanish.
- **Port risk:** the earlier `maddubs` prefill experiment was slower because `psignb` contended for the single port; `vpdpbusd` avoids that specific port but needs an unsigned/signed correction and per-block scales, adding work to a kernel that is only 13% of the token.

## 5. What would reopen it

1. A 4096/8192-context i8 phase measurement showing attention ≥25% of decode (long-context attention is the natural use case for i8 KV).
2. A kernel microbenchmark proving ≥2.5x on the QK dot with query quantization included.
3. A fidelity price in the i8-KV tier showing no additional top-1 loss beyond the existing 3.2e-4 KL.

Until then, do not build the VNNI attention path.

## 6. Exact commands

```text
cmake --build --preset release                         → 54/54 (unchanged)
.\build\release\bin\tge_tests.exe                      → 145/145 (unchanged)
tge generate --model Llama-3B --tokens <2048 ids> --n 16 --threads 6 --gpu off --kv f32|f16|i8
# phases: decode matvec 80/87/86%, attention 18/12/13%
python aikit.py check aegean-tge                        → 4/4
```

No code changes, no push, no merge.
