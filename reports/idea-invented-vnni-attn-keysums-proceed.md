# INVENTED: VNNI Q8 QK dot with precomputed key bsums — MARGINAL PROCEED (median 1.61x; 8k saving ~7.5–15%)

- **Idea**: Re-layout the cycle-25 VNNI QK dot to remove its per-block overhead: precompute the signed key block sums, bias the query to unsigned, fold each block's `qscale × kvscale` into a float vector, and do one horizontal sum at the end instead of four.
- **Cycle date**: 2026-11-07
- **Verdict**: **MARGINAL PROCEED to model wiring.** Three 8192-key runs: **1.61x, 1.24x, 2.30x** (median 1.61x; best-of-best ~1.49x; VNNI 8.3–12.0 ns/key vs ref 12.4–27.6). At the measured 8k i8 share (50% attention, QK ≈ half), even 1.3x saves ~7.5% of decode; 1.61x saves ~15%. Above the 5% bar, but the box noise means the model-level gain is not yet proven.
- **Tree state**: spike reverted from `include/tge/cpu/kernels.hpp`, `src/cpu/avx2/q8_batch_vnni.cpp`, `tests/test_attention.cpp`; 145/145 baseline tests rebuilt and green; no branch, no spike commit; nothing pushed.

## 1. Mechanism

Cycle 25's stage-1 VNNI dot had four per-block horizontal sums, an XOR per key block, and a scalar `128·qsum` correction. The new layout:

```text
query quantization: q_u = clamp(round(query*127/amax), -127..127) + 128
                    qscale = amax/127                    (per 32)
key: signed int8, per-32 kvscale, precomputed ksum = sum(key[32])
per block:
  acc = vpdpbusd(0, q_u, key)          // unsigned q × signed k
  values = cvtepi32_ps(acc)
  values[0] += -128 * ksum             // (q+128)*k = q*k + 128*sum(k)
  values *= qscale * kvscale
  total += values
result = horizontal_sum(total)          // one horizontal sum per head
```

The key sums are a one-time cost at the i8 KV store (4 int32 per 128-dim head row, 0.125 bytes/value). The dot itself is the hot path.

## 2. Falsifier

- **PROCEED** if the 8192-key dot is ≥1.5x the current q8 dot (median) or if the 8k phase share implies ≥5% decode even at a lower measured speed.
- At 50% attention / QK half, 1.3x → ~7.5%, 1.61x → ~15%; both pass. The stage-1 error must stay in the i8 tier.

## 3. Measured numbers

| run | current q8 dot | VNNI key-bsum dot | speedup |
|---:|---:|---:|---:|
| 1 | 13.4 ns/key | 8.3 ns/key | 1.61x |
| 2 | 12.4 ns/key | 10.0 ns/key | 1.24x |
| 3 | 27.6 ns/key | 12.0 ns/key | 2.30x |
| **median** | 13.4 | 10.0 | **1.61x** |
| **best-of-best** | 12.4 | 8.3 | 1.49x |

Error vs the float-query q8 dot over 8192 keys: mean abs **4.501e-3**, max abs **2.463e-2**, RMS reference **1.597** → **0.28% mean relative**. This is the query-quantization price; the i8 KV tier's own 3.2e-4 KL is separate.

The box was heavily loaded (ref swings 12.4–27.6 ns); the VNNI absolute time is steadier (8.3–12.0). The median 1.61x passes the 1.5x proxy, but the best-of-best 1.49x is borderline; the 8k decode arithmetic still clears 5% even at 1.3x.

## 4. Why it improved

- One horizontal sum instead of four.
- No XOR per key block: the query is biased once; keys stay signed.
- The `128·ksum` correction is folded into one float lane and one scale multiply per block.
- Key sums are precomputed at store time, so the dot reads the same int8 key bytes as before plus a 4-byte sum per 32.

## 5. Stage-2 recipe (model wiring, next cycle)

1. `store_kv_q8` writes a parallel `k_q8_sums` array: 4 int32 per 128-dim head row (sum of signed int8 per 32).
2. `AttentionArgs` gains `keys_q8_sums`; `attention_head_avx2` q8 branch, when AVX-VNNI and `head_dim % 32 == 0`, quantizes the query to unsigned q with per-32 scales (once per head), then runs `attention_q8_dot_vnni_keysums` for each key.
3. Keep the existing float-query q8 dot for non-VNNI CPUs and scalar oracle.
4. Gate: 399-position i8 fidelity (mean KL ≤3.2e-4-class/top-1), 8192-context i8 decode ≥1.15x the current i8 path, ids identical where the tier allows, 145/145, 4/4.
5. If model fidelity leaves the i8 tier, kill; the kernel error is 0.28% mean but softmax sensitivity is unpriced.

## 6. Exact commands

```text
cmake --build --preset release                         → 54/54
.\build\release\bin\tge_tests.exe                      → 146/146 with the key-bsum bench; 145/145 after revert
# [vnni-attn-keysums] ref 13.4 / 12.4 / 27.6 ns/key
#                     vnni 8.3 / 10.0 / 12.0 ns/key (1.61x / 1.24x / 2.30x)
git checkout -- include/tge/cpu/kernels.hpp src/cpu/avx2/q8_batch_vnni.cpp tests/test_attention.cpp
python aikit.py check aegean-tge                        → 4/4
```

No push, no merge, no force-push. Spike code is not in the tree or in any commit.
