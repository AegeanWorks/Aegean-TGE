# INVENTED: full vector tail in the GQA q8 value loop — KEEP kernel-only (kernel 1.1–2.3x best-of-3; decode 1.059x in matched pair)

- **Idea**: The GQA decode kernel's common (no new maximum) q8 value branch bounds its vector loop with `index + 32 <= head_dim` while stepping 8, so a 128-dim head processes only 104 of 128 values in AVX2 and the last 24 in scalar code. Make the vector bound cover the whole head.
- **Cycle date**: 2026-11-08
- **Verdict**: **KEEP kernel-only.** The vector tail is strictly less work: the kernel is faster in 3/3 best-of-3 rounds (0.621→0.562, 0.833→0.526, 0.771→0.335 ms at span 4096). In the clean model pair, decode matvec matched (10,520 vs 10,580 ms) while **attention is 1.32x faster and decode 1.059x faster**; ids identical. A second pair was drift-confounded but its attention also favored the fix (1.107x).
- **Tree state**: spike reverted from `src/cpu/avx2/attention_avx2.cpp` and `tests/test_attention.cpp` (temporary bench); 145/145 baseline tests rebuilt and green; `aikit.py check` 4/4; no branch, no spike commit; nothing pushed.

## 1. Mechanism

`attention_group_avx2`, q8 else-branch (the common path when the running maximum does not change):

```text
old: for (; index + 32 <= head_dim; index += 8) { ... 8-wide FMA ... }
new: for (; index <= head_dim - 8;      index += 8) { ... 8-wide FMA ... }
```

At `head_dim = 128` the old bound vectorized indices 0–103 and left 104–127 (18.75% of the head) to the scalar `beta * value * scale` tail. The fix vectorizes all 16 chunks. `TGE_Q8_FULLTAIL=0` restored the old bound for the A/B. The new-max branch already vectorized the full head.

## 2. Falsifier

- **KEEP** if the kernel is consistently faster with identical outputs/ids and the model shows a non-negative decode effect.
- **KILL** if the kernel is neutral or slower.

Measured: kernel 3/3 faster; model +5.9% in the matched pair.

## 3. Measured numbers

### Kernel benchmark — span 4096, i8, 3 query heads, best-of-3

| round | old bound | full vector tail | speedup |
|---:|---:|---:|---:|
| 1 | 0.621 ms | 0.562 ms | 1.10x |
| 2 | 0.833 ms | 0.526 ms | 1.58x |
| 3 | 0.771 ms | 0.335 ms | 2.30x |

The magnitude varies with box load, but the sign is consistent; the best-of-best is 0.621 → 0.335 ms.

### Model A/B — 4096 prompt, n=64, i8 KV

| pair | old decode | full decode | ratio | decode matvec | decode attention | ids |
|---:|---:|---:|---:|---:|---:|---|
| 1 | 4.47 t/s | 4.25 t/s | 0.951x | 10,378 → 11,429 ms | 3,436 → 3,104 ms (1.107x) | IDENTICAL |
| 2 | 4.37 t/s | 4.63 t/s | **1.059x** | 10,520 → 10,580 ms (matched) | 3,620 → 2,738 ms (**1.32x**) | IDENTICAL |

Pair 1 ran in strongly different windows (the full-tail prefill was 31% faster but its matvec 10% slower), so its decode ratio is not usable; its attention phase still favored the fix. Pair 2 is the clean comparison: matvec matched, attention 1.32x faster, decode +5.9%.

### Numerics

The tail's scalar `beta·v·scale + acc` becomes vector `fma(beta·v·scale, acc)` — the same per-element operation with FMA rounding. `attention_group_i8_matches_float` passes, all 145 tests pass, and generated ids are identical in both model pairs.

## 4. Interpretation

- The old bound was an unintended scalar tail, not a tuning choice: 18.75% of the value pass ran at scalar speed. Removing it is a strict reduction in work.
- The measured model gain (5.9% in the matched pair) is larger than the 18.75% × value-share estimate because the scalar tail also serializes the accumulator dependency chain.
- The fix is decode-only (the group kernel serves positions ≥256); prefill uses the per-token head path and is unaffected.

## 5. Disposition

**KEEP as a kernel fix**, report-only like the other integration candidates. It is small, numerically benign and additive to every decode configuration (all KV tiers that use the group kernel). Recommend folding it into the decode stack alongside the head+down kq8d scopes.

## 6. Exact commands

```text
cmake --build --preset release                         → 54/54
.\build\release\bin\tge_tests.exe                      → 145/145
TGE_Q8_FULLTAIL=0|on .\build\release\bin\tge_tests.exe → [q8-tail] 0.62/0.83/0.77 vs 0.56/0.53/0.34 ms
TGE_Q8_FULLTAIL=0|on tge generate --tokens <4096 ids> --n 64 --threads 6 \
  --gpu off --kv i8 --context 4352 --print-ids
git checkout -- src/cpu/avx2/attention_avx2.cpp tests/test_attention.cpp
python aikit.py check aegean-tge                        → 4/4
```

No push, no merge, no force-push. Spike code is not in the tree or in any commit.
