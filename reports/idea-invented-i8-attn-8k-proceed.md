# INVENTED: i8 attention at 8192 — PROCEED for VNNI attention (50% of decode, trigger met)

- **Idea**: Test the cycle-23 reopen trigger for an AVX-VNNI Q8 QK attention dot at long context. At 2048, i8 attention was only 13% of decode, so a QK-only 2x kernel saved ~3% and was killed. At 8192 the attention share should grow with context; this cycle measures it.
- **Cycle date**: 2026-11-07
- **Verdict**: **PROCEED to a VNNI attention stage-1 kernel.** At 8180-token context, i8 decode is **50% attention** (1,952 ms of 3,922 ms) and prefill is 50% attention; the ≥25% reopen trigger is met by 2x. A 2x QK dot (roughly half of attention) could save ~12.5% of 8k decode, well above the 5% bar. The cycle-23 kill stands for short context only.
- **Tree state**: measurement-only; a temporary `--tokens @file` CLI helper was added to exceed the Windows argv limit and reverted; 145/145 baseline tests rebuilt and green; no engine change; nothing pushed.

## 1. Mechanism / measurement

The reference model's 8192-token prompt does not fit the Windows 32k argv limit, so a temporary `--tokens @file` path read the id list from a file. The engine then ran:

```text
tge generate --model Llama-3.2-3B-Q4_K_M --tokens @ids8180.txt --n 8 \
  --threads 6 --gpu off --kv i8 --context 8192
```

Prompt length 8180 (the validator reserves the generated tokens). Phase timers split decode into matvec, attention and norm.

## 2. Measured numbers

| phase | time | share |
|---|---:|---:|
| prefill, 8180 tokens | 763,937 ms (10.7 t/s) | — |
| prefill matvec | 351,207 ms | 46% |
| prefill attention | 381,757 ms | 50% |
| prefill norm | 22,487 ms | 3% |
| decode, 7 tokens | 3,922 ms (1.78 t/s) | — |
| decode matvec | 1,924.5 ms | **49%** |
| decode attention | **1,951.8 ms** | **50%** |
| decode norm | 36.1 ms | 1% |

The box was heavily loaded (prefill 12.7 min; decode 1.78 t/s vs the earlier 7.6 t/s i8 figure), so the absolutes are not claimed. The **shares** are the evidence: attention rose from 13% at 2048 to 50% at 8192, as expected from the O(context) KV read per token.

## 3. Implication

- At 8192, QK is roughly half of the 50% attention phase; a 2x QK dot saves ~12.5% of decode, and a 1.5x QK dot ~8.3%. A free QK dot would save at most 25%.
- The 2048 kill is not overturned: at 2048 attention is 13%, so the QK-only ceiling is ~3%. The VNNI path should be **long-context only**.
- The i8 KV tier remains lossy (3.2e-4 KL, top-1 precedent); query quantization must be priced on top. The stage-1 kernel must include the query quantizer cost and a fidelity check, not just the dot.

## 4. Next step (stage 1, not built)

1. Prototype a Q8 query quantization per head (per-32 or per-head scale) and a `vpdpbusd`/`maddubs` QK dot.
2. Microbenchmark against the current widen+FMA q8 dot at 2048/8192 spans; require ≥2x including the query quantizer.
3. Price fidelity against the current i8 attention at 399 positions; the extra query quantization must not push the i8 tier below its 3.2e-4/top-1 precedent.
4. Gate: 8192 decode with the VNNI path ≥1.15x the current i8 path, same ids/fidelity tier; short-context dispatch can stay on the existing kernel.

## 5. Gates

- No engine code changed; the temporary `@file` helper was reverted.
- Tests: 145/145 after revert; `python aikit.py check aegean-tge` 4/4.
- Fidelity/ids: no numeric change; the measurement is read-only. The i8 tier remains the previously priced 3.2e-4.

## 6. Exact commands

```text
cmake --build --preset release                         → 54/54
.\build\release\bin\tge_tests.exe                      → 145/145
# temporary @file helper built, then:
tge generate --model Llama-3B --tokens @ids8180.txt --n 8 --threads 6 \
  --gpu off --kv i8 --context 8192
# decode phases: matvec 49%, attention 50%, norm 1%
git checkout -- apps/cli/main.cpp
python aikit.py check aegean-tge                        → 4/4
```

No push, no merge, no force-push. No spike code is in the tree or in any commit.
