# CONFIRM: kq8d head+down decode over 63 tokens at 4096 — 1.289x (matvec 1.47x), ids identical

- **Idea**: The 8180-token pairs used only 7 decoded tokens, where one background load swings the decode phase 4x (cycle 37). Measure the kq8d head+down decode at a 4,000-token prompt with **63 generated tokens** — long enough to average the load, still long enough in context for attention to matter.
- **Cycle date**: 2026-11-08
- **Verdict**: **CONFIRMED.** In the usable pair, decode is **3.36 → 4.33 t/s (1.289x)** with decode matvec **11,616 → 7,903 ms (1.470x)** and attention 6,864 → 6,371 ms (1.077x). Ids are identical over all 64 generated tokens. The second pair's baseline arm ran under a 4x load spike, but its drift-normalized direction agrees. This closes the decode-side gap left by cycle 37.
- **Tree state**: spike reverted from `include/tge/models/qwen2.hpp` and `src/models/qwen2.cpp`; 145/145 baseline tests rebuilt and green; `aikit.py check` 4/4; no branch, no spike commit; nothing pushed.

## 1. Setup

Re-applied only the two kq8d scopes (LM head + FFN down; block prefill attention is irrelevant to decode and was not included). `TGE_KQ8D_HEAD=1 TGE_KQ8D_SCOPE=down` vs both off, same binary, alternating order, at `--tokens <4000 ids> --n 64 --context 4096 --kv f32 --threads 6 --gpu off`.

## 2. Falsifier

- **CONFIRM** if decode ≥1.10x with identical ids and a consistent direction across the two pairs.
- **CHALLENGE** if the gain disappears at the longer sample.

Measured: 1.289x usable pair; direction consistent.

## 3. Measured numbers

### Pair 1 — baseline then kq8d (usable window)

| phase | baseline | kq8d | ratio |
|---|---:|---:|---:|
| decode (63 tokens) | 18,777 ms (3.36 t/s) | 14,536 ms (4.33 t/s) | **1.289x** |
| decode matvec | 11,616 ms | 7,903 ms | **1.470x** |
| decode attention (control) | 6,864 ms | 6,371 ms | 1.077x |
| decode norm | 264 ms | 231 ms | 1.142x |
| ids | — | — | **IDENTICAL** |

The attention phase (untouched by kq8d) was 7.7% faster in the kq8d arm, so the window slightly favored it; normalizing decode telemetry by attention gives a **20%** gain (`total/attn` 2.736 → 2.282).

### Pair 2 — kq8d then baseline (baseline arm loaded)

| phase | kq8d | baseline | ratio |
|---|---:|---:|---:|
| decode (63 tokens) | 12,271 ms (5.13 t/s) | 41,846 ms (1.51 t/s) | 3.397x (inflated) |
| decode matvec | 6,878 ms | 28,150 ms | 4.093x (inflated) |
| decode attention | 5,164 ms | 13,339 ms | 2.583x (inflated) |
| normalized `total/attn` | 2.376 | 3.137 | **1.320x** |

The baseline arm's attention is 2.58x slower, so the raw ratio is load drift. The normalized gain (32%) is consistent with pair 1's 20%.

### Cross-check with prior decode measurements

| measurement | window | raw decode | matvec |
|---|---|---:|---:|
| cycle 29 (head only) | 2048 / 127 tok | 1.330x, 1.394x | head matvec 3.3–3.6x |
| cycle 30 (head+down) | 2048 / 127 tok | 1.513x, 2.091x | — |
| **cycle 38 (head+down)** | **4000 / 63 tok** | **1.289x** | **1.470x** |
| cycle 38 normalized | 4000 / 63 tok | 1.20x / 1.32x | — |

The raw gain is smaller at 4k than at 2k because attention's share grows with context (37% vs ~20%), diluting the matvec-only win — the expected behavior.

## 4. Interpretation

- Over a 63-token sample the kq8d decode gain is stable and directly visible: matvec 1.47x, total 1.29x in a near-comparable window, 20% drift-normalized.
- The 8180 decode pair from cycle 37 is confirmed as unmeasurable, not contradictory: at 7 tokens, load noise dominates.
- Combined with cycles 36/37, the stack now has both sides measured: **prefill ~1.57x** (two opposite-order 8k pairs) and **decode ~1.29x at 4k** (63 tokens, ids identical).
- Ids remained identical in all four 64-token runs, consistent with the cycle-31 combo transcripts.

## 5. Disposition

**CONFIRMED — the kq8d decode component is real and stable at a measurable sample length.** Keep the stack as the integration target:
- block prefill attention: f32/f16, ~1.6–2.3x attention;
- kq8d LM head + down: decode matvec ~1.47x, decode ~1.29x at 4k;
- f32 KV primary, f16/i8 qualified at ~2–3 flips of top-1 margin (cycle 31).

Report-only; the human gate decides.

## 6. Exact commands

```text
cmake --build --preset release                         → 54/54
.\build\release\bin\tge_tests.exe                      → 145/145
TGE_KQ8D_HEAD=1 TGE_KQ8D_SCOPE=down tge generate --model Llama-3B \
  --tokens <4000 ids> --n 64 --threads 6 --gpu off --kv f32 --context 4096
TGE_KQ8D_HEAD=0 tge generate ... (same)
git checkout -- include/tge/models/qwen2.hpp src/models/qwen2.cpp
python aikit.py check aegean-tge                        → 4/4
```

No push, no merge, no force-push. Spike code is not in the tree or in any commit.
