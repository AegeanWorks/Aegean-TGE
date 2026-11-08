# CONFIRM: head+down kq8d frontier at 2,399 positions — passes narrowly (97.67%); head-only is the safe tier (98.83%)

- **Idea**: The head+down kq8d tier passed at 1,199 positions with 98.00% top-1. Double the corpus to 2,399 positions to halve the confidence interval on the integration acceptance gate, and measure head-only at the same size for the safer alternative.
- **Cycle date**: 2026-11-08
- **Verdict**: **Head+down still passes, but narrowly.** f32 head+down: **mean KL 7.874e-4, top-1 2,343/2,399 = 97.666%** (bar 97.6%, margin ≈1.6 flips). f16 head+down: 7.834e-4, 2,346/2,399 = 97.791% (margin ≈4.6 flips). Head-only f32: **1.915e-4, 2,371/2,399 = 98.833%** — a ~29-flip margin. The down scope costs ~1.2 points of top-1; the 1,199-position f32 reading (98.00%) was optimistic.
- **Tree state**: spike reverted from `include/tge/models/qwen2.hpp` and `src/models/qwen2.cpp`; 145/145 baseline tests rebuilt and green; `aikit.py check` 4/4; no branch, no spike commit; nothing pushed.

## 1. Setup

Combined corpus (`architecture.md` + `measurement-log.md` + `README.md`, 236 KB), `--context 3072 --tokens 2400` → **2,399 scored positions**, 6 threads, `--gpu off`. Three arms: f32 head+down, f16 head+down, f32 head-only.

## 2. Falsifier

- **PASS** if mean KL ≤1.6e-3 and top-1 ≥97.6%-equivalent.
- **REVISE** the tier recommendation if the margin collapses.

Both head+down arms pass, but the f32 margin is ~1.6 flips.

## 3. Measured numbers — 2,399 positions, context 3,072

| arm | mean KL | worst KL | top-1 | margin vs 97.6% |
|---|---:|---:|---:|---:|
| f32 head-only | **1.915e-4** | 1.488e-3 | **2,371/2,399 (98.833%)** | ~29 flips |
| f32 head+down | **7.874e-4** | 1.682e-2 | **2,343/2,399 (97.666%)** | ~1.6 flips |
| f16 head+down | **7.834e-4** | 1.031e-2 | **2,346/2,399 (97.791%)** | ~4.6 flips |

### 1,199 vs 2,399 positions

| arm | 1,199 top-1 | 2,399 top-1 | movement |
|---|---:|---:|---:|
| f32 head+down | 1,175/1,199 (98.00%) | 2,343/2,399 (97.67%) | −0.33 pt |
| f16 head+down | 1,172/1,199 (97.75%) | 2,346/2,399 (97.79%) | +0.04 pt |

The f32 estimate moved down by 0.33 points; f16 was stable. The correct statement of the tier is **~97.7% top-1**, not 98.0%.

Wall times: f32 head+down 930 s, f16 head+down 1,088 s, f32 head-only 1,204 s (window-dependent).

## 4. Interpretation

- The down scope roughly quadruples mean KL (1.9e-4 → 7.9e-4) and costs ~1.2 points of top-1 relative to head-only. That is the price of its 10–36% decode gain.
- Head+down sits **at** the 97.6% precedent rather than above it; a different corpus could tip it below. Head-only is comfortably safe and still carries the largest single speed win (head matvec 3.3–3.6x).
- f16's slightly better top-1 than f32 at this size is within near-tie noise; both are ~97.7–97.8%.
- The integration recommendation should split: **head-only as the safe default**, **head+down as the speed option with an explicit thin-margin note**. If a deployment prioritizes transcript stability over the extra down gain, use head-only.

## 5. Disposition

**CONFIRM with caveat.** The head+down tier remains the maximal passing kq8d set (2,399 positions: 97.67% f32, 97.79% f16), but its margin is ~2–5 flips, not the ~5–7 the 1,199 sample implied. Head-only is the recommended safe integration step; add down only where the ~1.2-point top-1 cost is acceptable.

## 6. Exact commands

```text
cmake --build --preset release                         → 54/54
.\build\release\bin\tge_tests.exe                      → 145/145
TGE_KQ8D_HEAD=1 TGE_KQ8D_SCOPE=down tge fidelity --model Llama-3B \
  --corpus <big corpus> --context 3072 --tokens 2400 --threads 6 --gpu off --kv f32|f16
TGE_KQ8D_HEAD=1 tge fidelity ... --kv f32 (head-only)
git checkout -- include/tge/models/qwen2.hpp src/models/qwen2.cpp
python aikit.py check aegean-tge                        → 4/4
```

No push, no merge, no force-push. Spike code is not in the tree or in any commit.
