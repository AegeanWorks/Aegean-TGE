# INVENTED: full kq8d head+down+attn stack — KILLED (KL 1.576e-3 passes, top-1 1,167/1,199 = 97.33% fails)

- **Idea**: Price the three-scope kq8d stack — LM head + FFN down + attention projections — on the 1,199-position corpus. Each scope passes fidelity individually or in the head+down pair; the open question was whether all three fit the class budget.
- **Cycle date**: 2026-11-08
- **Verdict**: **KILLED on top-1.** The full stack reaches **mean KL 1.576e-3** (just under the 1.6e-3 bar) but only **1,167/1,199 = 97.33% top-1**, below the 97.6% precedent. The boundary is narrow but real: **head+attn** also fails (1,168/1,199 = 97.41%), while **head+down** (cycle 31: 1,175/1,199 = 98.00%) and **attn alone** (cycle 39: 1,173/1,199 = 97.83%) pass. The maximal passing kq8d set remains **head+down**.
- **Tree state**: spike reverted from `include/tge/models/qwen2.hpp` and `src/models/qwen2.cpp`; 145/145 baseline tests rebuilt and green; `aikit.py check` 4/4; no branch, no spike commit; nothing pushed.

## 1. Mechanism

`TGE_KQ8D_SCOPE` accepts a comma list (`down,attn`); `TGE_KQ8D_HEAD=1` adds the LM head. The full stack routes the head, `w_down` and the four attention projections through `gemv_dequant_kq8_row_avx2`; prefill, attention kernels and gate/up remain float.

## 2. Falsifier

- **KEEP a scope set** if the 1,199-position mean KL ≤1.6e-3 **and** top-1 ≥97.6%-equivalent.
- **KILL** the set otherwise; no speed measurement is warranted for a failing set.

Measured: full stack fails top-1; head+attn also fails.

## 3. Measured numbers — 1,199 positions, f32 KV

| scope set | mean KL | worst KL | top-1 | verdict |
|---|---:|---:|---:|---|
| head + down (cycle 31) | 8.283e-4 | 1.682e-2 | **1,175/1,199 (98.00%)** | pass |
| attn alone (cycle 39) | 6.951e-4 | 1.278e-2 | **1,173/1,199 (97.83%)** | pass |
| **head + attn** | 8.886e-4 | 1.426e-2 | 1,168/1,199 (97.41%) | **fail** |
| **head + down + attn** | 1.576e-3 | 2.724e-2 | 1,167/1,199 (97.33%) | **fail** |

The two failing sets are only **2–3 flips** below the bar, and their mean KL is under 1.6e-3. **Top-1 is the binding constraint**, not mean KL.

## 4. Interpretation

- Scope errors stack on near-ties even when each scope's mean KL is small: head (1.9e-4 at 399 positions) + attn (7.0e-4 at 1,199) already costs ~5 flips relative to head alone, enough to drop below 97.6% despite a passing KL.
- The 399-position results from cycles 29–30 overestimated the margins; at 1,199 positions, head+down is the only two-scope set that clears both bars, and no three-scope set does.
- The attention-projection kernel is fast (2.9–3.1x cold) but its fidelity cost is not additive-safe with the head. It is a standalone alternative (97.83%) or an alternative pairing with down, not a third layer on the recommended stack.
- Since the failing sets fail fidelity, their speed was not measured; per the loop's gate, speed after a fidelity failure is moot.

## 5. Disposition

- **Kill the three-scope stack.** The recommended kq8d configuration remains **head + down**, f32 KV (98.00% top-1 at 1,199 positions).
- `attn` stays a documented opt-in scope (cycle 39) for workloads that prefer it over down, but it is not additive.
- If a wider quality budget ever becomes acceptable, the full stack is the fastest configuration measured at the kernel level; until then, it does not ship.

## 6. Exact commands

```text
cmake --build --preset release                         → 54/54
.\build\release\bin\tge_tests.exe                      → 145/145
TGE_KQ8D_HEAD=1 TGE_KQ8D_SCOPE=down,attn tge fidelity --model Llama-3B \
  --corpus <big corpus> --context 1536 --tokens 1200 --threads 6 --gpu off --kv f32
TGE_KQ8D_HEAD=1 TGE_KQ8D_SCOPE=attn tge fidelity ... (same)
git checkout -- include/tge/models/qwen2.hpp src/models/qwen2.cpp
python aikit.py check aegean-tge                        → 4/4
```

No push, no merge, no force-push. Spike code is not in the tree or in any commit.
