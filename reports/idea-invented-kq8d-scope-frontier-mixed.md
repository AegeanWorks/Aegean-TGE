# INVENTED: kq8d scope frontier — MIXED (up and gate+down pass fidelity; gate and f16+down fail; combo speed unresolved)

- **Idea**: Extend the cycle-19 down-only kq8d decode tier to the other FFN matrices: price `gate`, `up`, `gate+down`, `up+down`, and `f16 + down` at 399 positions. No new primary tier is promoted unless it passes the VNNI-class bar and has speed evidence.
- **Cycle date**: 2026-11-07
- **Verdict**: **MIXED.** `up` is the safest add-on (mean KL **1.421e-4**, top-1 **394/399 = 98.75%**); `gate+down` passes (8.451e-4, 390/399 = 97.74%); `gate` alone fails (1.636e-3, 385/399); `up+down` is one flip under the top-1 bar (8.313e-4, 389/399 = 97.49%); `f16 + down` fails (1.645e-3, 389/399). Combo speed was order-drift dominated, so **down-only remains the shipping opt-in tier**.
- **Tree state**: spike reverted from `include/tge/models/qwen2.hpp` and `src/models/qwen2.cpp`; 145/145 baseline tests rebuilt and green; no branch, no spike commit; nothing pushed.

## 1. Mechanism

The cycle-19 scoped kq8d path was extended with combinable scopes (`gate`, `up`, `down`, `gate+down`, `up+down`, `ffn`, `attn`, `all`). `forward` parses `TGE_KQ8D_SCOPE`, and only the named FFN matvecs pass `allow_kq8d`. All other paths stay float. The fidelity reference is `set_scalar_kernels(true)`, so the KL prices the kq8d error against the float path.

## 2. Falsifier

- **PROMOTE a scope** only if 399-position mean KL ≤1.6e-3, top-1 ≥97.6%-equivalent, ids identical, and an interleaved 2048 decode A/B shows a gain over the current down-only tier.
- **KEEP down-only** if no combo clears both fidelity and speed.

## 3. Measured numbers

### 399-position f32 fidelity

| scope | mean KL | worst KL | top-1 |
|---|---:|---:|---:|
| down (cycle 19) | 7.954e-4 | 1.581e-2 | 390/399 (97.74%) |
| **up** | **1.421e-4** | 1.993e-3 | **394/399 (98.75%)** |
| **gate+down** | **8.451e-4** | 8.904e-3 | **390/399 (97.74%)** |
| up+down | 8.313e-4 | 1.735e-2 | 389/399 (97.49%) |
| gate | 1.636e-3 | 3.134e-2 | 385/399 (96.49%) |
| f16 + down | 1.645e-3 | 1.711e-2 | 389/399 (97.49%) |

Reading:

- **up** is the safest single add-on: 1.4e-4 KL, 98.75% top-1.
- **gate+down** clears both bars at the same top-1 as down; the gate error is mostly hidden when combined with down.
- **gate alone** fails by a hair on KL (1.636e-3 vs 1.6e-3) and by 1.1 points on top-1.
- **up+down** is one flip under the top-1 bar; not promoted without a larger corpus.
- **f16 + down** fails the VNNI-class KL/top-1; the kq8d error stacks with f16 KV. No f16/i8 combo is safe.

### 2048 decode ordering (single runs, same binary)

```text
off        4.49 t/s
down       7.33 t/s
up+down    6.61 t/s
gate+down  6.22 t/s
up         3.22 t/s
```

The order is impossible (`up` alone cannot be slower than off by 28% while adding a faster kernel), so the box was drifting during the sequence. These numbers are **not** used for a combo speed claim. The cycle-19 interleaved down-vs-off pairs remain the speed evidence for the shipping tier (+9.6% and +35.6%).

### Ids

Thinking prompt A: `up`, `gate+down`, `up+down` all **IDENTICAL** to baseline. The 2048-token prompt ids were identical for down in cycle 19.

## 4. Verdict and disposition

**MIXED — keep down-only; map the frontier.** `up` and `gate+down` are fidelity-qualified candidates, but neither has interleaved speed evidence over down-only. `gate` alone and any f16/i8 combination fail the bar. The next step, if the combo is pursued, is a 3-pair interleaved `down` vs `gate+down`/`up+down` at 2048 on a quiet box; otherwise the down-only tier remains the shipping opt-in.

## 5. Reimplementation recipe (exact)

1. Re-use the cycle-19 `allow_kq8d` flag and `gemv_dequant_kq8_row_avx2` branch.
2. Parse `TGE_KQ8D_SCOPE` into tokens split on `+`; set `scoped_gate/up/down/attn`; `ffn` = gate+up+down, `all` = ffn+attn.
3. Pass the flags at the five `forward` matvec call sites; leave output/prefill/attention on float.
4. Gate: 399-position fidelity (mean KL ≤1.6e-3, top-1 ≥97.6%), thinking ids, 145/145, 4/4.
5. Do not promote a combo without interleaved 2048 speed pairs; single sequential runs are order-drift dominated.

## 6. Exact commands

```text
cmake --build --preset release                         → 54/54
.\build\release\bin\tge_tests.exe                      → 145/145
TGE_KQ8D_SCOPE=up|gate+down|up+down|gate tge fidelity ... --context 1024 --tokens 400 --kv f32
TGE_KQ8D_SCOPE=down tge fidelity ... --kv f16
TGE_KQ8D_SCOPE=... tge generate --tokens <2048 ids> --n 32 --threads 6 --gpu off
git checkout -- include/tge/models/qwen2.hpp src/models/qwen2.cpp
python aikit.py check aegean-tge                        → 4/4
```

No push, no merge, no force-push. Spike code is not in the tree or in any commit.
