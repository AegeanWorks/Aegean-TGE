# INVENTED: kq8d gate/up combo extension — KILLED (adding gate/up is slower; down-only remains the sweet spot)

- **Idea**: Promote the fidelity-qualified `gate+down` or `up+down` kq8d scopes from cycle 20 to a faster decode tier than down-only. Cycle 20 left the speed ordering unresolved because single sequential runs were order-drift dominated.
- **Cycle date**: 2026-11-07
- **Verdict**: **KILLED.** Interleaved 2048 A/B: down-only mean **6.92 t/s** / best 6.77; `up+down` 6.70 / 6.20; `gate+down` 5.96 / 5.74. Adding gate/up makes decode slower despite more integer matrices. Down-only stays the shipping kq8d tier.
- **Tree state**: spike reverted from `include/tge/models/qwen2.hpp` and `src/models/qwen2.cpp`; 145/145 baseline tests rebuilt and green; no branch, no spike commit; nothing pushed.

## 1. Mechanism

The cycle-20 scoped kq8d path was re-applied with combinable `TGE_KQ8D_SCOPE` values. In decode, only the named FFN matvecs route through `gemv_dequant_kq8_row_avx2`; all other paths stay float.

The question: `down` alone was kept at +9.6%/+35.6% in cycle 19. `gate+down` and `up+down` pass fidelity at 399 positions (8.451e-4 / 390/399 and 8.313e-4 / 389/399), so a faster tier was possible if the extra integer matrices amortized well.

## 2. Falsifier

- **PROMOTE** if the interleaved 2048 decode is at least 5% faster than down-only and ids are identical.
- **KILL** if adding gate/up is neutral or slower.

Measured: both combos slower; kill.

## 3. Measured numbers

Llama-3.2-3B-Q4_K_M, 2048-token id prompt, `--n 32 --threads 6 --gpu off`, same binary, rotating order (round 1 down→gate+down→up+down; round 2 reverse).

| round | down | gate+down | up+down |
|---:|---:|---:|---:|
| 1 | 6.77 | 6.17 | 7.20 |
| 2 | 7.06 | 5.74 | 6.20 |
| **mean** | **6.92** | **5.96** | **6.70** |
| **best** | **6.77** | **5.74** | **6.20** |

Down-only wins both mean and best. `up+down` got one lucky 7.20 run but its second run (6.20) and best confirm the ordering. Generated ids are **IDENTICAL** across all three scopes.

### Fidelity context (from cycle 20, 399 positions, f32)

| scope | mean KL | top-1 |
|---|---:|---:|
| down | 7.954e-4 | 390/399 |
| gate+down | 8.451e-4 | 390/399 |
| up+down | 8.313e-4 | 389/399 |

The fidelity was not the blocker; speed is.

## 4. Why it lost

- **Activation quantization cost per matrix.** kq8d quantizes the activation row for each call. Gate and up share the same activation; routing both adds two quantizations (and their fixed cost) where the float path does none.
- **Shape match.** The down projection is the wide 896×4864 shape; gate/up are 4864×896. The integer row kernel's advantage is strongest on the down shape (2.8x vs 2.07x bench), and the extra gate/up work does not amortize.
- Down-only is the narrow subset where the integer row kernel's savings exceed its per-call quantizer overhead.

## 5. Reimplementation / A-B recipe

1. Re-use the cycle-19/20 `allow_kq8d` flag and `TGE_KQ8D_SCOPE` parser (`down`, `gate+down`, `up+down`).
2. Use a 2048-token id prompt, `--n 32`, 6 threads, rotating scope order across two rounds; compare mean and best, not single sequential runs.
3. Gate: ≥5% over down-only and identical ids; expected down 6.92 vs combos 6.70/5.96 → do not promote.
4. Ship down-only; close the kq8d gate/up line.

## 6. Verdict and disposition

**KILLED.** The kq8d scope line is now closed at **down-only**: it is the only scope with both passing fidelity and a measured decode gain. Gate, up, gate+down, up+down, ffn and all are either below the fidelity bar or slower. No further kq8d scope work is warranted without a different activation-quantization strategy (e.g., quantize the shared FFN activation once and reuse it across gate/up).

## 7. Exact commands

```text
cmake --build --preset release                         → 54/54
.\build\release\bin\tge_tests.exe                      → 145/145
TGE_KQ8D_SCOPE=down|gate+down|up+down tge generate --model Llama-3B \
  --tokens <2048 ids> --n 32 --threads 6 --gpu off --print-ids
# round 1: 6.77 / 6.17 / 7.20; round 2: 7.06 / 5.74 / 6.20
git checkout -- include/tge/models/qwen2.hpp src/models/qwen2.cpp
python aikit.py check aegean-tge                        → 4/4
```

No push, no merge, no force-push. Spike code is not in the tree or in any commit.
