# INVENTED: Recommended stack end-to-end at 8180 — KEEP (1.56x total; prefill 1.57x, decode 1.14x, ids identical)

- **Idea**: Measure the combined stack from the report library in one binary: query-block prefill attention (cycles 33/34) plus kq8d on the LM head and FFN down (cycles 19/29/31). Component gains were measured separately; this cycle prices the whole configuration end-to-end at 8180 tokens.
- **Cycle date**: 2026-11-08
- **Verdict**: **KEEP as the integration target.** Prefill **691,979 → 442,070 ms (1.566x)** at 8180 tokens; decode **2.30 → 2.64 t/s (1.144x)** for 7 tokens; total wall (prefill + decode) **695.0 → 444.7 s (1.563x)**. Ids are identical on the 8180 prompt and on a 2048 prompt with 32 generated tokens. The untouched matvec phase stayed within 5%, so the window was fair.
- **Tree state**: spike reverted from `include/tge/cpu/kernels.hpp`, `include/tge/models/qwen2.hpp`, `src/cpu/avx2/attention_avx2.cpp`, `src/models/qwen2.cpp` and `apps/cli/main.cpp`; 145/145 baseline tests rebuilt and green; `aikit.py check` 4/4; no branch, no spike commit; nothing pushed.

## 1. Components

| component | scope | prior evidence |
|---|---|---|
| Block prefill attention | f32 KV, 64-token query blocks, head-parallel | cycle 33: attention 1.63–1.88x at 8k, ids identical |
| kq8d LM head | `output_logits` only, Q6_K integer row kernel | cycle 29: head matvec 3.3–3.6x, decode +33–39% |
| kq8d FFN down | single-token `forward`, `w_down` only | cycle 19: decode +10–36% |
| Combined kq8d | head + down | cycle 31: f32 8.283e-4 KL, top-1 98.00% at 1,199 positions, 3/3 prompts × 64 ids identical |

All three are opt-in and report-only. This cycle re-applies them behind `TGE_ATTN_BLOCK`, `TGE_KQ8D_HEAD` and `TGE_KQ8D_SCOPE=down`.

## 2. Falsifier

- **KEEP the stack** if 8180-token end-to-end wall is ≥1.3x faster with identical ids and the untouched phase shows no large window drift.
- **KILL** if the gains do not compose or ids move.

Measured: 1.56x; ids identical.

## 3. Measured numbers

### 8180-token end-to-end, f32 KV, 6 threads

| phase | baseline | stack | ratio |
|---|---:|---:|---:|
| prefill total | 691,979 ms (11.8 t/s) | 442,070 ms (18.5 t/s) | **1.566x** |
| prefill matvec | 221,972 ms | 233,488 ms | 0.951x (stack 5% slower) |
| prefill attention | 452,959 ms | 189,561 ms | **2.389x** |
| prefill norm | 15,224 ms | 16,944 ms | 0.899x |
| decode (7 tokens) | 3,038 ms (2.30 t/s) | 2,655 ms (2.64 t/s) | **1.144x** |
| decode matvec | 1,272 ms | 974 ms | **1.306x** |
| decode attention | 1,731 ms | 1,649 ms | 1.050x |
| total wall | 695.0 s | 444.7 s | **1.563x** |
| ids | — | — | **IDENTICAL** |

The matvec phase (untouched by block attention) was 5% *slower* in the stack arm, so the prefill ratio is not window-inflated. Decode matvec improved 1.31x from kq8d head+down; decode attention is within noise (the head/down scopes do not touch it).

### Id parity

- 8180 prompt, `--n 8`: **identical**.
- 2048 prompt, `--n 32`: **identical** (32/32).

## 4. Interpretation

- The stack composes: prefill gains come from block attention (2.39x in this window), decode gains from the kq8d head and down scopes (1.31x matvec).
- The prefill attention ratio (2.39x) is higher than cycle 33's 1.63–1.88x; the matvec control stayed within 5%, so this window was fair, but treat 1.56x as a window-specific headline rather than a guarantee. The direction and component magnitudes are consistent with the prior cycles.
- Decode is only 1.14x at 8k because attention is ~62% of the decode phase there and the kq8d scopes touch matvec only. At 2048 decode, where attention is ~20%, the head+down gain is larger (cycle 29/30).
- Fidelity of the stack is the kq8d combo's (f32 8.283e-4, top-1 98.00% at 1,199 positions); block attention changes only accumulation order and was id-identical in cycles 33/34 and again here.

## 5. Disposition

**KEEP as the recommended integration target, report-only.** Enabling the three spikes together gives ~1.56x end-to-end at 8k on f32 KV with identical ids. Integration should be staged: block prefill attention for f32/f16, then kq8d head, then kq8d down, each behind a flag, with the cycle-31 fidelity corpus as the acceptance gate. The human gate still decides.

## 6. Exact commands

```text
cmake --build --preset release                         → 54/54
.\build\release\bin\tge_tests.exe                      → 145/145
TGE_ATTN_BLOCK=0 TGE_KQ8D_HEAD=0 tge generate --model Llama-3B \
  --tokens @ids8180.txt --n 8 --threads 6 --gpu off --kv f32 --context 8192
TGE_ATTN_BLOCK=on TGE_KQ8D_HEAD=1 TGE_KQ8D_SCOPE=down tge generate ... (same)
git checkout -- include/tge/cpu/kernels.hpp include/tge/models/qwen2.hpp \
  src/cpu/avx2/attention_avx2.cpp src/models/qwen2.cpp apps/cli/main.cpp
python aikit.py check aegean-tge                        → 4/4
```

No push, no merge, no force-push. Spike code is not in the tree or in any commit.
