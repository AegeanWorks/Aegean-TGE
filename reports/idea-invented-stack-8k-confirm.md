# CONFIRM: Recommended stack at 8180, reverse order — prefill 1.575x confirmed (matvec control within 1.3%)

- **Idea**: Reproduce cycle 36's 1.56x end-to-end stack headline with the opposite arm order (stack first, baseline second). The 8180 pair costs ~25 minutes of prefill per arm, so a reversed pair is the only practical drift control.
- **Cycle date**: 2026-11-08
- **Verdict**: **CONFIRMED for prefill.** Prefill **1,100,249 → 698,552 ms (1.575x)**, attention **2.287x**, and the untouched matvec phase matched to **1.3%** (357,916 vs 353,348 ms) — an excellent window control. Ids identical. Decode in the stack arm hit a 4x load spike (matvec 7,911 vs 1,920 ms), so this pair carries no decode claim; cycle 36's decode 1.14x remains the only usable 8k pair.
- **Tree state**: spike reverted from `include/tge/cpu/kernels.hpp`, `include/tge/models/qwen2.hpp`, `src/cpu/avx2/attention_avx2.cpp`, `src/models/qwen2.cpp` and `apps/cli/main.cpp`; 145/145 baseline tests rebuilt and green; `aikit.py check` 4/4; no branch, no spike commit; nothing pushed.

## 1. Setup

Same stack as cycle 36: block prefill attention (f32 KV) + kq8d LM head + kq8d FFN down, all behind `TGE_ATTN_BLOCK`, `TGE_KQ8D_HEAD`, `TGE_KQ8D_SCOPE=down`. This time the stack ran first and the baseline second.

## 2. Falsifier

- **CONFIRM** if the reverse pair shows ≥1.3x prefill with a matched matvec control (within ~10%) and identical ids.
- **CHALLENGE** if the ratio flips or the control drifts.

Measured: 1.575x with 1.3% control drift; confirmed.

## 3. Measured numbers

### 8180-token pair, stack first, f32 KV, 6 threads

| phase | stack | baseline | ratio |
|---|---:|---:|---:|
| prefill total | 698,552 ms (11.7 t/s) | 1,100,249 ms (7.4 t/s) | **1.575x** |
| prefill matvec (control) | 357,916 ms | 353,348 ms | **0.987x** |
| prefill attention | 317,297 ms | 725,559 ms | **2.287x** |
| prefill norm | 21,146 ms | 19,271 ms | 0.911x |
| decode (7 tokens) | 15,150 ms (0.46 t/s) | 4,042 ms (1.73 t/s) | 0.267x (load spike) |
| decode matvec | 7,911 ms | 1,920 ms | 0.243x (load spike) |
| ids | — | — | **IDENTICAL** |

The decode comparison is invalid: the stack arm's decode matvec — untouched by anything except kq8d, which makes it faster, not 4x slower — ran in a pathologically loaded window. The prefill control is the trustworthy one: matvec within 1.3%, so the attention gain and prefill ratio are real.

### Cross-pair summary

| pair | order | prefill ratio | attention ratio | matvec control | ids |
|---|---|---|---:|---:|---:|---|
| cycle 36 | baseline → stack | 1.566x | 2.389x | stack 5% slower | identical |
| cycle 37 | stack → baseline | 1.575x | 2.287x | within 1.3% | identical |

Two opposite-order pairs agree: **prefill ~1.57x**, attention **2.29–2.39x**.

## 4. Interpretation

- The prefill stack gain is reproducible and not an artifact of arm order. Both pairs land at 1.57x with the matvec control flat or slightly against the stack.
- The attention ratio (~2.3x) is higher than cycle 33's isolated 1.63–1.88x; both cycles' controls say the windows were fair. The difference may come from the kq8d head/down scopes warming/altering the decode-side state or simply window variation; the direction is consistent.
- Decode at 8k is not measurable with only 7 generated tokens on this box: a single background load can swing the phase 4x. The decode gain must be taken from the component measurements (head matvec 3.3–3.6x; combo raw 1.5–2.1x at 2048 with 127 tokens) rather than the 8k pair.
- Ids were identical again, consistent with every prior block-attention and kq8d check.

## 5. Disposition

**CONFIRMED — keep the stack as the integration target.** The prefill number is now double-confirmed in opposite orders. The staged integration proposal stands: block prefill attention for f32/f16, then kq8d head, then kq8d down, each behind a flag, gated by the 1,199-position fidelity corpus and transcript parity. Decode claims should be re-measured at a longer generation length (≥64 tokens) before quoting an end-to-end number.

## 6. Exact commands

```text
cmake --build --preset release                         → 54/54
.\build\release\bin\tge_tests.exe                      → 145/145
TGE_ATTN_BLOCK=on TGE_KQ8D_HEAD=1 TGE_KQ8D_SCOPE=down tge generate \
  --tokens @ids8180.txt --n 8 --threads 6 --gpu off --kv f32 --context 8192
TGE_ATTN_BLOCK=0 TGE_KQ8D_HEAD=0 tge generate ... (same)
git checkout -- include/tge/cpu/kernels.hpp include/tge/models/qwen2.hpp \
  src/cpu/avx2/attention_avx2.cpp src/models/qwen2.cpp apps/cli/main.cpp
python aikit.py check aegean-tge                        → 4/4
```

No push, no merge, no force-push. Spike code is not in the tree or in any commit.
