# INVENTED: Combined kq8d decode tier (LM head + FFN down) — KEEP opt-in, f32-only (raw 1.51x/2.09x, ~15% drift-normalized)

- **Idea**: Combine the two measured kq8d decode scopes — the LM head (cycle 29, 3.3–3.6x on its matvec) and the FFN down projection (cycle 19, +10–36% decode) — and price the stack. They touch different matrices, so the question is whether fidelity still clears the class bar and whether the speed adds up.
- **Cycle date**: 2026-11-08
- **Verdict**: **KEEP as an opt-in f32 decode tier.** Raw decode is 1.51x and 2.09x faster in the two interleaved pairs with identical ids; the direct head timer confirms 3.15–3.67x on the head matvec, and the drift-normalized decode gain (decode time ÷ unaffected attention time) is **~15% in both pairs**. Fidelity: f32 **9.711e-4 / 390/399 (97.74%)** — at the VNNI-class bar; f16 is one flip short (389/399), i8 two short (388/399), so the combo is **f32-only**.
- **Tree state**: spike reverted from `include/tge/models/qwen2.hpp`, `src/models/qwen2.cpp`, `apps/cli/main.cpp` (including the temporary head timer); 145/145 baseline tests rebuilt and green; `aikit.py check` 4/4; no branch, no spike commit; nothing pushed.

## 1. Mechanism

The spike re-applies the cycle-19/29 plumbing in one binary:

- `Qwen2::matvec(..., bool allow_kq8d)` routes Q4_K/Q6_K single-row products to `gemv_dequant_kq8_row_avx2` (Q8_K activation × integer weights).
- `TGE_KQ8D_HEAD=1` enables it for `output_logits` (the 3072×128256 Q6_K LM head).
- `TGE_KQ8D_SCOPE=down` enables it for `w_down` in the single-token `forward` only.
- Prefill/`batch_matvec`, attention, gate/up and the KV tiers are untouched.

A temporary `phase_head_ns_` timer wraps only the `output_logits` matvec so the head gain can be measured directly, independent of overall box drift.

## 2. Falsifier

- **KEEP as opt-in** if the 399-position mean KL stays ≤1.6e-3 with top-1 ≥97.6%-equivalent, ids are identical, and decode is faster in both interleaved pairs.
- **Restrict the tier** if a KV tier drops below the top-1 precedent.
- **KILL** if fidelity leaves the bar or the head timer loses its gain when stacked with the down scope.

## 3. Measured numbers

### 399-position fidelity (context 1024)

| KV | mean KL | worst KL | top-1 |
|---|---:|---:|---:|
| f32, combo | **9.711e-4** | 1.682e-2 | **390/399 (97.74%)** |
| f16, combo | 9.166e-4 | 1.031e-2 | 389/399 (97.49%) |
| i8, combo | 1.210e-3 | 9.156e-3 | 388/399 (97.24%) |

Reference points from the single scopes: head-only f32 1.874e-4 / 393/399; down-only f32 7.954e-4 / 390/399. The combo's f32 result is exactly the down-only top-1 with the head error added; it sits at the bar. f16 and i8 each lose one further flip, so the combo is **f32-only**.

### Decode A/B (2048 prompt, n=128, interleaved order)

| pair | baseline | combo | ratio | head off → on | ids |
|---:|---:|---:|---:|---:|---|
| 1 | 2.67 t/s | 4.04 t/s | **1.513x** | 4,736 → 1,505 ms (3.15x) | IDENTICAL |
| 2 | 1.86 t/s | 3.89 t/s | **2.091x** | 5,562 → 1,517 ms (3.67x) | IDENTICAL |

The box drifted between arms; the unaffected attention phase was 8,849 → 6,897 ms in pair 1 (combo window 1.28x faster) and 12,674 → 7,093 ms in pair 2 (1.79x faster). Dividing total decode time by attention time (a same-run load control) gives:

| pair | baseline total/attn | combo total/attn | normalized combo gain |
|---:|---:|---:|---:|
| 1 | 5.377 | 4.558 | **15.2%** |
| 2 | 5.388 | 4.601 | **14.6%** |

So the honest speed statement is: both pairs faster, head directly 3.15–3.67x, drift-normalized decode ~15%. The raw 1.5–2.1x magnitudes are not a quiet-box claim.

## 4. Interpretation

- The two scopes are additive in mechanism (different matrices) and both pairs confirm the direction.
- The head dominates the combo: in this window the head is 4.7–5.6 s of a 5.5–7.5 s decode budget baseline, falling to ~1.5 s.
- The down scope's own gain is smaller and was measured on a different window; the combo's normalized ~15% is consistent with the head's ~20% of matvec share being cut by two-thirds.
- Fidelity is the limiter: the combo doubles the down-only KL and sits exactly at the top-1 bar, so it must stay f32-only and opt-in.

## 5. Reimplementation recipe (exact)

1. Re-use the cycle-19 `allow_kq8d` flag and the cycle-29 head call site.
2. Add a single-token `forward` scope flag (`TGE_KQ8D_SCOPE=down` for the spike; production: a `--kernel kq8d=head,down` flag) and pass it only at the `w_down` matvec.
3. Leave prefill, attention, gate/up and the output `batch_matvec` on the float path.
4. Gate: 399-position f32 fidelity (mean KL ≤1.6e-3, top-1 ≥97.6%-equivalent), ids identical, head timer ≥2x, two interleaved decode pairs faster, 145/145, 4/4.
5. Ship only with f32 KV; do not enable f16/i8 until a larger corpus resolves the one/two-flip top-1 gaps.

## 6. Disposition

**KEEP as an opt-in f32 decode tier, report-only.** The combo is the fastest measured decode configuration in the library (raw 1.5–2.1x, normalized ~15%, head 3.2–3.7x) at the fidelity bar. It should not be a default: the top-1 margin is zero at 399 positions, and the i8/f16 stacks fall below the precedent. A default-on decision needs a larger corpus and the human gate.

## 7. Exact commands

```text
cmake --build --preset release                         → 54/54
.\build\release\bin\tge_tests.exe                      → 145/145
TGE_KQ8D_HEAD=1 TGE_KQ8D_SCOPE=down tge fidelity --model Llama-3B \
  --corpus docs/architecture.md --context 1024 --tokens 400 --threads 6 \
  --gpu off --kv f32|f16|i8
TGE_KQ8D_HEAD=1 TGE_KQ8D_SCOPE=down tge generate --model Llama-3B \
  --tokens <2048 ids> --n 128 --threads 6 --gpu off
git checkout -- include/tge/models/qwen2.hpp src/models/qwen2.cpp apps/cli/main.cpp
python aikit.py check aegean-tge                        → 4/4
```

No push, no merge, no force-push. Spike code is not in the tree or in any commit.
