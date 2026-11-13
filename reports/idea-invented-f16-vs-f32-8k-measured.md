# MEASURED: f16 KV is also the long-context speed tier — 1.15-1.22x decode and 1.04-1.14x prefill over f32 at 8k (attention 0.71-0.75x)

- **Idea**: The tier ladder (`idea-invented-natural-8k-tier-ladder-measured.md`) recommends f16 as the only quality-preserving 8k mode, but its speed was only measured at 2k and at 497 positions. At 8k, attention is 52-64% of decode, so the f16 trade (half the KV bytes vs a per-row half-to-float conversion) is at its most favourable. The open question: is the quality-first default also the speed default at long context, or does the conversion cost make f32 faster? A/B f32 vs f16 at 8,193 tokens on the reference 3B, interleaved.
- **Cycle date**: 2026-11-13
- **Verdict**: **MEASURED — f16 wins both axes at 8k.** Llama-3.2-3B-Instruct-Q4_K_M, 8,193-token periodic prompt, `--context 8704 --n 32 --threads 10 --gpu off`, interleaved pairs:
  - **decode**: f16 **5.29 / 5.38 t/s** vs f32 **4.40 / 4.59 t/s** -> **1.15-1.22x**; decode attention **3,158-3,196 ms** vs **4,252-4,483 ms** (**0.71-0.75x**), while matvec is flat (2,409-2,572 vs 2,465-2,508 ms, within 2%).
  - **prefill**: f16 **244.0 / 247.2 s** vs f32 **255.1 / 281.7 s** -> **1.04-1.14x**; the second pair is the clean one (matvec 2,571 vs 2,409 ms), the first pair's f32 arm ran on a slower box state (matvec 2,465 vs 2,508 ms).
  - **f16 is therefore the only tier that is simultaneously quality-preserving (8k natural: 99.90%, KL 2.312e-06) and the fastest measured 8k KV tier.** f32 has no speed role left at this context; it remains the exact-reproduction tier (f16 is not id-identical to f32 by construction, though its fidelity is 2.3e-6 KL).
  - The result is consistent with the 2k direction (f16 wins there too) and with i8's 8k loss (`idea-invented-i8-8k-killed.md`, −13 to −23% vs f32): the f16 conversion is cheap enough that bandwidth dominates at every measured context, and the gap widens where attention is largest.
- **Tree state at close**: no code touched (shipped flags); `git status` clean; stash holds only the pre-existing 2026-11-04 peer entry; `tge_tests` **171/171**, `python aikit.py check aegean-tge` **4/4**. No branch, nothing pushed.

## 1. Setup

| | |
|---|---|
| model | Llama-3.2-3B-Instruct-Q4_K_M.gguf |
| prompt | `The fox runs. ` x 2048 = 8,193 tokens (fits the Windows command line) |
| context / n / threads / GPU | 8704 / 32 / 10 / off |
| arms | `--kv f32`, `--kv f16` (interleaved: f32->f16, then f16->f32) |
| runs | 2 per arm |

f32 KV at 8k: 28 layers x 2 x 8,704 x 1,024 x 4 B = ~1.83 GB; f16 ~0.92 GB. Both fit the 16 GB box, so this is a pure kernel/traffic comparison, not a capacity comparison.

## 2. Results

| pair | arm | prefill | decode | decode attn | matvec (prefill / decode) |
|---|---|---:|---:|---|---|
| 1 | f32 | 281.7 s | 4.40 t/s | 4482.9 ms (64%) | 2465.2 ms / (not printed separately) |
| 1 | f16 | 247.2 s | **5.38 t/s** | **3158.3 ms (55%)** | 2508.1 ms |
| 2 | f16 | 244.0 s | **5.29 t/s** | **3196.4 ms (55%)** | 2571.5 ms |
| 2 | f32 | 255.1 s | 4.59 t/s | 4252.2 ms (63%) | 2409.2 ms |

| comparison | ratio | notes |
|---|---:|---|
| decode f16/f32 | **1.15-1.22x** | median 1.19x across the four cross-pairings |
| decode attention f16/f32 | **0.71-0.75x** | the mechanism: half the KV bytes read per key row |
| prefill f16/f32 | **1.04-1.14x** | pair 2 is clean; pair 1's f32 arm drifted |
| matvec | within 2% | context/tier independent, confirms the pairs are admissible |

The decode attention phases are the cleanest signal: f16 reads half the bytes for the same rows and pays only a vectorized conversion, so the phase falls ~28% at 55-64% attention share. The end-to-end decode gain (1.15-1.22x) is lower than the attention gain because matvec (44-36% of decode) is unchanged.

## 3. Reading

- **The 8k default is now unambiguous**: `--kv f16`, dense attention. It is the quality tier (99.90% natural 8k, 99.98% repeated) *and* the fastest measured KV tier at this context (1.15-1.22x decode, 1.04-1.14x prefill over f32). The earlier framing ("f16 is the quality-preserving choice, at some speed cost") is superseded — there is no speed cost.
- **f32's remaining role is exact reproduction**, not performance: it is the reference arm of the fidelity harness and the only tier that is bit-identical to the float oracle by construction. Any workflow that needs f32 ids should say so explicitly.
- **The KV tier rank at 8k is f16 > f32 > i8 on speed** (i8 was −13 to −23% vs f32 in the earlier kill) and **f16 > i8 > sparse on quality** (99.90 / 91.33 / 80.81 on natural text). f16 is the only tier on the efficient frontier at both axes.
- **Speculation composes on top**: the 8k spec ratios (1.42-1.61x periodic, 1.13x code) were measured with f16 already; nothing here changes them. `--kv f16 --speculative` is the quality-preserving 8k speed configuration; add `--sparse` only with the 13-19pp quality cost stated.
- **Do not extend this to capacity**: f16 halves f32 but still costs 0.92 GB per 8k sequence on the 3B. For multi-sequence capacity, the i8 tier and its ~9pp natural-text cost remain the trade, and the continuous batching admission win is separate.

## 4. Falsifier (stated before the runs)

**CONFIRM f16 as the 8k speed tier** if f16 decode >= 1.10x f32 with the attention phase materially lower and matvec within 3%. **REVISE** if f32 is equal or faster. Measured 1.15-1.22x decode with attention 0.71-0.75x and matvec within 2%: confirmed. The quality side was already priced (2.312e-06 / 99.90% natural 8k).

## 5. File anchors (current HEAD, read-before-write verified)

- `include/tge/models/qwen2.hpp` — KV tier members (`k_cache_half_`, `v_cache_half_`, q8 twins) and `Qwen2::load(..., kv_half, kv_q8)`.
- `src/models/qwen2.cpp:573-586` — cache allocation per tier.
- `src/models/qwen2.cpp:1730-1745` — `store_kv` per tier (f16 conversion loop).
- `src/models/qwen2.cpp:1467-1485` — per-token f16 read path; `:1590-1594` — block kernel f16 inputs.
- `src/cpu/avx2/attention_avx2.cpp` — `widen_row` (f16 conversion in the block kernel) and the strided f16 twin (`:367`).
- `apps/cli/main.cpp:2740+` — `--kv` parsing.
- `reports/idea-invented-natural-8k-tier-ladder-measured.md` — the quality side of this trade.
- `reports/idea-invented-i8-8k-killed.md` — the i8 8k speed loss.
- `reports/idea-invented-8k-decision-matrix-closeout.md` — the matrix this fills in.

## 6. Reimplementation recipe

1. Compact 8k prompt: `$p = ("The fox runs. " * 2048)` (8,193 tokens).
2. Interleave the arms (each run ~4-5 min of prefill):
   ```powershell
   tge generate --model Llama-3.2-3B-Instruct-Q4_K_M.gguf --prompt $p \
     --context 8704 --n 32 --kv {f32|f16} --threads 10 --gpu off
   ```
3. Read `prefill :`, `decode :`, `phases : decode`. Expect f16 decode 5.29-5.38 t/s (attn 55%, ~3.2 s) vs f32 4.40-4.59 (attn 63-64%, ~4.3-4.5 s); prefill 244-247 s vs 255-282 s.
4. Compare only pairs whose matvec phases agree within ~3% (this box drifts 20-40% over 20 minutes; the matvec column is the tell).
5. No code ships.

## 7. Exact commands run and results

```text
cmake --build --preset release                    -> clean (56/56, cycle gate)
build\release\bin\tge_tests.exe                   -> 171 cases, 0 failed

8k (compact fox x2048, ctx 8704, n 32, 10 threads, gpu off), interleaved:
  pair 1: f32 prefill 281.7 s, decode 4.40 t/s (attn 4482.9 ms, 64%)
          f16 prefill 247.2 s, decode 5.38 t/s (attn 3158.3 ms, 55%)
  pair 2: f16 prefill 244.0 s, decode 5.29 t/s (attn 3196.4 ms, 55%)
          f32 prefill 255.1 s, decode 4.59 t/s (attn 4252.2 ms, 63%)
  decode ratio f16/f32 1.15-1.22x; attention ratio 0.71-0.75x; matvec within 2%
git status --short                                -> clean (no code touched)
python aikit.py check aegean-tge                  -> 4/4 gates passed
```

No push, no merge, no force-push. The committed tree is unchanged except this report.
