# MEASURED: fused vs non-fused speculation at 8k is a wash (1.01x median) — the fused chain's 2k advantage does not carry, and its extra draft stops paying

- **Idea**: The fused greedy chain (bonus row + five drafts, one six-row verify, no correction forward) beat the non-fused path by 1.29x at 2k (`idea-invented-fused-lock-six-draft-killed.md`: 49.16 vs 38.22 t/s). It is unconditional. At 8k the correction forward the fusion deletes is a full single-row pass over an 8k KV, so the heuristic should matter more — or, if the non-fused path's sixth draft accepts better, it may flip. A/B `--speculative` vs `--speculative --no-fuse` on the reference 3B at 8,193 tokens, same binary.
- **Cycle date**: 2026-11-13
- **Verdict**: **MEASURED — the two paths are equivalent at 8k (0.98-1.04x, median ~1.01x); the fused heuristic is neutral, not load-bearing.** Llama-3.2-3B-Instruct-Q4_K_M, 8,193-token periodic prompt, `--context 8704 --n 32 --kv f16 --threads 10 --gpu off --speculative`, interleaved:
  - **pair 1**: fused **9.35 t/s** (7 steps, 28 accepted, **4.00/step**, 34 emitted) vs no-fuse **9.76 t/s** (6 steps, 30 accepted, **5.00/step**, 35 emitted) -> no-fuse **1.044x**.
  - **pair 2**: no-fuse **9.99 t/s** (same 6/30 line) vs fused **10.19 t/s** (same 7/28 line) -> fused **1.020x**.
  - **They do the same row work**: fused = 7 steps x 6 rows = 42 row-equivalents; no-fuse = 6 steps x (6-row verify + 1-row correction) = 42. The paths trade one verified row per step against one extra accepted draft, and at 8k the trade is exactly even.
  - **The 2k advantage came from draft yield, not from the missing correction.** At 2k the fused chain accepted 5.05/step and the non-fused 6.00; at 8k the non-fused matcher still accepts 5.00/step while the fused row only reaches 4.00, so the non-fused path's 7-token steps offset its correction forward. The heuristic should not be marketed as an 8k win.
  - **No decision change**: keep the default (fused) since it is neutral at 8k and measured faster at 2k; do not recommend `--no-fuse` for long context on this evidence. Do not attribute 8k spec speed to the fused chain.
- **Tree state at close**: no code touched (both paths shipped); `git status` clean; stash holds only the pre-existing 2026-11-04 peer entry; `tge_tests` **171/171**, `python aikit.py check aegean-tge` **4/4**. No branch, nothing pushed.

## 1. Setup

| | |
|---|---|
| model | Llama-3.2-3B-Instruct-Q4_K_M.gguf |
| prompt | `The fox runs. ` x 2048 = 8,193 tokens |
| context / n / KV / threads / GPU | 8704 / 32 / f16 / 10 / off |
| arms | default (fused), `--no-fuse` (baseline path, same binary) |
| runs | 2 per arm, interleaved |

The fused path is `apps/cli/main.cpp:788-922`: `feed[0]` = last emitted token (bonus), `feed[1..5]` = five prompt-lookup drafts, one six-row verify, emit `accepted + 1`, chain on full accept. The non-fused greedy path (`:924-1070`) drafts six, verifies six, then pays a one-row correction forward after every step. `--no-fuse` is the documented same-binary A/B switch (`:784`, `:2169`).

One caveat: the first `--no-fuse` attempt loaded only **6,453 context** at `--context 8704` under transient memory pressure and exited before decode; the retry (after the prior process released memory) was clean and is the run reported. 8k runs on this 16 GB box occasionally need the retry.

## 2. Results

| pair | arm | decode t/s | spec line | tokens/step | steps x rows |
|---|---|---:|---|---:|---:|
| 1 | fused | 9.35 | 7 steps, 28 acc (4.00/step), 34 emitted | 4.86 | 7 x 6 = 42 |
| 1 | no-fuse | **9.76** | 6 steps, 30 acc (5.00/step), 35 emitted | 5.83 | 6 x (6+1) = 42 |
| 2 | no-fuse | 9.99 | 6 steps, 30 acc (5.00/step), 35 emitted | 5.83 | 42 |
| 2 | fused | **10.19** | 7 steps, 28 acc (4.00/step), 34 emitted | 4.86 | 42 |

Ratios: no-fuse/fused = 1.044 (pair 1), 0.980 (pair 2); median **1.01x**. Both arms are stable per path (fused 9.35/10.19, no-fuse 9.76/9.99), and the spec lines are identical within each arm, so the comparison is clean.

## 3. Reading

- **The fusion is neutral at 8k by construction of the trade**: one saved single-row correction (the fused benefit) against one fewer accepted draft per step (fused 4.00 vs non-fused 5.00 on this prompt). Both paths perform 42 row-equivalents for 32-35 tokens; the wall difference is under 5%.
- **Why 2k disagreed**: at 2k the fused chain accepted 5.05/step against the non-fused 6.00, and the missing correction forward dominated (49.16 vs 38.22 t/s, 1.29x). The acceptance gap at 8k is larger (4.00 vs 5.00) because the fused five-draft row drops the sixth continuation that the non-fused path gets; that is what cancels the 2k benefit.
- **Recommendation**: keep the default fused path (neutral at 8k, faster at 2k). Do not claim the fused chain as an 8k optimization, and do not recommend `--no-fuse` for long context; if a future long-context drafter raises fused acceptance toward 5-6/step, the fusion would regain its 2k edge — that is the reopen trigger.
- **Greedy 8k spec remains 1.4-1.6x** (the 9.4-10.2 t/s arms here vs the 5.2-5.7 t/s dense off from `idea-invented-spec-at-8k-measured.md`). The wash is between two speculative variants, not against dense.
- **Falsifier recorded for the library**: any fused-chain claim must show a >= 1.10x advantage over `--no-fuse` on the target workload. At 8k it fails that bar.

## 4. Falsifier (stated before the runs)

**KEEP the fused default as an 8k win** if fused >= 1.10x no-fuse. **RECOMMEND `--no-fuse` at 8k** if no-fuse >= 1.10x fused. **RECORD NEUTRALITY** otherwise. Measured 1.044x the other way and 0.980x back: neutral; keep the default, withdraw the 8k attribution.

## 5. File anchors (current HEAD, read-before-write verified)

- `apps/cli/main.cpp:788-922` — fused greedy chain (bonus row + five drafts, 6-row verify, `position = vpos + 7` on full accept).
- `apps/cli/main.cpp:924-1070` — non-fused greedy path (six drafts, 6-row verify, correction forward, `accepted == 6` chain at `:1047`).
- `apps/cli/main.cpp:2169-2171` — `--no-fuse`; `:2334-2336` — `TGE_NO_FUSE=1` env fallback.
- `include/tge/models/qwen2.hpp:180` — `kSpeculativeBatch = 6`.
- `reports/idea-invented-fused-lock-six-draft-killed.md` — the 2k fused measurement this cycle re-prices at 8k.
- `reports/idea-invented-spec-at-8k-measured.md` — the dense off baseline (5.21-5.68 t/s).

## 6. Reimplementation recipe

1. Compact 8k prompt: `$p = ("The fox runs. " * 2048)`.
2. Interleave the arms (each ~4-5 min of prefill):
   ```powershell
   tge generate --model Llama-3.2-3B-Instruct-Q4_K_M.gguf --prompt $p \
     --context 8704 --n 32 --kv f16 --threads 10 --gpu off --speculative [--no-fuse]
   ```
3. Read `decode :` and `spec :`. Expect fused 9.35-10.19 t/s with 7 steps / 4.00 accepted per step; no-fuse 9.76-9.99 t/s with 6 steps / 5.00 accepted per step. If the load line reports fewer than 8,704 context available, rerun after the prior process releases memory.
4. Compare within-path stability first: the spec lines should be identical across repeats of an arm; if not, the box drifted.
5. No code ships.

## 7. Exact commands run and results

```text
cmake --build --preset release                    -> clean (56/56, cycle gate)
build\release\bin\tge_tests.exe                   -> 171 cases, 0 failed

8k (compact fox x2048, ctx 8704, n 32, f16, 10 threads, gpu off, --speculative):
  pair 1: fused   9.35 t/s (7 steps, 28 acc, 4.00/step, 34 emitted)
          no-fuse 9.76 t/s (6 steps, 30 acc, 5.00/step, 35 emitted)   -> 1.044x
  pair 2: no-fuse 9.99 t/s (6 steps, 30 acc, 5.00/step, 35 emitted)
          fused  10.19 t/s (7 steps, 28 acc, 4.00/step, 34 emitted)   -> 0.980x
  (first no-fuse attempt failed at 6453 context under memory pressure; retried clean)
git status --short                                -> clean (no code touched)
python aikit.py check aegean-tge                  -> 4/4 gates passed
```

No push, no merge, no force-push. The committed tree is unchanged except this report.
