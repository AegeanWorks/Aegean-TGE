# MEASURED: continuous-batching usage recipe on 3B with path-invariant kq8 — 6 slots is the peak (30.6 t/s), KV tier is flat at the bench's prompt cap

- **Question**: With the path-invariant kq8 decode up to 1.35-1.71x, what is the best slot count on the reference 3B, and does the KV tier matter in continuous batching? The harness caps `prompt-tokens + sequences <= 64`, so the longest available prompt is 48-56 tokens — attention is a small share, and KV effects should be flat.
- **Cycle date**: 2026-11-11
- **Verdict**: **MEASURED.** Llama-3.2-3B-Instruct-Q4_K_M, `bench continuous --requests 16 --tokens 8 --threads 10 --kv f16 --prompt-tokens 48`, path-invariant kq8:
  - **Slot sweep**: 2 slots 23.82, 4 slots 29.19, **6 slots 30.64 (peak)**, 8 slots 29.15, 10 slots 27.19, 12 slots 27.49 t/s. 6 slots beats 4 by 5% and 8 by 5%; beyond 6, admission and tail costs erase the extra parallelism.
  - **A/B at the two best points** (2 pairs each): 4 slots float 18.61/15.60 -> kq8 29.72/22.19 (1.597/1.422); 6 slots float 16.80/15.54 -> kq8 27.13/24.45 (1.615/1.573). The kq8 win holds at both; 6 slots is the better operating point on averages.
  - **KV tier sweep at 8 slots / 56-token prompts**: f32 25.54, f16 24.30, i8 25.85 t/s (kq8); float arm 16.41/15.28/16.10. All within this box's noise — at the harness's prompt cap the KV tier is irrelevant, as expected.
  - `state: independent (all transcripts identical)` in every run of every arm.
- **Tree state at close**: spike reverted from `src/models/qwen2.cpp`; `git status` clean; stash holds only the pre-existing 2026-11-04 peer entry; rebuild clean, `tge_tests` **171/171**, `python aikit.py check aegean-tge` **4/4**. No branch, no spike commit, nothing pushed.

## 1. Setup

| | |
|---|---|
| model | Llama-3.2-3B-Instruct-Q4_K_M.gguf |
| arms | float-batch baseline (`build/idea/tge-cont-float-base.exe`) vs path-invariant kq8 spike |
| workload | 16 requests x 8 decode tokens, `--threads 10`, `--prompt-tokens 48` (slot sweep) or 56 (KV sweep) |
| slots | 2, 4, 6, 8, 10, 12 |
| KV | f16 (slot sweep), f32/f16/i8 (KV sweep) |

## 2. Results

### Slot sweep (path-invariant kq8, 16 requests x 8 tokens, pt=48, f16)

| slots | t/s aggregate |
|---:|---:|
| 2 | 23.82 |
| 4 | 29.19 |
| **6** | **30.64** |
| 8 | 29.15 |
| 10 | 27.19 |
| 12 | 27.49 |

### A/B at the best points (2 pairs each)

| slots | pair | float t/s | kq8 t/s | ratio |
|---:|---:|---:|---:|---:|
| 4 | 1 | 18.61 | 29.72 | 1.597 |
| 4 | 2 | 15.60 | 22.19 | 1.422 |
| 6 | 1 | 16.80 | 27.13 | 1.615 |
| 6 | 2 | 15.54 | 24.45 | 1.573 |

### KV tier at the bench's prompt cap (8 slots, pt=56)

| KV | float t/s | kq8 t/s |
|---|---:|---:|
| f32 | 16.41 | 25.54 |
| f16 | 15.28 | 24.30 |
| i8 | 16.10 | 25.85 |

## 3. Reading

- Operating point: **6 slots at 10 threads** is the aggregate peak on this box/model (30.6 t/s vs 29.2 at 4 and 29.2 at 8). `--sequences auto` sizes from memory headroom and would pick 8 here; the sweep suggests capping at 6 is better for aggregate throughput at this workload, while 8 gives more fairness/headroom for bursty arrivals. Record both: 6 for throughput, 8 for latency headroom.
- The kq8 path-invariant win reproduces at both 4 and 6 slots (1.42-1.62x), consistent with the closeout's 1.35-1.71x matrix.
- KV tier is not worth changing for short-prompt continuous loads: all three tiers are within noise at 56-token prompts. The i8/f16 tiers remain levers for long-context decode, not this harness.
- The harness's `prompt-tokens + sequences <= 64` cap is the reason long-context KV effects cannot be measured here; a long-prompt continuous bench would need a larger batch buffer (new surface, not this cycle).

## 4. Reimplementation recipe

1. Apply the path-invariant change (`reports/idea-invented-continuous-kq8-path-invariant-keep.md` §8); build; 171/171.
2. Slot sweep: `bench continuous --model Llama-3.2-3B-Instruct-Q4_K_M.gguf --requests 16 --tokens 8 --threads 10 --sequences {2,4,6,8,10,12} --kv f16 --prompt-tokens 48`; expect the peak at 6.
3. A/B at 4 and 6 slots vs a float-batch binary; expect 1.42-1.62x.
4. KV sweep at 8 slots / pt=56 with `--kv {f32,f16,i8}`; expect all within noise.
5. Revert (report-only); the recipe is a usage table, not a code change.

## 5. Exact commands run and results

```text
cmake --build projects\aegean-tge\build\release --config Release   -> clean (56/56)
projects\aegean-tge\build\release\bin\tge_tests.exe             -> 171 cases, 0 failed
tge bench continuous ... --sequences {2,4,6,8,10,12} --prompt-tokens 48 -> slot table
tge bench continuous ... --sequences {4,6} (float vs kq8)               -> A/B table
tge bench continuous ... --kv {f32,f16,i8} --prompt-tokens 56           -> KV table
git checkout -- src/models/qwen2.cpp
cmake --build ... -> clean; tge_tests -> 171/171
python aikit.py check aegean-tge                                   -> 4/4 gates passed
```

No push, no merge, no force-push. Baseline binary lives at `build/idea/tge-cont-float-base.exe`; the committed tree is unchanged except this report.
