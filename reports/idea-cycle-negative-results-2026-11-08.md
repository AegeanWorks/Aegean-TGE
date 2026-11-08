# Negative cycle: adaptive speculation, 256-row chunks, Q4_K metadata cache

- **Cycle date**: 2026-11-08
- **Verdict**: three ideas measured, all reverted. One separate win shipped from the same cycle (f16 block-attention row-widening hoist, see the measurement log).
- **Tree state at close**: all three spikes reverted; `git diff` contains only the f16 attention hoist, its docs and the ledger. Tests 146/146, `aikit.py check aegean-tge` 4/4. Nothing pushed.

## 1. Adaptive greedy speculation depth - KILLED (median forward-time ratio 1.006)

**Idea**: stop paying for a six-row verifier when the drafter accepts only one token. Use the last accepted prefix to set the next verify depth (`accepted + 1`, clamped 2..6), then add an 8-token cooldown after two one-accept steps.

**Mechanism measured**: low-hit prose (1.00 accepted/step over 63 tokens) went from six verifies to five verifies, but each short verify still streams the full weight matrix. Decode is weight-stream bound, so fewer rows do not remove the dominant cost. The cooldown also only removed one verify because the drafter abstained on most steps anyway.

**Falsifier**: forward-time ratio >1.05 on prose with identical ids and no loss on periodic prompts. Not met.

| pairs | fixed-6 forward / adaptive forward (sorted) | median |
|---:|---|---:|
| 8 | 0.763, 0.854, 0.955, 0.970, 1.006, 1.388, 1.628, 2.167 | **1.006** |

Ids were identical to fixed-6 and plain decode in every pair. The outliers show box variance, not a policy effect; the median says tie.

## 2. 256-row prefill chunk - REJECTED (loses at 2048, unclear at 4096)

**Idea**: 128 rows beat 64 at 2048/4096, so test the next doubling.

| prompt | 128-row chunks | 256-row chunks | ratio |
|---|---:|---:|---:|
| 2048 pair 1 | 27.6 t/s | 26.1 t/s | 0.946x |
| 2048 pair 2 | 33.8 t/s | 33.0 t/s | 0.976x |
| 4096 pair 1 | 20.9 t/s | 22.7 t/s | 1.086x |
| 4096 pair 2 | 24.7 t/s | 24.6 t/s | 0.996x |

Ids identical throughout. The 4096 median is ~1.04x, below the keep bar, and the buffer cost doubles the 128-row score/logit scratch (~+132 MB at context 4096). 128 stays.

## 3. Q4_K per-row scale-metadata cache - KILLED on the default thread count

**Idea**: the kq8 batch kernel re-unpacks each Q4_K row's 6-bit scales/mins once per token. Cache the unpacked 32-byte metadata per row and reuse it across the token loop.

**What happened**:
- At 6 threads (the isolated test pool), the cache is **bit-exact** and measured 1.163x, 1.170x, 1.222x, 1.277x, 1.365x on 3072x3072x64 (median **1.222x**). This looked like the cycle's win.
- At the default all-logical prefill pool, the same-binary `bench batch --kernel kq8` A/B reverses it:

| shape | direct | metadata cache | ratio |
|---|---:|---:|---:|
| 3072x3072, 64 tok | 3.563 ms | 3.769 ms | 0.945x |
| 3072x8192, 64 tok | 8.776 ms | 8.984 ms | 0.977x |
| 3072x3072, 128 tok | 8.393 ms | 8.401 ms | 0.999x |
| 3072x8192, 128 tok | 19.733 ms | 19.027 ms | 1.037x |

- End-to-end at 2048 tokens agreed with the default-pool result: six HEAD-vs-cache prefill pairs gave 1.181, 0.951, 0.979, 0.933, 1.015, 0.889 (median ~0.965x).

**Why**: the precompute is a serial per-row pass; with 16 prefill threads the extra metadata traffic and scalar unpack work compete with the dot instead of hiding under it. The 6-thread pool has enough headroom per worker to amortize it. The default is 16 prefill threads, so the cache is not shippable. Reopen only with a vectorized metadata prepack that wins at 16 threads.

## 4. 16-query value tile - KILLED (8-query tile stays)

**Idea**: the kept 8-query value tile keeps its accumulators in L1 (4 KiB at head_dim 128) but re-streams each value row 16 times per 128-query chunk. A 16-query tile is still L1-resident (8 KiB) and halves that to 8 streams.

**What happened**: tile-8 and tile-16 binaries (same source except `kValueTile`) were run in alternating pairs on Llama-3.2-3B-Q4_K_M, `--threads 6 --gpu off`, n=2. Attention ratios below are tile-8 time / tile-16 time, so >1 favors tile-16:

| prompt / KV | pairs | attention ratio (tile16 vs tile8) | ids |
|---|---|---|---|
| 2049 / f16 | 4 | 1.056, 0.927, 0.997, 0.845 -> **0.962** | identical 4/4 |
| 4081 / f16 | 3 | 1.022, 0.890, 0.836 -> **0.890** | identical 3/3 |

Median 0.927 across seven pairs. In the 2049 pairs the untouched matvec phase actually favored the tile-16 arm in 3/4 (1.051-1.109), so the attention shortfall is not a uniform window bias. The predicted row-traffic saving does not pay for the larger accumulator footprint.

**Verdict**: revert to tile 8. Tests 146/146 after the revert; no source change shipped. Detail in `docs/measurement-log.md`.

## 5. 128-row chunks above 4096 after the value tile - INCONCLUSIVE, 64 kept

**Idea**: the 8192-token chunk rule (64 rows above 4096) was chosen before the value tile, when a 128-query chunk's 64 KiB value-accumulator set lost 7-11%. The tile cut that set to 4 KiB, so 128 might now win and halve the number of chunks.

**What happened**: shipped binary (64 above 4096) vs a same-source binary with 128 everywhere, alternating pairs at 8161 tokens, f16 KV, Llama-3.2-3B-Q4_K_M, `--threads 6 --gpu off`, n=2. Ratios are 128/64; <1 favors 128.

| pair | total 64 → 128 (s) | ratio | matvec 64 → 128 (s) | attention 64 → 128 (s) | ids |
|---:|---|---:|---|---|---|
| 1 | 308.8 → 335.1 | 1.085 | 170.4 → 184.5 | 121.3 → 131.7 | identical |
| 2 | 482.1 → 401.3 | 0.832 | 288.9 → 222.3 | 173.3 → 159.2 | identical |
| 3 | 421.5 → 313.4 | 0.743 | 257.8 → 185.8 | 144.2 → 109.6 | identical |
| 4 | 259.8 → 302.5 | 1.164 | 152.1 → 174.1 | 92.1 → 111.0 | identical |

Median 0.958, geometric mean 0.940, but the spread (0.743-1.164) is smaller than the drift: the identical 64-row arm's matvec phase alone ranged **152.1-288.9 s (1.90x)** across the session. The phase ratios disagree within pairs, so the model-level pairs cannot resolve a ~5% effect.

The isolated kernel is the reliable datum. On the real prefill shape (`bench batch --format q4k --kernel kq8`, 10 repeats):

| shape | 64 tokens | 128 tokens | per-token |
|---|---|---:|---:|
| 3072x3072 (attention projections) | 3.462 ms, 348.9 GFLOPS | 7.199 ms, 335.6 GFLOPS | 64 is 3.8% better |
| 8192x3072 (FFN) | 9.255 ms, 348.0 GFLOPS | 18.559 ms, 347.1 GFLOPS | equal |

Per-token throughput is flat-to-slightly-worse at 128, so halving the chunk count does not reduce the dominant matvec work. **Verdict: keep the 64-row cap.** Ids identical 4/4; tests 146/146 after the revert. Reopen only on a quiet box (identical-arm matvec spread <10%) with at least six pairs, or after a matvec kernel that scales better at 128 tokens.

## 6. Exact commands (selection)

```text
# adaptive spec: 8 interleaved fixed-6 vs adaptive pairs on Llama-3.2-3B Q4_K_M
TGE_SPEC_FIXED_DEPTH=1 tge generate ... --speculative
tge generate ... --speculative                         # median forward ratio 1.006

# chunk sweep: TGE_PREFILL_CHUNK=128|256, ids checked
# metadata cache: TGE_KQ8_META=0|1 with bench batch and generate
tge bench batch --rows 3072 --cols 8192 --tokens 128 --format q4k --kernel kq8
tge generate --tokens @2048 --kv f32 --print-ids
```

No code from sections 1-3 is in the tree or any commit.
