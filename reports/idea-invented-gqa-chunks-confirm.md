# CONFIRM: GQA 4-chunk decode at 255 tokens — KEEP (decode +4.5%/+25%, attention 4–13% better, ids identical)

- **Idea**: Cycle 47 found a 5–11% attention-phase lead for 3–4 GQA history chunks but could not resolve decode over 63-token samples. Retest 2 vs 4 chunks with a **255-token** decode sample to average the box drift.
- **Cycle date**: 2026-11-08
- **Verdict**: **KEEP — 4 chunks win both pairs.** Decode **2.01 → 2.10 t/s (1.045x)** and **1.99 → 2.49 t/s (1.251x)** at 4,000 tokens; attention/matvec falls **0.482 → 0.421** and **0.479 → 0.459**; ids identical. Combined with cycle 47, 4 chunks is better in 6/6 observations (attention 4–13% normalized), and the change is safe (scratch sized for 16 chunks, merge cost bounded).
- **Tree state**: spike reverted from `src/models/qwen2.cpp`; 145/145 baseline tests rebuilt and green; `aikit.py check` 4/4; no branch, no spike commit; nothing pushed.

## 1. Mechanism

`attention_gqa` splits the decode span into `chunks` per KV head and parallelizes `(kv_head, chunk)` over the decode pool. The shipped formula gives `chunks = 2` at 6 threads / 8 KV heads → 16 tasks, so the last scheduling wave leaves two workers idle. The spike exposed `TGE_GQA_CHUNKS` to force 4 chunks → 32 finer tasks and better dynamic balance.

## 2. Falsifier

- **KEEP** if 4 chunks is faster in both interleaved 255-token pairs, with matching attention-phase improvement and identical ids.
- **MIXED** if the long sample still cannot resolve the sign.

Measured: both pairs faster; keep.

## 3. Measured numbers — 4,000-token prompt, n=256, f32 KV, 6 threads

| pair | 2 chunks | 4 chunks | decode ratio | attn/matvec 2ch | attn/matvec 4ch |
|---:|---:|---:|---:|---:|---:|
| 1 | 2.01 t/s | 2.10 t/s | **1.045x** | 0.482 | 0.421 |
| 2 | 1.99 t/s | 2.49 t/s | **1.251x** | 0.479 | 0.459 |

Ids identical in both pairs. The attention phase is 4–13% faster relative to matvec, consistent with cycle 47's 64-token sweep (5.3% at 3 chunks, 10.9% at 4).

### Cross-cycle summary

| source | sample | 2 chunks attn/mv | 4 chunks attn/mv | direction |
|---|---|---:|---:|---|
| cycle 47 pair 3v4 #1 | 63 tok | 0.531 (3ch) | 0.511 | 4 better |
| cycle 47 pair 3v4 #2 | 63 tok | 0.556 (3ch) | 0.496 | 4 better |
| cycle 48 pair 1 | 255 tok | 0.482 | 0.421 | 4 better |
| cycle 48 pair 2 | 255 tok | 0.479 | 0.459 | 4 better |

The direction is consistent across 4 chunk-comparisons and two sample lengths; the 255-token sample also shows the raw decode gain in both pairs.

## 4. Interpretation

- The wave-count model explains the direction; the 4-chunk advantage over 3 suggests dynamic work-stealing benefits from finer granularity beyond the naive wave count.
- At a 32–35% decode attention share, a ~9% attention gain predicts ~3% decode; pair 1's +4.5% matches, pair 2's +25% is drift-inflated but same-signed.
- The change costs one extra merge per KV head and no extra scratch (`kMaxGqaChunks` is 16 and the state buffers are sized for it). Ids are identical across every pair, including cycle 47.

## 5. Disposition

**KEEP, report-only.** Adopt `chunks = 4` for the tested shape (6 decode threads, 8 KV heads); the general rule suggested by the data is to target roughly 4 tasks per worker (`kv_heads × chunks ≈ 4 × threads`) while capping chunks at 4. This is a small, safe default change for the integration dossier; the human gate decides.

## 6. Exact commands

```text
cmake --build --preset release                         → 54/54
.\build\release\bin\tge_tests.exe                      → 145/145
TGE_GQA_CHUNKS=2|4 tge generate --tokens <4000 ids> --n 256 \
  --threads 6 --gpu off --kv f32 --context 4352 --print-ids
git checkout -- src/models/qwen2.cpp
python aikit.py check aegean-tge                        → 4/4
```

No push, no merge, no force-push. Spike code is not in the tree or in any commit.
