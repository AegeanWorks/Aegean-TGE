# INVENTED: 8-chunk GQA decode history split at long context — MIXED (attention phase ~1.09x better in 3/4 pairs; decode unresolved on a loaded box)

- **Idea**: Decode attention splits each KV head's history into 4 chunks at span >= 1024; the 2/3/4 sweep shipped 4. Eight chunks is the next untested point: 64 tasks instead of 32 on a 6-worker decode pool improves the last-wave utilization (4/6 vs 2/6), at the cost of twice as many online-softmax merge states per head.
- **Cycle date**: 2026-11-08
- **Verdict**: **MIXED — keep the shipped 4-chunk default.** The matvec-normalized attention phase favors 8 chunks in 3 of 4 interleaved pairs (median ~1.09x), but one pair contradicts strongly (0.814x) and the raw decode rates are unusable because the box's matvec phase moved up to 4.5x between arms. Ids are identical in all pairs; fidelity at 8 chunks is 1.042e-11 mean KL with 399/399 top-1.
- **Tree state**: spike reverted from `src/models/qwen2.cpp` (temporary `TGE_GQA_CHUNKS` override); branch `idea/gqa-chunks-6-8` deleted; `git status` clean; nothing committed except this report.

## 1. Mechanism

`Qwen2::attention_gqa` splits the decode span into `chunks` per KV head and dispatches `(kv_head, chunk)` tasks:

```text
base_chunks = ceil(threads*2 / kv_heads)
chunks      = span >= 1024 ? clamp(max(4, base_chunks), 1, 16) : base_chunks
```

At 6 threads and 8 KV heads: 4 chunks -> 32 tasks on 6 workers (5.33 waves, last wave 2/6), 8 chunks -> 64 tasks (10.67 waves, last wave 4/6). Finer chunks balance better but double the per-head merge work (`chunks * group * head_dim` floats) and the merge loop's overhead.

The spike exposes `TGE_GQA_CHUNKS` (0 = shipped rule, clamped to `kMaxGqaChunks=16`) and uses it in place of the computed count.

## 2. Falsifier

- **KEEP a new default (8)** if decode is >=3% faster in matched interleaved pairs, ids are identical, and the attention phase improves relative to the untouched matvec phase.
- **MIXED** if only the attention phase moves and raw decode cannot be resolved.
- **KILL** if the attention phase regresses.

Measured: attention phase favors 8 in 3/4 pairs; raw decode unresolved. MIXED.

## 3. Measured numbers — 2049-token prompt, n=64, f32 KV, 6 threads

Ratios are 4-chunk / 8-chunk normalized attention (`attn_ms / matvec_ms`); >1 favors 8 chunks.

| pair | 4ch decode t/s | 8ch decode t/s | 4ch matvec/attn (ms) | 8ch matvec/attn (ms) | attn ratio 4/8 |
|---:|---:|---:|---|---|---:|
| 1 | 0.77 | 1.40 | 67,701.6 / 13,162.8 (0.1944) | 38,153.6 / 6,323.4 (0.1657) | **1.173** |
| 2 | 0.39 | 1.77 | 142,359.3 / 20,146.2 (0.1415) | 30,049.7 / 5,224.2 (0.1738) | **0.814** |
| 3 | 1.67 | 4.84 | 31,297.6 / 6,060.3 (0.1936) | 10,799.0 / 1,956.6 (0.1812) | **1.069** |
| 4 | 4.75 | 4.36 | 10,795.4 / 2,204.3 (0.2042) | 11,940.0 / 2,201.9 (0.1844) | **1.107** |

Median **1.088**. The raw decode rates track the box, not the chunks: the untouched matvec phase ranged 10.8-142.4 s across arms (4.5x), so only the same-run normalized attention ratio is usable. Pair 2 is the strongest counter-signal; it ran in a 4.5x slower 4-chunk window but its normalized ratio still favors 4.

### Numerics

- Ids identical in all four pairs (`578 4062 14198 ... 520 39493 13` in every arm).
- `tge fidelity --kv f32` with 8 chunks, 399 positions: mean KL **1.042e-11**, worst 2.330e-10, top-1 **399/399** (shipped 4-chunk baseline: 1.020e-11 / 399/399). The merge-order difference is negligible.
- Suite 162/162 (chunk count only changes scheduling/merge order).

## 4. Interpretation

- The utilization model predicts a small attention gain from 8 chunks (last wave 4/6 vs 2/6) and the median agrees (~9%), but the merge cost and the box's 4.5x swing prevent a decode-level claim.
- The 3->4 step earlier measured a normalized attention lead of ~10.9%; an 4->8 lead of ~9% would predict only ~2-3% decode at a 30% attention share, below this box's resolution.
- The change is safe: scratch is already sized for 16 chunks, ids identical, fidelity unchanged.

## 5. Full reimplementation recipe

1. `src/models/qwen2.cpp`, above `clock_ns`: add the `gqa_chunk_override()` helper (`getenv("TGE_GQA_CHUNKS")`, `atoi`, clamp `0..kMaxGqaChunks`, `static`).
2. In `attention_gqa`, after the shipped `chunks` computation: `const int active_chunks = override_chunks > 0 ? override_chunks : chunks;` and use it for `chunk_size`/`actual_chunks`.
3. Bench script (PowerShell, one process per arm so the `static` env read is fresh): 2049-token prompt, `--n 64 --threads 6 --gpu off --kv f32 --context 4096 --print-ids`; parse `decode ... (Y tok/s)`, `phases : decode matvec/attn`, and `ids`; alternate 4/8 by pair and compute `attn/matvec` per arm.
4. To ship 8, change the long-span floor from `max(4, base_chunks)` to `max(8, base_chunks)`; keep `kMaxGqaChunks=16`.

## 6. Reopen trigger

- A quiet box (identical-arm matvec spread <10%) with ABBA x3 at 2049 and 4081, f32 and f16; adopt 8 if decode is >=3% faster in >=4/6 pairs and the normalized attention improves every time.
- Do not change the default on this run: the box moved more than the effect.
