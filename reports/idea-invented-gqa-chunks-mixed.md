# INVENTED: GQA decode chunk-count tuning — MIXED (attention 5–11% faster at 3–4 chunks; decode gain inside box noise; default unchanged)

- **Idea**: At 6 decode threads and 8 KV heads, the GQA split uses 2 history chunks per KV head → 16 tasks, so the last scheduling wave runs 4 of 6 workers (≈89% utilization). Test 3 chunks (24 tasks, 4 full waves) and 4 chunks (32 finer tasks) against the default 2.
- **Cycle date**: 2026-11-08
- **Verdict**: **MIXED — a real attention-phase lead, not a proven decode win.** Normalized by the untouched matvec phase, attention is **5.3% faster at 3 chunks and 10.9% at 4 chunks** (attn/matvec: 2ch 0.559, 3ch 0.531, 4ch 0.504). At a 35% decode attention share that predicts ~2–4% decode, which the raw pairs cannot resolve on this box (0.986x and 1.104x for 2→3; 0.823x and 1.556x for 3→4). Ids are identical in all pairs. Keep the default; re-test on a quiet box or inside the integrated stack.
- **Tree state**: spike reverted from `src/models/qwen2.cpp` (chunk-count env override); 145/145 baseline tests rebuilt and green; `aikit.py check` 4/4; no branch, no spike commit; nothing pushed.

## 1. Mechanism

`Qwen2::attention_gqa` splits the decode span into `chunks` per KV head and parallelizes `(kv_head, chunk)` over the decode pool:

```text
chunks = clamp((threads*2 + kv_heads - 1) / kv_heads, 1, kMaxGqaChunks)
threads=6, kv_heads=8 → chunks=2 → 16 tasks
```

With equal tasks and 6 workers, 16 tasks take three waves (6+6+4) — the last wave idles two workers. 24 tasks (3 chunks) is exactly four waves; 32 tasks (4 chunks) is finer-grained and can balance better under dynamic scheduling, at the cost of one or two extra online-softmax merges per head.

## 2. Falsifier

- **KEEP a new default** if decode is ≥3% faster in interleaved pairs and ids stay identical at the same time as the attention phase improves.
- **MIXED** if only the attention phase moves and decode cannot be resolved.

Measured: attention phase improves; decode cannot be resolved.

## 3. Measured numbers — 4096-token prompt, n=64, f32 KV, 6 threads

### 2 vs 3 chunks

| pair | 2 chunks | 3 chunks | decode ratio | attn/matvec 2ch | attn/matvec 3ch |
|---:|---:|---:|---:|---:|---:|
| 1 | 2.87 t/s | 2.83 t/s | 0.986 | 0.560 | 0.515 |
| 2 | 2.69 t/s | 3.45 t/s | 1.104 | 0.558 | 0.523 |

### 3 vs 4 chunks

| pair | 3 chunks | 4 chunks | decode ratio | attn/matvec 3ch | attn/matvec 4ch |
|---:|---:|---:|---:|---:|---:|
| 1 | 2.84 t/s | 3.45 t/s | 0.823 (3ch/4ch) | 0.531 | 0.511 |
| 2 | 4.28 t/s | 2.75 t/s | 1.556 | 0.556 | 0.496 |

### Normalized summary

| chunks | tasks | attn/matvec | attention speedup vs 2 |
|---:|---:|---:|---:|
| 2 | 16 | 0.559 | 1.000x |
| 3 | 24 | 0.531 | 1.053x |
| 4 | 32 | 0.504 | 1.109x |

Ids identical in all four pairs. The raw decode ratios move in opposite directions with the load, so only the same-run normalized metric is usable; it shows a consistent 5–11% attention improvement.

## 4. Interpretation

- The utilization model explains most of the direction (16 tasks → three waves with two idle workers), but the measured 4-chunk advantage over 3 suggests dynamic work-stealing and uneven chunk lengths favor finer granularity beyond the simple wave count.
- The decode-level effect is ~2–4% by arithmetic, below this box's ±20–35% drift; a quiet box or a longer sample is needed before changing a shipped default.
- The change is otherwise safe: ids identical in all pairs, extra merge states bounded by `kMaxGqaChunks`, scratch already sized for 16 chunks.

## 5. Disposition

**MIXED — keep the shipped default (2 chunks).** The lead is real in the attention phase and cheap to adopt later; record 4 chunks as the candidate for a quiet-box or integrated-stack retest. Do not change the default on normalized evidence alone.

## 6. Exact commands

```text
cmake --build --preset release                         → 54/54
.\build\release\bin\tge_tests.exe                      → 145/145
TGE_GQA_CHUNKS=2|3|4 tge generate --tokens <4096 ids> --n 64 \
  --threads 6 --gpu off --kv f32 --context 4352 --print-ids
git checkout -- src/models/qwen2.cpp
python aikit.py check aegean-tge                        → 4/4
```

No push, no merge, no force-push. Spike code is not in the tree or in any commit.
