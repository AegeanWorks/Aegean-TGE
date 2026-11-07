# INVENTED: Continuous-batching slots 4 → 8 — MIXED (median +34% with f16 KV, but memory pressure loses 21% in one pair)

- **Idea**: Raise continuous-batching concurrency from the current 4 KV slots to 8 under f16 KV, so one weight stream serves eight active decodes in a single `forward_batch` pass instead of two waves of four. The engine already exposes `--sequences`; this is a capacity/tuning spike, no code change.
- **Cycle date**: 2026-11-07
- **Verdict**: **MIXED.** Eight f16 slots delivered a **1.337x median** aggregate across five 4-vs-8 observations (range 0.790–1.873; 4/5 faster), with identical transcripts. But one pair lost 21% and 8-slot load spiked to 1.8 s, consistent with memory pressure from 3.6 GB of KV on a 16 GiB box. Eight f32 slots (7.2 GB KV) gained only 19% and doubled load. Keep **4 slots as the default**; make 8 f16 slots an opt-in only when the memory budget has headroom, and add `--sequences auto` as the next step.
- **Tree state**: measurement-only; no code change, no branch, no spike to revert. Report-only commit; tree clean; nothing pushed.

## 1. Mechanism

`bench continuous` keeps N per-sequence KV slots and decodes one token per active sequence in a single weight pass. With 4 slots and 8 requests, the load runs two waves; with 8 slots it admits all eight at once and reuses each weight read across eight decodes. Predicted aggregate gain from weight-stream sharing: +20–35%, at the cost of doubling KV memory.

At Llama-3B, context 4096:

| slots | f16 KV | f32 KV |
|---:|---:|---:|
| 4 | 1.8 GB | 3.6 GB |
| 8 | 3.6 GB | 7.2 GB |

The model is 1.87 GiB. The box has 16 GiB and routinely 2–4 GiB free, so 8 f32 slots is at the edge; 8 f16 slots is viable but not always safe.

## 2. Predicted gain / falsifier

- **KEEP as a preset** only if 8-slot aggregate is ≥1.15x 4-slot across paired runs, transcripts identical, and load stays within ~1.5x.
- **KILL as a default** if any pair loses, or load/memory pressure appears; 8 slots may still be a guarded opt-in.

## 3. Measured numbers

Llama-3.2-3B-Q4_K_M, `bench continuous --requests 8 --tokens 32 --threads 6`, context 4096, 248 decode tokens. First transcripts identical and `state: independent` in every run.

### f16 KV (5 observations)

| observation | 4 slots | 8 slots | ratio |
|---:|---:|---:|---:|
| 1 | 12.45 t/s | 16.65 t/s | 1.337 |
| 2 | 10.41 t/s | 19.50 t/s | 1.873 |
| 3 | 6.29 t/s | 4.97 t/s | 0.790 |
| 4 | 5.91 t/s | 10.30 t/s | 1.743 |
| 5 | 14.91 t/s | 17.58 t/s | 1.179 |
| **median** | — | — | **1.337** |

Load times: 4-slot 462–1467 ms; 8-slot 789–1809 ms. The 8-slot high end tracks the losing pair and the box's memory pressure.

### f32 KV (one pair)

| | 4 slots | 8 slots | ratio |
|---|---:|---:|---:|
| aggregate | 12.63 t/s | 15.04 t/s | 1.191 |
| wall | 19.64 s | 16.49 s | — |
| load | 931 ms | 2166 ms | — |

Eight f32 slots still wins aggregate but pays a 2.3x load and holds 7.2 GB of KV; not a safe default on this box.

## 4. Interpretation

- The weight-stream sharing is real: 4/5 f16 pairs and the f32 pair are faster with 8 slots, median +34% f16.
- The loss and the 1.8–2.2 s loads are memory-pressure signals, not a compute regression. The run that lost 21% had the 8-slot load at 1809 ms and the box already loaded.
- Therefore the current 4-slot default is the right **safe** choice; 8 f16 slots is a high-throughput opt-in with a budget check. Eight f32 slots should not be selected on a 16 GiB box.
- A budget-aware `--sequences auto` would capture the win when free memory is high and fall back to 4 under pressure:
  `slots = min(8, floor((usable - weights - non_kv_scratch) / kv_per_slot))`, with a reserve for the OS/page cache.

## 5. Reimplementation / use

No code change is needed today:

```text
tge bench continuous --model Llama-3B --requests 8 --tokens 32 \
  --threads 6 --sequences 8 --kv f16
```

Recommended policy:

1. Default remains `--sequences 4 --kv f32` for predictable memory.
2. High-throughput preset: `--sequences 8 --kv f16` only when free memory ≥ 4 GiB before load.
3. Do not use `--sequences 8 --kv f32` on this 16 GiB box.
4. Next code step: `--sequences auto` using the existing `plan`/budget numbers, with a 4-slot floor and an 8-slot ceiling; falsifier = aggregate ≥1.15x 4 slots on the 8×32 load without any run above the budget.

## 6. Gates

- **Code**: none changed; no build/test delta from this study.
- **Tests**: 145/145 on the current tree (last full run after the cycle-10 revert).
- **Project gate**: `python aikit.py check aegean-tge` → 4/4 after the report.
- **Fidelity**: f16 KV tier unchanged (mean KL 2.501e-07, 63/63) and all continuous transcripts identical; no arithmetic changed.

## 7. Exact commands

```text
tge bench continuous --model Llama-3B --requests 8 --tokens 32 --threads 6 --sequences 4 --kv f16
tge bench continuous --model Llama-3B --requests 8 --tokens 32 --threads 6 --sequences 8 --kv f16
tge bench continuous ... --sequences 4|8 --kv f32
python aikit.py check aegean-tge   → 4/4
```

No code changes, no push, no merge.
