# MEASURED: sparse geometry sweep — stride-only dilution is the efficient knob (8k stride 16: attention 0.31-0.34x, top-1 −2.25pp); narrowing the window costs 4-11pp for no extra speed

- **Idea**: The shipped sparse geometry is the paper's short-context defaults (`sink 4`, `window 512`, `stride 4`). At 8,193 positions that keeps ~2,436 rows (30%); the kept set grows linearly with context because only the window is fixed and the strided middle is span/stride. The 8k measurements (`idea-invented-sparse-8k-reopened.md`, `idea-invented-sparse-prefill-8k-measured.md`) show sparse is the long-context lever, but the geometry itself has never been swept on this CPU. Question: at a long context, does spending the kept budget on a longer stride (dilute the middle) or on a shorter window (shrink the recent tail) buy attention time more cheaply in quality? Price both axes with the shipped `--sparse-window`/`--sparse-stride` flags.
- **Cycle date**: 2026-11-12
- **Verdict**: **MEASURED — stride-only dilution wins the trade.** Quality priced on Qwen2.5-0.5B-Q8_0, 1,999 teacher-forced positions, ctx 2048, f16 KV; speed on Llama-3.2-3B-Q4_K_M, 8,193-token prompt, f16 KV, 10 threads.
  - **Quality ladder (mean KL / top-1 vs the full-scalar oracle)**: default 512/4 **8.724e-02, 1776/1999**; 512/8 **1.128e-01, 1747**; 512/16 **1.312e-01, 1731**; 256/8 **1.789e-01, 1646**; 256/16 **2.082e-01, 1604**; 128/8 **2.729e-01, 1553**.
  - **8k speed, clean pair (matvec matched within 1%)**: default 512/4 prefill 223.3 s (attn 64.0 s), decode 8.30 t/s (attn 1008 ms, 27%); 512/16 prefill **181.5 s** (attn **21.4 s**), decode **10.61 t/s** (attn **301 ms**, 10%) -> **prefill 1.23x, decode 1.28x, attention 0.31-0.33x**.
  - **Stride 8**: prefill 201.7 s (attn 38.0 s), decode 10.03 t/s -> 1.11x / 1.21x, attention ~0.55x, for a 1.29x KL / 1.45pp quality cost.
  - **Window narrowing is dominated**: 256/8 and 128/8 cost 6.5-11.0pp top-1 (and 2-3x KL) for attention reductions no better than 512/16, because most of the kept rows sit in the window and shrinking it removes the most recent context the model needs most.
  - **Context itself erodes the default tier**: at ctx 1024 the same corpus family priced at KL 2.6e-2 / 975/1023 (95.3%); at ctx 2048 the default is 1776/1999 (88.8%). The strided middle grows with span while the window does not.
- **Tree state at close**: no code touched (shipped flags only); `git status` clean; stash holds only the pre-existing 2026-11-04 peer entry; `tge_tests` **171/171**, `python aikit.py check aegean-tge` **4/4**. No branch, nothing pushed.

## 1. Setup

| | quality | speed |
|---|---|---|
| model | Qwen2.5-0.5B-Instruct-Q8_0 | Llama-3.2-3B-Instruct-Q4_K_M |
| corpus / prompt | `architecture.md` repeated twice -> 1,999 positions | `The fox runs. ` x 2048 -> 8,193 tokens |
| context / tokens | 2048 / 2000 | 8704 / n=32 |
| KV / threads | f16 / 10 | f16 / 10 |
| baseline arm | full-scalar oracle vs AVX2 (harness) | dense prefix |

Geometry set: `(window, stride)` = 512/4 (default), 512/8, 512/16, 256/8, 256/16, 128/8. `sink` stays 4 throughout. Each geometry keeps `4 + ceil((span - window - 4)/stride) + window` rows; at 1,999 positions that is 887 / 702 / 609 / 478 / 369 / 366; at 8,193 it is 2,436 / 1,476 / 996 / 1,252 / 756 / 1,140.

Fidelity command (per geometry):

```text
tge fidelity --model qwen2.5-0.5b-instruct-q8_0.gguf --corpus <arch x2> \
  --context 2048 --tokens 2000 --kv f16 --threads 10 --gpu off \
  --sparse --sparse-window W --sparse-stride S
```

Speed command (per geometry):

```text
tge generate --model Llama-3.2-3B-Instruct-Q4_K_M.gguf --prompt "<fox x2048>" \
  --context 8704 --n 32 --kv f16 --threads 10 --gpu off \
  --sparse --sparse-window W --sparse-stride S
```

## 2. Quality results (0.5B, ctx 2048, 1,999 positions)

| window / stride | kept @1999 | mean KL | worst KL | max abs diff | top-1 | drop vs default |
|---|---:|---:|---:|---:|---:|---:|
| 512 / 4 (default) | 887 (44%) | **8.724e-02** | 1.176e+01 | 1.825e+01 | **1776/1999 (88.85%)** | — |
| 512 / 8 | 702 (35%) | 1.128e-01 | 1.183e+01 | 1.836e+01 | 1747 (87.39%) | 1.45pp |
| 512 / 16 | 609 (30%) | 1.312e-01 | 1.197e+01 | 1.847e+01 | 1731 (86.59%) | **2.25pp** |
| 256 / 8 | 478 (24%) | 1.789e-01 | 1.125e+01 | 1.780e+01 | 1646 (82.34%) | 6.51pp |
| 256 / 16 | 369 (18%) | 2.082e-01 | 1.312e+01 | 1.774e+01 | 1604 (80.24%) | 8.61pp |
| 128 / 8 | 366 (18%) | 2.729e-01 | 1.089e+01 | 1.963e+01 | 1553 (77.69%) | 11.16pp |

Quality is monotone in kept count, but the *composition* matters more than the count: at similar kept counts, 512/16 (609 rows) beats 256/8 (478 rows) by 4.3pp top-1 — the longer window is worth far more than the extra middle rows. The default's own top-1 falls from 95.3% at ctx 1024 (§50: 975/1023) to 88.8% here, and the sweep explains why: the middle grows with context while the window is fixed.

## 3. Speed results (3B, 8k, n=32)

| run | geometry | prefill total / attn | decode t/s / attn | matvec (prefill / decode) |
|---|---|---:|---:|---|
| 1 | 512/4 (default) | 223.3 s / 64.0 s | 8.30 / 1008 ms (27%) | 146.0 s / 2635 ms |
| 2 | 512/8 | 201.7 s / 38.0 s | 10.03 / 544 ms (18%) | 149.4 s / 2465 ms |
| 3 | 512/16 | **181.5 s / 21.4 s** | **10.61 / 301 ms (10%)** | 144.6 s / 2538 ms |
| 4 | 256/8 | 207.6 s / 27.8 s | 6.74 / 637 ms (14%) | 164.6 s / 3848 ms |
| 5 | 128/8 | 225.3 s / 24.7 s | 7.93 / 524 ms (13%) | 183.5 s / 3279 ms |
| 6 | 512/4 (bracket) | 291.0 s / 72.2 s | 6.66 / 1127 ms (24%) | **200.1 s / 3413 ms** |

The box degraded sharply during the sweep: the closing default bracket's context- and geometry-independent matvec phase is 1.37x its opening value, so the raw totals for runs 4-6 are contaminated. Two clean readings survive:

- **Opening default vs 512/16 (runs 1 vs 3)**: matvec 146.0 vs 144.6 s (-0.96%) and 2635 vs 2538 ms (-3.7%), so the pair is admissible. Prefill 223.3 -> 181.5 s (**1.23x**), attention 64.0 -> 21.4 s (**0.33x**), decode 8.30 -> 10.61 t/s (**1.28x**), decode attention 1008 -> 301 ms (**0.30x**).
- **512/8 (run 2)**: matvec within 2.3% of opening default in prefill and 6.5% lower in decode (slightly favourable), so its 1.11x prefill / 1.21x decode is a fair estimate with attention ~0.55x.
- **Window-narrowed runs (4-5)** are reported but not compared on totals; their attention phases (27.8 s, 24.7 s) are no better than 512/16's 21.4 s, and their quality is 4.3-8.9pp worse, so they lose the trade regardless of drift.

Drift-normalized decode attention (attn / matvec) supports the ordering without relying on the box: 0.383 (default open), 0.221 (512/8), 0.118 (512/16), 0.166 (256/8), 0.160 (128/8), 0.330 (default close). Stride 16 cuts the attention share ~3x; the window variants do not.

## 4. Falsifier

Stated before the speed sweep: **KEEP-GRADE a long-context geometry** if it cuts 8k attention >= 2x with a top-1 drop <= 2.5pp on the 1,999-position corpus; **REJECT** any geometry whose quality cost is worse than 2.5pp. Result: 512/16 passes (attention ~3x, 2.25pp); 512/8 passes the quality bar but only halves attention (1.45pp); all window-narrowed variants fail the quality bar (6.5-11.2pp). The shipped default stays 512/4 (quality-first); the keep-grade recipe is a long-context mode.

## 5. Reading and recipe framing

- **The efficient knob is stride, not window.** Diluting the middle removes rows the model already sees sparsely; shrinking the recent window removes contiguous recent context, which is the highest-value part of the kept set. At matched kept counts the longer window wins by >4pp top-1.
- **The right long-context policy is a kept budget, not a fixed stride.** The sweep shows 512/16 at 8k keeps ~996 rows for a 2.25pp cost. A budget policy (`stride = max(2, ceil((span - sink - window) / budget))`, budget ~1,000 rows) would hold the kept count — and therefore attention cost — roughly constant with context, instead of letting it grow linearly as the default does. The recipe below is for that mode; it is a new config, not a change to the shipped default.
- **Usage now**: for 8k+ generation on this box, `--sparse --sparse-stride 16` cuts decode attention ~3x for ~2.3pp top-1 on the 2k corpus. `--sparse-stride 8` is the conservative step (~1.5pp, attention ~0.55x). Keep `--sparse-window 512` unless memory forces otherwise; do not shrink it for speed.
- **Quality caveat**: the tier is already lossy; at 8k the default's quality is unmeasured directly. The budget mode's quality at 8k would need a 8k-position fidelity run (long, not attempted this cycle) before shipping it as a default.

## 6. File anchors (current HEAD, read-before-write verified)

- `include/tge/models/qwen2.hpp` — `SparseConfig{sink=4, window=512, stride=4}` and its doc comment.
- `src/models/qwen2.cpp:222-260` — `build_sparse_index` (sink + strided middle + window), the geometry definition.
- `src/models/qwen2.cpp:1329-1342`, `:1357-1361` — sparse dispatch and `uses_sparse` threshold (`span > sink + window`).
- `apps/cli/main.cpp:2177-2180, 2233-2236` (generate) and `:2699+` (fidelity) — `--sparse`, `--sparse-window`, `--sparse-stride` in both parsers.
- `src/cpu/avx2/attention_avx2.cpp:263,367` — the strided kernels whose kept-set geometry this sweep prices.
- `reports/idea-invented-sparse-8k-reopened.md`, `idea-invented-sparse-prefill-8k-measured.md` — the 8k sparse pricing at default geometry.
- measurement-log §50 — the ctx-1024 default-geometry fidelity reference (KL 2.6e-2, 975/1023).

## 7. Reimplementation recipe

1. Quality (per geometry), 0.5B Q8_0, corpus = `docs/architecture.md` repeated twice (~23.9 KB), ctx 2048:

   ```text
   tge fidelity --model qwen2.5-0.5b-instruct-q8_0.gguf --corpus <corpus> \
     --context 2048 --tokens 2000 --kv f16 --threads 10 --gpu off \
     --sparse --sparse-window W --sparse-stride S
   ```

   Expect the §2 ladder: default 1776/1999; 512/8 1747; 512/16 1731; 256/8 1646; 256/16 1604; 128/8 1553 (deterministic per binary). Each run ~85-91 s.

2. Speed (per geometry), 3B Q4_K_M, compact 8k prompt, n=32:

   ```text
   tge generate --model Llama-3.2-3B-Instruct-Q4_K_M.gguf --prompt "<fox x2048>" \
     --context 8704 --n 32 --kv f16 --threads 10 --gpu off \
     --sparse --sparse-window W --sparse-stride S
   ```

   Expect 512/4 ~8.3 t/s decode (attn 27%), 512/8 ~10.0 (18%), 512/16 ~10.6 (10%), prefill attention 64.0 / 38.0 / 21.4 s. Each run costs ~3-4 min. Bracket the sweep and compare only pairs whose matvec phases agree within ~3%; this box drifts 20-40% over a 20-minute window.

3. Budget-mode recipe (new config, not shipped): add `int kept_budget = 0;` to `SparseConfig`; when `kept_budget > 0`, the effective stride is `max(2, ceil((span - sink - window) / kept_budget))` and the `window` is clamped to `span` when the span is short. The per-token and block-sparse kernels already take `stride` as a runtime argument, so only the dispatch sites need the computed value. Calibration from this cycle: budget ~1,000 rows at 8k maps to stride 16 with 2.25pp top-1 at the 2k proxy; a real ship would re-run the fidelity harness at 8k positions and set the budget from that.
4. No code ships this cycle; the shipped default stays 512/4.

## 8. Exact commands run and results

```text
cmake --build --preset release                    -> clean (56/56, cycle gate)
build\release\bin\tge_tests.exe                   -> 171 cases, 0 failed

fidelity (0.5B Q8_0, arch x2 corpus, ctx 2048, 1999 positions, f16, 10 threads):
  512/4  -> KL 8.724e-02, worst 1.176e+01, max|diff| 1.825e+01, 1776/1999 (83.99 s)
  512/8  -> KL 1.128e-01, worst 1.183e+01, max|diff| 1.836e+01, 1747/1999 (90.40 s)
  512/16 -> KL 1.312e-01, worst 1.197e+01, max|diff| 1.847e+01, 1731/1999 (91.29 s)
  256/8  -> KL 1.789e-01, worst 1.125e+01, max|diff| 1.780e+01, 1646/1999 (85.50 s)
  256/16 -> KL 2.082e-01, worst 1.312e+01, max|diff| 1.774e+01, 1604/1999 (84.90 s)
  128/8  -> KL 2.729e-01, worst 1.089e+01, max|diff| 1.963e+01, 1553/1999 (83.58 s)

8k speed (3B Q4_K_M, compact fox x2048, ctx 8704, n 32, f16, 10 threads):
  512/4   -> prefill 223.3 s (attn 64.0), decode 8.30 t/s (attn 1008 ms, 27%)
  512/8   -> prefill 201.7 s (attn 38.0), decode 10.03 t/s (attn 544 ms, 18%)
  512/16  -> prefill 181.5 s (attn 21.4), decode 10.61 t/s (attn 301 ms, 10%)
  256/8   -> prefill 207.6 s (attn 27.8), decode 6.74 t/s (attn 637 ms, 14%)   [drifting box]
  128/8   -> prefill 225.3 s (attn 24.7), decode 7.93 t/s (attn 524 ms, 13%)   [drifting box]
  512/4   -> prefill 291.0 s (attn 72.2 s), decode 6.66 t/s (attn 1127 ms, 24%)  [bracket, box 1.37x slower]
git status --short                                -> clean (no code touched)
python aikit.py check aegean-tge                  -> 4/4 gates passed
```

No push, no merge, no force-push. The committed tree is unchanged except this report.
