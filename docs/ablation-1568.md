# IDEA-1568 Portability Ablation Protocol (x86-AVX2 batch-one)

Source: arXiv:2610.31479 Abstract + §4. Four families on Apple batch-one single-request vs RTX4090: KV pruning to 1.75x, all speculative methods reduce throughput on Apple despite 2-6x server-GPU claims, CUDA designs partial, PTQ varies. Transferable ceiling 1.20x is a prior, not a promise.

## Question
What survives on x86-AVX2 CPU mmap-first batch-one single-request decode: KV-pruning/packing, speculative, ported kernels, PTQ?

## Workload (fixed)
- Single-request decode, batch-one. Models: Qwen2.5-0.5B Q8_0 (primary), Llama-3.2-3B Q4_K_M (scale check).
- Contexts: 512, 2048, 8192. Threads: pinned per ADR-0005 policy, back-to-back in one session.
- Metrics: decode t/s (best-of-N), prefill t/s, `tge fidelity` KL + top-1 vs scalar oracle, greedy ids identical.

## Families and hooks (no engine change in this step)
1. KV pruning: `--sparse` static sink-window-stride (qwen2.hpp:30) vs dense default; ADR-0008 is the oracle gate.
2. Speculative: `--speculative` prompt-lookup (main.cpp:1664) greedy identical-ids; record acceptance + t/s at batch-one.
3. Ported kernels: 1325 pack-on-load, 1539 clamped epilogue, 1540 M==1, 1558/1458 int-dot — measure per-shape (896x896, 4864x896, 2048 cubed), never one shape.
4. PTQ: `--kv f16|i8` priced (KL 1.658e-6 / 3.2e-4 precedent) + SHq LUT candidate; fidelity gates apply.

## Rules (from docs/benchmarks.md)
- Back-to-back ratios only; ±20% absolute noise, 29% worst observed. Best-of-N throughput, median latency.
- Every figure states model file + threads + shape. One sample is not evidence.
- Ablate before concluding (trap 4): whole-system delta is not attributed to one family.

## Kill criteria
- Speculative 1.66-2.0x (log §71/75) vanishes at batch-one x86 like Apple → deprioritize speculative for prefill gap, pivot to pruning/packing.
- KV pruning ≤20% back-to-back or KL/top-1 regress vs scalar oracle → reject port, keep dense+GQA.
- Ported CUDA kernels need async-copy/serving or PTQ varies beyond harness noise → mark non-transferable, do not port.
- Any family breaking greedy-id identity or fidelity KL gate → rejected for that family.

## Safety
Live bench skipped in this protocol commit per no-run constraint (full-system crash history). Existing log is the baseline. Future runs use safe subcommands only in back-to-back sessions; `measure.ps1` full runs remain gated by operator.
