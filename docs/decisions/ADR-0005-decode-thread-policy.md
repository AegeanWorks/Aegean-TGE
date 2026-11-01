# ADR-0005: Decode runs on fewer threads than the machine has

- Status: accepted
- Date: 2026-11-01

## Context

The engine defaulted every parallel kernel to `hardware_concurrency()`. That is
the right instinct for compute-bound work and the wrong one for decode.

Measured on the target machine (i5-14400F, 6 performance + 4 efficiency cores,
16 logical) with the real 0.5B Q8_0 model:

| Threads | 1 | 4 | 6 | 8 | 10 | 16 |
|---|---|---|---|---|---|---|
| Decode t/s | 16.6 | 32.5 | 32.3 | 30.5 | 29.5 | 21.8 |

Throughput plateaus at four to six workers and falls by a third at sixteen. Two
effects cause it:

1. Decode issues roughly 150 matrix-vector calls per token (24 layers x 6
   projections plus the output projection). Each call is a barrier. More workers
   means more per-call wake and join cost, and there are 150 of them per token.
2. The efficiency cores add contention without adding bandwidth. Once DRAM is
   saturated, extra workers only queue.

llama.cpp shows the same cliff on the same machine: 15.2 t/s at six threads,
10.1 at sixteen.

## Decision

`cpu::resolve_decode_threads()` returns the performance-core count, falling back
to physical cores and then to the logical count. Memory-bound decode uses that;
compute-bound GEMM keeps the full logical count.

Sizing each call to the matrix individually (fewer workers for a 128-row
projection than for a 4864-row one) was implemented and measured: median 27.5
t/s against 32.8 t/s for the flat rule. Rejected as unhelpful complexity.

## Consequences

- Decode runs at the measured optimum on hybrid parts without the caller having
  to know the topology.
- `tge info` reports both counts so the difference is visible rather than
  surprising.
- The rule is derived from core classes, not from a hardcoded number, so it
  transfers to other hybrid parts, but it has only been validated on one
  machine. A part with many more P-cores may want a different ratio.
- Prefill is unaffected: it is compute-bound and still uses every logical
  processor.
