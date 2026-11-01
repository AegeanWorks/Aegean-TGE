# ADR-0006: Q8 decode stays in the float domain

- Status: accepted (rejects an earlier proposal)
- Date: 2026-11-01

## Context

Earlier analysis claimed the Q8 decode kernel was arithmetic-bound. The
reasoning was: single-threaded decode reaches about 8.3 GB/s of weight traffic,
while a standalone f32 GEMV on the same machine reaches 18.4 GB/s, so the
difference had to be the cost of widening each int8 to int32 and then to float.

That reasoning is wrong, and it is worth recording why so it is not repeated.

The proposal that followed was standard practice in this field: quantize the
activation vector to int8 as well, then compute the inner product entirely in
integer domain with `maddubs`, removing the widening chain.

## What was built and measured

The kernel was implemented as proposed:

- Per 32-element block, the activation is quantized to int8 with its own scale.
- The inner loop uses `sign` + `maddubs` + `madd` and never converts to float
  until one vector accumulate per block.
- A first version reduced to a scalar per block; that was identified as wasteful
  and replaced with lane accumulation and a single horizontal reduction per row.

Result, 128 generated tokens, three runs each, same model and thread count:

| Kernel | Decode |
|---|---|
| Float activation (current) | 30.3 / 28.9 / 33.1 t/s |
| Integer activation | 31.0 / 30.4 / 32.9 t/s |

Inside the noise band. It also changed the greedy output: the token sequence
diverged from llama.cpp at position 14, where the float-activation version
matched for all 16.

## Decision

Revert. Keep the float-activation kernel, which is exact against the scalar
oracle and reproduces llama.cpp token for token.

## Why the hypothesis was wrong

Decode is not one GEMV. A token runs 24 layers of two norms, three projections,
rope, an attention pass, three more projections, and a final projection over
151,936 rows. The small projections have poor locality and each of the roughly
150 calls is a barrier. The gap between 8.3 GB/s and 18.4 GB/s is structural, not
arithmetic — comparing the two numbers was comparing a whole token against a
single kernel.

The levers that remain are therefore structural:

1. Batch prompt processing, which is currently one token at a time.
2. Remove per-call barriers: fuse QKV into one parallel region, fuse gate and up.
3. Revisit large pages once the privilege is available.

## Consequence

The engine keeps a property worth more than an unmeasurable 2 percent: its
greedy output is identical to a mature reference implementation on the same
weights. Any future change that breaks that must justify itself with a clear
measurement, not a plausible story about instruction counts.
