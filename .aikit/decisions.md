# Decisions

Why this project is built the way it is, oldest first. One entry per decision that changes a
contract, a dependency, a data format or the architecture; IDs are consecutive and never reused.

<!-- Entry format:
### ADR-NNN: <decision in one line>
- **Date**: YYYY-MM-DD
- **Status**: Accepted
- **Context**: <what forced the decision>
- **Decision**: <what was decided>
- **Consequences**: <what follows, including what it costs>
-->

### ADR-001: Greedy prompt-lookup speculative decoding
- **Date**: 2026-11-01
- **Status**: Accepted
- **Context**: Decode is memory-bound, and prompt-lookup speculation was the remaining speed lever the audits identified. It needs a verifier that returns the model's greedy token after each drafted position, which the existing batched prefill path can compute in one weight pass.
- **Decision**: Add `Qwen2::verify_draft`, which runs up to six tokens (five draft tokens plus one padding token at the batched kernel's group size) through the shared `prefill_run` path, computes logits for all six rows with the batched output projection, and returns the greedy argmax per position. Drafts are the five tokens following the latest earlier match of the last one to three tokens in prompt+output; speculation runs only for pure greedy (temperature 0, repetition penalty 1). Rejected draft suffixes stay in the KV cache and are overwritten by later tokens, so no rollback pass exists.
- **Consequences**: 1.66x decode (30.10 -> 49.92 t/s) on a repetitive story prompt with identical 127-token ids; short non-repetitive answers can lose slightly (ChatML: 24.56 vs 27.15 t/s); each model carries 3.6 MB of verify-logits scratch; sampled and penalized runs keep the original path.

### ADR-002: Speculation pins the per-head attention schedule
- **Date**: 2026-11-01
- **Status**: Accepted
- **Context**: The speculative verifier runs the batched prefill path while decode switches to grouped-query attention at 256 positions. The schedules agree on the fidelity corpus (top-1 497/497, mean KL 5.1e-12) but not by construction, so a near tie could accept a draft the sequential decode would not. Prefill-side GQA would align them, but a token-outer group kernel measured 2048 prefill -5% and decode -13% (510 prefill +33%), and the K/V rows were already in L2.
- **Decision**: `--speculative` sets the per-head schedule on the model (scalar keeps its scalar path), so the verifier and the decode use identical arithmetic. Normal generation keeps auto's grouped schedule. The prefill stays per-head; the group kernel stays head-outer and its comment is corrected.
- **Consequences**: Speculative decode re-measured 2.00x on the repetitive story prompt with identical ids; at 2048 it is 4% ahead of the per-head run but 12% behind auto, and its output agrees with both on the test prompt. The prefill GQA is not shipped, so the 2048 prefill remains the engine's relative weak spot.

### ADR-003: VNNI batch kernels are opt-in, with pre-quantized activations
- **Date**: 2026-11-01
- **Status**: Accepted
- **Context**: The VNNI batch kernel lost end to end because it re-quantized the activation rows on every call, six to seven times per layer. Quantizing each distinct activation once and reusing it across q/k/v and gate/up turns the 510 prefill from 163.1 to 194.5 t/s and the 2048 from 165.6 to 188.2 (the best run; a later 2048 run fell inside the machine noise). The cost is the Q8 activation grid: mean KL 1.568e-3, worst 2.254e-2, top-1 485/497, and free-running ids differ.
- **Decision**: `--vnni` selects the VNNI batch kernels and pre-quantizes once per activation; the default stays the f32 kernels. The decode's single-token VNNI is not claimed (one 510 run lost 29%).
- **Consequences**: Prefill gains 14-19% on the best run, still about 65% of llama.cpp's 2048 rate; the accuracy trade is explicit and opt-in; default output and numbers are unchanged; the fidelity harness can price the option with `--vnni`.

### ADR-004: f16 KV cache is opt-in
- **Date**: 2026-11-01
- **Status**: Accepted
- **Context**: The KV cache is f32 while llama.cpp uses f16, and the 2048 decode is the engine's weak spot. An f16 cache halves the KV (192 -> 96 MiB at 8192 tokens) and reads it with F16C. Measured against the f32 reference over 497 positions: mean KL 1.658e-6, worst 2.446e-5, top-1 497/497, with identical free-running ids on the 2048, story and speculative traces. The speed is mixed (510 prefill +58%, 2048 prefill +14%/-14%, decode -24%/+63% across runs; the machine moves ±20%).
- **Decision**: Add `--kv f16` (half storage, F16C attention reads); the default stays f32. The plan still prints the f32 KV size, so the budget check stays conservative.
- **Consequences**: The memory case is the reason to use it; the accuracy price is two orders of magnitude below the VNNI's (1.7e-6 vs 1.6e-3 KL); the speed is not a uniform win, so it is not the default; tests 58 -> 60 with the half round-trip and f16 attention differential.

### ADR-005: Continuous batching with per-sequence KV slots
- **Date**: 2026-11-01
- **Status**: Accepted
- **Context**: The sessions benchmark runs N models, so N copies of the weights stream independently. The model can instead own N KV slots and decode one token per slot in one batch pass, sharing the weight read. Eight requests of 32 tokens measure 58.28 t/s aggregate with four slots against 42.21 for four concurrent models (+38%), and the first transcript is token-identical to the single decode.
- **Decision**: Add `sequences` to `Qwen2::load` (default 1), `forward_batch` (one token per sequence with per-token sequence and position), and `bench continuous` (admit into freed slots, prefill the seed per slot, decode the active batch). The scheduler is minimal; there is no paged KV or mixed prefill+decode in one batch.
- **Consequences**: The weights are shared across the decode batch and the KV is `sequences` times the single cache; a newly admitted request's prefill still reads the weights separately; the default single-slot paths are unchanged; the batch parity is checked through the CLI transcript, not a unit test. The follow-up batched a new request's prompt with the active decodes in the same pass (`--prompt-tokens`): 68.98 t/s at 8 prompt tokens vs 62.34 at 1, still identical to `generate`; a 32-token prompt crowds the 64-row batch (54.08), and the KV remains fixed per slot rather than paged.
### ADR-006: int8 KV cache is opt-in
- **Date**: 2026-11-02
- **Status**: Accepted
- **Context**: At 2048 tokens the engine is compute-bound (~12 GB/s effective vs 26 GB/s wall) and the Llama-3B KV is 448 MiB f32. f16 halved it for +14% decode (5.80 -> 6.61 t/s). An int8 block cache (one float scale per 32) halves it again (448 -> 112 MiB) with widen-then-scale attention reads. Measured at 2048: 7.63 t/s (+32% vs f32, +15% vs f16); short-context text identical, long-context prefix identical. Fidelity over 31 positions: mean KL 3.166e-4, worst 1.234e-3, top-1 31/31 — between f16 (1.7e-6) and the accepted VNNI (1.6e-3), zero flips.
- **Decision**: Add --kv i8 (int8 storage, widen-then-scale attention reads in head and group kernels, scalar oracle parity); the default stays f32. The plan still prints the f32 KV size, so the budget check stays conservative.
- **Consequences**: The memory case is the reason to use it (long-context, multi-session); at short context the widen+scale is pure overhead (5.20 vs ~9 t/s) so f32 stays faster there; tests 84 -> 86 with head and group int8 differentials.
