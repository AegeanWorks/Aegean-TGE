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
