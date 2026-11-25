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

### ADR-007: Static sparse attention subset (opt-in)
- **Date**: 2026-11-03 (numbered for the `.aikit` sequence; full record lives at `docs/decisions/ADR-0008-static-sparse-subset.md`, whose filename is kept to avoid colliding with the parallel Vulkan `ADR-0007` draft)
- **Status**: Accepted
- **Context**: Latent Sparse Attention (arXiv 2610.42147) selects a kept set per query (sink + selected long-range + window); its indexer (distilled latents, GPU kernels, 10M context) cannot land here. Full record in `docs/decisions/ADR-0008-static-sparse-subset.md`; measurement in `docs/measurement-log.md` §50.
- **Decision**: `SparseConfig{enabled,sink=4,window=512,stride=4}` gathered in `attention_head` (f32 KV only), `uses_sparse` before `uses_gqa`, exact dense fallback otherwise; `--sparse` on `generate`/`fidelity`; default off.
- **Consequences**: Fidelity Q8_0 ctx1024: KL 2.6e-02, top-1 975/1023. Generate 8192: prefill attn phase lower both pairs, ids 31/31 identical; 512: ids identical via fallback. Kept opt-in for prefill-bound long prompts.

### ADR-008: Fail-loud error contracts for shapes, params, and schemas
- **Date**: 2026-11-04 (numbered for the `.aikit` sequence; the `docs/decisions/` filename series is separate)
- **Status**: Accepted
- **Context**: Differential testing (scalar vs AVX2, AVX2 vs reference) stays green while both sides compute the wrong thing: truncated block prefixes benchmark as exact, fused BPE merges forge control ids, NaN sampling params and merge ranks degrade silently, unknown GGUF value types desync the cursor, and a throwing test case used to abort the whole suite. Real Qwen/Llama vocabs and model shapes never hit these paths (full byte coverage, whole blocks, valid params), so loud errors cost the hot path one predictable branch and change nothing for valid inputs.
- **Decision**: Throw `std::invalid_argument` (or `runtime_error` at file load) for non-whole-block cols, negative/NaN/out-of-range sampling params, null buffers with size > 0, re-entrant `parallel_for`, attention shapes with stride < head_dim or ragged q8 blocks, unencodable BPE/SPM bytes, unknown GGUF value types, shape overflow, and non-finite SPM scores; exact-fit plan headroom admits weights. The documented "off" sentinels (top_k/top_p/min_p defaults, penalty 1.0, window 0) and degenerate `count <= 0` early-returns are preserved; the CLI catches `std::exception` at its generate path, so bad flags print clean errors. The test harness contains throwing cases and continues the suite.
- **Consequences**: Tests 75 -> 122 with one failing-first test per fix (RoPE odd-dim, alignment cap, plan math, argmax, byte budgets, GGUF strictness x2, sampling validation, seed-0, thread-pool x2, harness containment, BPE forgery/drop, SPM byte/NaN, quant-tail x2, attention shapes, forward smoke); `scripts\test.cmd` and `aikit.py check aegean-tge` stay 4/4 green. Still deferred: CLI numeric strictness (entangled `main.cpp`), VNNI/KQ8 fallback contract, Linux topology stub, GPU cache/loss/threading items, license-text paths, SPM greedy-vs-oracle question.

### ADR-009: kq8d decode scopes are opt-in, default off
- **Date**: 2026-11-08
- **Status**: Accepted
- **Context**: The LM head (3072x128256 Q6_K) and the FFN down projection still ran the float-dequant path. The integer kq8d row kernel is 2.9-3.6x faster on those shapes and decode is matvec-bound, but the activation quantization changes near-tie logits. Block prefill attention, the full q8 value tail and four GQA decode chunks are arithmetic-order or scheduling changes with identical ids, so they can ship as defaults.
- **Decision**: Add `--kq8d head|down|head,down|off` (default off) on `generate` and `fidelity`. `head` routes only `output_logits`; `down` only the single-token `w_down`; prefill batch paths and the attention projections stay float. The defaults ship query-block prefill attention for f32/f16 KV, the full 128-dimension q8 value tail, and four GQA history chunks per KV head above 1k positions; i8 prefill stays per-token.
- **Consequences**: `head`: 399-position mean KL 1.874e-4, top-1 393/399, and 98.83% at 2,399 positions; head matvec 3.3-3.6x, decode about 1.3x at 4k. `head,down`: 9.711e-4 / 390/399 and 97.67% at 2,399 (f16 97.79%); the three-scope stack fails at 97.33%. Default output is unchanged; the opt-in tiers are priced by `tge fidelity --kq8d` and documented in the README and CLI help.
### ADR-010: generate reads long prompts from a file
- **Date**: 2026-11-15
- **Status**: Accepted
- **Context**: The 12k-token prose speculation cell could not be measured because a prompt that long exceeds the Windows command-line limit (about 32k characters) when passed through `--prompt`.
- **Decision**: Add `generate --prompt-file PATH`. The file is read whole, in binary mode, into the same text that `--prompt` sets. A missing, unreadable or empty file, or combining it with `--prompt` or `--tokens`, exits 2 with a `generate:` message. `load_prompt_file` and `prompt_source_conflict` live in `apps/cli/cli_options.hpp` so unit tests can cover them without loading a model.
- **Consequences**: Prompts of any length can be used. The flag is opt-in, so default behaviour is unchanged. Three CLI option tests pin load, rejection and usage text.

### ADR-011: Adaptive greedy verification for isolated one-token drafts
- **Date**: 2026-11-24
- **Status**: Accepted
- **Context**: Greedy prompt-lookup speculation always verified six rows, even when an isolated one-token exact match offered little evidence for a long continuation. Shortening two-token, bigram or gapped matches broke the fused repetitive chain and measured -18.1 percent on the periodic control.
- **Decision**: Only a non-periodic one-token exact match uses a two-row greedy verifier; the second row predicts the successor/correction token. Two-token and longer exact matches, bigram/gapped drafts, period-1/2 locks, sampled speculation and fused steady state keep six rows. Short batches always materialize the correction forward and never enter fused state. The spec summary reports total verifier rows.
- **Consequences**: Generated ids remain byte-identical in interleaved A/B on Q8_0 0.5B/1.5B and Q4_K_M SmolLM2, including periodic and repeat-penalty controls. Natural-prompt decode medians improved 9.7 percent (22/24), 5.1 percent (12/12) and 6.4 percent (9/12); verifier rows fell 72->44, 18->6 and 48->24. Non-speculative and sampled paths are unchanged; weak proposals still pay one correction forward.

### ADR-012: Grouped Q8_0 decode projections for wide models
- **Date**: 2026-11-25
- **Status**: Accepted
- **Context**: Decode performs 169 thread-pool dispatches per token; an empty dispatch measures 4.62 us. Independent same-input Q8_0 projections currently pay separate joins. Grouping improved large Q8_0 models but not the 0.5B model, and inlining the group machinery perturbed fallback paths.
- **Decision**: Add a grouped Q8_0 GEMV kernel that keeps each matrix's row arithmetic and band factor but shares one dynamic-work pool dispatch. Use it for Q/K/V and gate/up only when embedding width is at least 1024, AVX2/FMA is present, and neither scalar kernels nor VNNI is active. Keep the helper out of line, use a stack task fast path with heap fallback, and retain sequential matvec for every other case.
- **Consequences**: Final interleaved decode medians improved 2.8 percent on Qwen2.5-Coder-1.5B Q8_0 (12/12 pairs) and 3.0 percent on train-run16a Q8_0 (11/12), with byte-identical ids. The 0.5B control was neutral at -0.1 percent and the unchanged Q4_K_M control was inside noise at -1.6 percent. Tests increased from 189 to 190 with a bitwise grouped-vs-individual differential and partial-block rejection coverage.

### ADR-013: Separate grouped and sequential decode forward loops
- **Date**: 2026-11-25
- **Status**: Accepted
- **Context**: After grouped Q8_0 dispatch was wired into the shared forward loop, unchanged small/Q4_K_M fallback models showed code-layout and per-layer branch penalties even though their kernels were not called. Moving only the group helper out of line reduced but did not eliminate the effect.
- **Decision**: Select grouped versus sequential decode once at the public forward boundary and instantiate two noinline forward_impl template specializations. The sequential specialization contains no grouping branches, task arrays or group calls; per-layer tensor checks remain only in the grouped specialization.
- **Consequences**: Fallback controls returned inside the noise band: Qwen2.5-0.5B +0.6 percent (6/12) and SmolLM2 Q4_K_M -1.3 percent (6/12), all ids identical. Wide Q8_0 gains remained: Coder-1.5B +2.7 percent (11/12) and train-run16a +3.4 percent (20/24). The source loop stays single-sourced while the generated hot paths are isolated.
