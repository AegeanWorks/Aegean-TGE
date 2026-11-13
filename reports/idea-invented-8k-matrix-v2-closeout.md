# CLOSEOUT v2: the 8k/12k decision matrix rebuilt — natural-text quality, workload-qualified speculation, and the retracted knobs

- **Idea**: `idea-invented-8k-decision-matrix-closeout.md` consolidated the first wave of 8k reports. Since then, eight cycles corrected or added material numbers: the natural-text tier ladder (which changed sparse's cost from 24-47pp to 13-19pp and i8's to 8.6pp), f16-vs-f32 speed, sampled speculation, prose speculation (which killed the lookup gate), the sampler-timer correction, the fused-vs-no-fuse wash, the 128-row chunk kill, and the two 12k cells (lock loss, code win). The v1 matrix's magnitudes are stale and it has no 12k column. Rebuild it as the authoritative decision table with a supersession ledger so no later report has to be read in isolation.
- **Cycle date**: 2026-11-13
- **Verdict**: **The quality-first 8k configuration is `--kv f16`, dense attention, with `--speculative` as the only quality-neutral accelerator (1.1-1.6x, workload- and acceptance-qualified). Every larger speed lever is lossy: sparse 13-19pp, i8 8.6pp, stride dilution 14pp. At 12k, f16 dense holds and speculation is acceptance-gated (code 1.33x, lock 0.84x); prefill dominates short outputs.** Full tables in §1-§3, supersessions in §4, open gaps in §5.

## 1. Quality at 8,199 positions (Qwen2.5-0.5B-Q8_0, eight distinct docs, candidate vs full-scalar f32-KV oracle)

| mode | mean KL | top-1 | cost | source |
|---|---:|---:|---:|---|
| f16 KV, dense | **2.312e-06** | **8191/8199 (99.90%)** | **−0.10pp** | natural-8k-tier-ladder |
| i8 KV, dense | 3.054e-02 | 7488/8199 (91.33%) | −8.57pp | natural-8k-tier-ladder |
| sparse 512/2 (~53% kept) | 1.376e-01 | 7096/8199 (86.55%) | −13.35pp | natural-8k-tier-ladder |
| sparse 512/4 (~30% kept) | 2.573e-01 | 6626/8199 (80.81%) | −19.09pp | natural-8k-tier-ladder |

The self-repeating `architecture.md x N` corpus that priced the first wave is **not** a valid substitute: it flattered i8 by 5.27pp (96.60 vs 91.33) and penalised sparse by 13.32pp (67.49 vs 80.81). Its numbers remain in the record only as corpus-specific readings. Speculation adds no quality row: greedy is verifier-exact (sparse vs spec+sparse ids byte-identical on code), sampled preserves the distribution.

## 2. Speed at 8k (Llama-3.2-3B-Q4_K_M, f16 KV unless stated, 10 threads, GPU off)

| lever | periodic (8,193) | prose (8,256) | code (8,104) | prefill effect | quality cost |
|---|---|---|---|---|---|
| dense off | 5.21-5.68 t/s; attn 54-56% | 5.78-5.86 | 6.52 | 205-213 s | none |
| `--speculative` | 8.09-8.38 (**1.42-1.61x**) | 6.91-7.25 (**1.18-1.25x**) | 7.34-7.46 (**1.13-1.14x**) | ~1.00x | none |
| sampled spec (temp 0.8/top-p 0.9) | 7.64-8.58 (**1.44-1.72x** vs sampled off 4.99-5.32) | — | — | ~1.00x | none |
| `--sparse` | 9.52-9.57 (**1.36-1.77x**); attn 24-25% | — | 10.49 (**1.61x**) | 1.06-1.15x | −13 to −19pp |
| `--speculative --sparse` | 15.61-16.36 (**2.2-3.0x**); 5.33/step | — | 11.12-11.92 (**1.71-1.83x**) | 1.10-1.19x | inherits sparse |
| f16 vs f32 KV | f16 **1.15-1.22x** decode, 1.04-1.14x prefill; attention 0.71-0.75x | — | — | f16 wins | f16 −0.10pp |
| block-sparse prefill (recipe) | decode unchanged | — | — | **162.4 s vs 205.3 dense (1.265x)**; 1.18-1.24x vs per-token sparse; verify 0.59x | = sparse (bit-identical) |
| chunk 128 above 4096 | — | — | — | **0.95-0.99x (killed)** | none |
| fused vs `--no-fuse` | **0.98-1.04x (wash)** | — | — | — | none |

Notes that must travel with the table:

- **Ratios are not portable without accepted/step.** Spec: 3.00-5.33 accepted/step at 8k; 3.17 on prose; 3.00 on code; sampled 3.29. The verify's six-row attention is the cost, and its per-row share falls with the batch.
- **Sampler cost is ~10 ms/run** (softmax 2.5 ms + cut 7.5 ms). The `timers sample` line on speculative decode is a double count (it encloses the verify forward); do not use it. There is no sampler bottleneck.
- **Prefill dominates short outputs**: at 8k a 16-32 token run is 205-213 s prefill vs 1.6-4.8 s decode, so whole-run wall follows the prefill column. Decode ratios matter only for long generations.
- **`--sparse` also routes the verifier off block attention** onto the per-token sparse kernel, which is why spec+sparse gains more than sparse alone on periodic text; the acceptance change (3.00 -> 5.33/step there) belongs to the priced sparse tier, not to a better drafter.

## 3. The 12k column (new)

| cell | off | spec | spec line | prefill | decode attn |
|---|---:|---:|---|---:|---:|
| 12,289-token lock-1 | 4.15 t/s | **3.49 t/s (0.84x)** | 4 steps, 1.50/step, 10 emitted | 407 s (attn 45%) | 67% |
| 12,284-token code | 3.87-4.00 t/s | **5.24 t/s (1.31-1.35x)** | 2 steps, 3.50/step, 9 emitted | 408-474 s (attn 44-46%) | 65-67% |

Speculation's 12k viability is **acceptance-gated**: 1.50/step loses, 3.50/step wins, and the break-even sits between them (8k break-even is below 3.00). Prefill at 12k is 26-30 t/s and attention is ~45% of it; the quadratic term is visible. The only 12k prompts testable are code and lock (prose needs ~43k chars, beyond the one-argument command-line limit).

## 4. Supersession ledger (v1 matrix rows that changed)

| v1 claim | status in v2 |
|---|---|
| sparse costs 24-47pp at 8k | **superseded**: 13-19pp on natural text; the 24-47pp rows were repetition-corpus readings |
| i8 costs 3.4pp at 8k | **superseded**: 8.57pp natural, and flat at 2k (90.05%) vs 8k (91.33%) |
| "sparse is the only large speed lever" | true as a **speed** statement; quality cost restated |
| `--sparse-stride 16` is a 2.25pp option | **withdrawn**: 14.37pp at 8,199 positions |
| spec 8k is 2.2-3.0x | **qualified**: 1.42-1.61x periodic, 1.18-1.25x prose, 1.13x code; 2.2-3.0x only for spec+sparse on periodic |
| f16 is quality-first "at some speed cost" | **superseded**: f16 is also the fastest KV tier (1.15-1.22x decode over f32) |
| "sampler is 53-55% of spec wall; batch the cut next" | **withdrawn**: timer double count; real sampler ~10 ms |
| "the lookup gate is the long-prompt policy" | **withdrawn for long prompts**: 8k prose spec wins 1.18-1.25x, so a t=0 last-window gate would disable it |
| (not in v1) | **new**: fused/no-fuse wash; chunk 128 kill; 12k lock loss; 12k code win |

## 5. Recommended configurations

1. **8k quality-first**: `--kv f16`, dense attention, add `--speculative` when accepted/step is ~3+ (periodic 1.5x, prose 1.2x, code 1.13x). This is the only free-on-both-axes configuration.
2. **8k speed-first, lossy accepted**: `--kv f16 --speculative --sparse` (periodic 2.2-3.0x, code 1.71-1.83x) with the 13-19pp natural-text cost stated. Add the block-sparse prefill recipe when ported for another ~1.2x prefill; it inherits the sparse tier exactly.
3. **8k capacity**: `--kv i8`, dense; KV quartered for 8.6pp top-1; decode speed at 8k is a measured loss, so not a speed mode.
4. **12k**: f16 dense; `--speculative` only after a sample confirms accepted/step (code 1.33x; lock 0.84x). Prefill dominates short outputs.
5. **Do not**: sparse-stride 16 as a general option; sparse window <512 for quality work; the lookup gate for long prompts; chunk 128; any sampler-side "bottleneck" work; the 2k i8 bar as an 8k claim; a self-repeating corpus for tier pricing.

## 6. Open gaps

- **Mixed f16-K/i8-V (or K-i8/V-f16)**: still the only path to a half-memory tier that might clear the 8k quality bar. Needs per-tensor tier flags across `attention_head`, `attention_group`, `attention_block`, and the scalar oracle, plus store/load — a multi-kernel feature, not a one-cycle spike.
- **Quality-preserving long-context prefill**: none on record. Attention is compute-bound (tiled ~0%, grouped slower, chunk free); the only measured prefill levers are lossy sparse variants.
- **Speculation at 12k on prose**: blocked by the command-line limit (a `--prompt-file` flag would unblock it); prose accepted 3.17/step at 8k, so a 12k win is plausible but unmeasured.
- **Drafter quality at long context**: acceptance, not the engine, is now the binding constraint for spec at 12k (1.50/step on lock). A better drafter is a different project.

## 7. File anchors (current HEAD, read-before-write verified)

- Quality/tier harness: `apps/cli/main.cpp:2520-2545` (reference vs candidate), `:2699+` (fidelity parser); `src/models/qwen2.cpp:117-134` (`store_kv_q8`), `:573-586` (allocation), `:1329-1362` (sparse dispatch), `:1580-1660` (block gate/fallback).
- Kernels: `src/cpu/avx2/attention_avx2.cpp:263,367` (strided sparse), `:660-864` (block + value tile).
- Spec: `apps/cli/main.cpp:788-922` (fused chain), `:924-1070` (non-fused), `:1081-1195` (sampled), `:2165-2171` (`--speculative`, `--no-fuse`).
- Chunk rule: `src/models/qwen2.cpp:33-38`, `:2406`.
- Consolidated reports: `idea-invented-natural-8k-tier-ladder-measured.md`, `idea-invented-f16-vs-f32-8k-measured.md`, `idea-invented-spec-at-8k-measured.md`, `idea-invented-spec-prose-8k-kills-lookup-gate.md`, `idea-invented-code-8k-spec-sparse-measured.md`, `idea-invented-sampled-spec-8k-measured.md`, `idea-invented-sampler-timer-correction.md`, `idea-invented-sparse-8k-reopened.md`, `idea-invented-sparse-prefill-8k-measured.md`, `idea-invented-block-sparse-prefill-keep.md`, `idea-invented-sparse-8k-quality-correction.md`, `idea-invented-i8-corpus-dependence-correction.md`, `idea-invented-fused-vs-nofuse-8k-measured.md`, `idea-invented-chunk128-8k-killed.md`, `idea-invented-spec-12k-killed.md`, `idea-invented-spec-12k-code-measured.md`.

## 8. Reimplementation recipe (rebuild any cell)

1. Gate: `cmake --build --preset release` + `tge_tests.exe` (171/171).
2. Quality: `tge fidelity --model qwen2.5-0.5b-instruct-q8_0.gguf --corpus <eight-docs> --context 8704 --tokens 8200 --threads 10 --gpu off [--kv f16|i8] [--sparse ...]`; expect §1. Each run 8-11 min.
3. Speed: `tge generate --model Llama-3.2-3B-Instruct-Q4_K_M.gguf --prompt <compact fox/code/prose> --context 8704 --n 16-32 --kv f16 --threads 10 --gpu off [--speculative] [--sparse] [--no-fuse]`; interleave pairs and compare only pairs whose matvec phases agree within ~3% (box drifts 20-40%/20 min).
4. 12k: lock `a`+` a`x12,288 or the 323-function code prompt, `--context 12800 --n 16`; one run per shell call (~7-8 min prefill).
5. Id parity: greedy spec is verifier-exact; compare `--print-ids` when claiming identical output. Sampled spec preserves distribution, not ids.
6. No code ships.

## 9. Exact commands run and results

This cycle ran no new measurements. Gate for the commit: `cmake --build --preset release` clean (56/56), `tge_tests` 171/171, `python aikit.py check aegean-tge` 4/4. Every number in §1-§3 is lifted from the committed report listed beside it; the corrections in §4 are the explicit supersessions of `idea-invented-8k-decision-matrix-closeout.md`.

No push, no merge, no force-push. The committed tree is unchanged except this report.
