# Idea / spike verdict index

This index preserves the verdict of every spike/idea report that was triaged
out of `reports/` (a git-ignored inbox with a 20-file cap). The full reports
were local scratch and are intentionally not committed; the decision from each
is recorded here so it remains reviewable. Generated 2026-11-19.

294 reports triaged.

| Idea | Report | Verdict | Decision |
|---|---|---|---|
| IDEA-1325 | `idea-1325-pack-on-load-split.md` | SPLIT | IDEA-1325: Pre-Packed Weight Matrix with Blocking Factors for Prefill - SPLIT (blocking/packing ships; persistent pack-on-load killed with no caller) |
| IDEA-1327 | `idea-1327-dual-accumulator-q8-gemv.md` | MEASURED | IDEA-1327: AVX2 Exact-Int8 Dual-Accumulator Q8 GEMV Dot - MEASURED AND KILLED |
| IDEA-1330 | `idea-1330-rht-probe-killed.md` | KILLED | IDEA-1330 STAGE-1: RHT Substrate Probe for TurboQuant KV - KILLED (RHT works exactly as theory predicts, and theory is not enough) |
| IDEA-1341 | `idea-1341-chunk-indexed-killed-analytic.md` | KILLED | IDEA-1341: Chunk-Indexed KV Selection with QShare Prefill Reuse - KILLED WITHOUT SPIKE (scored-sparsity family dead; QShare dies with selection) |
| IDEA-1382 | `idea-1382-moe-streaming-killed-analytic.md` | KILLED | IDEA-1382: NVMe Expert Streaming with Arena and Background Warm - KILLED WITHOUT SPIKE (no MoE; working set fits RAM 3x over) |
| IDEA-1383 | `idea-1383-kvarn-mtp-killed-analytic.md` | KILLED | IDEA-1383: Low-bit KV quants + multi-path speculative framework - KILLED WITHOUT SPIKE (falsifier tripped by the int4 verdict; draft half dead three times over) |
| IDEA-1384 | `idea-1384-qcow-split.md` | SPLIT | IDEA-1384: Paged Quantized KV with CoW Prefix Sharing - SPLIT (zero-copy killed analytic; quantized snapshot keep-grade recipe) |
| IDEA-1385 | `idea-1385-int4kv-killed.md` | KILLED | IDEA-1385 int4-KV sub-idea: 4-Bit KIVI-style KV Tier - KILLED (KL 4.6e-3 vs 1e-3, top-1 83/86) |
| IDEA-1386 | `idea-1386-ptq-export-killed-analytic.md` | KILLED | IDEA-1386: Sign-Aware PTQ with K-quant-aware GGUF Export - KILLED WITHOUT SPIKE (no write path; inference-only engine) |
| IDEA-1387 | `idea-1387-trie-drafter-spike-killed.md` | MEASURED | IDEA-1387: Online Majority-Vote Trie Drafter - SPIKED, MEASURED, KILLED (acceptance 3.45 → 3.18) |
| IDEA-1394 | `idea-1394-quantize-once-integrated.md` | INTEGRATED | IDEA-1394: Quantize-Activations-Once CPU Dispatch - INTEGRATED, CONFIRMED ON TODAY'S TREE (kq8 inside the 1e-9 bar, VNNI priced outside it by explicit gate) |
| IDEA-1395 | `idea-1395-kvarn-dominated-killed-analytic.md` | KILLED | IDEA-1395: Variance-Normalized KV Cache with Precision Tail - KILLED WITHOUT SPIKE (dominated on the size-KL frontier by shipped i8) |
| IDEA-1395 | `idea-1395-tail-flag-validation-measured.md` | MEASURED | IDEA-1395 tail as a real flag (`--kv-tail-tokens`) - MEASURED through the flag, then REVERTED per instruction |
| IDEA-1395 | `idea-1395-tail-slice-measured.md` | MEASURED | IDEA-1395 (slice: exact-fp32 KV tail over i8) - MEASURED: quality wins 10x, decode at parity; keep-grade recipe filed, spike reverted |
| IDEA-1451 | `idea-1451-scored-sparse-killed.md` | KILLED | IDEA-1451: Mid-Depth Unified KV Scoring and Eviction - SPIKED AS SCORED-MIDDLE PROXY, KILLED ON THE SPEED ARM (2x attention cost for +3 flips) |
| IDEA-1452 | `idea-1452-cisa-retrofit-superseded.md` | SUPERSEDED | IDEA-1452: No-Gradient Chunk-Indexed Sparse KV Retrofit (CISA) - SUPERSEDED BY 1341 (same paper, same C=16 mechanism, same analytic kill) |
| IDEA-1458 | `idea-1458-fused-kquant-killed-analytic.md` | KILLED | IDEA-1458: Fused K-Quant Dot with Unpack Reuse and Row-Interleaved Repack - KILLED WITHOUT SPIKE (shipped where it wins; 1-token bar fails at 7.85 vs 15+) |
| IDEA-1476 | `idea-1476-xlayer-probe-killed.md` | KILLED | IDEA-1476 STAGE-1: Cross-Layer KV Reconstruction Probe - KILLED (best affine residual 72-96%, zero viable layers) |
| IDEA-1477 | `idea-1477-spta-teacher-killed-analytic.md` | KILLED | IDEA-1477: Prefix-Conditioned Teacher Adaptation for Draft Models (SPTA) - KILLED WITHOUT SPIKE (training idea proposed to an inference-only engine; acceptance already saturated without it) |
| IDEA-1478 | `idea-1478-vers-stop-killed-analytic.md` | KILLED | IDEA-1478: Doubt-Marker-Gated Early Termination (VERS, Harness-Only) - KILLED WITHOUT SPIKE (accuracy-gated falsifier, zero accuracy carriers) |
| IDEA-1497 | `idea-1497-folded-bias-killed-analytic.md` | KILLED | IDEA-1497: Folded-Bias Compacted KV Cache - KILLED WITHOUT SPIKE (no far-context carrier; both readings land on dead precedent) |
| IDEA-1498 | `idea-1498-middepth-scorer-superseded.md` | SUPERSEDED | IDEA-1498: Single Mid-Depth Scorer for Shared KV Eviction Sets - SUPERSEDED BY 1451 (same paper, same mechanism, same measured kill) |
| IDEA-1499 | `idea-1499-rift-static-split-killed-analytic.md` | SPLIT | IDEA-1499: Static-Split Draft KV Cache with Periodic Refresh - KILLED WITHOUT SPIKE (RIFT twin of the dead 1526; no draft model, no 32K harness) |
| IDEA-1501 | `idea-1501-l2gqa-qow-superseded.md` | SUPERSEDED | IDEA-1501: L2-Blocked Online-Softmax GQA with Quantize-on-Write KV - SUPERSEDED (chunked GQA + qow KV ship) |
| IDEA-1502 | `idea-1502-rolling-hash-killed-analytic.md` | KILLED | IDEA-1502: Rolling-Hash Speculative Memory with Adaptive Gate - KILLED WITHOUT SPIKE (exact version already lost; no cross-request carrier) |
| IDEA-1514 | `idea-1514-antidiagonal-sparse-killed-analytic.md` | KILLED | IDEA-1514: Antidiagonal Block-Sparse Attention Port - KILLED WITHOUT SPIKE (analytic, measured) |
| IDEA-1515 | `idea-1515-ablation-profile-killed.md` | KILLED | IDEA-1515 STAGE-1: Per-Layer i8 Sensitivity Profile - KILLED (flat where it matters; joint effect irreducible) |
| IDEA-1515 | `idea-1515-budget-mixed-killed-analytic.md` | KILLED | IDEA-1515: Budget-Aware Mixed-Precision KV Tier (MultiAlloc) - KILLED analytic (no carrier on today's tree) |
| IDEA-1516 | `idea-1516-kivi-stage1-probe-proceed.md` | PROCEED | IDEA-1516 STAGE-1: KIVI Substrate Probe - PROCEED (premise holds; V flagged for stage-2 fidelity) |
| IDEA-1516 | `idea-1516-kivi-stage2a-killed.md` | KILLED | IDEA-1516 STAGE-2A: KIVI 2-Bit KV Tier, Format Fidelity - KILLED (KL 0.13 vs 1e-3 bar, top-1 75/86) |
| IDEA-1516 | `idea-1516-stage1b-vprobe-killed.md` | KILLED | IDEA-1516 STAGE-1B: V per-32 Variant Probe - KILLED (0.6-0.75x, needs <0.5x; KIVI line closed barring outlier-aware coding) |
| IDEA-1517 | `idea-1517-paged-kv-stage1-proceed.md` | PROCEED | IDEA-1517: Paged KV / prefix-hash - stage-1 probe **PROCEED** (fragmentation and indirection are not the blocker) |
| IDEA-1517 | `idea-1517-paged-kv-stage2-killed.md` | KILLED | IDEA-1517: Paged KV stage 2 (storage wiring) - KILLED (correct, but per-row lookup costs 6-15%; 2% bar fails) |
| IDEA-1517 | `idea-1517-paged-kv-stage2b-marginal.md` | MARGINAL | IDEA-1517: Paged KV stage 2b (row-pointer hoist) - MARGINAL (recovers to within 2%, but no speed win) |
| IDEA-1517 | `idea-1517-stage1-headroom-proceed.md` | MEASURED | IDEA-1517 STAGE-1: Continuous-Batch Prefix Sharing - Headroom Measured, Falsifier Re-based, EPIC PROCEED with Staged Recipe |
| IDEA-1517 | `idea-1517-stage1-populate-spike-measured.md` | MEASURED | IDEA-1517 STAGE-1: Populate-from-Batch Snapshot - MEASURED, PROCEED (bytes identical, overhead ~0%) |
| IDEA-1518 | `idea-1518-moe-streaming-killed-analytic.md` | KILLED | IDEA-1518: NVMe-to-RAM MoE Expert Streaming with 8-bit KV + Draft Routing - KILLED WITHOUT SPIKE (no MoE; other halves ship) |
| IDEA-1519 | `idea-1519-linear-recurrence-killed-analytic.md` | KILLED | IDEA-1519: Linear-Recurrence Attention with Streaming MXFP4 Experts - KILLED WITHOUT SPIKE (triple no-carrier) |
| IDEA-1526 | `idea-1526-draft-kv-refresh-killed-analytic.md` | KILLED | IDEA-1526: Decode-Time Draft-KV Refresh with Adaptive Rescoring - KILLED WITHOUT SPIKE (no decay; no draft KV) |
| IDEA-1539 | `idea-1539-1540-gemm-layer-superseded.md` | SUPERSEDED | IDEA-1539 + IDEA-1540: Broadcast GEMM Microkernel + Packed-B M==1 Fast Path - SUPERSEDED (the described layer is the shipped layer) |
| IDEA-1541 | `idea-1541-numa-dispatch-killed-analytic.md` | KILLED | IDEA-1541: NUMA-Aware Chunked Work-Stealing Decode Dispatch - KILLED WITHOUT SPIKE (precondition absent) |
| IDEA-1542 | `idea-1542-ngram-kv-superseded.md` | SUPERSEDED | IDEA-1542: Lossless Ngram Speculative Draft plus Priced KV Types - SUPERSEDED (both halves ship; sub-i8 KV residue named for the KV track) |
| IDEA-1553 | `idea-1553-paged-prefetch-split.md` | SPLIT | IDEA-1553: Paged KV with Bounded Prefetch and mmap Advice - SPLIT (advice-for-RSS killed as self-contradictory; paged half redirects to 1517/1585; Q4xQ8 arm already ships) |
| IDEA-1558 | `idea-1558-scatter-q40-spike-killed.md` | MEASURED | IDEA-1558: Scatter-Accumulate Exact-Int Dot for Q4_0 - SPIKED, MEASURED, KILLED (2.9-7.6x slower) |
| IDEA-1559 | `idea-1559-shq-lut-killed-analytic.md` | KILLED | IDEA-1559: Learnable Companded Grids (SHq) as Weight-LUT Quantization - KILLED WITHOUT SPIKE (no quantizer, no writer, no files, no registry kind) |
| IDEA-1568 | `idea-1568-ablation-closeout.md` | CLOSED | IDEA-1568: Portability Ablation Protocol - CLOSED (the run the deep-check asked for; triage CONFIRMED with two refinements) |
| IDEA-1569 | `idea-1569-inside-gemm-killed-analytic.md` | KILLED | IDEA-1569: Compression-Inside-GEMM with Per-Shape Autotune and Fused Gate/Up TwoMatMul - KILLED WITHOUT SPIKE (autotune target is bench-only; fusion Amdahl-bound below 1%) |
| IDEA-1576 | `idea-1576-chunksize-probe-killed.md` | KILLED | IDEA-1576 STAGE-1: Prefill Chunk-Size Sensitivity Probe (64 → 256) - KILLED (chunk size is not the lever) |
| IDEA-1581 | `idea-1581-compute-in-quant-superseded.md` | SUPERSEDED | IDEA-1581: Compute-in-quant CPU GEMM with fidelity oracle - SUPERSEDED (integer dot ships default-on) |
| IDEA-1582 | `idea-1582-rust-intdot-split.md` | SPLIT | IDEA-1582: Pure-Rust GGUF mmap with Int-Dot Tiers and Parity Gate - SPLIT (int-dot + parity ships; Rust rewrite dead: zero .rs files, wrong language) |
| IDEA-1585 | `idea-1585-block-table-split.md` | SPLIT | IDEA-1585: Block-Table Slot Mapping + KV Block Layout + Tiled Scheduler - SPLIT (headers ship, scheduler ships, wiring specified; 58.3 bar re-grounded at 2x) |
| IDEA-1586 | `idea-1586-portable-io-split.md` | SPLIT | IDEA-1586: Portable IO Shims with Trunk-Stream Prefetch - SPLIT (aligned-alloc shim + bitwise identity ship; O_DIRECT/trunk/expert/MXFP4 halves dead) |
| IDEA-1635 | `idea-1635-quant-registry-split.md` | SPLIT | IDEA-1635: Quant-Registry Dispatch with Coherence Harness - SPLIT (registry + harness ship; AVX512/repack/pread/TurboQuant-KV dead) |
| IDEA-1636 | `idea-1636-scale-pack-split.md` | SPLIT | IDEA-1636: Scale-Packing Micro-opt + Deterministic Block KV - SPLIT: prefix half duplicates CoW; scale half spiked, killed on split (1T +8%, 10T −4..−13%) |
| IDEA-1714 | `idea-1714-dequant-null-guards-spike-measured.md` | MEASURED | IDEA-1714 (flavored): Fail-Loud Null Guards on the K-Quant Batch Entries - SPIKED, MEASURED, REVERTED |
| -- | `idea-cycle-negative-results-2026-11-08.md` | RECORD | Negative cycle: adaptive speculation, 256-row chunks, Q4_K metadata cache |
| -- | `idea-decode-int-round2-killed.md` | KILLED | IDEA-1582 FOLLOW-UP: Decode Integer Path, Round-2 - KILLED again (1013/1023, deterministic max\|diff\| 0.66) |
| -- | `idea-decode-int-round3-captured-kill-confirmed.md` | CAPTURED | IDEA-DECODE-INT ROUND 3: Captured Real-Activation Differential - KILL CONFIRMED (rel ≤ 28.6%, p99 10%; mechanism found) |
| -- | `idea-decode-int-round4-granularity-probe.md` | PROCEED | IDEA-DECODE-INT ROUND 4: Activation Granularity Probe - PROCEED dot kernels (kernel exact, per-32 wins 2.6x) |
| -- | `idea-decode-int-round5-per32-reverted.md` | RECORD | IDEA-DECODE-INT ROUND 5: Per-32 Activation Dots - BUILT, FALSIFIED, REVERTED |
| -- | `idea-decode-int-round6-asymmetric-killed.md` | KILLED | IDEA-DECODE-INT ROUND 6: Asymmetric per-32 Activation - KILLED (0.68x, needs <0.5x) |
| -- | `idea-fusion-add-norm-killed-analytic.md` | KILLED | IDEA-FUSION: Residual-Add + RMSNorm Fusion - KILLED analytic (mechanism caps at ~1%) |
| -- | `idea-invented-2048-kv-ladder-closure.md` | MEASURED | INVENTED: 2048-History KV Ladder + Head-to-Head Closure - MEASURED (§45 suspension lifted; §29 i8 headline corrected; §41 mechanism confirmed) |
| -- | `idea-invented-8k-decision-matrix-closeout.md` | RECORD | CLOSEOUT: the 8k long-context decision matrix - every lever priced on speed and quality; f16 dense is the only free mode |
| -- | `idea-invented-8k-matrix-v2-closeout.md` | RECORD | CLOSEOUT v2: the 8k/12k decision matrix rebuilt - natural-text quality, workload-qualified speculation, and the retracted knobs |
| -- | `idea-invented-abstract-slot-tests-keep.md` | RECORD | INVENTED: abstract KV-slot property tests with a deterministic nonzero fixture - KEEP-GRADE (tests pass; both mutations caught) |
| -- | `idea-invented-attnsplit-measured.md` | SPLIT | INVENTED: Attention Sub-Phase Split Diagnostic - MEASURED (dots ~47% / softmax 3% / V-sum ~50%, flat; no lever, close the line) |
| -- | `idea-invented-auto-gate-blindspot-measured.md` | MEASURED | MEASURED: the lookup gate's blind spot is real - a late-drafting code prompt loses 1.88x (spec 22.5 vs off 12.0 t/s), so the gate must not be the only long-prompt policy |
| -- | `idea-invented-auto-spec-2k-measured.md` | MEASURED | MEASURED: `--speculative auto` probation at 2k context - periodic stays on (2.4-3.1x), prose pays 5-8% while it bails; probation length is not the lever |
| -- | `idea-invented-auto-spec-lookup-gate-keep.md` | RECORD | INVENTED: long-prompt lookup gate for `--speculative auto` - KEEP-GRADE (2k periodic 2.87x, 2k prose 1.06x with zero spec cost, short prompts unchanged) |
| -- | `idea-invented-auto-spec-two-tier-killed.md` | MEASURED | MEASURED: two-tier long-prompt auto-spec probe (gate + 1-2 step model probe) - KILLED; the free gate stays, and the blind-spot premise is corrected |
| -- | `idea-invented-batchcvtph-killed.md` | KILLED | INVENTED (1636 follow-up): Batched Scale Conversion in dot_row_q8 - KILLED (1T +12%, 10T −9%: the thread split repeats, shipped default loses) |
| -- | `idea-invented-batchsnap-kindtag-measured.md` | MEASURED | INVENTED: S6-Amendment Stage-1 Re-spike - Kind-Tagged Batch Snapshot + Same-Kind FNV Gate - MEASURED (Q8_0 MATCH 3/3, Q4_K refused-by-construction, kind refined to whole-model) |
| -- | `idea-invented-block-profile-analysis.md` | RECORD | ANALYSIS: block prefill attention internal profile - value pass 68%, QK 30%, softmax 3%; no AVX2 headroom |
| -- | `idea-invented-block-sparse-prefill-keep.md` | KEPT | INVENTED: block-sparse prefill (kept-set walk inside the 128-query block kernel) - KEEP-GRADE (8k prefill 1.27x dense / 1.18-1.24x per-token sparse; spec verify 1.24x; 2k ids bit-identical) |
| -- | `idea-invented-bpe-heap-encode-keep.md` | RECORD | INVENTED: BPE heap encoder - KEEP (192-400x on long pre-tokens, ids identical) |
| -- | `idea-invented-chunk128-8k-killed.md` | MEASURED | MEASURED: 128-row prompt chunks above 4096 are still a loss at 8k (0.95-0.99x) - the block kernel's K/V reuse is not the binding term |
| -- | `idea-invented-chunk96-8k-killed-analytic.md` | KILLED | INVENTED: 96-token prefill chunks above 4096 - KILLED ANALYTICALLY (per-token kq8 throughput is best at 64) |
| -- | `idea-invented-chunkcap-killed.md` | KILLED | INVENTED: L2-Cap GQA Chunk Sizing for Long-Context Decode - KILLED (parallelism knob, already saturated; L2 never binds) |
| -- | `idea-invented-chunkcap-longctx-killed.md` | KILLED | INVENTED: GQA Chunk-Count Sweep at 8k KV (L2-Cap Follow-up) - KILLED (plateau holds; +3.5% best, under the bar) |
| -- | `idea-invented-code-8k-spec-sparse-measured.md` | MEASURED | MEASURED: 8k decode on non-periodic code - spec alone is only 1.13x, sparse is the robust lever (1.61x), and spec+sparse stacks to 1.71-1.83x |
| -- | `idea-invented-coder-i8-qualify.md` | RECORD | INVENTED: Code-Tuning × i8 Robustness - QUALIFY (same arch/size/tokenizer: coder 3 flips↔9 flips by corpus, instruct flat 3↔3; domain interacts with quantization) |
| -- | `idea-invented-continuous-cap-lift-measured.md` | MEASURED | INVENTED: lift the continuous-bench prompt cap from 64 to the real 128 prefill rows - MEASURED (safe; kq8 win 1.55-1.67x at 96/120-token prompts) |
| -- | `idea-invented-continuous-kq8-3b-measured.md` | MEASURED | MEASURED: path-invariant kq8 continuous decode on the reference 3B model - 1.35-1.71x (median ~1.6x), audit clean, generate parity |
| -- | `idea-invented-continuous-kq8-batch-keep.md` | RECORD | INVENTED: run continuous-batch decode on the kq8 integer kernel (drop `float_only`) - KEEP-GRADE (+3.4% median, 10/13 pairs, transcripts identical 13/13) |
| -- | `idea-invented-continuous-kq8-closeout.md` | RECORD | CLOSEOUT: continuous-batch kq8 decode (path-invariant) - one decision matrix over six cycles; promote behind a flag, not by default |
| -- | `idea-invented-continuous-kq8-generate-parity-measured.md` | MEASURED | MEASURED: single-request prefill parity under the path-invariant change - identical ids and neutral timing (1-token prompt now also kq8) |
| -- | `idea-invented-continuous-kq8-leak-audit-measured.md` | MEASURED | MEASURED: occupancy-invariance (LEAK) re-audit for continuous-batch kq8 decode - FAILED at prompt-tokens 24 (solo vs busy diverge at output token 11); keep `float_only |
| -- | `idea-invented-continuous-kq8-longprompt-measured.md` | MEASURED | MEASURED: path-invariant kq8 continuous decode at 56-token prompts on 3B - 1.41-1.58x (median 1.57x), audit clean |
| -- | `idea-invented-continuous-kq8-path-invariant-keep.md` | RECORD | INVENTED: path-invariant continuous-batch kq8 decode (`tokens >= 1`, drop `float_only`) - KEEP-GRADE (audit now passes 3/3; +6% median throughput; thinking set identical 5/5) |
| -- | `idea-invented-continuous-kq8-prime-audit-measured.md` | MEASURED | MEASURED: prefix-prime admission path under path-invariant kq8 continuous decode - transcripts identical, primes engaged, +5-12% throughput |
| -- | `idea-invented-continuous-slots-kv-sweep-measured.md` | MEASURED | MEASURED: continuous-batching usage recipe on 3B with path-invariant kq8 - 6 slots is the peak (30.6 t/s), KV tier is flat at the bench's prompt cap |
| -- | `idea-invented-continuous-slots-mixed.md` | RECORD | INVENTED: Continuous-batching slots 4 → 8 - MIXED (median +34% with f16 KV, but memory pressure loses 21% in one pair) |
| -- | `idea-invented-determinism-audit-confirmed.md` | RECORD | INVENTED: Thread-Count Determinism Audit - CONFIRMED (14/14 identical ids, 0.5B + Llama-3B) |
| -- | `idea-invented-diverse-i8-flagged.md` | RECORD | INVENTED: i8 KV on Diverse Long-Form Text - CONDITIONAL keep (99.3% on Llama-3B; 94.2% on 0.5B with spread-out flips) |
| -- | `idea-invented-dot8acc-killed.md` | KILLED | INVENTED: 8-accumulator fused K-quant row dot (break the FMA dependency chain) - KILLED (median 0.93x; AVX2 register pressure) |
| -- | `idea-invented-exp-degree4-marginal.md` | MARGINAL | INVENTED: Degree-4 exp (one fewer FMA) - MARGINAL KEEP, kernel-only (3-6% SiLU; no end-to-end claim) |
| -- | `idea-invented-exp2-bitpath-killed.md` | KILLED | INVENTED: exp2 range-reduction path - KILLED (no consistent kernel win; fidelity fine) |
| -- | `idea-invented-f16-8k-confirm.md` | RECORD | INVENTED: f16 KV at 8k - CONFIRM (+11 to +19% decode; the auto rule scales with context as predicted) |
| -- | `idea-invented-f16-grouped-confirm.md` | RECORD | INVENTED (interaction follow-up): f16 Grouped-vs-Per-Head at 8k - CONFIRM (f16 follows f32: +33-44% grouped; the differentiator is dequant work, not traffic) |
| -- | `idea-invented-f16-kv-store-vector-killed.md` | KILLED | INVENTED: vectorize the f16 KV store with F16C - KILLED on the end-to-end bar (store 10x faster, only 0.45% of prefill) |
| -- | `idea-invented-f16-path-confirm.md` | RECORD | INVENTED: f16 Per-Head vs Grouped at 8k - CONFIRM (f16 follows f32: grouped wins +44%/+36%; path×KV matrix complete) |
| -- | `idea-invented-f16-qk-4acc-mixed.md` | RECORD | INVENTED: four-accumulator f16 QK dot in block prefill attention - MIXED (kernel +2.2-4.9%, end-to-end ~1%, ids/fidelity unchanged) |
| -- | `idea-invented-f16-qk-4acc-retest-killed.md` | KILLED | INVENTED retest: four-accumulator f16 QK dot - KILLED on a quiet box |
| -- | `idea-invented-f16-training-confirm.md` | RECORD | INVENTED (coder-i8 follow-up): f16 Training Robustness - CONFIRM (auto's default tier is training-proof: 199/199 both models, KL within 2.2x) |
| -- | `idea-invented-f16-vs-f32-8k-measured.md` | MEASURED | MEASURED: f16 KV is also the long-context speed tier - 1.15-1.22x decode and 1.04-1.14x prefill over f32 at 8k (attention 0.71-0.75x) |
| -- | `idea-invented-fidelity-corpus-audit-confirm.md` | RECORD | INVENTED: Fidelity Corpus-Sensitivity Audit - CONFIRM (ordering holds on all 3 corpora; i8 magnitude spans ~10x, no inversion) |
| -- | `idea-invented-flat-tokenizer-index-killed.md` | KILLED | INVENTED: Flat tokenizer string index - KILLED (1.22x median, under the 1.25x bar; correct but too small alone) |
| -- | `idea-invented-flip-locator-frontloaded.md` | RECORD | INVENTED: i8 Flip-Location Probe at 2048 Tokens - FRONT-LOADED, no compounding (2 flips at positions 26/38, 98% of KL mass in Q0) |
| -- | `idea-invented-fused-decode-matvecs-killed.md` | KILLED | INVENTED: fused per-layer decode projections (qkv and gate/up in one pool pass) - KILLED (median 1.01x end-to-end; matvec phase +5% median, below the 5% decode bar) |
| -- | `idea-invented-fused-lock-six-draft-killed.md` | KILLED | INVENTED: fused chain with the 6-draft lock drafter - KILLED analytically (the bonus row is structural; the two ways to build it cost more than the 1.17x token gain) |
| -- | `idea-invented-fused-six-draft-8k-killed.md` | KILLED | INVENTED: fused six-draft (7-row) lock chain revisited at 8k - KILLED for the long-context revival, but it wins on true period-1 locks at short context (2k lock 1.09x, 4k fox 1.06x) |
| -- | `idea-invented-fused-vs-nofuse-8k-measured.md` | MEASURED | MEASURED: fused vs non-fused speculation at 8k is a wash (1.01x median) - the fused chain's 2k advantage does not carry, and its extra draft stops paying |
| -- | `idea-invented-gpu-prefill-reopen.md` | RECORD | INVENTED (gpu-validate follow-up): GPU vs CPU on Long Prefill - REOPEN (offload loses 2.5x on 0.5B, 11% on Llama; 4.3GB moved per 0.5GB model = weights not persistent) |
| -- | `idea-invented-gpu-triage-correct.md` | RECORD | INVENTED (gpu-prefill follow-up): GPU Offload Triage - CORRECTED (weights ARE cached; the dominant cost is uncached output readback ~72%, not re-uploads) |
| -- | `idea-invented-gpu-validate-confirm.md` | RECORD | INVENTED: GPU Offload Validation - CONFIRM (ids identical, fidelity at dense floor both modes; slower on small shapes as designed) |
| -- | `idea-invented-gqa-8k-confirm.md` | RECORD | INVENTED: GQA-Grouped vs Per-Head Attention at 8k - CONFIRM (+14 to +51% grouped, ids identical; the auto ≥256 switch is validated at long ctx) |
| -- | `idea-invented-gqa-chunks-8-mixed.md` | SPLIT | INVENTED: 8-chunk GQA decode history split at long context - MIXED (attention phase ~1.09x better in 3/4 pairs; decode unresolved on a loaded box) |
| -- | `idea-invented-gqa-chunks-confirm.md` | RECORD | CONFIRM: GQA 4-chunk decode at 255 tokens - KEEP (decode +4.5%/+25%, attention 4-13% better, ids identical) |
| -- | `idea-invented-gqa-chunks-mixed.md` | RECORD | INVENTED: GQA decode chunk-count tuning - MIXED (attention 5-11% faster at 3-4 chunks; decode gain inside box noise; default unchanged) |
| -- | `idea-invented-gqa-exp-killed.md` | KILLED | INVENTED: Fast scalar exp for the GQA online softmax - KILLED (libm exp already 4.2-4.8 ns; polynomial twin 4.7-5.7 ns) |
| -- | `idea-invented-gqa-q8-tail-keep.md` | RECORD | INVENTED: full vector tail in the GQA q8 value loop - KEEP kernel-only (kernel 1.1-2.3x best-of-3; decode 1.059x in matched pair) |
| -- | `idea-invented-gqa-reuse-killed.md` | KILLED | INVENTED: Token-outer GQA reuse for i8 attention - KILLED (0.81-0.96x at 8k, 0.83-0.94x at 4k) |
| -- | `idea-invented-greedy-five-span-draft-killed.md` | KILLED | INVENTED: greedy five-span draft + sixth extension - KILLED (no effect on available prompts) |
| -- | `idea-invented-greedy-sampler-fastpath-killed.md` | KILLED | INVENTED: greedy sampler fast path (skip the vocab copy) - KILLED on share, but it exposed a 13x scalar argmax |
| -- | `idea-invented-greedy-softmax-killed.md` | KILLED | INVENTED: Greedy (temp<=0) Softmax Skip - KILLED analytic (fast path already ships; residual copy-skip ceilings ~0.2%) |
| -- | `idea-invented-grouped-block-attention-killed.md` | KILLED | INVENTED: grouped-head block-prefill attention (share K/V rows across GQA heads) - KILLED (attention 1.43x-1.60x slower, ids identical) |
| -- | `idea-invented-grouped-hoist-proceed.md` | PROCEED | INVENTED: Grouped i8 Widen-Hoist - PROCEED (−22%/-22%/-18% attn, bit-identical; hoist redundant per-head dequant into per-token L1 scratch) |
| -- | `idea-invented-grouped-i8-interaction-confirm.md` | RECORD | INVENTED (i8-geometry follow-up): Grouping × i8 Interaction - CONFIRMED on MHA (+24% per-head over grouped-i8); BONUS: per-head i8 beats per-head f32 everywhere (+13 to +24%) |
| -- | `idea-invented-grouped-i8-singlethread.md` | CLOSED | INVENTED (interaction follow-up): Grouped-vs-Per-Head i8, Single-Threaded - PARALLELISM OWNS THE GAP (algorithmic branch closed: tied at 1.8% with no threads involved) |
| -- | `idea-invented-grouped-i8-thermal-check.md` | RECORD | INVENTED (interaction follow-up): Grouped-vs-Per-Head i8, Thermally Controlled - DOWNSIZED (+6-7% real on 0.5B; Phi 2x gap withdrawn as thermal-confounded) |
| -- | `idea-invented-hoist-scaling-confirm.md` | RECORD | INVENTED (grouped-hoist follow-up): Group-Size Scaling - CONFIRM (Llama −18% ≈ 2/3 of 0.5B's −22%; Phi −5%/−4% consistent structural floor, not thermal noise) |
| -- | `idea-invented-i8-8k-killed.md` | KILLED | INVENTED: i8 KV at 8k Context - KILLED (i8 loses −13 to −23%: dequant compute dominates the byte saving, worse with context) |
| -- | `idea-invented-i8-attn-8k-proceed.md` | PROCEED | INVENTED: i8 attention at 8192 - PROCEED for VNNI attention (50% of decode, trigger met) |
| -- | `idea-invented-i8-corpus-dependence-correction.md` | RECORD | CORRECTION: i8 KV quality is corpus-dependent - 99.75% on repeated text vs 90.05% on prose at the same 2k context; the recorded 2k bar was corpus-flattered |
| -- | `idea-invented-i8-geometry-architectural.md` | RECORD | INVENTED (flagged puzzle): i8-Attention Geometry Scaling - ARCHITECTURAL (0.5B −18% / Llama +19% / Phi −6%, same window; mechanism open with narrowed candidates) |
| -- | `idea-invented-i8-kv-under-spec-measured.md` | MEASURED | MEASURED: i8 KV under speculation at 2k - slower than f16 (0.86-0.96x); the KV tier is not a spec multiplier |
| -- | `idea-invented-i8-longctx-confirmed.md` | RECORD | INVENTED: i8 (and f16) KV Fidelity at 2048-Token Context - CONFIRMED (i8 holds at 99.9% top-1; KL grows 4x, still beats shipped VNNI) |
| -- | `idea-invented-i8-size-ladder-conditional.md` | RECORD | INVENTED: i8 KV Diverse-Text Size Ladder - CONDITIONAL keep, boundary above 1.5B (monotonic in size, format-independent) |
| -- | `idea-invented-i8-vnni-attention-killed-analytic.md` | KILLED | INVENTED: AVX-VNNI Q8 KV attention dot - KILLED analytic (13% attention ceiling; QK-only 2x saves ~3%) |
| -- | `idea-invented-indexhoist-killed.md` | KILLED | INVENTED (nogather follow-up): Hoist Head-Invariant Sparse Index Build - KILLED (off/on pairs +9%/−0%/−4%: build cost is noise, dots+gather dominate) |
| -- | `idea-invented-int8qk-killed.md` | KILLED | INVENTED: Integer QK Dots for i8-KV Attention - KILLED (never beats widen-then-FMA: per-chunk reduction + query quant ≥ the maddubs saving at head_dim ≤128) |
| -- | `idea-invented-jacobi-killed.md` | KILLED | INVENTED: Jacobi Parallel Decoding - KILLED analytic (structurally dominated by shipped speculative: same verify primitive, worse accept policy, same guess family) |
| -- | `idea-invented-kchunk-retry-killed.md` | KILLED | INVENTED (trigger §49): K-Chunk Dot Reorder Retry - KILLED analytic (primary best-of-5 already fails the falsifier on clean data; code absent) |
| -- | `idea-invented-kq8-band1-killed.md` | KILLED | INVENTED: one band per worker in the kq8 prefill kernel - KILLED (0.93-0.99x; activation re-sweep is not the bottleneck) |
| -- | `idea-invented-kq8-pair-dot-killed.md` | KILLED | INVENTED: two-row Q4_K x Q8_K dot sharing the activation block - KILLED (+7% isolated at chunk sizes, 0.91-0.99x end-to-end) |
| -- | `idea-invented-kq8-split-sumi-killed.md` | SPLIT | INVENTED: split integer accumulators in the kq8 dots - KILLED (neutral; kernel is not add-chain-bound) |
| -- | `idea-invented-kq8-vnni-killed.md` | KILLED | INVENTED: AVX-VNNI Q4_K x Q8_K dot for prefill - KILLED (-24% at sustained sizes; the 16-bit scale madd is cheaper than per-chunk 32-bit multiply) |
| -- | `idea-invented-kq8-vnni-per32-killed.md` | KILLED | INVENTED: per-32 activation scales feeding a VNNI Q4_K dot (no weight multiply) - KILLED (0.62-0.99x; vpdpbusd + cvt per chunk loses to maddubs+madd) |
| -- | `idea-invented-kq8d-attn-keep.md` | RECORD | INVENTED: kq8d on the attention projections - KEEP as an opt-in scope (projection phase 1.31-1.84x, decode ~+5%) |
| -- | `idea-invented-kq8d-combo-keep.md` | RECORD | INVENTED: Combined kq8d decode tier (LM head + FFN down) - KEEP opt-in, f32-only (raw 1.51x/2.09x, ~15% drift-normalized) |
| -- | `idea-invented-kq8d-combo-largecorpus-keep.md` | RECORD | INVENTED: kq8d head/combo - larger-corpus validation - KEEP, all three KV tiers qualify (f32 98.00%, f16 97.75%, i8 97.83%) |
| -- | `idea-invented-kq8d-combos-killed.md` | KILLED | INVENTED: kq8d gate/up combo extension - KILLED (adding gate/up is slower; down-only remains the sweet spot) |
| -- | `idea-invented-kq8d-context-table-measured.md` | MEASURED | MEASURED: kq8d max tier context table - 1.41x at 2k, 1.33x at 4k, 1.21x at 6.6k; erosion is real but the tier stays worth it to 8k |
| -- | `idea-invented-kq8d-decode-4k-confirm.md` | RECORD | CONFIRM: kq8d head+down decode over 63 tokens at 4096 - 1.289x (matvec 1.47x), ids identical |
| -- | `idea-invented-kq8d-down-keep.md` | RECORD | INVENTED: Selective kq8d on the FFN down projection - KEEP as opt-in f32 decode tier (+10-36%, KL 7.95e-4, top-1 97.74%) |
| -- | `idea-invented-kq8d-ffn-down-killed.md` | KILLED | INVENTED: whole-FFN kq8d (`ffn,down`, gate+up+down integer) - KILLED on the fidelity gate by exactly one top-1 flip (97.49% vs 97.6%), despite +8% over `ffn |
| -- | `idea-invented-kq8d-ffn-keep.md` | RECORD | INVENTED (retest): kq8d on the FFN gate+up projections - KEEP as an opt-in tier (+1.20x median decode, KL 2.495e-4, 198/199) |
| -- | `idea-invented-kq8d-frontier-2400-confirm.md` | RECORD | CONFIRM: head+down kq8d frontier at 2,399 positions - passes narrowly (97.67%); head-only is the safe tier (98.83%) |
| -- | `idea-invented-kq8d-full-stack-killed.md` | KILLED | INVENTED: full kq8d head+down+attn stack - KILLED (KL 1.576e-3 passes, top-1 1,167/1,199 = 97.33% fails) |
| -- | `idea-invented-kq8d-full-stack-per32-keep.md` | RECORD | INVENTED: full-stack kq8d under per-32 activation scales (`head,ffn,down,attn`) - KEEP (393/399, KL 4.382e-4, ~1.42x decode) |
| -- | `idea-invented-kq8d-lmhead-keep.md` | RECORD | INVENTED: kq8d on the LM-head projection - KEEP as opt-in (3.3-3.6x head matvec, +33-39% decode, KL 1.9e-4) |
| -- | `idea-invented-kq8d-max-tier-thinking-set-measured.md` | MEASURED | MEASURED: thinking-set divergence of the per-32 kq8d max tier - 4/5 greedy prompts byte-identical, 1/5 diverges at token 40; sampled ids differ by design |
| -- | `idea-invented-kq8d-per32-activation-keep.md` | RECORD | INVENTED: per-32 activation scales in the kq8d row kernel - KEEP (whole-FFN stack now passes: 6.015e-4 / 392/399, ~1.25x; `ffn` KL drops to 9.05e-5) |
| -- | `idea-invented-kq8d-per32-q6k-max-tier-keep.md` | RECORD | INVENTED: per-32 activation scales for Q6_K + the `head,ffn,down` max tier - KEEP (394/399, KL 2.75e-4, ~1.38x decode) |
| -- | `idea-invented-kq8d-per32-verify-killed.md` | KILLED | INVENTED: per-32 batch verifier for spec+kq8d composition - KILLED (speed-neutral), and it corrects the previous report's mechanism claim |
| -- | `idea-invented-kq8d-scope-frontier-mixed.md` | RECORD | INVENTED: kq8d scope frontier - MIXED (up and gate+down pass fidelity; gate and f16+down fail; combo speed unresolved) |
| -- | `idea-invented-kq8d-scope-matrix-2k-measured.md` | MEASURED | MEASURED: kq8d scope matrix at 2k context on 3B - attn 1.10x, ffn+down 1.10x, full stack 1.41x; the tier's gain is context-stable |
| -- | `idea-invented-kq8d-shared-killed.md` | KILLED | INVENTED: Shared-activation kq8d for gate+up - KILLED (fidelity excellent, speed no gain; down-only stays) |
| -- | `idea-invented-kq8d-speculation-stacking-measured.md` | MEASURED | INVENTED: stacking the per-32 kq8d full stack with prompt-lookup speculation - MEASURED (MIXED: -6% where drafting works, +51-56% where it fails; verifier is float-only) |
| -- | `idea-invented-kq8d-tiers-repriced.md` | MEASURED | MEASURED: kq8d `head` / `down` / `head,down` re-priced against the fused-float default - KEEP (stack 12/12 positive, ~1.23x median; `head` alone is workload-dependent; order must be rotated) |
| -- | `idea-invented-kq8hoist-killed.md` | KILLED | INVENTED: Hoist Row-Invariant Weight-Scale Conversion in the kq8 Batch Path - KILLED (clean kernel trial: 5 interleaved pairs inside noise; attempt-1 voided on a stride bug) |
| -- | `idea-invented-kq8scratch-killed.md` | KILLED | INVENTED: Thread-Local kq8 Batch Activation Scratch - KILLED (off/on pairs +11%/-1%/-1%: the heap free-list already absorbs per-call allocs, no contention signal) |
| -- | `idea-invented-kv-auto-llama-measured.md` | MEASURED | INVENTED: --kv auto on the Reference Model (Llama-3.2-3B) - MEASURED (identical ids both arms, timing within noise; recipe holds unchanged) |
| -- | `idea-invented-kv-auto-proceed.md` | PROCEED | INVENTED: --kv auto (Context-Adaptive KV Tier) - PROCEED WITH RECIPE (fidelity-identical to manual, short takes f32, long takes f16; one false alarm resolved by interleaving) |
| -- | `idea-invented-kv-noinit-keep.md` | RECORD | INVENTED: No-init KV cache - KEEP (load 183 → 19 ms at 4096, ids identical) |
| -- | `idea-invented-kv-tier-quality-8k-measured.md` | MEASURED | MEASURED: long-context KV tier quality at 8k - f16 holds (99.98%), i8 fails its own bar (96.60%), sparse collapses (67.49%) |
| -- | `idea-invented-llama-f16-grouped-confirm.md` | RECORD | INVENTED (f16-path follow-up): Llama f16 Grouped-vs-Per-Head - CONFIRM uniformity (+4-6% decode, −33% attn; f16 follows f32 on the reference model too) |
| -- | `idea-invented-mha-bandwidth-measured.md` | MEASURED | INVENTED: MHA-vs-GQA Bandwidth-Model Test (Phi vs Llama, Short + 8k Ctx) - MEASURED, model refined (short confirms weights; 8k refutes bytes: decode is compute-bound) |
| -- | `idea-invented-mha-i8-split.md` | SPLIT | INVENTED: MHA i8 at 4k - SPLIT (decode KILLED at +1%: no MHA flip; prefill +17% twice, PROCEED-grade observation for kv-write savings) |
| -- | `idea-invented-mprefix-table-proceed.md` | PROCEED | INVENTED: Multi-Prefix CoW Clean Re-Spike - PROCEED (S4 trigger retired: 2 populates + 6 hits, zero unexplained, ids identical) |
| -- | `idea-invented-nattext-i8-confirm.md` | RECORD | INVENTED (grouped-i8 follow-up): Natural-Text Per-Head i8 - CONFIRM (+12.6%/+6.5% decode, attn −29 to −39%, ids byte-identical on prose) |
| -- | `idea-invented-nattext-llama-confirm.md` | RECORD | INVENTED (grouped-i8 follow-up): Llama Natural-Text Per-Head i8 - CONFIRM (+6%/+8%, ids identical; transfer matrix complete except one low-priority cell) |
| -- | `idea-invented-nattext-phi-confirm.md` | RECORD | INVENTED (nattext follow-up): Phi Natural-Text Per-Head i8 - CONFIRM (+11%/+13% with one recurring priced-class flip; matrix complete 6/6) |
| -- | `idea-invented-natural-8k-tier-ladder-measured.md` | MEASURED | MEASURED: natural-text 8k tier ladder - f16 99.90%, i8 91.33%, sparse 86.55%/80.81%; the self-repeating corpus flattered i8 and penalised sparse |
| -- | `idea-invented-nogather-prefill-proceed.md` | SPLIT | INVENTED (phasesplit follow-up): No-Gather Strided Prefill Attention - PROCEED (attn ~50% of dense at identical ids + digit-identical fidelity; gather fully removed) |
| -- | `idea-invented-noinit-combined-keep.md` | RECORD | INVENTED: Combined no-init KV + scratch - KEEP (load 175.3 → 8.7 ms, 20.1x) |
| -- | `idea-invented-normsplit-measured.md` | SPLIT | INVENTED: Norm-Phase Split Diagnostic - MEASURED (flat thirds 40/34/26; detect() 6.7us × ~100 calls/token is the only cross-cutting lever, ceilings 5%/0.7% - at/under bar, fix deprioritized) |
| -- | `idea-invented-penalty-spec-8k-measured.md` | MEASURED | MEASURED: repeat penalty kills speculation at 8k - acceptance collapses to 0.25/step and spec is neutral-to-negative (0.94-1.03x) |
| -- | `idea-invented-phi-i8-generalize.md` | RECORD | INVENTED: Phi-3.5-Mini i8 Architecture Test - GENERALIZE (3.8B holds 99.4% = Llama's 99.4% to the flip; size rule is architecture-independent) |
| -- | `idea-invented-preb-retry-killed.md` | KILLED | INVENTED (trigger S2): Pre-Packed-B GEMM Retry - KILLED analytic (no caller + code gone: a quiet-box rerun could only move bench GEMM) |
| -- | `idea-invented-prefill-block-attention-keep.md` | RECORD | INVENTED: Query-block prefill attention - KEEP (attention 1.63-1.88x, 8k prefill 1.20-1.44x, ids identical) |
| -- | `idea-invented-prefill-block-f16-i8-keep.md` | MARGINAL | INVENTED: Query-block prefill attention for f16/i8 KV - KEEP f32/f16, i8 marginal (f16 attention 1.47-1.83x, i8 1.02-1.07x) |
| -- | `idea-invented-prefill-chunk-mixed.md` | RECORD | INVENTED: Prefill chunk-size sweep - MIXED (128 wins at 2048, ties at 8192; keep 64) |
| -- | `idea-invented-prefill-i8-confirm.md` | RECORD | INVENTED (mha-i8 follow-up): Prefill-i8 at Long Context - CONFIRM (+12 to +25% on Phi best-of-3 and Llama ×2; KV-write savings land where K/V-reads don't) |
| -- | `idea-invented-prefill-thread-sweep-measured.md` | MEASURED | MEASURED: prefill thread sweep on 3B - the default (16 logical) is already the best; pt8 is 7-21% slower |
| -- | `idea-invented-q4k-fused-row-dot-keep.md` | RECORD | INVENTED: fused Q4_K row dot for decode - KEEP (kernel 1.763x, bit-identical) |
| -- | `idea-invented-q4k-r2-gemm-killed.md` | KILLED | INVENTED: two-row Q4_K prefill GEMM (row-interleaved without format repack) - KILLED (median 0.99x, bit-exact) |
| -- | `idea-invented-q5k-fused-row-dot-mixed.md` | RECORD | INVENTED: fused Q5_K row dot - MIXED (kernel 1.649x, end-to-end flat on the only Q5_K model) |
| -- | `idea-invented-q6k-fused-row-dot-keep.md` | RECORD | INVENTED: fused Q6_K row dot for decode - KEEP (kernel 1.315x, bit-identical) |
| -- | `idea-invented-q8micro-measured.md` | MEASURED | INVENTED (i8-geometry follow-up): q8 Dequant Microbench - MEASURED (1.6-2.2x compute tax, smooth across head_dim; arithmetic branch closed, structural branch narrowed) |
| -- | `idea-invented-qquant-killed.md` | KILLED | INVENTED: Vectorized Activation Quantization - KILLED at the probe gate (conversion share ceilings at ~5%, realistic port ~3-4% < bar; no spike built) |
| -- | `idea-invented-radix-top-p-sort-keep.md` | RECORD | INVENTED: LSD radix sort for the top-p candidate cut - KEEP-GRADE (2.56x sampler on large nuclei; flat decode +11.6% 0.5B; sampled ids identical) |
| -- | `idea-invented-sampled-sixth-draft-keep.md` | RECORD | INVENTED: use the sixth verifier row in sampled speculation - KEEP |
| -- | `idea-invented-sampled-spec-2k-measured.md` | MEASURED | MEASURED: sampled speculation at 2k on 3B - 2.22-2.33x (greedy 2.74x); the sampler becomes the dominant non-forward cost |
| -- | `idea-invented-sampled-spec-8k-measured.md` | MEASURED | MEASURED: sampled speculation at 8k - 1.44-1.72x with the 6-row softmax now 53-55% of spec wall; the chat path holds |
| -- | `idea-invented-sampled-speculation-cuts-keep.md` | RECORD | Idea 16 - Sampled speculation under top-k/top-p/min-p cuts |
| -- | `idea-invented-sampler-scans-killed.md` | KILLED | INVENTED: Sampler Scan Vectorization - KILLED (scans are ~1% of sampling; the 5% bar is unreachable even at zero cost) |
| -- | `idea-invented-sampler-split-measured.md` | SPLIT | INVENTED: Sampler Exp-vs-Logic Split - MEASURED (exp path unaccounted: only ~7ms of 99ms "sampling" is scans+exp; verify forwards are double-counted) |
| -- | `idea-invented-sampler-timer-correction.md` | RECORD | CORRECTION: the sampled-spec "53-55% sampler" is a timer double count - the real sampler bill is ~10 ms per run (0.2-0.3% of decode) |
| -- | `idea-invented-sampler-timers-measured.md` | MEASURED | INVENTED: Sampler Timer Overlap Fix (Measurement) - MEASURED (identity holds to 0.1 ms; true sampling 6.1% of spec decode, not 73.6%) |
| -- | `idea-invented-scratch-noinit-keep.md` | RECORD | INVENTED: No-init scratch buffers - KEEP (5.6% load standalone; ~2x of the residual after KV no-init) |
| -- | `idea-invented-sequences-auto-calibration-killed.md` | KILLED | INVENTED: Short 4-vs-8 calibration for `--sequences auto` - KILLED (opposite choices within minutes; variance > signal) |
| -- | `idea-invented-sequences-auto-cpu-killed.md` | KILLED | INVENTED: CPU-load guard for `--sequences auto` - KILLED (8 slots still win under synthetic CPU load) |
| -- | `idea-invented-sequences-auto-keep.md` | RECORD | INVENTED: Budget-aware `--sequences auto` - KEEP as opt-in (auto picks 8, +25% median aggregate) |
| -- | `idea-invented-sequences-auto-live-mixed.md` | RECORD | INVENTED: Live-headroom `--sequences auto` - MIXED (safety fixed; 8-slot throughput not reliable) |
| -- | `idea-invented-silu-rcp-killed.md` | KILLED | INVENTED: SiLU reciprocal + Newton - KILLED (kernel 22% slower; numerics fine) |
| -- | `idea-invented-smollm2-ladder-mixed.md` | RECORD | INVENTED: SmolLM2 Coverage + i8 Ladder Bottom Rung - MIXED (runs clean, 98% i8; size trend CONFIRMED in-Qwen on second corpus; cross-arch comparison stays confounded) |
| -- | `idea-invented-softmax-exp-vector-maxsum-keep.md` | RECORD | INVENTED: vectorize the AVX2 softmax kernel's max scan and total sum - KEEP-GRADE (kernel 2.48x; sampled ids identical) |
| -- | `idea-invented-softmax-normalize-pass-killed.md` | KILLED | INVENTED: drop the softmax normalization pass and scale each query's accumulator once - KILLED (ids/fidelity identical, no repeatable >3% gain) |
| -- | `idea-invented-sparse-8k-killed.md` | KILLED | INVENTED: Sparse Decode at 8k - KILLED (gather scales with kept count: −13 to −23%; gate narrows to a context window, no-gather-decode port becomes load-bearing) |
| -- | `idea-invented-sparse-8k-quality-correction.md` | RECORD | CORRECTION: sparse quality at 8k positions - the 2k proxy understated it 6x (stride 16 costs 14.4pp, not 2.25pp; even stride 2 leaves top-1 at 75.8%) |
| -- | `idea-invented-sparse-8k-reopened.md` | RECORD | INVENTED: strided sparse decode at 8k - REOPENED (1.36-1.77x plain, attention 3.0-3.8x faster; spec+sparse stacks to 1.6-1.75x over spec, ~2.2-3.0x over dense) |
| -- | `idea-invented-sparse-f16-proceed.md` | PROCEED | INVENTED: Strided Attention over f16 KV - PROCEED WITH RECIPE (combines the subset win with halved KV traffic; digit-identical to gathered sparse) |
| -- | `idea-invented-sparse-geometry-sweep-measured.md` | MEASURED | MEASURED: sparse geometry sweep - stride-only dilution is the efficient knob (8k stride 16: attention 0.31-0.34x, top-1 −2.25pp); narrowing the window costs 4-11pp for no extra speed |
| -- | `idea-invented-sparse-indexcache-killed.md` | KILLED | INVENTED: Sparse Shared Index Cache - KILLED (160K+ cache hits, zero timing change; the cost is the gather, not the build) |
| -- | `idea-invented-sparse-nogather-decode-measured.md` | MEASURED | INVENTED: No-Gather Strided Attention at Long-Context Decode (Llama-3B) - MEASURED (decode +13% vs gathered/+21% vs dense; ids identical; recipe holds) |
| -- | `idea-invented-sparse-nogather-llama-measured.md` | MEASURED | INVENTED: No-Gather Strided Attention on the Reference Model (Llama-3B) - MEASURED (0.49-0.52x dense attn, ids byte-identical; recipe generalizes) |
| -- | `idea-invented-sparse-nogather-proceed.md` | PROCEED | INVENTED: No-Gather Strided Attention - PROCEED WITH RECIPE (attn 0.53-0.58x dense at 2k, 0.46x at 4k; bit-identical; fidelity digit-identical) |
| -- | `idea-invented-sparse-phasesplit-measured.md` | SPLIT | INVENTED: Sparse Prefill Phase Split - MEASURED (build 4%, gather ~85%, dots ~11%: the gather is the cost, the index never was) |
| -- | `idea-invented-sparse-prefill-8k-measured.md` | MEASURED | INVENTED: strided sparse prefill at 8k - MEASURED (attention 0.61-0.83x, total prefill 1.06-1.15x; per-token fallback forfeits block reuse, so prefill gains ~1/3 of decode's) |
| -- | `idea-invented-spec-12k-code-measured.md` | MEASURED | MEASURED: 12k speculation is workload-dependent - code still wins 1.31-1.35x at 3.50 accepted/step, so the lock-prompt 0.84x is an acceptance artifact |
| -- | `idea-invented-spec-12k-killed.md` | MEASURED | MEASURED + KILLED: speculation at 12k is a loss on the only fit-capable 12k prompt (0.84x) - acceptance collapses to 1.50/step while attention is 65-67% |
| -- | `idea-invented-spec-at-2k-measured.md` | MEASURED | MEASURED: speculation at 2k context on the reference 3B - still 2.2-2.8x with identical ids; un-pinning attention to the grouped schedule is neutral |
| -- | `idea-invented-spec-at-4k-measured.md` | MEASURED | MEASURED: speculation at 4k context on 3B - ~1.9-2.1x (down from 2.74x at 2k); attention share reaches 29-33% |
| -- | `idea-invented-spec-at-8k-measured.md` | MEASURED | MEASURED: speculation at the 8k boundary on 3B - still 1.45-1.61x; attention reaches 52-56% of decode, and the 4k compression is not a law |
| -- | `idea-invented-spec-equivalence-sweep.md` | MEASURED | MEASURED: speculative transcript equivalence sweep - KEEP with a kernel-path caveat (6/8 identical; exact only with matching kernels) |
| -- | `idea-invented-spec-kq8d-killed.md` | KILLED | INVENTED: kq8d head + speculative decoding - KILLED (verify path is float-only by design; head scope unreachable) |
| -- | `idea-invented-spec-kv-compat-confirm.md` | RECORD | INVENTED: Speculative × Non-f32 KV Compatibility - CONFIRM (8/8 combos clean; temp-0 exact, temp-0.1 same shape) |
| -- | `idea-invented-spec-prose-8k-kills-lookup-gate.md` | MEASURED | MEASURED + KILLED: forced speculation wins on 8k prose (1.18-1.25x), so the zero-model lookup gate is wrong for long prompts |
| -- | `idea-invented-spec-tier-2k-measured.md` | MEASURED | MEASURED: kq8d max tier under speculation at 2k - neutral-to-negative (median 0.966x), confirming they are substitutes at long context too |
| -- | `idea-invented-spec12-wide-verify-10t-killed.md` | KILLED | INVENTED: 12-row speculative verify re-priced at the default 10-thread decode - KILLED again (median 1.089x at 10 threads; the 6-thread kill was not thread-limited) |
| -- | `idea-invented-spec12-wide-verify-killed.md` | KILLED | INVENTED: 12-row speculative verify (double-width draft batch) - KILLED (steps halve 11 -> 6, throughput flat at ~1.02x) |
| -- | `idea-invented-speculation-auto-probation-keep.md` | RECORD | INVENTED: `--speculative auto` with an accept-rate probation - KEEP as an opt-in mode (3.2-4.0x on repetitive workloads; prose bails after 9 steps and matches off) |
| -- | `idea-invented-speculative-benchmark-keep.md` | MEASURED | MEASURED: shipped speculative decoding on the standard benchmark - KEEP as a usage mode (1.95-2.15x on predictable prompts, ids identical) |
| -- | `idea-invented-stack-8k-confirm.md` | RECORD | CONFIRM: Recommended stack at 8180, reverse order - prefill 1.575x confirmed (matvec control within 1.3%) |
| -- | `idea-invented-stack-8k-keep.md` | RECORD | INVENTED: Recommended stack end-to-end at 8180 - KEEP (1.56x total; prefill 1.57x, decode 1.14x, ids identical) |
| -- | `idea-invented-tiled-attn-ceiling-killed.md` | KILLED | INVENTED (trigger S3): Vectorized Tiled Prefill Attention Retry - KILLED analytic (ceiling ~0% realistic, 2.2% generous, both under the 5% bar - no spike) |
| -- | `idea-invented-timers-fix-ship.md` | RECORD | INVENTED: Sampler Timer-Overlap Fix Validation - SHIP RECIPE (identity 225.88 vs 226.02, trajectories + ids byte-identical) |
| -- | `idea-invented-tokenizer-source-cache-killed.md` | KILLED | INVENTED: Tokenizer source sidecar - KILLED (quiet-box 1.24x, below the 1.25x bar; map build dominates) |
| -- | `idea-invented-tokenizer-state-cache-keep.md` | RECORD | INVENTED: Serialized flat tokenizer state - KEEP (warm tokenizer 107.6 → 19.2 ms, 5.6x) |
| -- | `idea-invented-top-p-avx2-exp-keep.md` | RECORD | INVENTED: AVX2 exp in the top-p/min-p sampler branch - KEEP-GRADE (sampler 1.68x on peaked logits; sampled ids identical) |
| -- | `idea-invented-top-p-histogram-killed.md` | KILLED | INVENTED: exponent-histogram threshold for the top-p sampler - KILLED (slower in both distributions; halving's early exit wins) |
| -- | `idea-invented-topk-radix-sort-killed.md` | KILLED | INVENTED: radix sort for the top-k sampler branch - KILLED (10.4x slower and wrong for negative logits) |
| -- | `idea-invented-value-tile-keep.md` | RECORD | INVENTED: 8-query value-accumulation tile in block prefill attention - KEEP (attention 1.32-2.41x, ids identical) |
| -- | `idea-invented-vector-argmax-keep.md` | RECORD | INVENTED: AVX2 first-index argmax shared by the sampler and speculative verifier - KEEP-GRADE (22.4x kernel; ~7% 0.5B speculative best-of; 3B neutral; ids identical) |
| -- | `idea-invented-vector-max-softmax-killed.md` | KILLED | INVENTED: vectorize the block-prefill softmax max scan - KILLED (sub-0.5% share, no resolvable gain) |
| -- | `idea-invented-vnni-asym-killed.md` | KILLED | INVENTED: VNNI Asymmetric (Zero-Point) Activation Quantization - KILLED (halves coarse KL to 5.97e-4 but misses the 1e-4 bar; t/s within noise) |
| -- | `idea-invented-vnni-attn-keysums-proceed.md` | MARGINAL | INVENTED: VNNI Q8 QK dot with precomputed key bsums - MARGINAL PROCEED (median 1.61x; 8k saving ~7.5-15%) |
| -- | `idea-invented-vnni-attn-stage1-killed.md` | KILLED | INVENTED: AVX-VNNI Q8 QK attention dot, stage 1 - KILLED (1.19x; 8k decode saving ~4.75%, under 5%) |
| -- | `idea-invented-vnni-attn-wiring-killed.md` | KILLED | INVENTED: VNNI i8 attention model wiring (stage 2) - KILLED (fidelity in class, but no reliable 8k decode gain) |
| -- | `idea-invented-vnni-bisect-measured.md` | MEASURED | INVENTED: VNNI Shape-Bisection Localization - MEASURED (distributed, not localized: smooth ladder, no cliff; k/v amplified, FFN diluted) |
| -- | `idea-invented-vnni-combo-killed.md` | KILLED | INVENTED: VNNI Asymmetric + Fine Combo (Kitchen Sink) - KILLED (2.97e-4, still 3x over the 1e-4 bar; road ends before the bar) |
| -- | `idea-invented-vnni-decode-confirm.md` | RECORD | INVENTED (1394 hole): VNNI Decode Trajectory Stability - CONFIRM (0 flips on 56 ids, fluent, no crash; decode under the flag pays ~5-13%) |
| -- | `idea-invented-vnni-double-accum-killed.md` | KILLED | INVENTED: VNNI Double-Accumulation Audit - KILLED (1.504e-03 vs 1.503e-03; combining precision exonerated; localization harness named next) |
| -- | `idea-invented-vnni-dump-measured.md` | MEASURED | INVENTED: VNNI Per-Matmul Dump Localization - MEASURED (depth-growing divergence, FFN-led; threading/outlier/group-tail dead; layer-gated VNNI named next) |
| -- | `idea-invented-vnni-error-split.md` | SPLIT | INVENTED: VNNI Error Split - PARTIALLY RETRACTED (qonly-floor was silent non-engagement; confound kills stand; see qonly re-run) |
| -- | `idea-invented-vnni-finescale-killed.md` | KILLED | INVENTED: VNNI Per-16 Finer Activation Scales - KILLED (KL halves to 8.1e-4 but misses the 1e-4 bar; 27% slower than coarse VNNI) |
| -- | `idea-invented-vnni-layergate-measured.md` | MEASURED | INVENTED: VNNI Layer-Gated Compounding Arbiter - MEASURED (intrinsic growth with depth, not compounding; late layers dominate on clean inputs) |
| -- | `idea-invented-vnni-qonly-rerun-measured.md` | MEASURED | INVENTED: VNNI qonly Re-Run WITH Engagement Proof - MEASURED (activation quant CONFIRMED at 1.515e-3; prior floor result retracted as silent non-engagement) |
| -- | `idea-invented-vnni-replay-measured.md` | MEASURED | INVENTED: VNNI Replay Audit - MEASURED (kernel arithmetic exonerated to 1.7e-5; divergence sits in the inputs; prior qonly floor flagged SUSPECT, re-run mandated) |
| -- | `idea-invented-vnni-statcorr-measured.md` | MEASURED | INVENTED: VNNI Activation-Statistic Correlation - MEASURED (maxdiff ~ maxact^0.6, R2 0.67; weight scales clean at 0.06; replay audit named as arbiter) |
| -- | `idea-thread-sweep-abba-parked.md` | RECORD | Thread Sweep ABBA - STILL INCONCLUSIVE, PARKED until cool box |
| -- | `idea-thread-sweep-inconclusive.md` | RECORD | THREAD SWEEP + QUEUE STATE - NO CHANGE (thermal drift, needs cool-box ABBA) |
| -- | `idea-vnni-default-probe-nochange.md` | RECORD | IDEA-XXXX: VNNI Default-Flip Probe - NO CHANGE (inconsistent sign, stays opt-in) |
| -- | `invented-1395b-tail-sweep-measured.md` | MEASURED | INVENTED-1395b: exact-tail length sweep (N=32/128 vs 64) - MEASURED: monotonic KL gains to 50x, decode parity at all N |
| -- | `invented-1395c-longctx-measured.md` | MEASURED | INVENTED-1395c: exact tail at long context + multi-domain corpus - MEASURED: ordering holds, saturation begins as tail fraction drops |
| -- | `invented-1395d-mprefix-clean-measured.md` | MEASURED | IDEA-1395d / S4-clean: multi-prefix table re-spike (populate-only-on-miss, single retired) - MEASURED: S4 misses explained, 6/6 hits, byte-identical |
| -- | `llama-cpp-comparison-2026-11-07.md` | RECORD | TGE vs llama.cpp on Llama-3.2-3B-Q4_K_M - 2026-11-07 |
| -- | `llama-cpp-comparison-2026-11-08.md` | RECORD | TGE vs llama.cpp - 2026-11-08 re-comparison (shipped baseline and the validated stack) |
| -- | `llama-cpp-comparison-2026-11-11.md` | MEASURED | MEASURED: llama.cpp re-comparison on the current tree - decode now ahead like-for-like (1.55x tg128, 1.14x tg32), prefill still 1.3-1.5x behind |
| -- | `llama-history-analysis-2026-11-08.md` | RECORD | Inference-engine performance history: what llama.cpp, ik_llama.cpp and vLLM changed, and what transfers to TGE |
| -- | `state-machine-hardening-2026-11-08.md` | RECORD | State-machine hardening: implicit states made explicit (2026-11-08) |
| -- | `verification-round-hoist-reverted.md` | RECORD | VERIFICATION ROUND - table + S1 hold, hoist reverted (effect not robust) |
| -- | `verification-round2-hoist-reverted.md` | RECORD | HOIST RE-VERIFY - FAILED AGAIN, STAYS REVERTED (3/3 losses, stable baseline) |
| -- | `winner-port-1517-s1-batch-populate-keep.md` | RECORD | WINNER PORT: 1517-S1 Batch Populate - LANDED, keep-grade (MATCH, overhead never-negative) |
| -- | `winner-port-1517-s2-admission-skip-keep.md` | RECORD | WINNER PORT: 1517 Stage-2 Admission Skip - LANDED, keep-grade (+75% bench, identical transcripts) |
| -- | `winner-port-1714-null-guards-keep.md` | RECORD | WINNER PORT: 1714 Null Guards - LANDED, keep-grade (contract pinned, zero cost) |
| -- | `winner-port-f16-table-skip-keep.md` | RECORD | WINNER PORT: f16 Prefix Table + Skip - LANDED, keep-grade (+57% f16 bench, tier-tagged) |
| -- | `winner-port-grouped-hoist-keep.md` | RECORD | WINNER PORT: Grouped-i8 Widen-Hoist - LANDED, keep-grade (−16% attn, identical ids) |
| -- | `winner-port-kv-auto-keep.md` | RECORD | WINNER PORT: --kv auto - LANDED, keep-grade (mapping exact on all arms) |
| -- | `winner-port-mprefix-table-keep.md` | RECORD | WINNER PORT: Multi-Prefix CoW Table - LANDED, keep-grade (1.72x prefill-side, identical ids) |
| -- | `winner-port-sparse-f16-keep.md` | RECORD | WINNER PORT: Sparse-f16 Strided - LANDED, keep-grade (0.55x best pair, tier-priced fidelity) |
| -- | `winner-port-sparse-nogather-keep.md` | RECORD | WINNER PORT: No-Gather Strided Sparse - LANDED, keep-grade (0.44x attn, bit-exact) |
