# Inference-engine performance history: what llama.cpp, ik_llama.cpp and vLLM changed, and what transfers to TGE

- **Date**: 2026-11-08
- **Question**: study the commit histories of llama.cpp and other inference engines to see how they improved performance, then map the patterns onto Aegean TGE.
- **Data**: llama.cpp `origin/master` fetched to `a691b234` (2026-11-08), **10,463 commits** since 2024-01-01; ik_llama.cpp **874 commits** (depth 800); vLLM last **500 commits** (Oct-Nov 2026); plus published comparison pages reachable by search. GitHub's API/pages returned 504 from this network, so the analysis uses local git metadata and search-cached pages; measured claims are attributed where they come from.
- **Bottom line**: TGE's per-row K-quant kernels are already at llama.cpp's level (the Q6_K/Q4_K bsum-hoisted dot is transcribed from it). The remaining CPU gap is **layout**: llama.cpp and especially ik_llama.cpp win prefill by packing **8-16 output rows into interleaved weight tiles** so one Q8_K activation load feeds many output rows. That is the single highest-value untried lever for TGE, and it is distinct from the float "pre-packed B" GEMM killed earlier.

## 1. How the histories were mined

- `git log` over the fetched histories, filtered by performance keywords (`perf|speed|faster|optimi|throughput|latency`), then by CPU-relevant themes (excluding CUDA/Vulkan/Metal/HIP/SYCL/WebGPU).
- llama.cpp: **372** perf-labeled commits (3.6% of all commits), **177** after removing GPU/backend-only work, spread **73 (2024) / 122 (2025) / 156 (2026)**.
- Theme mentions in the full 10,463-commit log: quant **495**, gemm/matmul **421**, attention/flash **234**, kv-cache **251**, batch **183**, thread/parallel **138**, simd/vector **124**, repack/interleave **62**, MoE **185**, speculative **102**, GPU **2,448**.
- ik_llama.cpp: 874 commits, **297** matching perf/kernel terms, **42** AVX/VNNI commits, **20+** repack/R8/R16 commits.
- vLLM: 500 commits, 151 perf/kernel matches, overwhelmingly GPU/MoE/serving.
- Sampled diffs and full commit bodies; pulled the llama.cpp Q6_K AVX2 patch and TGE's counterparts to compare techniques directly.

## 2. The playbook in the histories

### 2.1 Kernel specialization per quant format (largest CPU bucket)

- llama.cpp `2dd84169` (2026-04): *ggml-cpu: optimize avx2 q6_k* - hoists the Q6_K -32 correction out of the inner loop using the Q8_K block sums (`bsums`): one `madd_epi16(q8sums, scales)` plus a shift per superblock replaces four `maddubs` + four `sub` per 128 weights. The same bsum trick is in the Q4_K min correction.
- llama.cpp `07d15723` (2025-03): IQ1 mul_mat_vec with **BMI2** (`pext`) for bit extraction.
- llama.cpp `85c55223` (2026-08): AVX2 batched GEMM for grid IQ quants, with an 8x8 Q8_K panel kernel.
- llama.cpp `5c3d0f18` (2024-09): IQ4_NL sgemm plus Q4_0 AVX dot improvements.
- A public discussion (#26775) reports an SSE4.1 Q6_K dot lifting token generation **3.5 -> 6.6 t/s (+89%)** on an old N4000, purely from kernel specialization.
- ik_llama.cpp has **42** AVX/VNNI kernel commits, including `_mm512_dpbusd_epi32` versions for IQ4/IQ5/IQ6 (`b7f4411`), AVX-512+VNNI GEMM for IQ5_KS_R16 (`44db69c`), and an AVX-512 VNNI path tuned for Cascade Lake (`1a2a860`).

**Pattern**: when a quant format becomes hot, write a dedicated inner loop for it instead of a generic dequant-then-dot path. The gains are 1.3-2x per kernel in the commit record, and up to 2-3x on formats the generic path handles badly.

### 2.2 Repacked / interleaved weight layouts (R8/R16): the prefill lever

- llama.cpp `1e58ee13` (2024-11): convert Q4_0 into `Q4_0_X_Y` repacks.
- llama.cpp `ea5d7478` (2024-08): Q4_0/Q8_0 sgemm via **4xN and Mx4** tiles.
- llama.cpp `3a0d1053` (2025-12): Q4/Q8 tiled GEMM.
- ik_llama.cpp: `Q4_0_R8` GEMM on vanilla AVX2 (`659d691`), `MXFP4_R8` (`5ae7ce9`), `Q8_0_R16` (`796f780`, then extended to Q4_0/Q5_0/Q6_0/IQ4_NL/MXFP4 in `0a81bfe`), `IQ3_KS_R16` (`7a31ec4`), `IQ4_KS_R16`/`IQ5_KS_R16` (`2c22783`, `1b18524`), `Q2_0_R8` (`b9c47ba`), plus converter fixes (`0488a1f`).
- ik_llama's published comparisons show the result on **prefill**: Q6_K pp512 is **3.29x** llama.cpp on Zen4, **2.54x** on AVX2 (Ryzen 5975WX), **3.86x** on M2-Max NEON; Q4_K about **1.96x** on AVX2. Decode gains are smaller (1.3-1.6x).
- An independent C-Kernel-Engine roadmap states the same conclusion from the outside: "llama.cpp's faster path uses repacked/interleaved layouts so the hot loop computes a block of rows and columns together", and the first llama.cpp-style win it reproduces is an **8x8 repacked Q4_K GEMM**.

**Pattern**: pack N output rows' quant blocks together. Then, per Q8_K activation block, load the activation once and compute N row dots, keeping scale/min metadata and accumulators in registers. This converts a row-at-a-time GEMV into a register-blocked GEMM, raising FMA density and amortizing activation loads. It is the dominant CPU prefill optimization in both engines and is independent of SIMD width (it helps AVX2, not just AVX-512).

### 2.3 Tiling and register blocking for GEMM vs GEMV

- llama.cpp `ea5d7478` (4xN, Mx4), `3a0d1053` (tiled Q4/Q8 GEMM), `a05ca936` (loop tiling in the scalar path).
- An independent head-to-head (fak vs llama.cpp) summarizes the roofline: **decode is bandwidth-bound** (more cores do not help; per-core dot quality dominates), while **prefill is compute-bound** and the residual is GEMM micro-kernel quality (register blocking and accumulation structure), not instruction-set access.

**Pattern**: optimize GEMV and GEMM differently. TGE's decode is already at the bandwidth roofline for its layout; the open end is prefill GEMM density.

### 2.4 Attention and KV scheduling

- llama.cpp `9f682fb6` (2026-02): **flash attention split across KV for faster token generation** - the same class of fix as TGE's GQA history-chunk split; TGE just shipped four chunks (5-13% attention, 1.05x decode).
- llama.cpp `16bcc125` (2025-11): pad the KV cache size to 256 for alignment; `2fbde785` (2026-01): optimize KQ mask construction; `2c4fb692` (2024-03): defrag moves.
- ik_llama.cpp: sparse FA for DSA/MiniMax (`ef45f22`, `d7548ca`), SWA compression, MLA decode micro-optimizations (`9462969`), flash-attention work-buffer reservation.

**Pattern**: keep attention work proportional to the useful KV and align the cache; TGE already has f32/f16/i8 KV, GQA chunks, block prefill, and CoW prefix. The remaining attention wins in the histories are model-specific (MLA/DSA) or memory-tier-specific.

### 2.5 Activation quantization is a first-class hot kernel

- ik_llama.cpp `000c4da` (2026-10): *CPU: tweak activation quantization*; `b25c3fd` (2026-11): *faster q8_k_r16* - "process two 16-row tiles per pass to relieve pressure on FMA execution and load ports"; `796f780`/`0a81bfe`: Q8_0_R16 setup and extension.
- llama.cpp `85c55223` includes vectorized IQ panel decode and a "Q8_K activation contract" for batched GEMM.

**Pattern**: once weights are repacked, the activation quantizer and its layout become the bottleneck; it deserves its own optimization and port-pressure analysis.

### 2.6 Thread balance for small and leftover work

- ik_llama.cpp `9c312c2` (2026-11): split leftover row units by tokens (prefill path), balance small matrices across threads (decode path), and fuse conv+SiLU.
- ik_llama.cpp `1a3c785`: `pool_1d` ran everything on thread 0; split rows across threads.
- ik_llama.cpp `e34d163`: RMS norm split rows over more dimensions so >4 threads participate.

**Pattern**: separate "big GEMM" parallelism from "small tensor" parallelism; rebalance leftovers. TGE's 4-chunk GQA change (16 -> 32 tasks on 6 workers) is the same lesson and measured 5-13% on attention.

### 2.7 Fusion of elementwise + quantization

- vLLM `e336407` (2026-11): fuse the FFN-side MXFP8 activation quant into post-attention norm and SwiGLU; `c2a6042`: vectorize `silu_and_mul_per_block_quant` for large inputs.
- ik_llama.cpp `9c312c2`: fused conv + SiLU in DeltaNet layers.
- llama.cpp `5764d7c6`: per-layer projections in the first layer (graph-level restructuring).

**Pattern**: eliminate a read/write pass of the activation by emitting the quantized form directly from norm/SiLU. The payoff is modest on CPU but reliable.

### 2.8 Serving and memory (mostly not transferable to TGE's single-process CLI)

- vLLM: CUDA graphs, PagedAttention/KV blocks (`518c624` reduces per-block CPU cost), chunked prefill, prefix caching, cross-layer KV sharing for MLA, fused MoE kernels.
- llama.cpp: high-throughput server mode (`225e7a14`), speculative server scheduling (`f896d2c3`), prompt/vocab/model loading speedups (`c13e04e1`, `824eaecf`), weight eviction after each layer (`05967042`).
- TGE already has continuous batching with KV slots, CoW prefix reuse, chunked prefill, and speculative decoding. Paged KV was tried and killed in the report library; weight eviction and startup speed are not bottlenecks at 1.87 GiB / 215 ms.

### 2.9 Sampling, elementwise ISA breadth, and numerical approximations

- llama.cpp `15665`/`15704`/`10803`: bucket-sort reuse, distribution-sampler, and penalty-sampler optimizations; `4b0afb3` (ik): SIMD sigmoid/SiLU; `7551`: AVX-512 exp; `13107`: FP16/BF16 conversion APIs.
- TGE's sample share is <1%, SiLU is already AVX2, and exp/reciprocal variants were priced (degree-4 exp marginal, exp2 flat, reciprocal slower). No action.

## 3. What TGE already has (from the 48-cycle report library)

| History pattern | TGE equivalent | Status |
|---|---|---|
| Q6_K/Q4_K bsum-hoisted dots | `dot_q6k_q8k`/`dot_q4k_q8k` match llama.cpp's optimized inner loops | shipped (prefill) |
| Integer decode row kernels | kq8d head + FFN down (opt-in) | shipped opt-in |
| Attention split across KV | GQA decode chunks (now 4) + query-block prefill | shipped |
| KV quantization / f16 cache | f32/f16/i8 tiers | shipped |
| Continuous batching / prefix cache | continuous slots, CoW multi-prefix | shipped |
| Speculative decoding | prompt-lookup + SKIP-lock + fused verify | shipped |
| Quantized prefill GEMM | **only per-row integer dots, no row-interleaved tile** | **gap** |
| Repacked weight layouts | none (no R8/R16 formats) | **gap** |
| Fused norm/SiLU + quantization | separate passes | **gap (small)** |
| Activation-quant kernel tuning | `quantize_row_q8k` exists; not profiled | **gap (small)** |
| Small-work thread balance | 4-chunk GQA shipped; prefill band split | partial |
| Paged KV / weight eviction / load speed | paged KV killed; load 215 ms | not a priority |

## 4. Prioritized transfer list

### P1. Row-interleaved (R8) quantized prefill GEMM - highest value

- **Mechanism**: repack Q4_K/Q6_K matrices at load into groups of 8 output rows, with the 8 rows' blocks interleaved per column block. Quantize the prefill activations to Q8_K once (already done), then per column block load one activation block and compute 8 row dots, holding 8 integer accumulators and the scale/min metadata in registers. This is the R8 pattern from ik_llama/Q4_0_R8 and the 8x8 Q4_K tile identified by C-Kernel-Engine.
- **Why it is new here**: the earlier "pre-packed B GEMM" kill (measurement log S2) was a **float** 1024-cubed GEMM with a persistent B cache; the quantized K-quant row-interleave operates on Q8_K x K-quant integer dots and targets the prefill matvec phase.
- **Expected**: 1.2-1.6x on the prefill matvec phase (40-50% of prefill), i.e. ~10-20% prefill; the histories show larger gains for formats with worse generic kernels, and this stays measured rather than assumed.
- **Falsifier**: kernel >=1.3x on the 2048x3072 Q4_K prefill shape; model prefill >=10% at 2048 and 4096 with identical ids and fidelity unchanged; otherwise kill and keep the per-row kernel.
- **Tests**: differential kernel vs the current `gemv_dequant_batch_kq8_avx2` (exact integer result), full suite, prefill A/B at 512/2048/4096, ids and fidelity gates.

### P2. Fuse SiLU + Q8_K activation quantization (and RMSNorm + Q8_K)

- **Mechanism**: emit the quantized activation while the elementwise kernel still has the values in registers, instead of writing floats and re-reading them in `quantize_activations`.
- **Expected**: 1-3% prefill; zero fidelity change if the same arithmetic order is kept.
- **Falsifier**: >=1% prefill at 2048 with identical quantized bytes/ids; otherwise revert.
- **Evidence**: vLLM `e336407`/`c2a6042`, ik `9c312c2` (conv+SiLU fusion).

### P3. Profile and tune the Q8_K activation quantizer

- **Mechanism**: measure `quantize_row_q8k` ns/row at the three prefill call sites; apply the ik q8_k_r16 lesson (two tiles per pass to relieve FMA/load-port pressure) if the share is measurable.
- **Expected**: 1-2% on kq8 paths; zero if the quantizer is already at the memory roofline.
- **Falsifier**: quantizer >=10% of prefill matvec time and >=3% faster; otherwise close.

### P4. Prefill small-work balance

- **Mechanism**: split the last row band by tokens and the norm over rows, as ik `9c312c2`/`e34d163` do; measure idle time at 512-token prefill.
- **Expected**: 1-3% at 512; likely neutral at 2048+.
- **Falsifier**: >=2% at 512 with identical ids.

### P5. AVX-VNNI for K-quant dots (conditional)

- **Mechanism**: `vpdpbusd` replaces `maddubs+madd` where the operand/scale structure allows. For Q4_K/Q6_K the per-16 sub-block scales complicate a direct port; an R8-interleaved layout may make a VNNI tile natural.
- **Decision**: do not prototype standalone; revisit only if P1 adopts a layout where scales are uniform per VNNI group. The CPU has AVX-VNNI, so the instruction is available.

### Do not pursue

- i-quant formats (IQ2/IQ3/IQ4_KS/R16): they are where ik_llama's largest prefill multiples come from, but they require a quantizer pipeline and matching model files; the local reference model is Q4_K_M.
- AVX-512 paths: the i5-14400F has AVX-512 fused off; the histories' Zen4 AVX-512 wins do not apply.
- NUMA replication/pinning: single socket; already covered by the pinning studies.
- Paged KV, weight eviction, load-speed work: measured as non-bottlenecks or already killed.
- GPU/MoE/MLA work: out of scope for this CPU dense model.

## 5. What this means for the llama gap

The 2026-11-08 comparison put TGE prefill at 0.72-0.85x of llama.cpp at 512 and 0.49-0.64x at 2048, and decode at parity with the shipped stack at 512. P1 attacks exactly the prefill deficit: a 1.3x matvec phase would move 2048 prefill from ~0.64x to roughly 0.72-0.75x, and a 1.6x phase toward ~0.8x. Full parity would still need the repacked formats and AVX-512-class kernels that ik_llama uses; on this CPU the realistic ceiling is the AVX2 R8 path.

Decode is bandwidth-bound: the histories' decode wins come from fewer streamed bytes per token (lower-bit formats), repacked loads, or AVX-512. TGE's kq8d tiers already reduce head/down work; further decode gains require format changes, not scheduling.

## 6. Proposed next cycles (in order)

1. **P1 kernel prototype**: R8-interleaved Q4_K prefill dot vs the current kernel on 2048x3072 and 4096x3072, differential-exact, quiet-box best-of-3. Kill/proceed gate at 1.3x.
2. **P2 fusion**: SiLU+quantize and RMSNorm+quantize, measured against the current two-pass path with byte-identical Q8_K output.
3. **P3 quantizer profile**: instrument `quantize_row_q8k` share at 2048/4096 prefill; only optimize if >=10% of matvec.

## 7. Reproduction

```text
# llama.cpp history (commit metadata only)
git -C <llama.cpp> fetch --shallow-since=2024-01-01 --filter=blob:none origin master
git -C <llama.cpp> log origin/master --since=2024-01-01 --format='%h|%ad|%s' --date=short
# perf slice
... | Select-String 'perf|speed|faster|optimi|throughput|latency'
# ik_llama.cpp
git clone --filter=blob:none --depth 800 https://github.com/ikawrakow/ik_llama.cpp
git -C ik_llama log --format='%h|%ad|%s' --date=short
# vLLM recent serving history
git clone --filter=blob:none --depth 500 https://github.com/vllm-project/vllm
```

Cited commits: llama.cpp `2dd84169`, `07d15723`, `85c55223`, `5c3d0f18`, `1e58ee13`, `ea5d7478`, `3a0d1053`, `9f682fb6`, `16bcc125`, `2fbde785`, `a05ca936`, `225e7a14`, `f896d2c3`, `05967042`, `c13e04e1`, `824eaecf`, `15665`, `15704`, `10803`; ik_llama.cpp `659d691`, `5ae7ce9`, `796f780`, `0a81bfe`, `7a31ec4`, `2c22783`, `1b18524`, `b9c47ba`, `b25c3fd`, `000c4da`, `9c312c2`, `1a3c785`, `e34d163`, `4b0afb3`, `1a2a860`, `b7f4411`, `44db69c`; vLLM `e336407`, `c2a6042`, `518c624`, `b02e228`.

No engine code changed; this report is analysis for the next integration/optimization cycles.
