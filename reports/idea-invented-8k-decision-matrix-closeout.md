# CLOSEOUT: the 8k long-context decision matrix — every lever priced on speed and quality; f16 dense is the only free mode

- **Idea**: Five cycles this session priced 8k behaviour one lever at a time: speculation (`idea-invented-spec-at-8k-measured.md`), sparse decode (`idea-invented-sparse-8k-reopened.md`), sparse prefill (`idea-invented-sparse-prefill-8k-measured.md`), the block-sparse prefill kernel (`idea-invented-block-sparse-prefill-keep.md`), the code workload (`idea-invented-code-8k-spec-sparse-measured.md`), the geometry sweep on a 2k proxy (`idea-invented-sparse-geometry-sweep-measured.md`), and the 8k quality corrections (`idea-invented-sparse-8k-quality-correction.md`, `idea-invented-kv-tier-quality-8k-measured.md`). The reports individually are accurate but their guidance contradicts: sparse was first a "robust lever", then a 24-47pp quality cost; stride 16 was a 2.25pp option, then 14.4pp. This cycle consolidates everything measured on 3B/Qwen-0.5B at 8k into one speed x quality matrix, records which earlier guidance is superseded, and names the open lever. No new measurements; every number below is from a committed report this session.
- **Cycle date**: 2026-11-13
- **Verdict**: **f16 + dense is the only mode that is free on both axes at 8k.** Sparse is the only large speed lever and it costs 24-47 top-1 points at 8,199 positions; i8 costs 3.4pp and fails its own 2k bar; speculation is 1.13-3.0x depending on workload but quality-neutral; the block-sparse prefill kernel is a keep-grade speed recipe *inside* the lossy sparse mode. Detailed matrix in §1, supersession ledger in §2, recommended configurations in §3.

## 1. The matrix

### Quality at 8,199 positions (Qwen2.5-0.5B-Q8_0, `architecture.md` x9; candidate vs full-scalar f32-KV oracle)

| mode | mean KL | top-1 | cost vs oracle | source |
|---|---:|---:|---:|---|
| f16 KV, dense attention | 1.145e-06 | 8197/8199 (99.98%) | **−0.02pp** | kv-tier-quality-8k |
| i8 KV, dense attention | 1.657e-02 | 7920/8199 (96.60%) | **−3.40pp (bar fail)** | kv-tier-quality-8k |
| sparse 512/2 (~53% kept) | 1.250e+00 | 6216/8199 (75.82%) | −24.16pp | sparse-8k-quality-correction |
| sparse 512/4 default (~30%) | 1.630e+00 | 5533/8199 (67.49%) | −32.49pp | sparse-8k-quality-correction |
| sparse 512/16 (~12%) | 2.382e+00 | 4356/8199 (53.13%) | −46.85pp | sparse-8k-quality-correction |

Speculation is greedy-verifier-exact on top of any tier (sparse and spec+sparse ids byte-identical on code, `idea-invented-code-8k-spec-sparse-measured.md`), so it adds no quality row. The block-sparse prefill kernel is bit-identical to the per-token sparse path (`idea-invented-block-sparse-prefill-keep.md`), so it inherits the sparse row exactly.

### Speed at 8k (Llama-3.2-3B-Q4_K_M, f16 KV, 10 threads, GPU off, n=16-32)

| lever | periodic fox (8,193 tok) | code (8,104 tok) | prefill effect | notes |
|---|---|---|---|---|
| off (dense, per-token decode) | 5.21-5.68 t/s; attn 54-56% | 6.52 t/s; attn 52% | dense 205-213 s | reference |
| `--speculative` | 8.09-8.38 (**1.42-1.61x**); 3.00/step | 7.34-7.46 (**1.13-1.14x**); 3.00/step | ~1.00x (verify is decode-side) | quality-neutral |
| `--sparse` | 9.52-9.57 (**1.36-1.77x** vs dense); attn 24-25% | 10.49 (**1.61x**); attn 28% | per-token sparse 175-197 s (**1.06-1.20x**) | 24-47pp quality cost |
| `--speculative --sparse` | 15.61-16.36 (**2.2-3.0x** vs dense off); 5.33/step | 11.12-11.92 (**1.71-1.83x**); 2.71/step | 171-192 s (1.10-1.19x) | inherits sparse quality |
| sparse geometry 512/16 | 10.61 t/s (1.28x over sparse default); attn 10% | — | 181.5 s (1.23x over default geometry) | quality 53.13% (worst row) |
| block-sparse prefill (spike, reverted) | decode unchanged; prefill 162.4 s (1.265x dense, 1.18-1.24x per-token sparse); spec verify 1.24x | — | attention 27.1 s vs 66.7 dense / 57.2 per-token | recipe only; quality = sparse |

"vs dense off" is the same-session ratio where measured; all prefill totals are prefill-dominated runs (16-32 generated tokens), so whole-run wall tracks the prefill column, not the decode column.

## 2. Supersession ledger (what changed this session)

| earlier claim | where | status now |
|---|---|---|
| "sparse is the workload-robust 8k lever" | `idea-invented-sparse-8k-reopened.md` | **superseded on quality**: speed robust, quality 24-47pp lost. Read as a speed claim only. |
| "`--sparse-stride 16` costs 2.25pp top-1 at 8k" | `idea-invented-sparse-geometry-sweep-measured.md` §4 | **withdrawn**: 14.37pp measured at 8,199 positions. The 2k proxy understated the default tier too (88.85% vs 67.49%). |
| "i8 holds at 2048 context (99.9%, bar 98.6%)" | `idea-invented-i8-longctx-confirmed.md` | **does not transfer to 8k**: 96.60%, 2.0pp below the bar. i8 is a capacity mode, not a quality tier. |
| "the normative sparse gate: prefill-bound long prompts" | ADR-007 / measurement-log §50 | **still true as a speed statement; incomplete as a recommendation**: at 8k it is a 24-47pp quality trade. |
| "speculative 8k is 2.2-3.0x" | `idea-invented-spec-at-8k-measured.md` | **workload-qualified**: 1.42-1.61x periodic, 1.13x code; both at 3.00 accepted/step, the difference is the verify share and baseline. |

## 3. Recommended configurations at 8k (this box, 3B Q4_K_M)

1. **Quality-first (default recommendation)**: `--kv f16`, dense attention. Free on quality (99.98%), 205-213 s prefill, 5.2-6.5 t/s decode. Add `--speculative` only when the prompt echoes structure (code, templates, repeated text): 1.13-1.61x with identical ids.
2. **Speed-first, lossy accepted**: `--kv f16 --speculative --sparse`. Periodic 15.6-16.4 t/s (2.2-3.0x), code 11.1-11.9 t/s (1.71-1.83x), prefill 1.10-1.19x. Quality is the sparse row: 67.49% top-1 at 30% kept. State the cost when quoting the number.
3. **Memory-constrained**: `--kv i8`, dense attention. KV quartered for ~3.4pp top-1; decode speed at 8k is a measured loss (`idea-invented-i8-8k-killed.md`), so this is not a speed mode.
4. **Prefill-dominated 8k runs (short outputs)**: the whole-run wall is the prefill column. Sparse mode + the block-sparse prefill recipe (when ported) is the measured prefill lever: 162-181 s vs 205-213 s. Without the lossy mode, no prefill lever is on record that preserves quality at 8k (the tiled/sparse/"move less" prefill attempts are killed or lossy).
5. **Do not use**: sparse geometry narrower than window 512 for quality work (256/8 and 128/8 cost 6.5-11.2pp at 2k and worse at 8k), `--sparse-stride 16` as a general option, or the 2k i8 bar as an 8k claim.

## 4. The gap this matrix exposes

There is no measured, quality-preserving 8k speed lever. f16 is free but not faster; speculation is the only quality-neutral accelerator and it needs draftable text; everything that cuts the 8k attention phase (sparse, stride dilution, i8 KV) is lossy. The named candidates and their state:

- **Block-sparse prefill kernel**: keep-grade, 1.18-1.27x prefill — but only inside sparse mode, so it accelerates a lossy configuration. A port would make option 2 faster, not option 1.
- **Scored middle selection** (IDEA-1451): quality edge +3 flips at 2k, speed +64-114% attention (killed). At 8k the scoring pass reads every key row, which is ~half of dense attention's row traffic, so it cannot beat dense on speed; not reopened.
- **i8 with finer scales (per-16) or mixed K/V tiers**: would need to recover +2pp top-1 at 8k; the current 96.60% is already the per-32 result. A mixed f16-K/i8-V or i8-K/f16-V probe is the cheapest next experiment (separate buffers already exist in the attention args, but the `half` flag currently couples K and V in every kernel), and it is the only path to a half-memory tier that might clear the 8k bar.
- **Long-context attention kernel work**: killed by the compute-bound ceilings (tiled attn ~0-2%, grouped block 1.4-1.7x slower). Nothing on record reopens them.

## 5. File anchors (current HEAD, read-before-write verified)

- `include/tge/models/qwen2.hpp` — `SparseConfig{sink=4, window=512, stride=4}`, KV tier load signature.
- `src/models/qwen2.cpp:1329-1362` — sparse dispatch and `uses_sparse`.
- `src/models/qwen2.cpp:1580-1660` — block kernel gate and per-token fallback.
- `src/cpu/avx2/attention_avx2.cpp:263,367,660` — strided sparse kernels and the dense block kernel.
- `apps/cli/main.cpp:2177-2236` (generate) and `:2699+` (fidelity) — sparse flags and fidelity harness.
- Reports consolidated: `idea-invented-spec-at-8k-measured.md`, `idea-invented-sparse-8k-reopened.md`, `idea-invented-sparse-prefill-8k-measured.md`, `idea-invented-block-sparse-prefill-keep.md`, `idea-invented-code-8k-spec-sparse-measured.md`, `idea-invented-sparse-geometry-sweep-measured.md`, `idea-invented-sparse-8k-quality-correction.md`, `idea-invented-kv-tier-quality-8k-measured.md`.

## 6. Reimplementation recipe (rebuild the matrix)

1. Gate: `cmake --build --preset release` + `tge_tests.exe` (expect 171/171).
2. Quality rows: build the 8k corpus (`architecture.md` x9, 107,624 bytes) and run one fidelity per mode, candidate vs the built-in oracle:
   `tge fidelity --model qwen2.5-0.5b-instruct-q8_0.gguf --corpus <corpus8k> --context 8704 --tokens 8200 --threads 10 --gpu off [--kv f16|i8] [--sparse --sparse-window 512 --sparse-stride S]`. Each run ~8-11 min; expect the §1 quality rows exactly (deterministic per binary).
3. Speed rows: 3B Q4_K_M, compact prompts (`The fox runs. ` x 2048 = 8,193 tokens; the 213-function code prompt = 8,104 tokens), `--context 8704 --n 16-32 --kv f16 --threads 10 --gpu off [--speculative] [--sparse]`. Each run ~3-4 min of prefill; interleave arms and compare only pairs whose matvec phases agree within ~3% (the box drifts 20-40% over a 20-minute window).
4. Id parity: any arm compared to another must match `--print-ids` byte for byte (sparse vs spec+sparse matched on code).
5. No code ships; the only non-shipped artifact referenced is the block-sparse prefill recipe in `idea-invented-block-sparse-prefill-keep.md` §6.

## 7. Exact commands run and results

This cycle ran no new measurements. The gate was re-run for the commit: `cmake --build --preset release` clean, `tge_tests` 171/171, `python aikit.py check aegean-tge` 4/4. All numbers are lifted from the committed reports listed in §5; the two contradictory proxies are corrected against the 8,199-position runs recorded in `idea-invented-sparse-8k-quality-correction.md` and `idea-invented-kv-tier-quality-8k-measured.md`.

No push, no merge, no force-push. The committed tree is unchanged except this report.
