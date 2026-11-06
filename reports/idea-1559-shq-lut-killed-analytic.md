# IDEA-1559: Learnable Companded Grids (SHq) as Weight-LUT Quantization — KILLED WITHOUT SPIKE (no quantizer, no writer, no files, no registry kind)

- **Idea ID**: IDEA-1559 (dossier `scout-2026-11-goal-lgpapers.md`: arXiv:2611.02966 — 2-param family SHq(α,c)=sinh(αu)+c sampled uniform, E4M3-snapped LUT drop-in, no storage/decode change; offset lets FP8 sign select grid vs mirror; calibrate (α,c) per model/width with RTN, no retraining; CPU LUT-decode portable)
- **Cycle date**: 2026-11-06
- **Verdict**: KILLED on the record, no spike built (dead-idea rule: the dossier's falsifier is a calibration-gain comparison that cannot be constructed on today's tree — no carrier at any of its four legs). No branch created, no code touched, nothing to revert.
- **Tree state at close**: `git status` clean (no diff, no untracked), stash unchanged (only the pre-existing `leftover peer work shelved 2026-11-04`, untouched), `master` ahead of `origin/master` by 51 before this report. Nothing pushed (human gate).

## 1. Mechanism (as dossiered)

Per model/width, calibrate two scalars (α,c) with round-to-nearest (RTN, no retraining) over the SHq family SHq(α,u)=sinh(αu)+c; snap the sampled grid to E4M3; decode weights through the LUT at inference (drop-in: no storage or decode-path change); the offset lets the FP8 sign bit select grid vs mirror, doubling distinct values at 3b (paper: Qwen3-8B 3b gives 16 distinct values vs 8/9/13 for fixed grids). Reported: lowest held-out KL in 45/50 settings over 5 models (3B–35B+MoE, 2–6b), 4.0–34.4% over best fixed at 3/4b, NVFP4 beaten 40.7–67.4% at 4.5bpw, sign-bit removal +21–38% KL at 3/4b, one grid per model suffices. TGE relevance claimed: a weight-LUT format (distinct from the shipped f16/i8 KV tiers) with CPU-portable LUT decode.

## 2. Predicted gain (with the math) — and why none of it is reachable here

Paper-side, the gain is KL at fixed bit-width (up to −38% vs fixed grids) plus LUT-decode replacing per-element dequant ALU. For TGE that would mean a new weight format cheaper than dequantize-then-dot with better fidelity than fixed K-quants. Every leg of that story needs a carrier the tree does not have (§5). A kernel-only LUT prototype (quantize random rows to a hand-picked grid in-test, LUT-decode, differential) would price a kernel, not the dossier's falsifier — which is explicitly a *calibration* win (≥3% over best fixed grid on Qwen2.5-Coder-1.5B with RTN) plus mirror-gain survival through E4M3 plus no accuracy inversion. Pricing the kernel while the calibration leg is unbuildable would manufacture a number for a gate it cannot enter.

## 3. Falsifier (from the dossier) — unconstructible, all four legs missing

"Calibrate (α,c) on Qwen2.5-Coder-1.5B with RTN — reject if win over best fixed grid <3% (GPTQ noise floor), or mirror gain vanishes after E4M3 rounding, or KL win repeats the 2b-MoE inversion (lower KL but −1.2pp accuracy)."

| leg | carrier needed | tree status |
|---|---|---|
| RTN quantizer + (α,c) scan + KL harness | offline calibration pipeline writing candidate weight files | absent: engine is read-only by design, zero GGUF write/export/convert/imatrix paths anywhere in `src/` (same kill family as IDEA-1386) |
| SHq-quantized model files | GGUFs carrying SHq grids (the dossier's own gate model is Qwen2.5-Coder-1.5B) | absent: all 8 local models are standard Q4_K_M/Q8_0/Q5_K_M; Coder-1.5B exists only as Q4_K_M |
| LUT-grid weight kind in the loader/kernels | new `QuantKind` + `to_quant_kind` mapping + dequant path; loader must accept the type | absent: registry closed at 7 fixed GGUF block formats (Q8_0/Q4_K/Q5_K/Q6_K/Q4_0/Q5_0/Q5_1); unknown types throw at load (`type_supported` gate) |
| SHq/sinh/compand/grid code | any implementation foothold | absent: zero matches in `src/` + `include/` (the only "LUT" hits are comments on the Q5_0 bit-spread unpack table — the opposite direction, a decode helper for a fixed format) |

Distinct from IDEA-1386 (generic sign-aware PTQ + K-quant-aware GGUF export — killed analytic for the same read-only reason) per the dossier's own relationship note: this is the SHq family + calibration-KL scan + LUT mechanism. The kill compounds rather than duplicates: 1386 closed the exporter door; 1559 has no other door (no files, no registry kind, no code).

## 4. File anchors (current `master`, read but untouched)

- `src/models/qwen2.cpp:105-127` (`to_quant_kind`: closed 7-kind mapping), `:739-804` (loader `type_supported` gates + row-stride contract — unknown types throw, never run mis-sliced), `:681-711` (supported-type matrix).
- `src/cpu/scalar/dequant.cpp:240-332`, `src/cpu/avx2/dequant_avx2.cpp:401-433` (per-kind dequant dispatch — no grid/LUT kind exists to add a case to without files first).
- `src/` grep for `write_gguf|export_gguf|imatrix|quantize_file|save_gguf` → zero code matches (only the 1386 report text mentions them).
- `src/` + `include/` grep for `SHq|sinh|compand|lut_decode` → zero code matches.
- `reports/idea-1386-ptq-export-killed-analytic.md` (the read-only precedent: "weights arrive pre-quantized; the engine never produces a weight file").
- Local models: `Llama-3.2-3B-Instruct-Q4_K_M`, `qwen2.5-coder-1.5b-instruct-q4_k_m` (gate model present, wrong format), plus 6 standard-quant files — none SHq.

## 5. What was done (ANALYZE only — no spike per the dead-idea rule)

1. Read the dossier body against the current tree; mapped each falsifier leg to its required carrier (§3).
2. Carrier greps (§4): SHq family zero, writer/exporter zero, registry closed, local files standard-only.
3. Gates for the record on the unmodified tree: `cmake --build --preset release` clean (54/54); `ctest --preset release` pass; `tge_tests.exe` 141/141; `python aikit.py check aegean-tge` 2/4 (test + secrets pass; hygiene + guidelines fail pre-existing — C++-only repo, README `reports` drift, same pair as every recent cycle).
4. No fidelity run owed (no arithmetic exists to price); no bench A/B owed (no second path exists to pair); no thinking-set diff owed (same binary, zero behavior change).

## 6. Measured numbers — none owed (nothing unshipped to measure)

The applicable evidence is the carrier inventory in §3 (four zeros) plus the green gates above. No benchmark belongs to a kill verdict whose falsifier cannot be constructed. For orientation only: the dossier's paper-side numbers (45/50 lowest-KL settings, 4.0–34.4% over fixed at 3/4b) are receipts-with-source, not independently verified facts, and they describe the paper's pipeline — not this engine.

## 7. Full reimplementation recipe (deliberately NOT recommended — revival requires the pipeline first, never the kernel first)

Do not build the LUT kernel first: without files it is untestable performance art (same rule that left Q5_K/Q4_0 scalar-oracle-only until a file containing them appeared, §28l). Revisit only as a pipeline, in this order: (1) an offline RTN calibrator producing SHq-quantized GGUFs (outside this engine — the 1386 verdict stands: the engine never writes weight files); (2) a calibration KL scan on a local model (Coder-1.5B per the dossier) showing ≥3% win over best fixed grid with no accuracy inversion, E4M3-snapped; (3) only then a differential-first LUT-decode kernel round (scalar oracle from the grid definition, memcmp-gated AVX2, fidelity ≤ VNNI 1.6e-3, stash-built A/B). To re-verify this kill: re-run the four greps in §4 and confirm they are still zero, plus `ls` the models dir for non-standard quants.

## 8. Verdict and why

**KILLED without spike.** SHq is a pipeline idea (calibrate → write files → LUT-decode) proposed to a read-only engine with no pipeline: no quantizer, no writer, no registry kind, no files, no code foothold. The dossier's own falsifier cannot be run, and running a different, easier falsifier (kernel differential on synthetic grids) would not answer the stated gate. Any future revival starts at leg 1 (offline calibrator + files + ≥3% KL proof), never at the kernel.

## 9. Exact commands run and their results

- Carrier greps (§4): SHq/sinh/compand zero; writer/exporter zero (code); QuantKind closed at 7; local models all standard quants.
- `cmake --build --preset release` → 54/54; `ctest --preset release` → pass; `tge_tests.exe` → 141/141.
- `python aikit.py check aegean-tge` → 2/4 (pre-existing hygiene + guidelines fails). No branch, no edits, no bench, no fidelity, no push.

(End of file.)
