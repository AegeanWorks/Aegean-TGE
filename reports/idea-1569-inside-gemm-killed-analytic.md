# IDEA-1569: Compression-Inside-GEMM with Per-Shape Autotune and Fused Gate/Up TwoMatMul — KILLED WITHOUT SPIKE (autotune target is bench-only; fusion Amdahl-bound below 1%)

- **Idea ID**: IDEA-1569 (dossier `scout-2026-11-goal-gemma.md`: gemma.cpp `matmul-inl.h` MMDecompress-inside-LoopKC + `compress-inl.h` SFP-8b/NUQ formats + `dot-inl.h` compensated dots + `matmul_static_sfp.cc` multi-ISA build — MMAutoTune 7 params per (M,K,N) vs aegean fixed 144/256/256; TwoMatMul gate+up in one A-stream pass with single callback, halving FFN streaming in prefill)
- **Cycle date**: 2026-11-06
- **Verdict**: KILLED on the record, no spike built (dead-idea rule: both halves miss their production target on today's tree — no carrier to tune, no traffic to fuse away). No branch, no code touched, nothing to revert.
- **Tree state at close**: `git status` clean (no diff, no untracked), stash unchanged (only the pre-existing `leftover peer work shelved 2026-11-04`, untouched), `master` ahead of `origin/master` by 59 before this report. Nothing pushed (human gate).

## 1. Mechanism (as dossiered, per half)

Decompress weights inside the GEMM K-loop (no separate dequant pass) + per-shape autotuned blocking (MMAutoTune over 7 params per (M,K,N)) beating the fixed 144/256/256 + fused gate/up TwoMatMul running both FFN projections in one activation-stream pass. Claimed: prefill gap pp512 261→352 via autotuned MC/KC. Must-be-true: autotune finds MC/KC beating fixed blocking on the 14400F; port keeps GGUF quants (SFP/NUQ need offline conversion — mechanism not format, so no-retraining holds); compensated dots give a decode-GEMV analog.

## 2. Predicted gain — and why neither half reaches production

Autotune half: the tuned object (`gemm_packed<144,256,256>`, `src/cpu/avx2/gemm_avx2.cpp:26-29`) is bench-only. Production prefill never calls f32 GEMM — it routes Q8_0/K-quant/F32-per-token through `batch_matvec` (`src/models/qwen2.cpp:1511-1638, :1893-1981`), whose batch kernels (kq8 integer, q8, VNNI) carry their own format-fixed blocking (K-quant sub-blocks, 6-token groups), not MC/KC panels. An autotune sweep over MC/KC would optimize `bench gemm`, not prefill — the dossier's pp512 261→352 gap dates from the f32-GEMM prefill era and does not exist in that form (today's prefill is kq8 integer at 53+ t/s per 2k tokens on the reference model).

TwoMatMul half: gate and up each stream `batch_h_` plus their own weights (`:1626-1628` continuous, `:1969-1971` chunked prefill). Fusing shares one activation read per layer per chunk: 64 rows × 3072 × 4 B ≈ 786 KB for Llama-3B — L2-resident — against ~50 MB of Q4_K gate+up weights per layer from page cache/RAM. Amdahl ceiling ≈ 1.5% of layer traffic at L2-vs-RAM cost ratio, realistically <0.5% end-to-end — an order of magnitude below the 5% bar by construction. Worse, the production path it would fuse is the kq8 integer batch kernel (per-row activation quantization), so fusion is a new kernel with reordered accumulation (fidelity price owed) for unmeasurable gain — the wrong trade in both directions.

## 3. Falsifier (from the dossier) — resolved without building

"`bench batch` prefill shapes <5% over current blocking after autotune sweep" — moot: `bench batch`/production never calls the blocked kernel; sweeping MC/KC pairs a production workload against a bench-only kernel. "Fused gate/up `tge fidelity` KL>1e-6 vs f32" — moot with prejudice: no fused kernel exists to price, and §2 caps its win below 0.5%, so there is no gain a KL price could buy. "End-to-end prefill within machine noise" — conceded in advance: TwoMatMul lands inside noise by the traffic arithmetic above.

## 4. File anchors (current `master`, read but untouched)

- `src/cpu/avx2/gemm_avx2.cpp:26-29` (kMc/kKc/kNc 144/256/256 — the fixed blocking an autotune would sweep; bench-only), `:89-206` (`pack_a`/`pack_b` + `gemm_packed` — the inside-loop packing discipline already ships transiently, per 1539/1540).
- `src/models/qwen2.cpp:1511-1638` / `:1893-1981` (production batch routing: seven separate `batch_matvec` calls per layer/chunk — the fusion points, with the kq8 integer path as default).
- `apps/cli/main.cpp:198-201` + `tests/test_tensor.cpp:109-110` (the only `gemm_avx2`-family callers: `bench gemm` dispatch + unit tests — re-verified this cycle; `src/models` zero).
- Precedents: `idea-1325-pack-on-load-split.md` (f32-GEMM carrier proof), `idea-1458-fused-kquant-killed-analytic.md` (fused-dot family), `idea-1539-1540-gemm-layer-superseded.md` (microkernel/panel/edge accounting).

## 5. What was done (ANALYZE only — no spike per the dead-idea rule)

1. Read the dossier body against the current tree; split per half (§2) with the traffic arithmetic that kills the fusion and the caller inventory that kills the autotune.
2. Carrier greps (§4): blocked kernel bench-only confirmed this cycle; production batch routing + kq8 default confirmed; fusion points located (activation re-read quantified, not assumed).
3. Gates for the record on the unmodified tree: `cmake --build --preset release` clean (54/54); `ctest --preset release` pass; `tge_tests.exe` 141/141; `python aikit.py check aegean-tge` 2/4 (test + secrets pass; hygiene + guidelines fail pre-existing — no flake this cycle).
4. No fidelity run owed (no new arithmetic); no bench A/B owed (no production target to pair); no thinking-set diff owed (same binary).

## 6. Measured numbers — none owed (inventory + one Amdahl bound, both load-proof)

No benchmark belongs to a kill whose halves miss production. The operative numbers are structural, not sampled: fusion saves one ≤786 KB L2-resident activation re-read per layer per chunk against ~50 MB/layer weight traffic (Llama-3B Q4_K) → <0.5% end-to-end ceiling, inside the ±20% box noise by 40x. For orientation: the dossier's 261→352 pp512 gap is a pre-kq8 f32-GEMM number; today's prefill runs quantized-batch kernels the dossier's autotune cannot address.

## 7. Full reimplementation recipe (deliberately NOT recommended — revival requires a production target, never a sweep first)

Do not sweep MC/KC: it tunes a bench kernel. Do not fuse gate/up: sub-0.5% ceiling with a fidelity price attached. Revive only if f32 GEMM ever returns as the production prefill kernel (quantized batch abandoned — not on any roadmap), and then only as: bench sweep of blocking per prefill shape + differential-first fused kernel with KL ≤1e-6 vs f32 + end-to-end prefill ≥5% paired. To re-verify this kill: re-run the §4 caller inventory (bench+tests only, `src/models` still zero) and recompute the §6 traffic ratio for the widest model on disk.

## 8. Verdict and why

**KILLED without spike.** The autotune half optimizes a kernel production never calls; the fusion half eliminates a re-read the cache hierarchy already absorbs. Distinct from 1325 (persistent packing), 1458 (fused K-quant dot), and 1558 (scatter bins) per the dossier's own relationship note — judged independently, killed independently, same family discipline (ported-kernel distrust until priced, per 1568/1515).

## 9. Exact commands run and their results

- Dossier read + carrier inventory (§4: `gemm_avx2` callers bench+tests only; `batch_matvec` seven-call routing; fusion-point traffic math).
- `cmake --build --preset release` → 54/54; `ctest --preset release` → pass; `tge_tests.exe` → 141/141.
- `python aikit.py check aegean-tge` → 2/4 (pre-existing hygiene + guidelines fails). No branch, no edits, no bench, no fidelity, no push.

(End of file.)
