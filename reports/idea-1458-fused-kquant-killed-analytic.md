# IDEA-1458: Fused K-Quant Dot with Unpack Reuse and Row-Interleaved Repack — KILLED WITHOUT SPIKE (shipped where it wins; 1-token bar fails at 7.85 vs 15+)

- **Idea ID**: IDEA-1458 (dossier `scout-2026-11-goal-ikllama.md`: ik_llama.cpp fused per-block K-quant dequant-dot, unpack-once-reuse across columns, R8 row-interleaved repack at load, fused gate/up single pass; header-claimed 150–350% PP over mainline)
- **Cycle date**: 2026-11-06
- **Verdict**: KILLED on the record, no spike built (dead-idea rule: the headline mechanism ships where it wins, and its falsifier's GB/s arm already fails on today's tree). No branch created, no code touched, nothing to revert.
- **Tree state at close**: `git status` clean (no diff), stash unchanged (pre-existing peer work only), `master` ahead of `origin/master` by 16 after this report. Nothing pushed (human gate).

## 1. Mechanism (as dossiered)

Fuse per-block K-quant (Q4_K/Q6_K) dequantization into the dot (no scratch
round-trip), quantize activations once and reuse across columns/rows, repack
weights row-interleaved (R8) at load when batch width ≥ 32, and fuse gate/up
projections into a single activation-stream pass — all on the AVX2 path,
mmap'd GGUF layout unchanged (repack is a load-time buffer).

## 2. Predicted gain (with the math) — and what today’s tree already takes

The dossier cites "aegean §7 dequantize-then-dot at 6.71 GB/s vs Q8_0 21.85"
and demands Qwen2 896×4864 Q4_K 1-token rise toward 15+ GB/s. Against today's
tree that premise is stale in three parts:

- Fused integer dot: SHIPPED where it wins — `gemv_dequant_batch_kq8_avx2`
  (maddubs integer path, per-call activation quant reused across all rows;
  `qwen2.cpp:982-987` routes Q4_K/Q6_K prefill batches with tokens>1 through
  it). Measured today at 6.27 GB/s vs 4.61 float on the 896×4864×6 prefill
  shape (+36%).
- 1-token shape: deliberately NOT wired — `qwen2.cpp:975-981` records the
  measurement (quantizer costs more than it saves at tokens==1: 0.8x
  end-to-end) plus the LEAK occupancy-invariant rule (forward_batch forces
  float so identical requests can't diverge on near-tie flips). The dossier's
  headline target is exactly the rejected shape.
- Unpack/gate-up reuse: SHIPPED in the forms that matter — per-call activation
  quant shared read-only across bands in kq8; cross-matrix quantized-activation
  reuse for q/k/v and gate/up in the VNNI path (`kernels.hpp` Q8BatchArgs
  contract). The unshipped residue (R8 load-time repack; kq8 gate/up single
  pass) attacks non-bottlenecks (§6).

## 3. Falsifier (from the dossier)

Port scratch kernel; `bench gemv` + `fidelity` on Q8_0 model — adapted to the
K-quant reality: kill if no repeatable >5% GB/s gain best-of-3 on the
dossier's own 896×4864 Q4_K shape toward 15+ GB/s, or mean KL > 1e-4 vs scalar
oracle, or top-1 < 497/497, or ids diverge under `--speculative`.

## 4. File anchors (current `master`, read but untouched)

- `src/cpu/avx2/dequant_batch_kq8_avx2.cpp` — shipped integer path (batch +
  single-row `kq8d` decode variant, bench-only for the latter).
- `src/models/qwen2.cpp:975-989` — routing with the 0.8x + LEAK comments;
  `:1548-1551`, `:1711-1714`, `:1928-1930` store sites; `:2097-2101` reset.
- `src/cpu/avx2/dequant_batch_avx2.cpp` — float dequant path (the 1-token
  model path).
- `include/tge/cpu/kernels.hpp` — `Q8BatchArgs.quantized` cross-matrix reuse
  contract; kq8 entries marked bench-only.

## 5. What was done (ANALYZE + bench on unmodified binary — no spike)

1. Re-read the dossier against the current tree; mapped each sub-mechanism to
   shipped / deliberately-rejected / no-carrier (§2).
2. Gates: `cmake --build --preset release` clean; `ctest` pass;
   `python ../../aikit.py check aegean-tge` 2/4 (test + secrets pass; hygiene
   + guidelines fail pre-existing).
3. `bench batch --format q4k` best-of-5 at the dossier's 896×4864 shape (cols
   must be %256; 896 is not, so rows=896/cols=4864): 1-token float (model
   path) vs kq8d (bench-only integer) vs 6-token float vs kq8.
4. `fidelity --kernel kq8` on Llama-3.2-3B-Q4_K_M (86-pos corpus, ctx 512):
   prices the shipped integer path against the scalar oracle.
5. No thinking-set diff owed: same binary throughout, zero behavior change.

## 6. Measured numbers (today's tree, unmodified binary, best-of-5)

| shape | kernel | best | note |
|---|---|---|---|
| 896×4864×1 (dossier target) | float (model path) | 0.460 ms / **5.33 GB/s** | ≈ the cited §7 6.71 — premise stale, gap unmoved on this path |
| 896×4864×1 | kq8d (bench-only) | 0.312 ms / **7.85 GB/s** | 1.47x kernel win, still **~half the 15+ bar** |
| 896×4864×6 | float | 0.531 ms / 4.61 GB/s | prefill baseline |
| 896×4864×6 | kq8 (ships in model) | 0.391 ms / **6.27 GB/s** | +36%, the shipped win |

Fidelity (`--kernel kq8`, Llama-3B): mean KL **1.620e-11**, worst 1.289e-10,
top-1 **86/86** — the numerics arms (KL ≤ 1e-4, top-1) PASS with four orders
of margin. The GB/s arm kills: even the tree's best integer kernel reaches
7.85 on the dossier's shape vs the demanded 15+. Kernel-level 1.47x exists but
system-level 1-token is quantizer-bound (the recorded 0.8x), and repack attacks
address math, not the quantizer — wrong bottleneck. End-to-end 1-token implied
≈ 4.3 GB/s (0.8 × 5.33), widening the gap further.

Residue audit: R8 repack needs a load-time full-model copy — doubles resident
weight memory on a 16 GiB single-DIMM box at 58% use, breaking the mmap-first
+ budget design for a sub-bottleneck; kq8 gate/up single-pass fusion would save
re-reading ~3.5 KB of activation per pair against ~2.3 MB weight streams
(<1% by construction). Both correctly unshipped.

## 7. Reimplementation recipe (deliberately NOT recommended — kill-grade)

For the record: the shippable subset already ships (`kq8`/`kq8d` +
`qwen2.cpp` routing + quantized reuse). The unshipped residue would be: (a)
load-time R8 repack buffer with budget accounting + repacked GEMV variants, or
(b) wiring `kq8d` into the 1-token model path behind the LEAK/occupancy
invariant. Per §6, (a) breaks the memory design for a non-bottleneck and (b)
contradicts two recorded measurements plus a correctness invariant. Do not
build either; the recipe for the verdict itself is §5 steps 3–4 (rerun the
four bench points + fidelity to confirm 7.85 < 15 and KL ~1e-11).

## 8. Verdict and why

**KILLED without spike.** The dossier's "must be true" premises do not survive
today's tree: the fused integer dot ships where it wins (+36% prefill), the
1-token target was measured (0.8x) and deliberately kept float behind a
correctness invariant, unpack reuse ships in both its forms, and the remaining
R8 residue has no carrier and attacks the wrong bottleneck (quantizer-bound,
not layout-bound). The falsifier's GB/s arm fails at 7.85 vs 15+ with the
tree's own best kernel; KL/ids arms pass but cannot save a speed idea that
misses its bar by 2x. Revisit only on a multi-channel box with a fixed
decode-quantizer amortization AND a repeal of the LEAK invariant — three
independent changes, none visible today.

## 9. Exact commands run and their results

- `cmake --build --preset release` → 54/54; `ctest` → pass.
- `bench batch --rows 896 --cols 4864 --tokens {1,6} --format q4k [--kernel kq8d|kq8] --repeat 5` → table in §6 (first attempt with cols=896 correctly rejected: Q4_K needs cols%256).
- `fidelity --model Llama-3B-Q4_K_M --corpus <86-pos> --context 512 --tokens 200 --threads 1 --gpu off --kernel kq8` → KL 1.620e-11, 86/86 (89 s).
- `aikit.py check` → 2/4 (pre-existing fails). No spike branch, no edits, no push.
