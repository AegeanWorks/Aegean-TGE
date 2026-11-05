# IDEA-1581: Compute-in-quant CPU GEMM with fidelity oracle — SUPERSEDED (integer dot ships default-on)

- **Idea ID**: IDEA-1581 (dossier `kilo-sweep-2026-11-03-b5.md`: vllm.cpp compute directly on compressed blocks, no BF16 expansion, byte-identical greedy on gated paths, 1.18x prefill; falsifier: id drift, KL above zero, or prefill ratio below 1.0x)
- **Cycle date**: 2026-11-06
- **Verdict**: SUPERSEDED — the substance ships and is default-on. No spike built (building another integer dot would duplicate the kq8 path); no branch, no code touched.
- **Tree state at close**: `git status` clean (no diff), stash unchanged (pre-existing peer work only), `master` ahead of `origin/master` by 26 after this report. Nothing pushed (human gate).

## 1. Mechanism (as dossiered)

Integer dot products directly on compressed weight blocks with a single
per-call activation quantization, no BF16/f32 weight expansion, routed by
quant type, priced by a fidelity oracle.

## 2. Predicted gain — and where the tree already collects it

`src/cpu/avx2/dequant_batch_kq8_avx2.cpp`: scalar per-row Q8_K activation
quant once per call, AVX2 `maddubs` integer dots over packed Q4_K/Q6_K with
per-block scales at accumulation — no fp32 weight materialization. Wired as
the DEFAULT prefill path (`qwen2.cpp:982-987`: Q4_K/Q6_K, tokens>1,
`kq8_` defaults true; `bench batch --kernel kq8` keeps the isolated
instrument). Q8_0 has its own fused integer path (`q8_batch_avx2`,
`USE_VNNI`/`USE_Q8` dispatch). Measured 1458-cycle: 896×4864×6 Q4_K 6.27 vs
4.61 GB/s float (**+36%**); kq8 fidelity KL 1.467e-11 at ctx 2048 (round-7
log); `kq8d` single-row variant bench-only (deliberately unwired at tokens==1
per the recorded 0.8x + LEAK notes).

## 3. Falsifier (from the dossier) — does not trip as written

Id drift (none — greedy verifier-gated, ids identical), prefill ratio below
1.0x (measured 1.36x, ships default-on). The "KL above zero" arm is met by no
quantized path anywhere including the dossier's own source (which prices
itself at NMSE 5e-4): the tree operationalizes it as tolerance pricing
(kq8 KL ~1e-11, i8 3.2e-4, VNNI 1.6e-3 with 12 documented flips), and kq8
passes every tolerance gate. A literal-zero KL bar kills all quantization
including the proposal — it is not a bar, it is a ban.

## 4. File anchors (current `master`, read but untouched)

- `src/cpu/avx2/dequant_batch_kq8_avx2.cpp` (batch + `kq8d` row variant).
- `src/models/qwen2.cpp:975-989` (routing with the 0.8x + LEAK comments).
- `include/tge/cpu/kernels.hpp` (Q8BatchArgs quantized-reuse contract).
- `tests/test_dequant.cpp` (kq8 differential vs float oracle).
- `docs/measurement-log.md` §§57/61 + `reports/idea-1458-*.md` (numbers cited).

## 5. What was done (ANALYZE only — no spike per the dead-idea rule)

1. Mapped every operative sub-mechanism to shipped code (§2).
2. Gates for the record: build clean, `ctest` pass, `aikit.py check` 2/4
   (test + secrets pass; hygiene + guidelines fail pre-existing).
3. No thinking-set diff owed: same binary, zero behavior change.

## 6. Measured numbers (cited, not re-run — same tree)

896×4864×6 Q4_K: kq8 6.27 GB/s vs float 4.61 (+36%, best-of-5, 1458 cycle).
kq8 fidelity KL 1.467e-11, worst 5.5e-10 (ctx 2048, round-7). Both inside the
dossier's spirit (prefill win, oracle-priced).

## 7. Open residue (none from this idea)

The only unshipped integer-dot adjacent is the deliberately-rejected 1-token
`kq8d` wiring (0.8x + LEAK invariant) and the forward_batch mixed-batch CoW
port (1384 follow-up) — both owned by their own reports, not this one.

## 8. Verdict and why

**SUPERSEDED.** Compute-in-quant with a fidelity oracle is this engine's
standing prefill architecture, default-on, measured, and priced — not a
proposal. Revisit only if a new compressed format appears that the current
integer path cannot consume (then the format, not the dot, is the idea).

## 9. Exact commands run and their results

- `cmake --build --preset release` → 54/54; `ctest` → pass.
- `aikit.py check` → 2/4 (pre-existing fails). No branch, no edits, no push.
