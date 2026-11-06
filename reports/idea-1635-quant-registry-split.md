# IDEA-1635: Quant-Registry Dispatch with Coherence Harness — SPLIT (registry + harness ship; AVX512/repack/pread/TurboQuant-KV dead)

- **Idea ID**: IDEA-1635 (dossier `kilo-sweep-2026-11-03-b8.md`: bitnet.c quant-registry + capability-gated dispatch + x8 repacked layouts + mmap/pread LRU + TurboQuant KV; what-must-be-true centers on AVX512 BW/VNNI kernels matching refs on qwen2.5-3b-q4_0)
- **Cycle date**: 2026-11-06
- **Verdict**: SPLIT. (a) Quant-registry dispatch + coherence harness: SUPERSEDED — `to_quant_kind` routing, `kernels()` dispatch set, scalar-oracle differentials (141 tests) and the fidelity oracle-vs-production harness all ship. (b) AVX512 kernels, x8 repack, pread LRU, TurboQuant KV: KILLED — no AVX512 kernels exist (detection only; box ISA lacks it), no repack buffers (mmap-first), no pread path (same MoE-streaming family), TurboQuant KV dead via the int4/int2 verdicts. No spike built (nothing unmeasured remains); no branch, no code touched.
- **Tree state at close**: `git status` clean (no diff), stash unchanged (pre-existing peer work only), `master` ahead of `origin/master` by 33 after this report. Nothing pushed (human gate).

## 1. Mechanism (as dossiered, per part)

Registry mapping quant types to kernels with capability gating; a coherence
harness checking kernels against scalar/AVX2 refs; AVX512 BW/VNNI fast paths;
x8 row-interleaved repacked layouts; mmap/pread LRU expert streaming;
TurboQuant KV tiers.

## 2. Predicted gain — and the per-part disposition

| part | status | grounds |
|---|---|---|
| quant-registry dispatch | SUPERSEDED | `to_quant_kind` + per-kind entries + `kernels()` set + capability gates (`features.f16c/avx_vnni`, VNNI opt-in) ship |
| coherence harness | SUPERSEDED | scalar-oracle differentials per kernel family + `tge fidelity` oracle-vs-production + FNV gates ship |
| AVX512 BW/VNNI kernels | KILLED | detection only (`cpu.cpp` flags); zero `_mm512` sites; target box (i5-14400F, Raptor Lake, AVX512 fused off) cannot execute them — dead code on arrival |
| x8 repacked layouts | KILLED | no repack buffers (only Q5_0 bit-spread unpack LUTs, the opposite direction); mmap-first design excludes load-time copies (same R8 analysis, 1458 report) |
| pread LRU streaming | KILLED | zero pread/O_DIRECT/io_uring matches; same no-MoE/no-pressure family as 1382/1518 kills |
| TurboQuant KV | KILLED | sub-8-bit KV 0-for-2 on this tree (2-bit KL 0.13; int4 KL 4.6e-3 vs 1e-3 bar) |

## 3. Falsifier (from the dossier) — resolved per part

"Kernel divergence vs scalar or AVX2 refs" — no divergence possible where no
kernels exist (AVX512); shipped kernels pass (differentials green, fidelity
priced). "GPU-vs-CPU mismatch" — no GPU backend in scope (CPU-only engine).
"pread slower than mmap" — moot with no pread path and no pressure (fits RAM
3x over).

## 4. File anchors (current `master`, read but untouched)

- `src/core/cpu.cpp:208-209,254-289` (AVX512 detection + OS-gate disabling),
  `include/tge/core/cpu.hpp:23-31` (flags) — detection without kernels.
- Registry/dispatch: `to_quant_kind` sites + `kernels()` set + `set_*` gates.
- `reports/idea-1516-kivi-stage2a-killed.md`, `idea-1385-int4kv-killed.md`,
  `idea-1382-moe-streaming-killed-analytic.md` — the kills this compounds.
- `docs/measurement-log.md` — i8/VNNI/f16 pricing receipts.

## 5. What was done (ANALYZE only — no spike per the dead-idea rule)

1. Carrier greps: AVX512 (detection only, no `_mm512`), repack (unpack LUTs
   only), pread/uring (zero) — each part classified shipped vs dead.
2. Gates for the record: build clean, `ctest` pass, `aikit.py check` 2/4
   (test + secrets pass; hygiene + guidelines fail pre-existing).
3. No thinking-set diff owed: same binary, zero behavior change.

## 6. Measured numbers (inventory — the parts that can run already did)

Registry/coherence: 141/141 tests green incl. per-family differentials;
fidelity precedents f16 1.7e-6, VNNI 1.6e-3, i8 3.2e-4. Dead parts: zero
carrier matches; box AVX512 absent by silicon (Raptor Lake); RAM headroom 3x.

## 7. Reimplementation recipe (deliberately NOT recommended — split verdict)

Shipped parts need no recipe (in-tree). Dead parts must not be built:
AVX512 kernels (no ISA on target), repack (breaks mmap-first for a
layout-bound non-bottleneck), pread streaming (no pressure), TurboQuant KV
(sub-8-bit 0-for-2). Revisit any single part only with its own carrier
first (ISA on box, repack budget, pressure, tensor evidence) — per the
stage-1 pattern, never engine-first.

## 8. Verdict and why

**SPLIT.** A bundle whose shippable half ships and whose remainder fails
four independent carrier checks is not a task — it is inventory. The loop's
honest output is the map (§2), so no future cycle re-treads the bundle.

## 9. Exact commands run and their results

- Carrier greps (AVX512/repack/pread) → detection-only / unpack-only / zero.
- `cmake --build --preset release` → 54/54; `ctest` → pass.
- `aikit.py check` → 2/4 (pre-existing fails). No branch, no edits, no push.
