# IDEA-1539 + IDEA-1540: Broadcast GEMM Microkernel + Packed-B M==1 Fast Path — SUPERSEDED (the described layer is the shipped layer)

- **Idea IDs**: IDEA-1539 (XNNPACK broadcast-A + contiguous-B, MRxN tile, clamp epilogue; falsifier: ported microkernel vs shipped `bench gemm` at 512/1024/2048 cubed, no codegen-toolchain violation) + IDEA-1540 (onnxruntime packed-B 16-col zero-padded + M==1 decode path; falsifier: packed-B variant vs `bench gemm`/`gemv` on 896x896 and 4864x896, E-core throughput holds). One report: the two dossiers describe one kernel layer, and the tree matches it point for point.
- **Cycle date**: 2026-11-06
- **Verdict**: SUPERSEDED — no spike built (a port would retype the shipped kernel: same broadcast, same 16-wide packed B, same stack-buffer edges, same dedicated-GEMV decode). No branch created, no code touched, nothing to revert.
- **Tree state at close**: `git status` clean apart from PRE-EXISTING, UNCOMMITTED peer spike (verified byte-identical to the saved baseline at cycle start AND end — untouched). `master` ahead of `origin/master` by 46 after this report (reports only). Nothing pushed (human gate).

## 1. Mechanism (as dossiered, both halves)

1539: broadcast-A + contiguous-B operand discipline, 6x16-class MRxN tile, clamp epilogue, per-ISA TU flags. 1540: 16-column zero-padded B panels, dedicated M==1 decode path (not GEMM reuse), kernel-set selection, edge handling without masked vector code.

## 2. Predicted gain — already collected, point by point

| dossier demand | tree | anchor |
|---|---|---|
| `_mm256_broadcast_ss` A + fp32 fmadd accumulate | verbatim in the dot row | `gemm_avx2.cpp:48-50` (also `gemv`/`q8_batch` siblings) |
| contiguous-B panel loads | packed B panels, 64-B aligned | `gemm_avx2.cpp:32, 89-118` |
| 6x16 tile, 12 accumulators, no spills | exactly as specified | `gemm_avx2.cpp:16-19, 38-42` |
| 16-col B panels, blocked GEMM | `kCols/kMc/kKc/kNc` + `gemm_packed<144,256,256>` (144+256 KiB in 2 MiB L2) | `gemm_avx2.cpp:26-29, 120-206` |
| edges without masked vector code | stack-buffer tile + scalar copy | `gemm_avx2.cpp:72-86`; zero `maskload/maskstore/blendv/mask` in `src/cpu/avx2` (grepped) |
| M==1 decode fast path (not GEMM) | decode routes to the dedicated `gemv_*` family; GEMM is never called with M==1 | `qwen2.cpp` matvec dispatch (`gemv_q8/_q8_0/_f32`, batch variants) |
| per-ISA TU flags, no codegen | hand-written TU + CMake ISA flags | `CMakeLists.txt` (`-mavx2 -mfma -mf16c` / `/arch:AVX2`), generator-free by design |

Not shipped and correctly so: the clamp epilogue (no fused min/max consumer exists anywhere — activations are separate kernels), the codegen template (same emitted shape without the toolchain), explicit M==1 GEMM-side routing (decode never reaches GEMM).

## 3. Falsifiers (from both dossiers) — nothing to run them against

1539's port-vs-shipped A/B presupposes a port that differs; a retyped identical kernel can only tie within noise (and identical numerics — same op order). 1540's packed-B-vs-tree A/B compares the tree against itself (the tree IS packed-B with stack-buffer edges). E-core throughput is guarded by existing H-notes.

## 4. File anchors (current `master`, read but untouched)

- `src/cpu/avx2/gemm_avx2.cpp` (microkernel, packing, edge rule)
- `src/cpu/avx2/gemv_avx2.cpp`, `src/models/qwen2.cpp` (dedicated decode)
- `CMakeLists.txt` (per-TU ISA flags), `tests/test_gemm.cpp` + `bench gemm/gemv` (differentials + kernel harness)

## 5. What was done (ANALYZE only — no spike per the dead-idea rule)

1. Read both dossier bodies against the current tree, line by line (§2).
2. Verified the two negative claims (no clamped epilogue consumers, no masked vector loads) by grep.
3. Gates for the record: build clean, `ctest` pass, `aikit.py check` 2/4
   (test + secrets pass; hygiene + guidelines fail pre-existing).
4. No thinking-set diff owed: same binary, zero behavior change.

## 6. Measured numbers — none owed (nothing unshipped to measure)

The applicable measurements already exist on record: `bench gemm/gemv`
differentials green, `gemv_avx2` 6.5-7.4 GB/s-class decode paths priced in
the H-notes. No new benchmark belongs to a supersede verdict.

## 7. Open residue: none

The only unshipped fragments are non-requirements (clamp epilogue without a
consumer, generator without a toolchain gap, M==1 GEMM routing for calls
that never happen). There is no follow-up to name.

## 8. Verdict and why

**SUPERSEDED, jointly.** Two dossiers describing one shipped kernel layer,
down to the tile size, the broadcast intrinsic, the edge-copy idiom, and the
decode-not-GEMM routing. Porting either would be transcription, not
improvement.

## 9. Exact commands run and their results

- `git diff` baseline re-verified IDENTICAL at cycle start and end (peer
  spike untouched throughout).
- `cmake --build --preset release` → 54/54; `ctest` → pass.
- `aikit.py check` → 2/4 (pre-existing fails). No branch, no edits, no push.
