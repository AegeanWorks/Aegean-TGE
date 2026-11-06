# IDEA-1325: Pre-Packed Weight Matrix with Blocking Factors for Prefill — SPLIT (blocking/packing ships; persistent pack-on-load killed with no caller)

- **Idea ID**: IDEA-1325 (dossier `scout-2026-11-tge-kernels.md`: FBGEMM PackMatrix CRTP + PackBMatrix::isPrePacked + packedBufferSize + BlockingFactors MCB/KCB/NCB/MR/NR + PackAWithRowOffset; constant weights packed once into blocked tiles, packed-GEMM for multi-token prefill, unpacked fallback for single-token decode)
- **Cycle date**: 2026-11-06
- **Verdict**: SPLIT. (a) Blocking factors + packing discipline: SUPERSEDED — `pack_a`/`pack_b` + `gemm_packed<144,256,256>` + 6x16 microkernel ship verbatim (jointly recorded in the 1539/1540 report). (b) Persistent pack-on-load buffer: KILLED — no production caller exists on today's tree, the mmap-first design bans the load-time copy family, and the bandwidth roof is already closed at 97% on the kernel that actually runs prefill. No spike built (dead-idea rule: falsifier fails by construction, nothing unshipped to measure); no branch, no code touched, nothing to revert.
- **Tree state at close**: `git status` clean (no diff), stash unchanged (only the pre-existing `leftover peer work shelved 2026-11-04`, untouched), `master` ahead of `origin/master` by 49 before this report. NOTE: at cycle start the tree carried a PRE-EXISTING UNCOMMITTED peer spike (`spike-batchsnap` driver + `batch_snapshot_*` plumbing in `apps/cli/main.cpp`, `include/tge/models/qwen2.hpp`, `src/models/qwen2.cpp`); it was never touched by this cycle and was gone by cycle end via concurrent peer revert, not by any command below. Nothing pushed (human gate).

## 1. Mechanism (as dossiered, both halves)

FBGEMM-style: pack each constant weight matrix ONCE at load into blocked tiles sized by blocking factors (bit-identical reorder, no retraining; activation packing fuses the row-offset computation), stream the packed layout in a packed-GEMM path for multi-token prefill (M >= 32), keep the unpacked path for single-token decode. Dossier value score 4, Med-High impact / High effort; concept-only clean-room extraction (no code copied).

## 2. Predicted gain (with the math) — and why it is zero here

The dossier's amortization story assumes prefill runs F32 GEMM over the packed weights, so one pack per load replaces one pack per call. On today's tree that caller does not exist:

- Production prefill (`Qwen2::batch_matvec`, `src/models/qwen2.cpp:914-1019`) routes Q8_0 through `gemv_q8_batch` (or VNNI/GPU), K-quants through `gemv_dequant_batch` (or the shipped kq8 integer path for tokens > 1), and F32 through a per-token `gemv_f32` loop — never through `gemm_avx2`. `gemm_f32` (dispatch `src/core/dispatch.cpp:17`) is called only by `bench gemm` (`apps/cli/main.cpp:145-222`) and the unit tests.
- The packing the dossier wants to hoist already ships as transient per-call work inside `gemm_packed` (`src/cpu/avx2/gemm_avx2.cpp:89-115` pack, `:143-201` blocking loop). A persistent buffer would therefore speed bench-only F32 GEMM at best, and exactly zero production prefill tokens.
- Even if F32 GEMM were the prefill kernel, the roof is closed on the kernel that actually runs: `bench batch` ffn reports 295 GFLOPS against a ~305 GFLOPS streaming roof at the measured wall — 97% (measurement-log §28ak, bandwidth-bound on weights). Packing more cannot move a bandwidth-bound kernel; fewer bytes (VNNI int8, opt-in) or fewer passes (speculation) are the recorded levers, not layout.

Stated falsifier (dossier): pack one Q-Proj-sized matrix; `bench gemm --m 1024 --n 1024 --k 1024` + 510/2048-token prefill + fidelity; kill if packed prefill < 5% faster best-of-3, or packed memory exceeds budget, or KL != ~0 / any top-1 flip. Fails by construction: packed prefill gain is 0% (no packed-prefill path exists to A/B), and building one is new-engine work that breaks mmap-first for a non-bottleneck.

## 3. Falsifier (from the dossier) — resolved per half

- "Packed prefill >= 5% best-of-3" — moot: prefill never calls the packed kernel (carrier grep §5). A tie-by-absence is not a 5% win.
- "Packed memory fits budget headroom" — moot with no caller, and the direction is wrong: one 64-B-aligned packed copy per weight tensor duplicates the mmap'd weights outside the file mapping, against the mmap-first rule that killed the same family in 1458 (R8 repack) and 1382/1518 (streaming/pread).
- "KL ~0 / no top-1 flip" — a reorder-only pack would pass trivially (same floats, same order of accumulation inside the unchanged microkernel), so the numerics arm cannot save a zero-gain layout change.

## 4. File anchors (current `master`, read but untouched)

- `src/cpu/avx2/gemm_avx2.cpp:89-115` (`pack_a`/`pack_b`, transient per call), `:16-29` (kRows/kCols 6/16, kMc/kKc/kNc 144/256/256), `:120-206` (`gemm_packed`), `:205-214` (`gemm_avx2` + unblocked bench control).
- `include/tge/cpu/kernels.hpp:19-56` (`GemmArgs`/`GemmConfig`/`GemmFn`, shipped + two bench controls; addressing contract: unaligned-only).
- `src/core/dispatch.cpp:17` (`set.gemm_f32 = &cpu::gemm_avx2`); callers of `gemm_f32`: `apps/cli/main.cpp:145-222` (bench) + `tests/test_tensor.cpp:46` + `test_cpu.cpp:49` only (grepped).
- `src/models/qwen2.cpp:914-1019` (`batch_matvec`: Q8_0 / K-quant / F32-per-token dispatch, zero `gemm` calls); `:1559-1767`, `:2043-2194` (forward_batch / prefill call sites of `batch_matvec`).
- `src/memory/allocator.cpp`, `src/memory/budget.cpp` (no packed-weight buffer; budget headroom logic, no caller).
- `reports/idea-1539-1540-gemm-layer-superseded.md` (the shipped microkernel/panel/edge half, jointly recorded), `docs/measurement-log.md` §28ak (97% roof), §§12/34/44 + §57/§61-63 (integer-lane context: the real prefill lever was kq8, shipped as default).

## 5. What was done (ANALYZE only — no spike per the dead-idea rule)

1. Re-read the dossier mechanism against the current tree; mapped shipped (blocking + transient packing + microkernel) vs dead (persistent pack-on-load) (§2).
2. Carrier greps: `pack_a|pack_b` — 4 matches, all transient inside `gemm_avx2.cpp`; `prepack|packedBuffer|isPrePacked` — zero; `gemm_f32` callers — bench + tests only; `batch_matvec` body — zero `gemm` calls (F32 takes the per-token `gemv_f32` loop).
3. Gates for the record on the unmodified tree: `cmake --build --preset release` clean (54/54); `ctest --preset release` pass; `tge_tests.exe` 141/141; `python aikit.py check aegean-tge` 2/4 (test + secrets pass; hygiene + guidelines fail pre-existing — C++-only repo, README `reports` drift, same as the 1635/1636 cycles).
4. No fidelity run owed (no arithmetic exists to price); no `bench` A/B owed (no second path exists to pair); no thinking-set diff owed (same binary, zero behavior change).

## 6. Measured numbers (inventory — nothing unshipped to measure)

No new benchmark belongs to a split verdict with a zero-carrier half. The applicable numbers already exist on record:

- Packing/blocking ships and is measured: `bench gemm` cubed + unpacked/unblocked controls separate packing from blocking (measurement-log §1 ablation); the microkernel/panel/edge accounting is closed in 1539/1540.
- The prefill kernel that matters is not F32 GEMM: `bench batch` ffn at 295 GFLOPS vs ~305 roof = 97% (§28ak); post-kq8 head-to-head at 71% prefill / 78% decode vs pinned llama (§62) is a quantized-batch story, unaffected by F32 packing.
- Suite/gates this cycle: build 54/54, `ctest` pass, 141/141 cases, aikit 2/4 (pre-existing fails only).

## 7. Full reimplementation recipe (deliberately NOT recommended — rebuild only to re-verify the kill)

To reproduce the negative result (do not ship): add one 64-B-aligned `AlignedBuffer` per weight tensor sized for the padded blocked layout, pack once at load with the `pack_a`/`pack_b` order, thread a packed-GEMM entry through dispatch with an unpacked fallback, and route F32 prefill through it. Gate on the dossier's own falsifier: `bench gemm` 1024-cubed + 510/2048-token prefill best-of-3 + `tge fidelity` vs scalar. Expected: bench-only delta at best, 0% production prefill (quantized models never call it), extra resident memory outside the mmap, KL ~0 (reorder-only). Revisit ONLY if F32 GEMM ever becomes the production prefill path (quantized weights abandoned — not on any roadmap) AND a packed-resident budget is approved against mmap-first; until both hold, any revival re-proves the caller first.

## 8. Verdict and why

**SPLIT: blocking/packing SUPERSEDED, persistent pack-on-load KILLED.** The dossier's implementable half is already the engine (same tile, same panels, same blocking constants, same edge idiom — see 1539/1540). Its novel half has no production caller to accelerate, attacks a kernel that is already at 97% of its streaming roof, and costs resident memory against the mmap-first invariant for a bench-only win. The loop's honest output is the map above, so no future cycle re-treads the bundle without a caller first.

## 9. Exact commands run and their results

- Baseline: `git status` showed the pre-existing peer spike (3 modified files); content peeked read-only, never edited. By cycle end the peer reverted it concurrently — `git status` clean, `git stash list` unchanged (only the 2026-11-04 shelved peer work), no command of this cycle touched it.
- `cmake --build --preset release` → 54/54 clean (peer spike present at build time; analysis is path-based, not binary-based).
- `ctest --preset release` → pass (1.47 s); `tge_tests.exe` → 141/141, 0 failed.
- `python aikit.py check aegean-tge` → 2/4 (test + secrets pass; hygiene + guidelines fail pre-existing, same pair as prior cycles).
- Carrier greps (§5): pack transient-only / prepack zero / gemm_f32 bench-only / batch_matvec gemm-free.
- No branch created, no edits made, no benchmarks run beyond the gate suite, no push.

(End of file.)
