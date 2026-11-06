# IDEA-1586: Portable IO Shims with Trunk-Stream Prefetch — SPLIT (aligned-alloc shim + bitwise identity ship; O_DIRECT/trunk/expert/MXFP4 halves dead)

- **Idea ID**: IDEA-1586 (dossier `kilo-sweep-2026-11-03-b6.md`: kimi-k3-in-c `k3_portable_io.h` (O_DIRECT intercept at CreateFileA + FILE_FLAG_NO_BUFFERING, pread via ReadFile + OVERLAPPED, posix_memalign → _aligned_malloc) + `k3_trunk.h` (pin-largest-first + uniform ring, trunk prefetch overlap, 71.75 → 42.27 s/tok 1.70x measured, int4-on-trunk rejected 17.4% vs 0.96% weight-error); sibling of IDEA-1519, disjoint files by dossier construction)
- **Cycle date**: 2026-11-06
- **Verdict**: SPLIT. (a) Portable aligned-alloc shim: SUPERSEDED — ships as `AlignedBuffer` (`_aligned_malloc` on Windows, `posix_memalign` elsewhere). (b) Scalar-to-AVX2 bitwise identity discipline: SUPERSEDED — ships as the differential-memcmp + unaligned-only addressing contract (kq8, Q4_K/Q6_K/Q5_0/Q5_1, attention, norms all gated). (c) O_DIRECT intercept / pread-overlapped bypass: KILLED — zero carrier, and directionally wrong (bypassing the page cache decode re-reads every token; TGE wants residency, §§26c/62). (d) Trunk pack + pin-largest-first ring + overlap: KILLED — zero trunk/pin/ring carriers; trunk-pack layout forks the GGUF mmap (banned family: 1458/1325/1553); the source's own int4-on-trunk is rejected by its own numbers. (e) Frequency-aware expert cache + fused MXFP4 matmuls: KILLED — dense-only decoder, zero MoE/expert/MXFP4 matches (compounds IDEA-1519 + 1382/1518). No spike built (dead-idea rule: the KL arm is moot with no new numerics, the overlap arm needs a layout that does not exist). No branch, no code touched, nothing to revert.
- **Tree state at close**: `git status` clean (no diff, no untracked), stash unchanged (only the pre-existing `leftover peer work shelved 2026-11-04`, untouched), `master` ahead of `origin/master` by 53 before this report. Nothing pushed (human gate).

## 1. Mechanism (as dossiered, per half)

Windows-portable IO shims (aligned allocation across CRT/POSIX, O_DIRECT-style cache-bypass reads, overlapped pread) plus fixed-order trunk streaming (pin-largest-first resident set, uniform ring eviction, prefetch overlapped with compute) plus a frequency-aware expert cache plus fused MXFP4/AVX2 matmuls — all held to scalar-to-AVX2 bitwise identity, for byte-identical output at every RAM budget. Source claim: 1.70x prefetch-overlap win; int4-on-trunk rejected by the source itself.

## 2. Predicted gain — and the per-half disposition

| half | disposition | grounds |
|---|---|---|
| aligned-alloc shim | SUPERSEDED | `AlignedBuffer` ships both branches + multi-GB guard (§5) |
| bitwise-identity discipline | SUPERSEDED | memcmp differentials + unaligned-only contract ship everywhere hot |
| O_DIRECT / pread-overlapped | KILLED | zero carrier; anti-optimal for a re-read-every-token decoder |
| trunk pack + pin/ring + overlap | KILLED | zero carrier; forks mmap; overlap presumes the absent layout |
| expert cache + MXFP4 | KILLED | dense-only; zero MoE/MXFP4 matches (1519 family) |

The dossier's 1.70x is cold-start-shaped (71.75 → 42.27 "s/tok" units are first-touch-dominated, the same shape §26c diagnosed as a 43 s fault-in and fixed with an explicit 489 ms prefetch step). TGE's fault-in story is closed and measured; there is no second cold-start win behind it.

## 3. Falsifier (from the dossier) — resolved without building

"Any id mismatch or KL above zero vs tge scalar oracle on N-token decode" — moot: no new numerics exist to price (all five halves are absent-or-shipped, none modified); shipped paths already price at floors (kq8 1.4e-11, norms/RoPE/SiLU full-program 1e-11, f16-KV 1.7e-6). "Prefetch overlap below 1.3x on tge NVMe host" — unconstructible: overlap is a property of a streaming layout, and no trunk layout exists to overlap; File::prefetch (whole-file, up-front, reported) plus OS readahead is the shipped equivalent for an mmap-first engine whose weights must be resident anyway.

## 4. File anchors (current `master`, read but untouched)

- `src/memory/allocator.cpp:15-53` (`normalize_alignment` + `AlignedBuffer`: `_aligned_malloc` Windows / `posix_memalign` elsewhere + multi-GB cap — the portable shim half, verbatim).
- `include/tge/cpu/kernels.hpp:11-16` (addressing contract: unaligned-only, no vector-pointer-over-buffer — the identity discipline's load-bearing rule, §28i) + per-family memcmp differentials (`q4k/q6k/q50/q51`, kq8 generic triangulation, attention, norm/rope).
- `src/models/gguf.cpp:434` (CreateFileMappingW — the Windows mapping the POSIX source's "conditional pass" port requires), `:597` (PrefetchVirtualMemory — shipped fault-up-front).
- Greps: `MXFP4|mxfp4|expert|MoE|k3_|trunk|O_DIRECT|pread|OVERLAPPED` → zero code matches in `src/` + `include/` (only "spread" homonyms: attention fan-out, Q5_0 bit-spread LUT).
- Precedents: `idea-1519-linear-recurrence-killed-analytic.md` (zero MXFP4/MoE), `idea-1553-paged-prefetch-split.md` (advice-for-RSS arithmetic kill — this cycle compounds it: bypass ≠ residency), `idea-1382-moe-streaming-killed-analytic.md`.

## 5. What was done (ANALYZE only — no spike per the dead-idea rule)

1. Read the dossier body against the current tree; split the bundle per half (§2), keeping the dossier's sibling-disjointness from 1519 intact (different files, different mechanism — judged independently, killed independently).
2. Carrier greps (§4): shim ships, identity ships, everything else zero.
3. Gates for the record on the unmodified tree: `cmake --build --preset release` clean (54/54); `ctest --preset release` pass; `tge_tests.exe` 141/141; `python aikit.py check aegean-tge` 2/4 (test + secrets pass; hygiene + guidelines fail pre-existing — no flake this cycle).
4. No fidelity run owed (no new arithmetic); no bench A/B owed (no second path/layout); no thinking-set diff owed (same binary).

## 6. Measured numbers — none owed (inventory + one cold-start note)

No benchmark belongs to a split whose dead halves have no carrier and whose live halves already ship with numbers (kq8 1.58–2.22x kernel / 1.6–1.83x prefill at KL floor; prefetch 489 ms storage-bound, same total bytes — §§26c/58/61). Orientation only: the source's 71.75 → 42.27 "s/tok" pair is two orders above any steady-state TGE rate on any model (short decode 10–75 t/s, prefill 100–280 t/s across windows) — cold fault-in plus tiny-model fixed costs dominate it, the diagnosed shape, not a transferable overlap ratio.

## 7. Full reimplementation recipe (deliberately NOT recommended — per half)

Shipped halves need no recipe (in-tree: `AlignedBuffer`, differential suite, `File::prefetch`). Dead halves must not be built: O_DIRECT bypass (evicts what decode re-reads), trunk repack (breaks mmap-first for a layout-bound non-bottleneck — same R8 analysis as 1458/1325), expert cache (no experts), MXFP4 (no format, no ISA path on this box — AVX2+FMA+F16C+VNNI only). Revive any single half only with its carrier first (a trunk layout behind a bench flag with overlap ≥1.3x paired + fidelity; an MoE model file that loads; an MXFP4 tensor the loader accepts) — engine-first, never. To re-verify this split: re-run the §4 greps and confirm still-zero, plus `tge info` ISA line unchanged.

## 8. Verdict and why

**SPLIT: shims + identity SUPERSEDED (both ship verbatim), bypass + trunk + expert + MXFP4 KILLED (zero carriers, one anti-optimal direction, one self-rejected source-number, one absent model class).** Nothing in the bundle is both unshipped and buildable; the loop's honest output is the map, so no future cycle re-treads it without a carrier first.

## 9. Exact commands run and their results

- Carrier greps (§4): MXFP4/MoE/expert/trunk/k3_/O_DIRECT/pread/OVERLAPPED zero (code); shim + identity ship.
- `cmake --build --preset release` → 54/54; `ctest --preset release` → pass; `tge_tests.exe` → 141/141.
- `python aikit.py check aegean-tge` → 2/4 (pre-existing hygiene + guidelines fails). No branch, no edits, no bench, no fidelity, no push.

(End of file.)
