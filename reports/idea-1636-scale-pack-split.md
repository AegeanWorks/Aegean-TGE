# IDEA-1636: Scale-Packing Micro-opt + Deterministic Block KV — SPLIT: prefix half duplicates CoW; scale half spiked, killed on split (1T +8%, 10T −4..−13%)

- **Idea ID**: IDEA-1636 (dossier `kilo-sweep-2026-11-03-b8.md`: "Scale-packing micro-opt with deterministic block KV (single cvtph, shuffle+permute, bit-identical 1.32x at d28?) + deterministic block prefix cache for stable pins")
- **Cycle date**: 2026-11-06
- **Verdict**: SPLIT. (a) Prefix-cache half: dead on arrival — duplicates the shipped cross-sequence CoW prefix (byte-identical, 3x prefill-side, FNV-gated 7/7), no spike. (b) Scale half (`cvtph` single-instruction scale decode in `dot_row_q8`): spiked, bit-identity proven exhaustively, then KILLED on a thread-count split — +8.1% single-threaded but −4% to −13% on the shipped 10-thread default (5/5 runs same direction). A change that speeds a non-default config while slowing the default is not shippable. Spike fully reverted; only this report committed.
- **Tree state at close**: `git status` clean (no diff), spike branch created and deleted with nothing committed, stash unchanged (pre-existing peer work only), `master` ahead of `origin/master` by 22 after this report. Nothing pushed (human gate).

## 1. Mechanism

(a) Deterministic block prefix cache for stable pins: reuse already-computed
KV for repeated prompt prefixes. (b) Replace the portable scalar
bit-twiddling `half_to_float` (branchy subnormal loop, ~10–15 uops) on the Q8
decode hot path with one F16C `cvtph` lane-0 conversion per 32-weight block:
`set_epi64x(0, bits)` + `cvtph` + `cvtss_lane0`, gated at runtime on
`features.f16c`, scalar path unchanged.

## 2. Predicted gain (with the math)

Per Q8 block: ~28 uops of widen/cvt/fmadd + ~12 uops of scale decode. cvtph
cuts the scale leg to ~4 uops → ~20–25% off the per-block front-end; decode
on Q8_0 is ~96% matvec, so up to ~+5–8% end-to-end IF the OoO engine was
front-end bound. Falsifier (stated): kill if <5% decode gain best-of-3 on the
Q8_0 model, or any finite-value bit difference, or any ids divergence.

## 3. Falsifier (stated; prefix half needs none — duplication, not measurement)

Scale half: <5% gain best-of-3 → kill; any finite-input bit difference vs
`half_to_float` → kill; any ids/fidelity divergence → kill. Prefix half:
killed by code duplication (anchors §4).

## 4. File anchors (current `master`, read before writing)

- Prefix: `src/models/qwen2.cpp:52-54` (`g_shared_prefix` + hit/saved
  counters), `:1826-1864` (prefix-hit reuse with accounting), `:2002-2026`
  (populate on first prefill); measurement-log §2026-11-05 ("SKIP-locked
  drafter + fused verify + cross-sequence CoW prefix", 3x prefill-side).
- Scale: `include/tge/cpu/kernels.hpp:276-307` (`half_to_float`, exact incl.
  subnormals); `src/cpu/avx2/q8_gemv_avx2.cpp:27-67` (`dot_row_q8`, the sole
  spiked function); `CMakeLists.txt:36-43,84-94` (per-TU `-mavx2 -mfma
  -mf16c`; test TUs carry no ISA flags by default).
- Tests: `tests/test_q8.cpp` (`half_conversion_matches_known_values`,
  q8 differential style).

## 5. What was done (prefix: analyze only; scale: temporary spike, fully reverted)

1. `git checkout -b spike-1636-cvtph-scale` (no commit ever made on it).
2. Prefix half mapped to shipped CoW (§4) — no code touched for it.
3. Scale spike (3 files): file-local `half_to_float_cvtph` in
   `q8_gemv_avx2.cpp` (header untouched — MSVC `/arch:AVX2` F16C uncertainty,
   no intrinsics in shared headers); `dot_row_q8` gains `use_cvtph` (resolved
   once per `gemv_q8_0_avx2` call from `features.f16c`); `CMakeLists.txt` adds
   `tests/test_q8.cpp` to the ISA-flagged list (reverted with everything);
   exhaustive `cvtph_matches_half_to_float_exhaustive` test (duplicated 3-line
   conversion, all 65536 inputs, bitwise compare).
4. Build: first failure taught that test TUs lack ISA flags
   (`target specific option mismatch`) → scoped flags to the test TU.
5. Exhaustive test first run: 1022 mismatches, first at 0x7C01 — the entire
   NaN-payload range; all 64514 finite/subnormal/infinite inputs bit-identical.
   Test refined to assert identity on non-NaN + enumerate NaN divergence
   (both sides stay NaN — unobservable to argmax/top-1/fidelity unless the
   model is NaN-poisoned under either path). Rationale recorded, not hidden.
6. `ctest` pass; `tge_tests` 142/142 (141 + 1 new). Qwen2.5-0.5B-Q8_0 A/B:
   spike best-of-3 at 1T and 10T (saved binary), stashed, rebuilt baseline,
   best-of-3 both, interleaved 10T×2 + 1T×1 confirmation pairs, fidelity on
   spike, fox + haiku ids on both binaries.
7. Reverted all 3 files, verified empty diff, deleted branch. Final gates on
   master: build clean, `ctest` pass, `aikit.py check` 2/4 (test + secrets
   pass; hygiene + guidelines fail pre-existing).

## 6. Measured numbers (Qwen2.5-0.5B-Q8_0, `--gpu off`, warm page cache)

| config | spike | baseline | delta |
|---|---|---|---|
| decode 1T best-of-3 (31 tok) | 22.99 (22.10/22.94/22.99) | 21.26 (21.01/21.26/21.11) | **+8.1%** |
| decode 1T interleave | 21.79 | 20.88 | +4.4% |
| decode 10T best-of-3 | 84.53 (82.92/84.53/80.83) | 87.84 (87.14/85.21/87.84) | **−3.7%** |
| decode 10T interleave ×2 | 74.08 / 75.75 | 85.14 / 86.70 | −13% / −12.7% |
| fidelity (86 pos, ctx 512) spike | KL 2.121e-11, 86/86 | KL 2.121e-11 (recorded) | identical to the digit |
| exhaustive cvtph | 64514/64536 identical; 1022 NaN-payload-only diffs | — | gate passes as refined |
| thinking set (fox + haiku) | ids byte-identical both binaries | — | pass |

Reading: 1T faster in 4/4 runs (+4 to +8%), 10T slower in 5/5 runs (−4 to
−13%). The split is structural, not noise: the lane-crossing extract sits on
the per-block `total` fmadd critical path; single-thread front-end savings
win, ten-thread shared-bandwidth/sync execution loses. No overlap within
either config's run sets. The shipped default is 10 threads (H3) — the spike
regresses the configuration users actually run.

## 7. Full reimplementation recipe (kill-grade — rebuild only to re-verify)

1. `q8_gemv_avx2.cpp`: file-local `half_to_float_cvtph` (set_epi64x + cvtph +
   cvtss lane 0); `dot_row_q8(..., bool use_cvtph)` selecting
   cvtph/manual per block; resolve `cpu::detect().features.f16c` once per
   `gemv_q8_0_avx2` call (needs `tge/core/cpu.hpp` include in that TU).
2. `CMakeLists.txt`: add `tests/test_q8.cpp` to `tge_set_isa_flags(AVX2 ...
   )` (test TUs otherwise lack `-mf16c`).
3. `test_q8.cpp`: exhaustive loop over all 65536 halves comparing cvtph-lane-0
   vs `half_to_float` bitwise; pass iff mismatches ⊆ NaN inputs.
4. Gate: 1T + 10T best-of-3 decode on the Q8_0 model + interleaved pairs +
   fidelity + both thinking prompts — expect the §6 split to reproduce.
   Follow-up direction (not this cycle): batch 8 scales per cvtph to amortize
   the extract, or restrict to `threads == 1` dispatch (not recommended —
   special-casing the non-default path).

## 8. Verdict and why

**SPLIT: prefix half dead by duplication; scale half killed on the thread
split.** The prefix cache is not an idea awaiting implementation — it ships,
measured, gated. The scale micro-opt is real, bit-safe (exhaustively proven
modulo NaN payloads), and faster where it doesn't matter (1T is not the
default) while slower where it does (10T default, 5/5 runs). The falsifier's
5% bar is met at 1T and missed with sign flipped at 10T — shipping either
half would be wrong (duplication) or a default-path regression. Revisit the
scale direction only with a batched-cvtph variant that clears 10T, or if the
default thread policy changes. MSVC `/arch:AVX2` F16C-intrinsic availability
is an open porting question for any revival (runtime gate already protects
execution; compile needs verification there).

## 9. Exact commands run and their results

- `git checkout -b spike-1636-cvtph-scale` → new branch.
- Spike (q8_gemv TU + CMakeLists + test_q8).
- `cmake --build` → ISA-flag error on test TU → scoped flags → clean 54/54.
- `tge_tests` → 141/142 fail (NaN payloads) → refined gate → 142/142.
- Spike decode best-of-3 (1T + 10T); saved binary; `git stash push` (3 files).
- Rebuilt baseline; best-of-3 both; interleaved 10T×2 + 1T×1; spike fidelity
  (KL 2.121e-11 = baseline digit-for-digit); fox + haiku ids both binaries
  (identical).
- `git stash pop`; `git checkout -- <3 files>` → diff empty;
  `git checkout master`; `git branch -D spike-1636-cvtph-scale`.
- Final: build clean, `ctest` pass, `aikit.py check` 2/4 (pre-existing fails).
- No push.
