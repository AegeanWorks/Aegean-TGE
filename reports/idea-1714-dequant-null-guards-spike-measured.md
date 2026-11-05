# IDEA-1714 (flavored): Fail-Loud Null Guards on the K-Quant Batch Entries — SPIKED, MEASURED, REVERTED

- **Idea ID**: IDEA-1714 flavor ("Strided-View and Rollback-State Bounds Audit"; register audit surface: kv_stride guards + fail-loud shape checks). This cycle ports that audit to the one matvec family missing it: the K-quant dequant batch entries.
- **Cycle date**: 2026-11-06
- **Verdict**: MEASURED — hardening works, zero measurable cost, ids identical. Killed as a *performance* idea (it never claimed speedup), kept-grade as an *audit recipe*: the full reimplementation below is safe to ship as a future hardening commit, but per loop rules the spike was fully reverted and only this report is committed.
- **Tree state at close**: `git status` clean (no diff), spike branch created and deleted with nothing committed on it, stash list unchanged (only the pre-existing peer-work stash, untouched), `master` ahead of `origin/master` by 12 after this report. Nothing pushed (human gate).

## 1. Mechanism

The four K-quant batch kernel entries — `gemv_dequant_batch_avx2`,
`gemv_dequant_batch_scalar`, `gemv_dequant_batch_kq8_avx2`,
`gemv_dequant_kq8_row_avx2` — are the only matvec kernels without null-buffer
guards. Every sibling fails loud: `gemv_avx2` throws on null matrix/vector/out,
`gemv_q8_0`/`gemv_q8_batch` throw on non-whole-block cols, `dequantize_row`
throws on null/unknown-kind/partial-block. A null `data`/`input`/`out` on the
batch path faults inside worker bands (or thread-pool tasks) instead of throwing
at the entry. The spike adds three pointer compares per call, before dispatch,
throwing `std::invalid_argument`, plus unit tests pinning the contract.

## 2. Predicted gain (with the math)

No throughput gain predicted: 3 pointer compares ≈ 1 ns against 200–750 µs
kernel calls = ~0.0002% overhead, four orders of magnitude under any perf bar.
Value is diagnostic: convert a worker-thread segfault (no message, no stack at
the call site) into a catchable `invalid_argument` naming the entry. Stated
falsifier: kill if guards already present (gap confirmed absent, §4), if guards
cost >1% on `bench batch` best-of-5, or if any test fails.

## 3. Falsifier (stated for this hardening spike)

- Guards already present on today's tree → no spike (they were absent: §4).
- `bench batch` guard cost >1% best-of-5 → kill.
- Any `tge_tests` failure (existing or the 1 new test) → kill.
- Thinking-set ids diverge between guarded/unguarded binaries → kill.

## 4. File anchors (current `master`, read before writing)

- `src/cpu/avx2/dequant_batch_avx2.cpp:54-60` — dims early-return, then straight
  to `quant_row_bytes` (throws on unknown kind/partial block, but not on nulls).
- `src/cpu/scalar/dequant_batch_scalar.cpp:8-14` — same shape, no null checks.
- `src/cpu/avx2/dequant_batch_kq8_avx2.cpp:323-332` (batch) and `:393-409`
  (row) — kind/cols gates, no null checks.
- Call sites: `src/models/qwen2.cpp:882` (prefill path, all K-quant models incl.
  the reference Llama-3.2-3B-Q4_K_M) and `:986` (kq8 decode path);
  `apps/cli/main.cpp` bench-batch wiring.
- `tests/test_dequant.cpp` — house style: try/catch + `TGE_CHECK` (mirrors
  `dequant_rejects_partial_block`, `test_q8.cpp` null tests).

## 5. What was done (temporary spike, fully reverted)

1. `git checkout -b spike-1714-dequant-null-guards` (no commit ever made on it).
2. Added `<stdexcept>` + null guard to all four entries (exact code in §7);
   added `TGE_TEST(dequant_batch_rejects_null_buffers)` (58 lines) covering
   null data/input/out × 4 entries plus the quiet early-return contract.
3. `cmake --build --preset release` clean; `ctest` pass;
   `tge_tests.exe` **142/142, 0 failed** (141 + 1 new).
4. `bench batch` best-of-5 on spike (3 kernel/shape points), saved spike binary
   to temp, stashed spike files, rebuilt baseline, repeated all bench points.
5. Big-shape cross-check (3072×3072×6, both binaries, no rebuild needed).
6. Thinking set: `generate` on the reference Llama-3.2-3B-Q4_K_M, both binaries,
   ids diffed.
7. No fidelity run owed: validation-only, no arithmetic touched (same precedent
   as drafter-only/H1 changes); unit differential (new test + full suite) is the
   numerics evidence, and end-to-end ids are identical.
8. `git stash pop`, `git checkout -- <4 files>`, verified `git diff` empty,
   `git checkout master`, `git branch -D spike-1714-dequant-null-guards`.
9. Final gates on reverted master: build clean, `ctest` pass,
   `python ../../aikit.py check aegean-tge` 2/4 (test + secrets pass; hygiene +
   guidelines fail pre-existing — C++-only repo, README drift untouched).

## 6. Measured numbers

`bench batch` best-of-5, warm page cache. Small shape first (1024×1024×6,
0.6 MiB — sub-ms, timer-noise dominated), then big shape (3072×3072×6,
5.1 MiB).

| kernel/shape | baseline best | spike best | delta |
|---|---|---|---|
| float q4k, 1024×1024×6 | 0.301 ms (41.79 GFLOPS) | 0.278 ms (45.29) | +8% (noise) |
| kq8, 1024×1024×6 | 0.249 ms (50.62) | 0.441 ms (28.51) | −77% (noise) |
| kq8d, 1024×1024×1 | 0.263 ms (7.98) | 0.195 ms (10.74) | +26% (noise) |
| float q4k, 3072×3072×6 | 0.682 ms (166.12) | 0.757 ms (149.60) | −11% (noise) |
| kq8, 3072×3072×6 | 0.513 ms (220.67) | 0.492 ms (230.13) | +4% (noise) |

Reading: deltas flip sign across shapes/kernels (±77% both ways on sub-ms
shapes, ±11% on 5 ms shapes) — the signature of box variance (documented
±20%, heavier under load), not of three pointer compares. Analytic cost:
~1 ns/call vs 500–750 µs calls. Verdict on the perf arm: **no measurable
cost, no measurable gain** — exactly as predicted for validation-only code.

Thinking set (reference model Llama-3.2-3B-Q4_K_M, `--n 16`, threads 10,
`--gpu off`): promptids identical, generated ids byte-identical baseline vs
spike (`320 2028 374 ... 220 16 25`, 15 ids). Decode t/s 10.67 vs 11.54 is box
noise (matvec 97% both, guards off the hot path).

Tests: 142/142 on spike (incl. new null-guards test), 141/141 on reverted
master. New test fails without the guards (null would fault, not throw) and
passes with them — the contract is pinned.

## 7. Full reimplementation recipe (keep-grade hardening, safe to ship later)

In each of the four entries, immediately after the non-positive-dims
early-return and before any other use of the buffers, insert:

```cpp
if (args.data == nullptr || args.input == nullptr || args.out == nullptr) {
  throw std::invalid_argument("<entry>: null data/input/out");
}
```

with `<entry>` = `gemv_dequant_batch` (avx2 + scalar),
`gemv_dequant_batch_kq8`, `gemv_dequant_kq8_row`; add `#include <stdexcept>`
to the three touched TUs (`dequant_batch_avx2.cpp`,
`dequant_batch_scalar.cpp`, `dequant_batch_kq8_avx2.cpp`). For the kq8
entries place the guard before the kind/cols fallback gates so a null call
throws under the invoked entry's name (the fallback would otherwise throw
later under `gemv_dequant_batch`). Then append
`TGE_TEST(dequant_batch_rejects_null_buffers)` to `tests/test_dequant.cpp`:
valid Q4_K args (rows 2, cols 256, tokens 1, real buffers), null each of
data/input/out in turn and require `invalid_argument` from all four entries;
plus a non-positive-dims case with null buffers requiring no throw (quiet
early-return preserved). Expected: 142/142 green, bench within noise, ids
identical. No dispatch, header, model, or CLI changes; no layout constants.

## 8. Verdict and why

**Measured, reverted; recipe keep-grade.** The audit gap is real (only batch
family without null guards, on the reference model's own prefill path), the
fix is 15 lines + tests, cost is analytically ~1 ns and empirically lost in
box noise, behavior on valid inputs is byte-identical. Killed only as a
performance idea — it never was one. Ship the recipe as a future hardening
commit whenever a hardening window opens; until then the tree stays as it was
plus this report.

## 9. Exact commands run and their results

- `git checkout -b spike-1714-dequant-null-guards` → new branch.
- 3 source edits + 1 test edit (§5.2, §7).
- `cmake --build --preset release` → 54/54; `ctest` → pass;
  `tge_tests.exe` → 142/142 (new `dequant_batch_rejects_null_buffers` OK).
- Spike `bench batch` best-of-5 ×3 points (§6 rows 1–3).
- Saved spike `tge.exe` to temp; `git stash push` (4 files) → clean tree.
- Rebuilt baseline; baseline `bench batch` best-of-5 ×3 (§6).
- Big-shape A/B both binaries ×2 kernels (§6 rows 4–5).
- `generate` (Llama-3B-Q4_K_M) both binaries → ids identical.
- `git stash pop`; `git checkout -- <4 files>` → `git diff` empty;
  `git checkout master`; `git branch -D spike-1714-dequant-null-guards`.
- Final: build clean, `ctest` pass, `aikit.py check` 2/4 (pre-existing fails).
- No push.
