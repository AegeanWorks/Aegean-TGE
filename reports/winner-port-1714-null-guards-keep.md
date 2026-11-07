# WINNER PORT: 1714 Null Guards — LANDED, keep-grade (contract pinned, zero cost)

- **Source recipe**: `reports/idea-1714-dequant-null-guards-spike-measured.md` (keep-grade hardening, killed as perf idea).
- **Port commit**: `0301bbf` (`feat(aegean-tge): fail-loud null guards on K-quant batch entries`), 4 files (+73/−0): `src/cpu/avx2/dequant_batch_avx2.cpp`, `src/cpu/scalar/dequant_batch_scalar.cpp`, `src/cpu/avx2/dequant_batch_kq8_avx2.cpp` (3-pointer fail-loud guards after the quiet early-return), `tests/test_dequant.cpp` (+58 contract test). Nothing pushed (human gate).
- **Verdict**: WINNER (hardening win) — kept. 3 pointer compares convert worker-band segfaults into catchable `invalid_argument`s; valid-input behavior is byte-identical by construction.

## 1. What landed

Each of the four K-quant batch entries (`gemv_dequant_batch` AVX2 + scalar, `gemv_dequant_batch_kq8`, `gemv_dequant_kq8_row`) throws `std::invalid_argument` naming the entry on null data/input/out — placed after the non-positive-dims quiet early-return (contract preserved) and before any buffer use (kq8: before the kind/cols fallback gates, so a null call throws under the invoked entry's name). New `dequant_batch_rejects_null_buffers` test: nulls each buffer × all four entries (must throw), plus null-buffers-with-empty-dims (must NOT throw).

## 2. Remeasure (this tree; pre-port binary saved)

- Tests: 144/144 green (new test passes; fails without guards by design).
- `bench batch` 3072×3072×6 best-of-5 float: baseline 0.539 ms vs port 0.454 ms; earlier pair 0.491 vs 0.535; kq8 pair 0.738 vs 0.481 — sign flips every pair on sub-ms shapes (box noise, as the recipe documented; analytic cost ~1 ns vs 500+ µs calls).
- Thinking set: story ids byte-identical to every prior run (`11 1052 572 ...`).
- No fidelity run owed (validation-only precedent: no arithmetic touched).

## 3. Gates

Build 54/54, `tge_tests` 144/144. `aikit.py check aegean-tge` 2/4 (test + secrets pass; hygiene + guidelines fail pre-existing).

## 4. Why it stays

Keep-grade hardening with the contract pinned by test and zero valid-input delta: the cheapest insurance in the tree. Killed as a performance idea (it never claimed speedup) — kept as code.

## 5. Queue note (no code)

The nogather-prefill spike recipe is superseded by winner #5: the strided port already covers the prefill chunk path (all sparse paths funnel through the swapped `attention_head`; the fox A/B measured prefill attn 0.38-0.53x). No separate prefill-nogather port is owed. Next: 1517 follow-ups / sparse-f16 prefill-shape check — same treatment.

(End of file.)
