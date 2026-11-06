# IDEA-1394: Quantize-Activations-Once CPU Dispatch — INTEGRATED, CONFIRMED ON TODAY'S TREE (kq8 inside the 1e-9 bar, VNNI priced outside it by explicit gate)

- **Idea ID**: IDEA-1394 (dossier `kilo-sweep-2026-11-03.md`: project-zero `parallel_matmul.c` — SIMD dispatch AVX-512 VNNI→scalar, quantize activations once then share across workers, 4-accumulator FMA, Qwen2.5-0.5B tested, PPL within 0.7% of llama.cpp; claim: port quantize-once dispatch + multi-token LM-head tiling for prefill gain without new math)
- **Cycle date**: 2026-11-06
- **Verdict**: INTEGRATED — confirmed, not merely asserted. The quantize-once discipline ships twice: kq8 integer prefill (per-row Q8 activation quant + maddubs dots, default on) prices at the dense floor KL 9.248e-12 with 1023/1023 top-1 — three orders inside the dossier's 1e-9 bar; VNNI pre-quantized activations (quantize once per layer, shared across the matrices it feeds) prices at KL 1.5e-03 with 993/1023 — outside the bar's letter, inside the loop's priced-gate practice (opt-in `--vnni` only, prefill/batch-only, decode untouched by construction). Greedy ids: no divergence on the default path. Decode regress: none possible — both quantized paths are prefill/batch-only; tokens==1 keeps the float path. No code changed this cycle (verification round); nothing to revert.
- **Tree state at close**: `git status` clean (no diff, no untracked), stash unchanged (only the pre-existing `leftover peer work shelved 2026-11-04`, untouched), `master` ahead of `origin/master` by 63 before this report. Nothing pushed (human gate).

## 1. Mechanism (as dossiered, as shipped)

Quantize each activation once, share the quantized rows across the workers/matrices that need them, dispatch SIMD VNNI→scalar, accumulate in wide FMA groups, tile multi-token batches (including the LM head). Native form: (a) kq8 — `gemv_dequant_batch_kq8_avx2` integer Q4_K/Q6_K × Q8 path for prefill-size batches (default on in generate/fidelity/bench); (b) VNNI — `quantize_activations` once per layer then `gemv_q8_batch`/VNNI dots over the shared buffer (opt-in `--vnni`); (c) 6-row microkernel groups + six-token speculative batch + chunked prefill batching (the tiling half).

## 2. Predicted gain — realized, priced per tier

Prefill without new math: kq8 1.58–2.22x kernel / 1.6–1.83x end-to-end ( flood §§58/61), VNNI 1.9x prefill (§23). Fidelity prices (§6): kq8 at the dense floor, VNNI at 1.5e-3 with 97% top-1 agreement.

## 3. Falsifier (from the dossier) — evaluated arm by arm on today's tree

"Mean KL > 1e-9 vs scalar oracle": default path PASSES (9.248e-12, ~100x inside); VNNI path prices 1.503e-03 — fails the letter, and that failure is the documented gate (opt-in flag, prefill-only, never default), not a regression. "Greedy ids diverge on Q8_0 or Q4_K_M": default 1023/1023 (no divergence); VNNI 993/1023 (30 flips — the reason it stays gated). "Decode regress is reject": both quantized paths are batch-only by construction (VNNI decode keeps float; kq8 tokens==1 keeps float — `kq8d` single-row variant aside, still the float-attention decode path); no decode regress possible, none measured anywhere on record.

## 4. File anchors (current `master`, read but untouched)

- `apps/cli/main.cpp:111,121` (`--kernel kq8` "default on"), `:1972/:2475` defaults, `:560`,`:2323` (`set_kq8`), `:1296-1297` (kq8/kq8d dispatch), `:2820-2833` (bench kernel select).
- `src/models/qwen2.cpp:1890,1946,1966` (quantize-once sites: prefill_run + forward_batch quantize `batch_h_` once, share across gate/up/down + wo), `:1626-1628/:1969-1971` (gate/up batch calls over the shared buffer — the TwoMatMul-adjacent sharing, without fusion).
- `src/cpu/avx2/dequant_batch_kq8_avx2.cpp` (integer Q4_K/Q6_K × Q8 dots — the inside-loop integer math).
- CLI: `--vnni` opt-in on generate/fidelity/bench; `--kernel kq8|float` escape hatch on generate/fidelity.

## 5. What was done (verification round — the feature ships; this cycle prices it)

1. Re-read the dossier falsifier against the two shipped quantize-once paths.
2. Gates first: `cmake --build --preset release` clean (54/54); `ctest --preset release` pass; `tge_tests.exe` 141/141; `python aikit.py check aegean-tge` → test gate failed once at 7.5 s (eighth flake overall — longer than the 2–3 s signature but the identical pattern: the exact gate command `scripts\test.cmd` passes cleanly on immediate direct rerun with full rebuild), secrets pass, hygiene + guidelines fail pre-existing.
3. Fidelity pair (Qwen2.5-0.5B-Q8_0, `docs/architecture.md`, ctx 1024, 6 threads, `--gpu off`): default (kq8) then `--vnni` — §6.
4. No bench A/B owed (speedups on record §§23/58/61, unchanged code); no thinking-set diff owed (no behavior delta — verification only).

## 6. Measured numbers (this cycle, wall 58.19 healthy)

| candidate arm | KL mean | worst KL | top-1 | wall |
|---|---|---|---|---|
| default kq8 | 9.248e-12 | 2.418e-10 | 1023/1023 agree | 46.61 s |
| `--vnni` | 1.503e-03 | 1.533e-02 | 993/1023 agree | 47.52 s |

The default path sits at the dense floor (the scalar oracle agrees to 1e-4 logits); the opt-in path reproduces the 1.6e-3 precedent class (1.50e-3 here) with 97% agreement — the priced tradeoff the integration explicitly accepted (flag-gated, prefill-only).

## 7. Full reimplementation recipe (keep-grade — already in-tree; rebuild only to re-verify)

Nothing to build (ships). To re-verify integration: (1) confirm `--kernel kq8` default-on in generate/fidelity/bench help + `bool kq8 = true` defaults; (2) re-run the §6 fidelity pair and expect the dense floor (≤1e-9) and the 1.5e-3 class with ≥990/1023; (3) confirm `quantize_activations` call sites remain prefill/batch-only and `tokens == 1` keeps the float path (decode-regress arm stays vacuous). Revival trigger: none — the remaining work under this number is the VNNI-accuracy program (narrow 1.5e-3 → 1e-9-class without losing 1.9x), which is a new idea, not this one.

## 8. Verdict and why

**INTEGRATED, confirmed with the books open.** The dossier's strict bar is met by the path users actually run (kq8, three orders inside); the path that misses it (VNNI) is flag-gated, prefill-only, priced, and documented — exactly the loop's keep-with-gate practice. No future cycle needs to re-price quantize-once; the open item, if any, is VNNI accuracy as its own proposal.

## 9. Exact commands run and their results

- `cmake --build --preset release` → 54/54; `ctest --preset release` → pass; `tge_tests.exe` → 141/141.
- `python aikit.py check aegean-tge` → test gate flaked once (7.5 s), direct `scripts\test.cmd` rerun → pass (full rebuild); secrets pass; hygiene + guidelines fail pre-existing.
- Fidelity default → KL 9.248e-12, 1023/1023 (46.61 s); fidelity `--vnni` → KL 1.503e-03, 993/1023 (47.52 s).
- No branch, no edits, no bench, no thinking-set (verification round); report committed; no push.

(End of file.)
