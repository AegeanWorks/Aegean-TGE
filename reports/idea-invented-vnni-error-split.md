# INVENTED: VNNI Error Split — PARTIALLY RETRACTED (qonly-floor was silent non-engagement; confound kills stand; see qonly re-run)

> **RETRACTION 2026-11-06 (same day, next cycle)**: the arm-(3) verdict below — "activation quantization exonerated (qonly at floor 9.248e-12)" — is WITHDRAWN. The qonly fidelity run never engaged its path (all 15 `quantize_activations` call sites gate on `vnni_`; the spike set only its own flag, so every matmul silently took the float path — digit-identical floor explained exactly). The engaged re-run (`idea-invented-vnni-qonly-rerun-measured.md`, with stderr engagement proof + `, qonly` label) prices qonly at KL 1.515e-03, 992/1023: **activation quantization CONFIRMED as the error; int dots add nothing measurable.** Arms (1) corpus confound killed and (2) engagement confinement STAND (independent measurements, unaffected). The replay round's kernel-arithmetic exoneration (1.7e-5 int-vs-int) also stands. Do not cite the exoneration below; cite the re-run. (Minor erratum fixed at the same time: the roundtrip unit test lived in `tests/test_q8.cpp`, not `test_attention.cpp` as §7.6 originally said.)

- **Idea**: Invented (no dossier; follow-up the 1394 verification round explicitly deferred: "VNNI accuracy program... is a new idea, not this one"). Decompose VNNI's priced 1.5e-3 error into activation-quant vs int-dot halves with a temporary `--vnni-qonly` flag (dequantize the shared Q8 buffer back to float, run the float Q8 dots), plus kill the corpus confound (was kq8's floor short-corpus-only?) and the engagement confound (where does VNNI actually run?).
- **Cycle date**: 2026-11-06
- **Verdict**: MEASURED, SPLIT NAMED. (1) Corpus confound killed: kq8-on-real-K-quants full-corpus fidelity = KL 1.006e-11, 1023/1023 (Llama-3B-Q4_K_M, arch corpus, ctx 1024) — true int8-activation floor, not a short-corpus artifact. (2) Engagement confined: `--vnni` on Llama-3B = digit-identical 1.006e-11 (VNNI engages zero Q8_0 weights there; the file is Q4_K/Q6_K/F32 only) — the error lives entirely in the Q8_0-weight VNNI dots. (3) Quant exonerated: `--vnni-qonly` on 0.5B-Q8_0 = KL 9.248e-12, 1023/1023 — digit-identical to the dense floor; activation rounding contributes nothing measurable. Therefore the full 1.5e-3 (30 flips) is in the int-dot path (per-block float accumulation of exact partials, bias-corrected dpbusd, fp16-scale application, horizontal sums). Next spike named in §8 (dot-hygiene audit: double/compensated accumulation first). Spike fully reverted (6 files + test); nothing pushed (human gate).
- **Tree state at close**: `git status` clean (no diff, no untracked), stash unchanged (only the pre-existing `leftover peer work shelved 2026-11-04`, untouched). Spike branch created (zero commits) and deleted; all spike files reverted with empty diff verified. `master` ahead of `origin/master` by 64 before this report (plus the 1394-correction commit this cycle). Nothing pushed (human gate).

## 1. Mechanism (as spiked)

`--vnni-qonly` (generate + fidelity): reuse the exact `quantize_q8_batch` buffer, `dequantize_q8_batch` it (new 20-line exact inverse) to a thread-local float buffer, run the production float `gemv_q8_batch`/`gemv_q8` dots. Everything else identical (same buffers, same kernels, same order) — the ONLY delta vs `--vnni` is int-vs-float dots on identical inputs; the only delta vs default is activation rounding. Falsifier (stated): qonly ≈ 1.5e-3 class → activation quant convicted (next: finer scales); qonly ≈ floor → int dots convicted (next: dot-hygiene audit); qonly diverging from both → spike broken, kill it (fail-safe); Llama-3B kq8 full-corpus >> floor → the floor was a short-corpus artifact, re-price everything.

## 2. Predicted gain — diagnostic, not product

No t/s claimed (qonly adds a dequant pass; it exists to split blame). Value: converts the VNNI-accuracy program from a two-suspect search into a one-suspect audit, with the corpus and engagement confounds already dead.

## 3. Falsifier outcome — quant exonerated, dots convicted, confounds dead

(1) kq8 full-corpus floor HOLDS (1.006e-11, §6.1) — not a short-corpus artifact; int8 activations per se are clean. (2) VNNI-on-Q4_K_M == default digit-for-digit (§6.2) — engagement confined to Q8_0 weights; wasted quantize calls elsewhere are harmless. (3) qonly == floor digit-for-digit (§6.3) — activation rounding contributes nothing; the int-dot path owns all 1.5e-3 + 30 flips. (4) Fail-safe not triggered (qonly matched floor exactly, the predicted clean outcome).

## 4. File anchors (master = pre-spike; spike reverted)

- `src/cpu/avx2/q8_batch_vnni.cpp:24-66` (`quantize_block` symmetric per-32 + `quantize_q8_batch`; the spike's `dequantize_q8_batch` lived beside it, reverted), `:134-209` (VNNI dots: exact `lane_sums` bias correction, per-block float fmadd accumulation, `horizontal_sum` — the convicted region).
- `src/models/qwen2.cpp:1008-1033` (`quantize_activations` per-layer sharing), `:1683-1742` (single-token forward quantizes too — the 1394 correction), `:938-1006` (`batch_matvec` Q8_0 dispatch: VNNI int dots vs float dequant dots — the fork the spike straddled; kq8 branch untouched).
- `include/tge/cpu/kernels.hpp:109-142` (Q8BatchArgs/quantized-buffer contract), `:273-274` (`kQ8BlockElements = 32` — the granularity the exoneration covers).
- CLI: `--vnni`, `--kernel kq8|float` (generate/fidelity/bench); the spike's `--vnni-qonly` (generate + fidelity only, reverted).
- Grounds: 1394 report (1.5e-3 price + 993/1023), §58 (31-pos kq8 floor — extended to full corpus here), §6 numbers below.

## 5. What was done (temporary spike, fully reverted)

1. `git checkout -b spike-vnni-qonly` (zero commits ever made on it; deleted at close).
2. Spike (6 files): `dequantize_q8_batch` decl+impl; `Qwen2::vnni_qonly_` + setter; `quantize_activations` gate widened (`vnni_ || vnni_qonly_`); qonly branches in `matvec` + `batch_matvec` Q8_0 paths (thread-local float scratch, production float kernels); `--vnni-qonly` on generate + fidelity (trailing defaulted params, no signature breaks); `vnni_qonly_roundtrip_bounds_error` unit test (per-value error ≤ half block scale, deterministic). One stray dead line (`kv_auto` local, copied from the prior cycle's recipe draft) caught by grep-diff and removed before build — tree-diff discipline held.
3. `cmake --build --preset release` clean (54/54); `ctest --preset release` pass; `tge_tests.exe` 142/142 (141 + 1 new).
4. Box health gate: wall 60.33 + aikit 2/4 with test PASS 17.8 s (no flake this cycle).
5. Fidelity ×4 (arch corpus, ctx 1024, 6 threads, gpu off): Llama-3B default; Llama-3B `--vnni`; 0.5B `--vnni-qonly` (+ the 0.5B `--vnni` = 1.503e-03 priced in the 1394 cycle for the comparison).
6. Thinking set: default spike paths (qonly off → identical binary behavior) fluent — reasoning + haiku recorded; one `--vnni-qonly` story: ids byte-identical to the f32 story (consistent with floor KL).
7. Reverted all 6 files (`git checkout --`, empty diff, branch deleted); final gates on reverted master: build clean, `ctest` pass, `tge_tests` 141/141.

## 6. Measured numbers

| run | KL mean | worst KL | top-1 | wall | reads as |
|---|---|---|---|---|---|
| Llama-3B default (kq8 on Q4_K/Q6_K) | 1.006e-11 | 3.113e-10 | 1023/1023 | 290.06 s | TRUE kq8 full-corpus floor |
| Llama-3B `--vnni` | 1.006e-11 | 3.113e-10 | 1023/1023 | 313.89 s | digit-identical: VNNI engages nowhere |
| 0.5B `--vnni` (1394 cycle) | 1.503e-03 | 1.533e-02 | 993/1023 | 47.52 s | the priced error |
| 0.5B `--vnni-qonly` (spike) | 9.248e-12 | 2.418e-10 | 1023/1023 | 33.30 s | digit-identical to dense floor |

Reading, in falsifier order. kq8's floor is real on real K-quants at full length (extends §58's 31-pos price; max|diff| 8.3e-05 both Llama runs — same computation twice). VNNI's error is confined to Q8_0 weights (no Q8_0 tensors in the file → zero delta, wasted quantize and all). Activation rounding is exonerated to the digit (qonly == floor: same KL, same worst, same max|diff| class 1.9e-04, same 1023/1023) — the 30 flips and 0.77 max|diff| of full VNNI come entirely from the int-dot path. qonly story ids == f32 story ids (consistent).

## 7. Full reimplementation recipe (diagnostic spike — rebuild to re-verify; product recipe in §8)

1. `kernels.hpp`: decl `dequantize_q8_batch(values, scales, tokens, cols, out)` beside `quantize_q8_batch`.
2. `q8_batch_vnni.cpp`: exact inverse (out = values[i] * scales[i/32]), same guards as quantize.
3. `qwen2.hpp`: `bool vnni_qonly_ = false;` + `set_vnni_qonly`.
4. `qwen2.cpp`: widen the `quantize_activations` early-out to `(!vnni_ && !vnni_qonly_)`; in both Q8_0 branches (`matvec`, `batch_matvec`) insert before the VNNI branch: if `vnni_qonly_` and buffer present → thread-local float dequant → production float `gemv_q8`/`gemv_q8_batch`.
5. `main.cpp`: `--vnni-qonly` no-value flag on generate + fidelity; trailing `bool vnni_qonly = false` params on `command_generate`/`command_fidelity`; `set_vnni_qonly` beside `set_vnni`.
6. `tests/test_q8.cpp`: `vnni_qonly_roundtrip_bounds_error` (error ≤ half block scale, nonzero).
7. Gate: build 54/54, 142/142 tests; fidelity qonly == floor digit-for-digit (§6 row 4 vs dense floor); remove entirely afterward.

## 8. Verdict and why — plus the named next spike

**MEASURED: quant exonerated, dots convicted, confounds dead.** The VNNI-accuracy program is now a dot-hygiene audit with four eliminated suspects (activation quant, weight format, corpus length, K-quant path) and three live ones inside `gemv_q8_batch_vnni`: (a) per-block float fmadd accumulation over 96 blocks (hypothesis: row-wide int32-exact accumulation like the per-block exactness extended, or double/compensated combining); (b) the bias-correction path (exact in theory — verify empirically with a correction-off variant, which is only valid on zero-mean blocks); (c) fp16 weight-scale application order vs the oracle's dequant order. Named next spike: variant (a) first (smallest diff: widen the accumulator, keep everything else), falsifier — kill if KL stays above 1e-4 (no meaningful move toward the 1e-9 bar) or prefill t/s regresses >5% vs VNNI. Revival trigger: none needed — the audit is specified; the only open question it must answer first is which of (a)–(c) moves KL, priced the same way (0.5B-Q8_0, arch corpus, ctx 1024).

## 9. Exact commands run and their results

- `git checkout -b spike-vnni-qonly` → branch (zero commits, deleted at close).
- 6-file spike (§5.2 + stray-line removal) + 1 test.
- `cmake --build --preset release` → 54/54; `ctest --preset release` → pass; `tge_tests.exe` → 142/142 on spike, 141/141 after revert.
- `tge gguf` on Q4_K_M → Q4_K/Q6_K/F32 only, zero Q8_0 (engagement proof).
- `python aikit.py check aegean-tge` → test PASS (17.8 s, no flake); secrets pass; hygiene + guidelines fail pre-existing. Wall 60.33 healthy.
- Fidelity Llama-3B default → 1.006e-11, 1023/1023 (290.06 s); `--vnni` → identical digits (313.89 s).
- Fidelity 0.5B `--vnni-qonly` → 9.248e-12, 1023/1023 (33.30 s); qonly story ids == f32 story; reasoning + haiku fluent.
- `git checkout -- <6 files>` → clean; `git checkout master`; `git branch -D`; final master gates green; 1394-correction commit + this report; no push.

(End of file.)
