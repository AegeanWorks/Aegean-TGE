# INVENTED: No-Gather Strided Attention at Long-Context Decode (Llama-3B) — MEASURED (decode +13% vs gathered/+21% vs dense; ids identical; recipe holds)

- **Idea**: Invented follow-up validating the keep-grade no-gather recipe where it was never measured: long-context DECODE on the reference model (Llama-3.2-3B-Q4_K_M, 2050-token history, 63-token decodes) — same spike, same strided kernel, re-applied verbatim from the recipe. The 1568 round had found gathered sparse decode +21% vs dense on Llama; the question was whether removing the gather (per token, per head, per layer) preserves or extends that win.
- **Cycle date**: 2026-11-06
- **Verdict**: MEASURED — extends, not just preserves. Decode 63 tok @2050 history (single runs, box-bracketed): dense 7.66 t/s, gathered 8.19 t/s (+7%), strided 9.29 t/s (+21% vs dense, +13% vs gathered). Prefill re-confirmed on the reference model in passing (attn 0.49–0.52x dense across three interleaved rounds). Ids byte-identical gathered-vs-strided over the full 63-token decode (file-compared, 315 chars each). Differential unit test bit-exact again. No new recipe — the original stands, now covering prefill (both models) and long-context decode (reference model).
- **Tree state at close**: `git status` clean (no diff, no untracked), stash unchanged (only the pre-existing `leftover peer work shelved 2026-11-04`, untouched). Spike branch created (zero commits) and deleted; all 4 spike files (+1 test) reverted with empty diff verified. `master` ahead of `origin/master` by 83 after this report. Nothing pushed (human gate).

## 1. Mechanism (as re-applied — verbatim recipe, no deviations)

`attention_head_static_sparse_avx2` (three-segment strided dots + shared vectorized softmax + three-segment weighted accumulate, f32-only), decl, dispatch swap in `attention_head`'s sparse branch, bit-exact differential test. Decode uses the identical code path (one position per call, full history as span) — no decode-specific changes exist or were needed.

## 2. Predicted gain — confirmed with a clean ordering: dense < gathered < strided

Predicted: removing the per-token gather (one index build + kept-rows memcpys per layer per head per token) should beat gathered decode while preserving its numerics. Measured: 7.66 < 8.19 < 9.29 t/s. The gather was pure overhead in decode too (subset math identical, verified by identical ids), worth ~13% at this shape/history.

## 3. Falsifier outcome — all arms pass; no model gate, no decode exception

(1) Strided decode ≥ gathered decode: 9.29 vs 8.19 (+13%) ✓ — direction holds; small-sample single runs, but consistent with the mechanism (less work per token, same math) and with prefill's larger wins. (2) Ids identical over 63 decoded tokens: file-compared byte-identical ✓ (both binaries, same prompt/flags). (3) Differential re-run: bit-exact ✓. Prefill re-confirmation (not required but free): 0.49–0.52x dense attn, end-to-end prefill wins all rounds.

## 4. File anchors (master = pre-spike; spike reverted)

Same four files as the original recipe round (no-gather-proceed §4/§7), applied verbatim — the re-application itself required zero adaptations, which is recorded as evidence for the recipe's precision.

## 5. What was done (temporary spike, fully reverted)

1. `git checkout -b spike-nogather-decode` (zero commits ever made on it; deleted at close).
2. Re-applied the 4-file recipe + differential test (no deviations).
3. `cmake --build --preset release` clean (54/54); `ctest --preset release` pass; `tge_tests.exe` 142/142 (141 + 1 new).
4. Box health: wall 56–57 bracketing; aikit 2/4 (test PASS, no flake; pre-existing hygiene + guidelines fails).
5. Llama-3B @2050 prompt + 63-token decode: dense vs strided (interleaved where practical); gathered baseline + full-ids file via stash dance; strided ids to file; file-level `identical: True` check.
6. Thinking set on spike binary (non-sparse paths untouched): story ids byte-identical to baseline; reasoning + haiku fluent.
7. Reverted all 4 files (`git checkout --`, empty diff verified), `git checkout master`, `git branch -D spike-nogather-decode`; final gates on reverted master: build clean, `ctest` pass, `tge_tests` 141/141.

## 6. Measured numbers (Llama-3B-Q4_K_M, 2050-token fox prompt, ctx 4096, 6 threads, --prefetch)

| run | prefill (attn) | decode 63 tok | ids (63) |
|---|---|---|---|
| dense | 53.8 t/s (attn 7348.6) | 7.66 t/s | — (reference arm) |
| gathered (master, stash-built) | 49.2 t/s (attn 8605.5) | 8.19 t/s | `578 4062 13876 … 927` (315 chars) |
| strided (spike) R1/R2/R3 | 55.0/55.2/55.5 t/s (attn 3818.9/3819.6/3887.4) | 9.29 t/s | byte-identical file |

Attn ratios strided/dense per round: 0.52/0.50/0.49. Prefill ratios gathered/dense: 1.17x here (box-loaded; the 1568 round saw the same direction). Decode gains: gathered +7%, strided +21% vs dense — the ordering dense < gathered < strided holds end to end.

## 7. Full reimplementation recipe (no new recipe — the original stands, now validated on prefill ×2 models + long decode)

Rebuild per no-gather-proceed §7 (verbatim; this cycle is the evidence it transfers without modification). Gate for this validation: Llama-3B long-decode strided ≥ gathered (measured +13%), ids identical over the full decode (file-compared), differential bit-exact. To re-verify: same §6 protocol. Open threads unchanged: human review/merge of the original recipe (human gate); the per-matmul dump harness question for VNNI is a different program and untouched by this cycle.

## 8. Verdict and why

**MEASURED: the keep-grade recipe extends to long-context decode on the reference model with the predicted ordering (dense < gathered < strided) and identical outputs throughout.** Nothing about the original verdict changes except its evidence base, which now covers: unit differentials (bit-exact), 0.5B prefill (0.53x), Llama prefill (0.49–0.52x), Llama long decode (+13% over gathered), and fidelity digit-identity (2.597e-02/975 both arms). The loop's honest output is the confirmation plus the numbers; pushing remains unrequested and unperformed.

## 9. Exact commands run and their results

- `git checkout -b spike-nogather-decode` → branch (zero commits, deleted at close).
- 4-file spike + 1 test (verbatim recipe; built clean first try, differential green first try).
- `cmake --build --preset release` → 54/54; `ctest --preset release` → pass; `tge_tests.exe` → 142/142 on spike, 141/141 after revert.
- Wall 56–57; `aikit.py check` → 2/4 (test PASS, no flake; pre-existing hygiene + guidelines fails).
- Llama-3B dense vs strided (§6 table); stash dance → master gathered (49.2 t/s, attn 8605.5) + full ids file; strided ids file; file-level `identical: True`.
- Story ids byte-identical; reasoning + haiku fluent.
- `git checkout -- <4 files>` → clean; `git checkout master`; `git branch -D`; final master gates green; report committed; no push.

(End of file.)
