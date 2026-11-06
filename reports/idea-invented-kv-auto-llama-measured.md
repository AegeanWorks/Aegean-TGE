# INVENTED: --kv auto on the Reference Model (Llama-3.2-3B) — MEASURED (identical ids both arms, timing within noise; recipe holds unchanged)

- **Idea**: Invented follow-up validating the keep-grade kv-auto recipe (f16 above 512 declared context, f32 at/below, i8 never) on the reference model Llama-3.2-3B-Instruct-Q4_K_M.gguf, where it was never measured (original validation was 0.5B-Q8_0 only). No new mechanism: re-applied the committed recipe verbatim to test whether it transfers unchanged.
- **Cycle date**: 2026-11-06
- **Verdict**: MEASURED — recipe holds verbatim on the reference model, no adaptations, no model gate. Long (2050-token fox, ctx 4096): auto ids byte-identical to manual `--kv f16` across all four interleaved runs (`578 4062 14198 … 578 4062`, 12 ids each); prefill 53.2/53.1 vs 53.2/51.3 tok/s, attn 6074/5978 vs 5946/6131 ms, decode 9.43/8.71 vs 9.28/9.40 — all within box drift (both arms slow together in round 2, same signature as the 0.5B false-alarm analysis). Short (story, ctx 256): auto ids byte-identical to manual `--kv f32` (`11 304 264 2678 … 315 279`, 12 ids each). No fidelity re-run: both arms are already priced on this exact model (2048-closure ladder: f16 KL-class holds, top-1 agreement held), the rule is unit-pinned (new `auto_kv_half_threshold` test, 9 exact boundaries, green then reverted), and identical ids prove the same binary paths execute.
- **Tree state at close**: `git status` clean (no diff, no untracked), stash unchanged (only the pre-existing `leftover peer work shelved 2026-11-04`, untouched). Spike branch created (zero commits) and deleted; all 4 spike files reverted with empty diff verified. `master` ahead of `origin/master` by 87 at cycle close (verified via rev-list count). Nothing pushed (human gate).

## 1. Mechanism (as re-applied; no deviations from the committed recipe)

`Qwen2::auto_kv_half(context_limit)`: `declared = context_limit > 0 ? context_limit : kDefaultContext (4096)`; `return declared > kAutoKvHalfContext (512)`. Four CLI sites (generate/fidelity/continuous/sessions): `kv_auto` bool, `"auto"` branch, post-parse resolve (`kv_half = auto_kv_half(ctx); kv_q8 = false`), error strings extended to `f32|f16|i8|auto`. Reference arm (fidelity) untouched (stays f32). i8 never auto (size-ladder conditional stands). The budget-shrink edge case from the original report is unchanged and untested here (same code shape, no new behavior).

## 2. Predicted gain — confirmed by identity, not by speedup

No new speedup claimed: auto selects the same binary path manual flags select, so the 2048-closure numbers (+9–16% at 2k) transfer by construction. What this round proves is the selection is correct on a GQA model with a different tokenizer and head geometry: long context → f16 path, short/capped context → exact f32 path.

## 3. Falsifier (stated) — all arms pass

Kill on any ids divergence vs the manual same-format arm, or timing outside box noise, or a needed model-specific adjustment to the recipe. Result: (a) long auto == manual f16 ids byte-identical (4/4 runs); (b) long timing within drift (interleaved pairs move together; the 0.5B false-alarm lesson applied pre-emptively, so no false verdict was recorded); (c) short auto == manual f32 byte-identical; (d) unit test pins all nine boundaries (0/1/511/512/513/1024/4096/32768) plus the constexpr. No bug found, no recipe change needed.

## 4. File anchors (master = pre-spike; spike reverted)

- `include/tge/models/qwen2.hpp:182-197` (rule decl + constexpr beside `kv_half()`).
- `src/models/qwen2.cpp:25` (`kDefaultContext = 4096`), new `auto_kv_half` beside `set_sparse_config`.
- `apps/cli/main.cpp` (4 sites: generate locals ~1973/parse ~2069/resolve pre-`command_generate`; fidelity locals ~2476-77/parse ~2490-2521/resolve pre-`command_fidelity`; bench continuous locals/parse/resolve; bench sessions likewise; help + error strings intact).
- `tests/test_attention.cpp` (`auto_kv_half_threshold` — green in-spike, reverted).
- Grounds: kv-auto recipe (0.5B identity + falsifiers), 2048-closure (Llama f16/f32 ladder), §41 (threshold math).

## 5. What was done (temporary spike, fully reverted)

1. `git checkout -b spike-kvauto-llama` (zero commits ever made on it; deleted at close).
2. Spike (4 files + 1 test): verbatim re-application of the committed recipe — no deviations, no model-specific adjustments, no debug prints needed.
3. `cmake --build --preset release` clean (54/54); `ctest --preset release` pass; `tge_tests.exe` 142/142 (141 + 1 new rule test).
4. Box health: wall 60.81 (healthy).
5. Long pair (Llama-3B, 2050-tok fox, ctx 4096, 6 threads, prefetch): auto R1 then manual f16 (paired), then interleaved auto/f16/auto/f16 — all 4 ids byte-identical; timing within drift.
6. Short pair (Llama-3B story, --context 256): auto == manual f32, byte-identical ids.
7. Thinking set on the spike binary (auto default ctx → f16 path): reasoning ids recorded, haiku fluent — no degenerate output on the auto-selected tier.
8. `python aikit.py check aegean-tge` → 2/4 (test PASS, no flake; pre-existing hygiene + guidelines fails).
9. Reverted all 4 files (`git checkout --`, empty diff verified); `git checkout master`; `git branch -D spike-kvauto-llama`; final gates on reverted master: build clean, `ctest` pass, `tge_tests` 141/141; tree clean, stash intact.

## 6. Measured numbers (Llama-3B-Q4_K_M, 6 threads, prefetch, gpu off)

Long (2050 tokens, ctx 4096, 11–12 tok decode, interleaved in run order):

| arm | prefill tok/s (attn ms) | decode tok/s | ids (12 shown) | verdict |
|---|---|---|---|---|
| auto #1 | 56.1 (5832.7) | 9.39 | `578 4062 … 578 4062` | — |
| f16 #1 | 52.7 (6029.6) | 8.99 | identical | within noise |
| auto #2 | 53.2 (6073.8) | 9.43 | identical | — |
| f16 #2 | 53.2 (5946.3) | 9.28 | identical | within noise |
| auto #3 | 53.1 (5977.9) | 8.71 | identical | drift pair, both slow |
| f16 #3 | 51.3 (6131.3) | 9.40 | identical | drift pair, both slow |

Attn share ~11% at 2k on Llama (vs ~20% on 0.5B at the same length — wider model, matvec-dominated). All 12-token ids byte-identical across all four runs (and all four runs' phases match within ±noise).

Short (story, ctx 256, 12 tok): auto == manual f32, byte-identical (`11 304 264 2678 … 315 279`).

Thinking set (spike binary, default ctx → f16): reasoning fluent with ids recorded; haiku about the sea fluent ("Golden light on rippling blue / Morning's gentle kiss").

## 7. Full reimplementation recipe (keep-grade — unchanged from the original; this cycle adds the reference-model evidence, not new instructions)

Rebuild exactly per `idea-invented-kv-auto-proceed.md` §7 (no deviations needed on Llama-3B — verified this cycle). Gate additions from this round: (a) 142/142 tests incl. the 9-boundary rule test; (b) Llama-3B long auto == manual f16 ids (interleaved, any order); (c) Llama-3B short auto == manual f32 ids; (d) timing within box drift (interleave; never trust paired runs on this box). Do not add model-specific branches (none needed). Do not re-price fidelity on Llama (both arms already priced: closure ladder + this cycle's identity proof).

## 8. Verdict and why

**MEASURED: the keep-grade recipe transfers verbatim to the reference model with zero adaptations and zero divergences.** The product question from the original report (ship as default vs keep as flag) now has both models' evidence: identical selection behavior, identical outputs, noise-level timing. Nothing here changes the recipe — that IS the finding, and for a cross-model validation that is the best possible outcome.

## 9. Exact commands run and their results

- `git checkout -b spike-kvauto-llama` → branch (zero commits, deleted at close).
- 4-file spike + 1 test (§7 of the original report, verbatim; no deviations, no debug prints needed).
- `cmake --build --preset release` → 54/54; `ctest --preset release` → pass; `tge_tests.exe` → 142/142 on spike, 141/141 after revert.
- Wall 60.81 bracketing; `aikit.py check` → 2/4 (test PASS, no flake; pre-existing hygiene + guidelines fails).
- Long: paired auto/f16 then interleaved auto/f16/auto/f16 — all ids byte-identical; timing within drift (§6 table).
- Short: auto == manual f32 ids byte-identical.
- Thinking set (auto tier): reasoning ids + fluent haiku.
- `git checkout -- <4 files>` → empty diff; `git checkout master`; `git branch -D`; final master gates green; report committed; no push.

(End of file.)
