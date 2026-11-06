# INVENTED: No-Gather Strided Attention on the Reference Model (Llama-3B) — MEASURED (0.49–0.52x dense attn, ids byte-identical; recipe generalizes)

- **Idea**: Invented validation round for the keep-grade no-gather recipe (reports/idea-invented-sparse-nogather-proceed.md): re-apply the strided kernel verbatim from the recipe on the reference model (Llama-3.2-3B-Instruct-Q4_K_M, larger hidden/head dims, GQA architecture, K-quant weights with f32 KV) and verify the win transfers beyond the 0.5B-Q8_0 development shape — prefill A/B, short decode, ids parity vs gathered, plus a bit-exact differential re-run.
- **Cycle date**: 2026-11-06
- **Verdict**: MEASURED — transfers cleanly. Llama-3B @2050 tokens, interleaved best-of-3: dense attn 7348.6/7577.3/7970.2 ms vs strided 3818.9/3819.6/3887.4 ms → ratios 0.52/0.50/0.49 (the win is *larger* and more stable than on 0.5B, and strided degrades less under box load: +2% vs +8% across the window). Gathered master baseline in-window: 8605.5 ms (1.1–1.17x dense — reproduces the gathered-sparse penalty). Ids byte-identical gathered-vs-strided (`578 4062 13876 38835 34208 916`). Differential unit test green again (bit-exact on 600/1024/2048). Decode (5–7 tokens): identical within noise (8.10 vs 8.10 t/s) — no claim made or needed there; sparse decode wins were already recorded on Llama (§1568 round, +21% gathered) and the strided path can only remove overhead from them. Spike fully reverted; nothing pushed (human gate).
- **Tree state at close**: `git status` clean (no diff, no untracked), stash unchanged (only the pre-existing `leftover peer work shelved 2026-11-04`, untouched). Spike branch created (zero commits) and deleted; all 4 spike files (+1 test) reverted with empty diff verified. `master` ahead of `origin/master` by 81 after this report. Nothing pushed (human gate).

## 1. Mechanism (as re-applied from the recipe)

Verbatim re-application of the keep-grade recipe: `attention_head_static_sparse_avx2` (three-segment strided dots + shared vectorized softmax + three-segment weighted accumulate, f32-only with half/q8 throw), decl, dispatch swap in `attention_head`'s sparse branch, bit-exact differential test. No adaptations were needed for Llama-3B (larger hidden/head dims, GQA layout, K-quant file with f32 KV all flow through the same per-head path — the sparse branch bypasses GQA grouping by existing design, unchanged).

## 2. Predicted gain — confirmed, slightly stronger than on 0.5B

Predicted: same 0.5–0.6x band (mechanism is shape-independent: eliminate the gather memcpys, keep the halved dots). Measured: 0.49–0.52x, plus better load behavior (strided +2% vs dense +8% across a drifting window — consistent with less memory traffic under contention).

## 3. Falsifier outcome — passes on all arms (validation bar, not kill bar)

(a) Strided attn ≤ dense attn on Llama-3B: 0.49–0.52x across three interleaved rounds — passes with the same margin as 0.5B; no model gate needed. (b) Ids byte-identical gathered-vs-strided on the 2k prompt (`578 4062 13876 38835 34208 916` both binaries — stash-built master for the gathered arm). (c) Differential unit test bit-exact again (re-ran, not assumed). Decode recorded as-is (identical within run noise on 5–7 tokens; the decode claim stays with the 1568 round's gathered +21%, which the strided path can only improve by removing the gather).

## 4. File anchors (master = pre-spike; spike reverted)

Same four files as the original recipe round (no-gather-proceed §4/§7): `src/cpu/avx2/attention_avx2.cpp` (kernel), `include/tge/cpu/kernels.hpp` (decl), `src/models/qwen2.cpp` (dispatch swap), `tests/test_attention.cpp` (differential). This cycle is a verbatim re-application — no deviations were required, which is itself a finding about the recipe's precision.

## 5. What was done (temporary spike, fully reverted)

1. `git checkout -b spike-nogather-llama` (zero commits ever made on it; deleted at close).
2. Re-applied the 4-file recipe + 1 test from the report (no deviations needed).
3. `cmake --build --preset release` clean (54/54); `ctest --preset release` pass; `tge_tests.exe` 142/142 (141 + 1 differential).
4. Box health: wall 61.72 bracketing; aikit 2/4 (test PASS, no flake; pre-existing hygiene + guidelines fails).
5. Llama-3B @2050 prefill A/B, interleaved best-of-3 (§6 table); master gathered baseline via stash dance (attn 8605.5 + matching ids); story + reasoning thinking set on the spike binary.
6. Reverted all 4 files (`git checkout --`, empty diff verified), `git checkout master`, `git branch -D spike-nogather-llama`; final gates on reverted master: build clean, `ctest` pass, `tge_tests` 141/141.

## 6. Measured numbers (Llama-3B-Q4_K_M, 2050-token fox prompt, ctx 4096, 6 threads, --prefetch)

| run | prefill | attn | decode (5–6 tok) |
|---|---|---|---|
| dense R1 / R2 / R3 | 53.8 (38114) / 51.7 (39680) / 49.1 (41756) t/s (ms) | 7348.6 / 7577.3 / 7970.2 ms | 8.10 t/s |
| strided R1 / R2 / R3 | 55.0 (37269) / 55.2 (37118) / 55.5 (36931) t/s (ms) | 3818.9 / 3819.6 / 3887.4 ms | 8.10 t/s |
| gathered (master, stash-built, 1 run) | 49.2 (41695) t/s | 8605.5 ms | — |

Attn ratios strided/dense: 0.52 / 0.50 / 0.49. Ids (2k prompt, `--print-ids`): `578 4062 13876 38835 34208 916` — identical on both binaries. Decode: identical within noise (short decodes; no claim). End-to-end prefill: strided wins all three rounds (+2% to +13%; the gap widens as the box loads, consistent with the traffic explanation).

## 7. Full reimplementation recipe (no new recipe — the original stands, now validated on two models)

This cycle adds no code to the keep-grade recipe (revert was total); it adds *evidence*: re-apply verbatim per the no-gather-proceed §7, then re-verify with the §6 protocol on Llama-3B (3-round interleaved A/B expecting attn ≤ 0.6x dense with the same direction every round; gathered-vs-strided ids byte-identical; differential test bit-exact). To re-verify this validation: same §6 table within box noise. Next (not this cycle): master-side alternatives remain untouched — applying the recipe still awaits human review/merge per the human gate.

## 8. Verdict and why

**MEASURED: the recipe generalizes verbatim to the reference model with equal-or-better margins (0.49–0.52x vs 0.53–0.58x), identical ids, and a passing differential — with zero adaptations required, which validates the recipe's precision as written.** No model gate, no shape gate, no additional work identified; the keep-grade standing from the prior round is strengthened, not qualified. The loop's honest output is the confirmation plus the numbers; pushing remains unrequested and unperformed.

## 9. Exact commands run and their results

- `git checkout -b spike-nogather-llama` → branch (zero commits, deleted at close).
- 4-file spike + 1 test (verbatim recipe re-application; built clean first try, differential green first try).
- `cmake --build --preset release` → 54/54; `ctest --preset release` → pass; `tge_tests.exe` → 142/142 on spike, 141/141 after revert.
- Wall 61.72; `aikit.py check` → 2/4 (test PASS, no flake; pre-existing hygiene + guidelines fails).
- Llama-3B 2k A/B best-of-3 interleaved (§6 table; strided wins every round).
- Stash dance → master gathered baseline (272... i.e. 49.2 t/s, attn 8605.5, ids identical) → unstash → rebuild.
- Story ids byte-identical to baseline; reasoning fluent.
- `git checkout -- <4 files>` → clean; `git checkout master`; `git branch -D`; final master gates green; report committed; no push.

(End of file.)
