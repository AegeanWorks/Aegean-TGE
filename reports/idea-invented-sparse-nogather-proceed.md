# INVENTED: No-Gather Strided Attention — PROCEED WITH RECIPE (attn 0.53–0.58x dense at 2k, 0.46x at 4k; bit-identical; fidelity digit-identical)

- **Idea**: Invented, queued by the phase-split round (gather ≈ 85% of sparse attn; index build ≈ 4%). A strided AVX2 attention kernel that walks the static sink/stride/window kept set in place instead of materializing compacted buffers — same kept rows in the same order as `build_sparse_index`, so the ONLY delta vs the gathered path is the removed memcpy traffic + per-call allocation.
- **Cycle date**: 2026-11-06
- **Verdict**: PROCEED WITH RECIPE (keep-grade — not committed as code per loop rules; the spike is fully reverted). All three falsifier arms pass with margin: (1) 2k prefill attn 0.53–0.58x dense across three interleaved rounds (bar was ≤1.0x / within-10%: cleared by ~45%); (2) 2k ids byte-identical to gathered sparse (`576 3974 13876 38835 34208 916`, both binaries) + unit differential bit-exact on synthetic shapes; (3) fidelity `--sparse` digit-identical to gathered (KL 2.597e-02, worst 7.867, max|diff| 11.67, 975/1023 — both arms, i.e. the numerics are unchanged by construction AND by measurement). Bonus: the win grows with context (4k attn 0.46x dense; end-to-end prefill flips from -7% to +13% vs dense between 2k and 4k).
- **Tree state at close**: `git status` clean (no diff, no untracked), stash unchanged (only the pre-existing `leftover peer work shelved 2026-11-04`, untouched). Spike branch created (zero commits) and deleted; all spike files reverted with empty diff verified. `master` ahead of `origin/master` by 79 before this report. Nothing pushed (human gate).

## 1. Mechanism (as spiked)

`attention_head_static_sparse_avx2(args, sink, window, stride)`: three-segment walk (sink verbatim [0,sink_end), strided middle [sink_end,window_start) step stride, window tail [max(window_start,sink_end),span)) through the resident K/V cache with the existing AVX2 dots + shared vectorized softmax + weighted accumulate — the dense kernel's exact structure with the loop bounds replaced. Dispatch: `attention_head`'s sparse branch calls it directly instead of build+gather+generic kernel (the `kv_stride`/`args.span` contract is unchanged; f32-only, guarded by the same condition as before). No index build, no scratch buffers, no memcpy.

## 2. Predicted gain — met and exceeded at every step

Predicted: remove the ~85% gather share → attn approaches the kept-fraction floor (~44% of dense dots at 2k, less the per-call overhead that remains). Measured: 0.53–0.58x dense at 2k (within noise of the kept fraction, i.e. the overhead is essentially gone — the residual over 0.44x is per-call dispatch, not data movement), 0.46x at 4k (kept ~35%). End-to-end prefill flips sign: strided BEATS dense at both lengths (R1 334.7 vs 310.8; 4k +13%).

## 3. Falsifier outcome — all three arms pass

(1) 2k prefill attn vs dense, interleaved best-of-3: dense 1033/1136/1257 ms vs strided 585/661/670 ms → ratios 0.57/0.58/0.53, consistent direction across a drifting box (matvec drifted +8–33% across the window on both arms; the attn ratio is load-independent). Bar was "within 10% of dense" — cleared by ~45%. (2) Ids: `576 3974 13876 38835 34208 916` on both binaries for the same 2k prompt (gathered master via stash-built binary, strided on spike) — plus the unit differential is bit-exact by construction on synthetic shapes. (3) Fidelity `--sparse` (0.5B-Q8_0, arch corpus, ctx 1024, 6 threads, gpu off): spike 2.597e-02 / 7.867 / 11.67 / 975/1023 vs master 2.597e-02 / 7.867 / 11.67 / 975/1023 — digit-identical, i.e. the strided path prices exactly the shipped sparse tier (no new numerics exist to price).

## 4. File anchors (master = pre-spike; spike reverted)

- `src/cpu/avx2/attention_avx2.cpp` (strided kernel beside `attention_head_avx2` — reverted).
- `include/tge/cpu/kernels.hpp` (`attention_head_static_sparse_avx2` decl — reverted).
- `src/models/qwen2.cpp` (`attention_head` sparse branch: 1 call replaces build+gather+dispatch — reverted).
- `tests/test_attention.cpp` (`sparse_strided_matches_gathered_exactly`: gather+dense-AVX2 vs strided on identical inputs, bit-exact including scores — green, reverted with the spike).
- Grounds: phase-split (85/4 split), index-cache kill (sharing proven, no-op), 1568 close-out (the original +35%/-7% gap), 1517 staged recipe (this is a wiring-candidate for the paged track's attention half).

## 5. What was done (temporary spike, fully reverted)

1. `git checkout -b spike-sparse-nogather` (zero commits ever made on it; deleted at close).
2. Spike (4 files + 1 test): strided kernel (~90 lines, three-segment dots + shared softmax + three-segment accumulate); decl; 8-line dispatch swap; bit-exact differential test (spans 600/1024/2048 — all engaged: identity-verified, not just smoke-tested).
3. `cmake --build --preset release` clean (54/54); `ctest --preset release` pass; `tge_tests.exe` 142/142 (141 + 1 new).
4. Box health: wall 56.97→58.38 bracketing the window; aikit 2/4 (test PASS, no flake; pre-existing hygiene + guidelines fails).
5. 2k prefill A/B (dense vs strided, interleaved best-of-3, 0.5B-Q8_0): §6 table.
6. Master baseline via stash dance (the same tools, honest comparator): master 2k sparse prefill (272.0 t/s) + ids + fidelity; restored spike, compared, reverted.
7. Long-context check (4k fox, ctx 8192): dense 259.4 vs strided 292.1 t/s prefill (+13%), attn 4316.9 vs 1972.4 (0.46x).
8. Fidelity `--vnni`-free `--sparse` pair (§6; digit-identical both arms).
9. Thinking set on spike binary (non-sparse paths untouched by construction, VNNI file untouched): story ids byte-identical to baseline; reasoning fluent.
10. Reverted all 4 files (`git checkout --`, empty diff verified), `git checkout master`, `git branch -D spike-sparse-nogather`; final gates on reverted master: build clean, `ctest` pass, `tge_tests` 141/141.

## 6. Measured numbers (0.5B-Q8_0, 6 threads, gpu off; box drift noted per pair)

2k prefill (fox ×200, ctx 4096, interleaved):

| round | dense prefill (attn) | strided prefill (attn) | attn ratio |
|---|---|---|---|
| R1 | 310.8 t/s (1033 ms) | 334.7 t/s (585 ms) | 0.57x |
| R2 | 275.2 t/s (1135.9 ms) | 289.6 t/s (661.0 ms) | 0.58x |
| R3 | 246.0 t/s (1256.9 ms) | 268.7 t/s (670.2 ms) | 0.53x |
| master gathered (stash) | — | 272.0 t/s (1456.3 ms attn) | 1.35x dense (the gap being closed) |

4k prefill (fox ×380, ctx 8192): dense 259.4 (attn 4316.9) vs strided 292.1 (attn 1972.4) → 0.46x attn, +13% end-to-end (win grows as kept fraction falls: 44% → ~35%). Decode arms (7–8 tokens): within run noise (sparse paths don't move short decode — no claim made).
Fidelity `--sparse` (both binaries, arch corpus, ctx 1024): 2.597e-02 / 7.867 / 11.67 / 975/1023, digit-identical.
Unit differential: bit-exact (0.0 diff) on spans 600/1024/2048 incl. tail/edge shapes.
Thinking set: story ids identical to baseline; reasoning fluent.

## 7. Full reimplementation recipe (keep-grade — rebuild exactly this; it is the implementation)

1. `kernels.hpp`: add the decl from §4 (args + sink/window/stride; f32-only contract in the comment).
2. `attention_avx2.cpp`: add `attention_head_static_sparse_avx2` per §5.2 — three-segment dots (dense `dot_avx2` per kept row), shared vectorized softmax over kept count, three-segment weighted accumulate; `validate_attention_args` + explicit half/q8 throw (unreachable via caller gate).
3. `qwen2.cpp`: in `attention_head`'s sparse branch, replace build+gather+generic call with the single strided call (keep the same `span > sink+window` gate and f32-only conditions bit-for-bit).
4. `test_attention.cpp`: add the bit-exact differential test (gather+dense-AVX2 vs strided, identical inputs; assert `==`, not tolerance).
5. Gate: build 54/54, 142/142 tests (differential must be bit-exact — any nonzero diff is a bug, not noise); 2k A/B reproduces §6 ratios within box noise (±20% absolutes, same direction); 8k check confirms the win grows; fidelity `--sparse` digit-matches the tier (2.597e-02/975); story ids identical; push only with human approval (never force-push; reports-only until then).
6. Explicitly NOT included (do not smuggle in): GQA-grouped strided paths (grouped kernels stay dense — separate idea), cross-head index sharing (killed as unnecessary by the index-cache round), any change to the dense path.

## 8. Verdict and why

**PROCEED WITH RECIPE.** Every gate the loop demands is green with margin: best-of-3 A/B at 0.53–0.58x (bar: ≤1.0x), end-to-end prefill wins at both lengths (R1–R3 + 4k), bit-identical outputs on synthetic AND real prompts (unit differential + 2k ids + digit-identical fidelity), full suite green, thinking set fluent. This is the rare spike that clears keep-grade on all arms at once: it deletes code paths (no index, no scratch, no gather loop) while making the engine faster. The remaining work is product, not research: review, human approval, and a merge — none of which this loop does unasked.

## 9. Exact commands run and their results

- `git checkout -b spike-sparse-nogather` → branch (zero commits, deleted at close).
- 4-file spike + 1 test (§5.2; no mid-spike fixes — built clean first try, differential green first try).
- `cmake --build --preset release` → 54/54; `ctest --preset release` → pass; `tge_tests.exe` → 142/142 on spike, 141/141 after revert.
- Wall 56.97→58.38; `aikit.py check` → 2/4 (test PASS, no flake; pre-existing hygiene + guidelines fails).
- 2k A/B best-of-3 (§6 table); master baseline via stash dance (272.0 t/s, attn 1456.3, ids `576 3974 13876 38835 34208 916` — identical to strided).
- 4k check (dense 259.4/4316.9 vs strided 292.1/1972.4); fidelity pair (digit-identical §6); thinking set (story ids identical, reasoning fluent).
- `git checkout -- <4 files>` → clean; `git checkout master`; `git branch -D`; final master gates green; report committed; no push.

(End of file.)
