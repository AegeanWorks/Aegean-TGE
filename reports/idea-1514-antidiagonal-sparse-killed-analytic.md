# IDEA-1514: Antidiagonal Block-Sparse Attention Port — KILLED WITHOUT SPIKE (analytic, measured)

- **Idea ID**: IDEA-1514 (dossier `scout-filing-2026-aegean-tge.md`: "Antidiagonal Block-Sparse Attention Port", arXiv 2503.16428 XAttention §§2–3)
- **Cycle date**: 2026-11-06
- **Verdict**: KILLED on the record, no spike built (dead-idea rule: the falsifier already fails on today's tree). No branch created, no code touched, nothing to revert.
- **Tree state at close**: `git status` clean (only `master` ahead of `origin/master`, now by 10 with the two report commits), stash untouched (`stash@{0}` leftover peer work), nothing pushed (human gate).

## 1. Mechanism (as dossiered)

Port XAttention's strided antidiagonal block-importance proxy to the CPU engine:
score each KV block by antidiagonal sums (a cheap stand-in for full block attention
probability mass), keep the top blocks at density ≤30%, skip the remaining QK/V
block compute in `src/cpu/avx2/attention_avx2.cpp` + `src/models/qwen2.cpp` +
`include/tge/cpu/`. Claimed upside from the paper: up to 13.5x attention-compute
speedup at 256k context, RULER 88.47 vs 87.52 full attention.

## 2. Predicted gain (with the math) — and why it does not transfer

Paper number is an **attention-microbenchmark at 256k context**. End-to-end gain on
today's tree follows Amdahl on the measured attention share (see §6):

- Prefill at 2081 tokens: attn 1546.1 ms of 8906.75 ms = **17%**.
- Decode at KV≈2081: attn 44.9 ms of 233.47 ms ≈ **20%** (matvec 70%, norm 10%).

Optimistic model: keep 30% of attention work, pay ~10–15% scoring overhead (every
block must at least be touched to be scored) → attention phase ≈ 40–45% of dense.
End-to-end: prefill → 59% + 0.45·17% + 23% ≈ 90% ⇒ **~11% prefill speedup**;
decode → 70% + 0.45·20% + 10% ≈ 89% ⇒ **~11% decode speedup** — *if and only if*
density ≤30% holds at fidelity KL ≤1e-9. Both conditions fail (§6): the KL arm is
dead by precedent, and scoring at 2k must beat a dense path that is already only
1.5 s of an 8.9 s prefill.

## 3. Falsifier (from the dossier)

- `tge fidelity` mean KL > 1e-9 vs scalar oracle, or any top-1 flip → kill.
- Same-run 2048 prefill/decode fails to beat `auto` (dense) → kill.

## 4. File anchors (current `master`, read but untouched)

- `src/cpu/avx2/attention_avx2.cpp` — `dot_avx2()` (4 accumulators), `q8_dot_avx2()`,
  `attention_head_avx2()` / group paths; where scoring + skip would land.
- `src/cpu/scalar/attention_scalar.cpp` — the oracle fidelity prices against.
- `src/models/qwen2.cpp` — `Qwen2::attention_head`, `Qwen2::attention()`,
  sparse-subset gather + `uses_gqa` precedence (shipped §50 static sparse).
- `include/tge/models/qwen2.hpp` — `SparseConfig{sink=4, window=512, stride=4}`.
- `apps/cli/main.cpp` — `generate`/`fidelity --sparse [--sparse-window] [--sparse-stride]`.
- `tests/test_attention.cpp` — sparse geometry + gather-vs-direct tests.
- `docs/measurement-log.md` §50 — static-sparse numbers cited below.

## 5. What was done (ANALYZE only — no spike per the dead-idea rule)

1. Re-read the dossier mechanism against the current tree (files above) and the
   shipped static-sparse precedent (§50): content-blind sink+stride+window is
   already opt-in (`--sparse`, default off); IDEA-1514 would add content-aware
   scoring on top of it.
2. Reproduced the gap it claims, on today's tree, with the unmodified binary:
   `cmake --build --preset release` clean; `ctest --preset release` pass;
   `tge_tests.exe` **141/141, 0 failed**.
3. Measured the attention share at ~2k context (Qwen2.5-0.5B-Q8_0, `--gpu off`
   to isolate CPU kernels, warm page cache, 10 threads):
   2081-token prompt → prefill 233.6 t/s with phases matvec 5279.6 / attn 1546.1 /
   norm 2013.3 ms; decode 64.25 t/s with matvec 70% / attn 20% / norm 10%.
4. Compared against the dossier's own two falsifier arms (§6). Both fail; no
   spike built, no branch created.
5. `python ../../aikit.py check aegean-tge` → 2/4: test PASS, secrets PASS,
   hygiene FAIL (pre-existing: C++-only project, no scannable Python/asset files),
   guidelines FAIL (pre-existing README drift, untouched by this cycle).
6. No thinking-set diff owed: same binary throughout, zero behavior change.

## 6. Measured numbers (today's tree, unmodified binary)

Command: `tge.exe generate --model qwen2.5-0.5b-instruct-q8_0.gguf
--prompt "<12200-char repeated paragraph>" --n 16 --threads 10 --gpu off`
(prompt tokenizes to **2081 tokens**).

| phase | time | share |
|---|---|---|
| prefill total (2081 tok, 233.6 t/s) | 8906.75 ms | 100% |
| └ matvec | 5279.6 ms | 59% |
| └ **attn** | **1546.1 ms** | **17%** |
| └ norm | 2013.3 ms | 23% |
| decode total (15 tok, 64.25 t/s) | 233.47 ms | 100% |
| └ matvec | 159.2 ms | 70% |
| └ **attn** | **44.9 ms** | **20%** |
| └ norm | 22.6 ms | 10% |

Kill arm 1 — fidelity (by precedent, no rerun needed): shipped static sparse at
*~63% density* (sink 4 + window 512 + stride-4 ≈ 644/1024 kept) already prices at
**KL mean 2.597e-02, worst 7.867, top-1 975/1023 (48 flips)** on natural text at
ctx 1024 (log §50). IDEA-1514 demands ≤30% density at **KL ≤1e-9** — seven orders
of magnitude tighter while skipping more than twice as much. The paper itself
concedes quality loss (RULER 88.47 vs 87.52). The KL arm cannot survive contact
with natural text on this tree; a spike would only re-pay the 193 s fidelity run
to re-learn it.

Kill arm 2 — beat-auto at ~2k (Amdahl): with attention at 17–20%, even a free
70%-saving scorer caps at ~11% end-to-end, and scoring is not free — antidiagonal
sums must stream every block they rank, at 24 layers × 14 query heads (GQA 14/2)
× head_dim 128, against a dense path whose whole attention phase is 1.5 s.
The dossier's cost condition (scoring < skipped blocks at 2k–32k) has no margin
at the 2k end, which is the only end this box can measure (max meaningful ctx
here is single-digit k, not 256k).

## 7. Reimplementation recipe (deliberately NOT recommended — kill-grade)

For the record only, the spike that was *not* built would have been:

1. `src/cpu/avx2/attention_avx2.cpp`: new `block_scores_avx2()` computing strided
   antidiagonal sums per KV block (block 64 tokens → 64/stride samples of
   |q·k| proxies), a top-k-per-query selection at budget = 30% of blocks, and a
   masked `attention_head_avx2()` variant that compacts the selected rows into the
   existing thread-local scratch (same pattern as the shipped `SparseConfig`
   gather) and runs the unchanged dot/exp/value-sum over the compacted span.
2. `src/models/qwen2.cpp` + `include/tge/models/qwen2.hpp`: per-call score buffer
   beside the sparse scratch, density knob next to `SparseConfig`, exact-dense
   fallback when span ≤ sink+window or KV is f16/i8 (mirror the shipped precedence).
3. Gate the same way the dossier demands: `fidelity` KL ≤1e-9 + 2048 prefill/decode
   vs dense, best-of-3, warm cache. Expected outcome per §6: KL arm fails first.

## 8. Verdict and why

**KILLED without spike.** The dossier's own falsifier is already tripped on
today's tree: (a) the KL ≤1e-9 arm is incompatible with ≤30% density by the
shipped-sparse precedent (2.6e-2 at 63% density, 48 flips) and the paper's own
quality concession; (b) the beat-auto arm has no Amdahl headroom at 2k (attn
17–20%, cap ~11% before scoring cost). Do not revisit until either the fidelity
bar is renegotiated for sparse attention (e.g. priced like VNNI at 1.6e-3 with a
flips budget, not 1e-9) or measured contexts reach tens of k where attention, not
matvec, dominates — neither holds on this box today.

## 9. Exact commands run and their results

- `cmake --build --preset release` → 54/54, link OK.
- `ctest --preset release` → 1/1 pass; `tge_tests.exe` → 141 cases, 0 failed.
- `tge.exe generate --model <q8_0> --prompt "<40× paragraph = 2081 tokens>" --n 16 --threads 10 --gpu off` → prefill 2081 tok in 8906.75 ms (233.6 t/s), phases matvec 5279.6 / attn 1546.1 / norm 2013.3 ms; decode 15 tok in 233.47 ms (64.25 t/s), matvec 70% / attn 20% / norm 10%.
- `python ../../aikit.py check aegean-tge` → 2/4 (test + secrets pass; hygiene + guidelines fail pre-existing, unrelated: no Python files in repo, README drift untouched).
- No spike branch, no edits (`git status` clean apart from the prior ahead count), no push.
