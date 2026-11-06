# INVENTED (trigger S3): Vectorized Tiled Prefill Attention Retry — KILLED analytic (ceiling ~0% realistic, 2.2% generous, both under the 5% bar — no spike)

- **Idea**: The parked S3 reopen trigger ("vectorized tiled inner loop matching AVX2 compute rate"): re-tile prefill attention as Q-block × K-block online-softmax, but with AVX2-vectorized dots and exp instead of the scalar code that killed S3 at +49%. Tiling eliminates the per-(query,head) score-row write+read while keeping FLOPs identical.
- **Cycle date**: 2026-11-06
- **Verdict**: KILLED analytic — the retry's own prerequisite (match AVX2 compute rate) yields ~nothing by construction. Two independent ceilings, both under the 5% bar: (1) realistic ≈0%: attention runs at ~4.4 GB/s against a ~55 GB/s roof — deeply compute-bound, so eliminating score traffic saves no time; (2) generous 2.2%: even pretending attention is fully traffic-bound, score traffic is ~10% of attention traffic and attention is ~21.5% of prefill. No spike per the dead-idea rule (falsifier fails on today's tree); no branch, no edits, nothing to revert.
- **Tree state at close**: `git status` clean, no diff, stash unchanged (only the pre-existing `leftover peer work shelved 2026-11-04`). `master` ahead of `origin/master` by 92 after this report. Nothing pushed (human gate).

## 1. Mechanism (as proposed, not built)

Q-block(8) × K-block(128) online-softmax tiling in `attention_batch`, AVX2-vectorized inner dots (reuse the dequant+dot idiom) and vectorized exp, dense path only. Same K reads, same V reads, same exp/dot FLOPs as the shipped kernel; the ONLY delta is the removed score-row traffic (write + re-read of one `span`-float row per (query, head) in `scores_`).

## 2. Predicted gain — the ceiling math that kills it

Recorded shape (verify-then-build S3, same tree): 2043-token repetitive prefill, dense attn 12.83 s of 59.6 s wall (attn = 21.5% of prefill). Verified geometry this cycle (`generate` header): 24 layers, 896 dim, 14/2 heads (H=14, kv_dim=128 confirmed), i.e. head_dim=64.

- Score traffic per layer: Σ over queries 2·span·4B ≈ T²·H·4B = 2043²×14×4 ≈ 234 MB; ×24 layers ≈ 5.6 GB.
- KV traffic per layer: K read + V read = 2·(T²/2)·kv_dim·4B ≈ 2.14 GB; ×24 ≈ 51 GB.
- Attention traffic ≈ 57 GB in 12.83 s ≈ 4.4 GB/s vs the measured 54–61 GB/s roof → compute-bound by >10x. Eliminating score traffic removes bytes from a non-binding resource: expected time saving ≈ 0%.
- Generous ceiling (grant traffic-bound, contrary to measurement): 5.6/(51+5.6) ≈ 10% of attention × 21.5% of prefill ≈ **2.2% end-to-end < 5% bar**.

Falsifier (stated): kill the retry if perfect score elimination ceilings below 5% end-to-end on the recorded 2043-token shape. Both ceilings do (0% and 2.2%). The S3 scalar result (+49% worse) is now explained twice over: it paid full vectorization loss to eliminate traffic that wasn't binding.

## 3. Falsifier outcome — kill, with the integration surface noted

Beyond the ceiling, the retry would have to preserve the GQA path, the static-sparse path, the scalar-oracle path, and the f16/i8 KV paths in `attention_head` — a large integration surface for a ~0% ceiling. Not attempted (dead-idea rule). Revival trigger ONLY: a prefill shape where attention is traffic-bound (measured ≥50% of roof) AND score traffic exceeds 25% of attention traffic — never observed; today's shape is 4.4 GB/s and 10%.

## 4. File anchors (current `master`, read but untouched)

- `src/models/qwen2.cpp:1420-1466` (`attention_batch`: per-(query,head) delegation with `scores_` row slices) and `:1135-1207` (`attention_head`: dense `kernels().attention_head(args)` with `args.scores`, plus sparse/scalar/KV-format arms the retry would have to preserve).
- Verify-then-build S3 entry in `docs/measurement-log.md` (12.83 s / 59.6 s, +49% scalar, ids byte-identical 31/31).
- `docs/benchmarks.md` + `bench mem` roof (54–61 GB/s reference band).

## 5. What was done (ANALYZE only — no spike per the dead-idea rule)

1. Re-read the S3 mechanism against the current tree; confirmed `scores_` materialization still present and the kernel path unchanged since S3 (only reports + measurement-log committed since; `git status` clean throughout).
2. Verified geometry live on the 0.5B reference (`generate` header: 24 layers, 896 dim, 14/2 heads).
3. Computed both ceilings (§2) from recorded S3 numbers + roof band.
4. Gates on the unmodified tree: `cmake --build --preset release` 54/54; `tge_tests.exe` 141/141; `aikit.py check` 2/4 (test + secrets pass; hygiene + guidelines fail pre-existing, same pair as every prior cycle).
5. No fidelity run owed (no arithmetic exists to price); no bench A/B owed (no second path exists to pair); no thinking-set diff owed (same binary, zero behavior change).

## 6. Measured numbers (ceilings — no new benchmark belongs to an analytic kill)

| quantity | value | source |
|---|---|---|
| dense attn / prefill wall, 2043-tok | 12.83 s / 59.6 s (21.5%) | S3 entry, same tree |
| score traffic, full prefill | ≈5.6 GB | T²·H·4B·24 (§2) |
| KV traffic, full prefill | ≈51 GB | 2·(T²/2)·128·4B·24 (§2) |
| attn traffic rate vs roof | ≈4.4 GB/s vs 54–61 GB/s | measured, compute-bound |
| realistic ceiling | ≈0% (non-binding traffic) | §2 |
| generous ceiling | ≈2.2% end-to-end | 10% of attn × 21.5% |
| loop bar | 5% | falsifier, unmet by 2.3–5x |

## 7. Full reimplementation recipe (deliberately NOT recommended — rebuild only to re-verify the kill)

To reproduce the negative result (do not ship): implement Q(8)×K(128) online-softmax tiling in `attention_batch` with AVX2 dots + vectorized exp, dense-f32 path only; gate on 2043-token prefill attn time vs 12.83 s with byte-identical ids. Expected: ≈12.8 s (traffic wasn't binding; FLOPs unchanged), i.e. a large diff for ~0%. Revisit ONLY if the §3 revival trigger is measured on a real shape (traffic-bound attention + score share >25%); until both hold, any tiling revival re-proves the ceiling first.

## 8. Verdict and why

**KILLED analytic: the S3 retry cannot clear the bar even perfect.** Its best possible outcome (AVX2-rate parity, all score traffic gone) is bounded at ~0% realistic / 2.2% generous, because attention at this shape is compute-bound by >10x and scores are ~10% of its traffic. The loop's honest output is the ceiling, so no future cycle re-treads tiling without the §3 trigger firing first. Next-cycle candidates: S2 pre-packed-B is likewise caller-less (1325 — skip unless a production GEMM caller appears); the live front is the 1517 epic's stage-2 skip harness and the human-gated ports (mprefix-table, sparse no-gather, batchsnap kind tags).

## 9. Exact commands run and their results

- `generate --model <0.5B-Q4_K_M> --prompt "hi" --n 1` → geometry line (24 layers, 896 dim, 14/2 heads).
- Carrier re-read: `scores_` materialization + `attention_head` dense/sparse/scalar/KV arms (§4).
- `cmake --build --preset release` → 54/54; `tge_tests.exe` → 141/141; `aikit.py check aegean-tge` → 2/4 (test + secrets pass; pre-existing hygiene + guidelines fails).
- No branch created, no edits made, no benchmarks beyond the gate suite, report committed, no push.

(End of file.)
