# INVENTED: VNNI Asymmetric (Zero-Point) Activation Quantization — KILLED (halves coarse KL to 5.97e-4 but misses the 1e-4 bar; t/s within noise)

- **Idea**: Invented, queued by the qonly re-run (activation quant confirmed at 1.515e-3): replace symmetric per-32 activation quantization (single scale, XOR-0x80 bias) with asymmetric min/max per-32 blocks (uint8 values, fp32 scale + fp32 min offset per block; dot = Σ(q·w)·sa + min·Σw, all × sw), so one-sided activation distributions stop wasting half their dynamic range on an empty tail.
- **Cycle date**: 2026-11-06
- **Verdict**: KILLED on the KL arm of its own falsifier (5.97e-4 vs the 1e-4 bar — 6x miss); t/s guard passes (0.96x of coarse VNNI on the same 70-token prefill). Direction confirmed (2.5x better than symmetric, better than per-16 fine at 8.1e-4), magnitude insufficient: quadratic-style extrapolation of the granularity ladder says the residual is structural, not granular. Named follow-ups (not built): asymmetric+fine combined blocks (estimated ~3e-4, still above bar — likely insufficient), then per-channel scales or a different quarrel entirely (the VNNI-int-dot path itself may simply be the wrong place to spend accuracy budget, given kq8's floor on K-quant models). Spike fully reverted (5 files + test); nothing pushed (human gate).
- **Tree state at close**: `git status` clean (no diff, no untracked), stash unchanged (only the pre-existing `leftover peer work shelved 2026-11-04`, untouched). Spike branch created (zero commits) and deleted; all spike files reverted with empty diff verified. `master` ahead of `origin/master` by 75 before this report. Nothing pushed (human gate).

## 1. Mechanism (as spiked)

`quantize_q8_batch` gains a min-offset output (`min` + `scale=(max-min)/255` per 32 values, uint8 bit patterns, degenerate constant blocks exact via `min`); VNNI int dots drop the XOR-0x80 bias and instead add `min×Σw` (weight sum hoisted per block, horizontal add of lane_sums) before the per-block float combine: `acc += (idot·sa + min·sumw)·sw`. Internal fallback path (unquantized input) fills the new buffers identically. `Q8Quantized`/`Q8BatchArgs` gain a parallel `offsets` array; `quantize_activations` allocates and threads it through both matvec call sites (with null-guards so a missing buffer falls back to float, never to garbage).

## 2. Predicted gain — direction right, magnitude short

+9 flips recovered vs coarse is wrong-side reading — flips: asymmetric 992/1023 (31) vs coarse 993/1023 (30): one MORE flip (noise). The real gain is distributional: KL 1.503e-3 → 5.972e-4 (2.5x), worst-KL 1.53e-2 → 6.69e-3 (2.3x), max|diff| 0.77 → 0.46. Same flip count at 2.5x lower mean KL means the tail shrank but the decision boundary crossings didn't — the residual errors sit exactly on top-1 boundaries, which no symmetric/asymmetric tweak at this granularity will clear without moving the bar two more orders.

## 3. Falsifier outcome — KL arm fires (kill), t/s arm passes

Kill bar was KL ≤ 1e-4: measured 5.97e-4 (6x miss). t/s bar was within 5% of coarse VNNI on the same 70-token prefill: measured 225.3 vs 235.2 t/s (0.96x) — passes; decode 77.9 vs 74.5 (faster, noise). Float baseline on the same prompt: 291.9 prefill — both int paths trail float here (small-model ALU-bound window, not the load-bound regime where VNNI's 1.9x was priced), so the guard reads as "asym costs nothing vs coarse", not as a product claim.

## 4. File anchors (master = pre-spike; spike reverted)

- `src/cpu/avx2/q8_batch_vnni.cpp` (asymmetric quantizer + dot rewrite + fallback offsets — reverted).
- `include/tge/cpu/kernels.hpp` (`quantize_q8_batch` overload + `quantized_offsets` + `Q8Quantized::offsets` — reverted).
- `include/tge/models/qwen2.hpp` (`q8_offsets_` — reverted) + `src/models/qwen2.cpp` (alloc + thread-through at both call sites — reverted).
- `tests/test_q8.cpp` (`vnni_asym_matches_scalar_reference` differential, tight `1e-5·cols+1e-4` — green, reverted).
- Grounds: qonly re-run (1.515e-3 = the error to beat); finescale (0.54x per halving); double-accum (combining exonerated); replay (kernel arithmetic exonerated).

## 5. What was done (temporary spike, fully reverted)

1. `git checkout -b spike-vnni-asym` (zero commits ever made on it; deleted at close).
2. Spike (5 files + 1 test): asymmetric quantizer (this cycle's only new math); kernel rewrite (no-XOR dots + hoisted weight sums + offset term); buffer plumbing (alloc/thread/predicate); differential test with per-32-block exact dequant reference.
3. `cmake --build --preset release` clean (54/54); `ctest --preset release` pass; `tge_tests.exe` 142/142 (141 + 1 new).
4. Box health: not separately re-gated (same window class; wall 64.19 bracketing on the gates run).
5. Fidelity `--vnni` (0.5B-Q8_0, arch corpus, ctx 1024, 6 threads, gpu off): §6 row 1.
6. t/s guard (70-token fox prompt, 0.5B-Q8_0): asym 225.3/77.9 vs coarse 235.2/74.5 (stash-built baseline) vs float 291.9/77.0; story ids byte-identical to the f32 baseline; reasoning fluent.
7. Reverted all 5 files (`git checkout --`, empty diff verified), `git checkout master`, `git branch -D spike-vnni-asym`; final gates on reverted master: build clean, `ctest` pass, `tge_tests` 141/141; `python aikit.py check aegean-tge` 2/4 (test PASS, no flake; hygiene + guidelines fail pre-existing).

## 6. Measured numbers (0.5B-Q8_0, arch corpus, ctx 1024, 6 threads, gpu off)

| run | KL mean | worst KL | max\|diff\| | top-1 | wall |
|---|---|---|---|---|---|
| coarse `--vnni` (on record) | 1.503e-03 | 1.533e-02 | 0.77 | 993/1023 | 47.52 s |
| asymmetric `--vnni` (spike) | 5.972e-04 | 6.687e-03 | 0.46 | 992/1023 | 32.51 s |
| dense floor (reference arm) | 9.248e-12 | 2.418e-10 | 1.9e-04 | 1023/1023 | — |

70-token prefill t/s (same prompt, 6 threads): float 291.9 → coarse 235.2 (0.81x) → asymmetric 225.3 (0.77x float, 0.96x coarse). Decode: 77.0 → 74.5 → 77.9 (noise). Story short-prompt ids identical across all three (the flips live in the natural-text corpus, not toy prompts).

## 7. Full reimplementation recipe (kill-grade — rebuild to re-verify, then STOP unless the bar moves)

Rebuild: §5.2 (asymmetric quantizer with degenerate-constant handling; VNNI dot without XOR plus hoisted per-block weight sums + offset term; parallel offset buffers through `quantize_activations` into both matvec call sites with null-guards; differential test with exact dequant reference). Gate: build 54/54, 142/142 tests (differential must pass with the tight bound — it guards the integer math, which is exact by construction), fidelity `--vnni` reproduces §6 row 2 within run noise (and still misses 1e-4 — that IS the expected result). Do NOT extend to asymmetric+fine (estimated ~3e-4, still misses; the ladder says stop). Revisit ONLY if the bar itself is renegotiated (e.g., "beat coarse by ≥2x at t/s parity" instead of an absolute KL — asymmetric clears that bar cleanly), or if per-channel scales get their own falsifier.

## 8. Verdict and why

**KILLED on the KL arm (6x miss) with the t/s arm passing.** Asymmetric quantization is confirmed as the single biggest VNNI-accuracy lever measured so far (2.5x, bigger than finer blocks at 1.9x), and it is also confirmed as insufficient: the remaining error needs a different shape of solution (per-channel adaptation, or conceding that int8 activations on this path cannot reach keep-grade and spending the budget on kq8-style integer paths for Q8_0 instead). The VNNI-accuracy program's honest state after eleven related cycles: quantizer identified (this report), dots exonerated (replay), combining exonerated (double-accum), layout mapped (bisect/layergate), distribution characterized (statcorr/dump), next step ambiguous — which is itself a finding worth recording: stop digging here; the bar needs renegotiating before more spikes.

## 9. Exact commands run and their results

- `git checkout -b spike-vnni-asym` → branch (zero commits, deleted at close).
- 5-file spike + 1 test (§5.2; zero mid-spike bugs — the 8x-broadcast lesson from the finescale round was applied pre-emptively with scalar combining).
- `cmake --build --preset release` → 54/54; `ctest --preset release` → pass; `tge_tests.exe` → 142/142 on spike, 141/141 after revert.
- Wall 64.19 bracketing; `aikit.py check` → 2/4 (test PASS, no flake; pre-existing hygiene + guidelines fails).
- Fidelity `--vnni` (asym) → KL 5.972e-04, worst 6.687e-03, 992/1023 (32.51 s).
- 70-token story ×3 (float 291.9/77.0; coarse 235.2/74.5 via stash dance; asym 225.3/77.9); asym story ids == f32 baseline; reasoning fluent.
- `git checkout -- <5 files>` → clean; `git checkout master`; `git branch -D`; final master gates green; report committed; no push.

(End of file.)
