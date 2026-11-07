# IDEA-DECODE-INT ROUND 4: Activation Granularity Probe — PROCEED dot kernels (kernel exact, per-32 wins 2.6x)

- **Idea**: Follow-up to the round-3 kill-confirmation, which isolated the defect to activation quantization (broad heavy tails vs per-256-block scales) but left two questions: (a) is the kq8d dot kernel itself exact, and (b) does finer activation granularity actually fix the roundtrip on live data? Both answered here with temp in-matvec probes (reverted).
- **Cycle date**: 2026-11-07
- **Verdict**: PROCEED WITH PER-32 DOT KERNELS. Kernel exonerated (bit-exact vs exact arithmetic, rel max 6.4e-06); per-32 symmetric activation quant cuts roundtrip error 2.9% → 1.1% p99 on the same live rows. The activation error is isolated as the dominant term (weights identical both arms). Prescribed next: per-32 activation dot kernels (Q4_K/Q6_K × Q8-per32), gated by row-level rel p99 < 2% + fidelity ≤ 1.6e-3 with top-1 held + the bench pulse re-verified.
- **Tree state at close**: clean, no diff, stash untouched (pre-existing entry only). No throwaway branch (spike developed in-tree, reverted via `git checkout --`, verified clean + rebuilt green). Nothing pushed (human gate).

## 1. Measured numbers (0.5B-Q4_K_M, story prompt, n=16, live decode rows)

Quantizer roundtrip (360 gate/up rows, cols 4864, amax ≤ 36.2):
- Real kq8d quantizer (per-256-block 8-bit, signed-amax, mantissa rounding — replicated exactly): rel max 0.030, p99 0.029.
- Per-32 symmetric candidate: rel max 0.012, p99 0.011.
- Mechanism note: an earlier pass of this probe used a fictional 2-bit baseline (rel p99 189%) — caught by comparing against round 3's row-level numbers, corrected to the faithful replica above. Never trust a straw-man baseline; the replica must reproduce the fielded behavior first (here: 2.9% vs round 3's row-level 10% — same order, dot amplification expected).

Kernel-vs-exact (same rows; float dot of dequantized values vs kq8d row output, 168 samples): rel max 6.4e-06, p99 4.0e-06, worst abs 7.6e-06. The kernel is exact; all row-level error is quantization, none is arithmetic.

## 2. Falsifier outcome — proceed, with the bar set for the build

The probe bars were: (a) kernel-vs-exact rel p99 < 1e-4 — MET (4e-6, 25x margin); (b) per-32 candidate p99 < 2% with the fielded quantizer ≥ 2x worse — MET (1.1% vs 2.9%). The dot-kernel build is green-lit with its own falsifier (stated in §5).

## 3. What remains (explicit prescription, not attempted here)

Per-32 activation dot kernels: new 8-bit activation layout (per-32 scales), Q4_K/Q6_K × Q8-per32 maddubs dots mirroring the existing lane structure, temp bench-flag wiring first. Gate, in order: (1) unit differential vs float on synthetic + captured-real rows (exact-equality where deterministic); (2) the row-level capture harness from round 3: rel p99 < 0.02 (vs today's 0.10); (3) model fidelity ≤ 1.6e-3 with top-1 1023/1023 (the bar round 2 set); (4) kq8d bench pulse re-verified + end-to-end decode A/B vs float. Kill the build at the first gate that fails — no fidelity run is owed for a kernel that loses its differential.

## 4. What was done (temporary, fully reverted)

1. Temp dual probe in `Qwen2::matvec` (env `TGE_KQ8D_QGRAIN=1`): quantizer roundtrip comparison + kernel-vs-exact reference dots (~150 lines, reverted).
2. Build 54/54, `tge_tests` 145/145 with the probe compiled in; story generate runs for capture.
3. `git checkout -- src/models/qwen2.cpp` → empty diff verified; rebuild 54/54; tests 145/145; aikit 2/4 (pre-existing fails).
4. Report committed; no push.

## 5. Exact commands run and their results

- Probe spike added; build 54/54; suite 145/145.
- `TGE_KQ8D_QGRAIN=1 tge generate --model 0.5B-Q4_K_M --prompt "Once upon a time" --n 16` → §1 tables (360 + 168 samples).
- Revert → empty diff; rebuild 54/54; tests 145/145.
- No fidelity run owed (round-2 numbers stand: KL 7.35e-4, 1013/1023; this probe is differential-only by design).

(End of file.)
