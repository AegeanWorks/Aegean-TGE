# IDEA-1395: Variance-Normalized KV Cache with Precision Tail — KILLED WITHOUT SPIKE (dominated on the size–KL frontier by shipped i8)

- **Idea ID**: IDEA-1395 (dossier `kilo-sweep-2026-11-03.md`: KVarN per-head Hadamard post-RoPE with 128-token tiles + kv-tail-tokens exact F16/BF16 merged in one FP32 softmax; claimed Qwen3.6-27B median KLD ≈ 0.0009 at 33% size; falsifier: median KLD > 2e-3 or flips > 7/497, retraining required is reject)
- **Cycle date**: 2026-11-06
- **Verdict**: KILLED on the record, no spike built (dead-idea rule: taking the dossier's own claimed number at face value, the shipped i8 tier dominates it on both axes — 25% size at KL 3.2e-4 vs 33% at KL 9e-4. A new tier must beat the frontier, not merely approach it). No branch created, no code touched, nothing to revert.
- **Tree state at close**: `git status` clean (no diff), stash unchanged (pre-existing peer work only), `master` ahead of `origin/master` by 36 after this report. Nothing pushed (human gate).

## 1. Mechanism (as dossiered)

Variance-normalized (symmetric, scale-by-spread) low-bit KV with a per-head
Hadamard post-RoPE over 128-token tiles plus exact-precision tail tokens,
merged in one FP32 softmax.

## 2. Predicted gain — and why the frontier already contains better

| tier | size (of f32) | median/mean KL | status |
|---|---|---|---|
| f32 (reference) | 100% | ~0 | ships |
| f16 KV | 50% | 1.658e-6 | ships, kept |
| i8 KV | 25% | 3.2e-4, top-1 31/31 | ships, kept |
| KVarN (claimed) | **33%** | **≈9e-4** | dossier claim |
| int4 asymmetric (measured) | ~14% | 4.6e-3 (`6b0c84b`) | killed |
| int2 asymmetric (measured) | ~8% | 1.3e-1 (`1fc2869`) | killed |

KVarN's claimed point (33%, 9e-4) is Pareto-dominated by shipped i8 (25%,
3.2e-4): smaller AND ~3x more accurate. Building it would add a tier that
loses to an existing flag on both axes. The symmetric-scaling core is also
directly measured on this tree's tensors (RHT probe, `0b4097e`): symmetric
4-bit underperforms asymmetric everywhere (K 0.058–0.069 vs 0.033–0.046) and
Hadamard only recovers theory (~17x worse than i8) — consistent with, not
contradicting, the domination reading.

## 3. Falsifier (from the dossier) — trips on the frontier, not a bug

"Median KLD > 2e-3 or flips > 7/497" is the bar for KVarN to clear; the
deeper cut is that i8 clears every bar KVarN sets (3.2e-4 < 1e-3 << 2e-3,
31/31) at fewer bytes. Retraining-required is reject (none needed — nothing
is built).

## 4. File anchors (current `master`, read but untouched)

- `src/models/qwen2.cpp` (`kv_q8_` paths), `apps/cli/main.cpp` (`--kv i8`).
- `docs/measurement-log.md` (f16/i8/VNNI pricing precedents).
- `reports/idea-1516-kivi-stage2a-killed.md`, `idea-1385-int4kv-killed.md`,
  `idea-1330-rht-probe-killed.md` (sub-8-bit measurement record).

## 5. What was done (ANALYZE only — no spike per the dead-idea rule)

1. Placed the dossier's claimed operating point on the measured size–KL
   frontier (§2); it is dominated by a shipped flag.
2. Cross-checked the mechanism against the RHT probe's symmetric-scaling
   measurements (same transform family, worse-than-asymmetric everywhere).
3. Gates for the record: build clean, `ctest` pass, `aikit.py check` 2/4
   (test + secrets pass; hygiene + guidelines fail pre-existing).
4. No thinking-set diff owed: same binary, zero behavior change.

## 6. Measured numbers (cited — no new measurement needed or owed)

Dossier claim: KLD ≈ 9e-4 at 33% size (Qwen3.6-27B). Shipped i8: KL 3.2e-4
at 25% size (Llama-3B, 31/31). Caveat recorded honestly: different models —
but domination by 3x on KL at 0.76x the bytes leaves no plausible model
effect to reverse it (model scaling improves absolute KL, not the
tier ordering, which the int2/int4/RHT series confirms is stable).

## 7. Reimplementation recipe (deliberately NOT recommended — kill-grade)

For the record: per-head Hadamard over 128-token tiles + symmetric low-bit
RTN + exact tail + merged FP32 softmax, priced against i8 at matched bytes.
Do not build: the frontier analysis (§2) shows the ceiling before any code
— the design point loses to `--kv i8` on both axes. Revisit only with a
published size–KL point that beats shipped i8 somewhere (smaller at ≤ KL,
or same size at strictly better KL with margin).

## 8. Verdict and why

**KILLED without spike.** New tiers must beat the frontier. KVarN's claimed
point does not — it is bigger and worse than a flag that already ships. The
symmetric-scaling mechanism underneath is additionally measured-inferior to
the asymmetric status quo on this tree's own tensors.

## 9. Exact commands run and their results

- Frontier analysis (§2 table from record + dossier claim).
- `cmake --build --preset release` → 54/54; `ctest` → pass.
- `aikit.py check` → 2/4 (pre-existing fails). No branch, no edits, no push.
